"""SYNAPSE DB — a única base de dados do MIND.

Princípio fundamental: TUDO local. Nada sai do hardware ou rede do
utilizador — sem upload cloud, sem transmissão externa de nenhum tipo.

SQLite em modo WAL (obrigatório desde o início). Acessível por CORTEX,
CEREBELLUM e todos os NEURONS. Dupla função:
  1. Operacional — estado do ciclo, código, relatórios, decisões
  2. Manutenção/retreino — exportação filtrada em JSONL para fine-tuning
"""

import json
import os
import sqlite3
import threading
from typing import Optional

# --- Schema completo -------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'in_progress',
    final_functionality_pct REAL
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    phase TEXT NOT NULL,
    component TEXT NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    full_output TEXT,
    duration_seconds REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    functionality_pct REAL,
    failures TEXT,
    improvements TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    component TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);
"""

# --- Schema da extensão HIPPOCAMPUS (camada de apoio de ML) ---------------
# Criado sempre, mesmo com ML_ENABLED=false — ter as tabelas vazias não custa
# nada e permite acumular histórico assim que a camada for activada.
ML_SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    consumer TEXT NOT NULL,          -- 'cortex' ou 'cerebellum'
    cycle_id INTEGER,
    features TEXT NOT NULL,          -- JSON
    label REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);

CREATE TABLE IF NOT EXISTS ml_model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    consumer TEXT NOT NULL,
    model_path TEXT NOT NULL,
    validation_metric REAL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 0,
    training_samples_count INTEGER
);

CREATE TABLE IF NOT EXISTS ml_predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL,
    consumer TEXT NOT NULL,
    prediction REAL,
    llm_final_decision TEXT,         -- mede concordância ML vs LLM
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
);
"""


class SynapseDB:
    """Camada de acesso à SYNAPSE DB.

    Thread-safe através de um lock — os backups e o ciclo podem tocar na
    base concorrentemente. O modo WAL permite leituras concorrentes com
    escritas, mas serializamos as escritas por segurança.
    """

    def __init__(self, db_path: str = "synapse.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        # `check_same_thread=False` porque o backup corre noutra thread.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Activa WAL e cria o schema se ainda não existir."""
        with self._lock:
            # Modo WAL obrigatório — melhor concorrência e recuperação.
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(SCHEMA)
            self._conn.executescript(ML_SCHEMA)
            self._conn.commit()

    # --- Ciclos -----------------------------------------------------------
    def create_cycle(self, task: str) -> int:
        """Cria um ciclo novo e devolve o seu id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO cycles (task) VALUES (?)", (task,)
            )
            self._conn.commit()
            return cur.lastrowid

    def update_cycle(
        self,
        cycle_id: int,
        status: Optional[str] = None,
        final_pct: Optional[float] = None,
        task: Optional[str] = None,
    ) -> None:
        """Actualiza estado, percentagem final e/ou tarefa de um ciclo."""
        sets, params = [], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if final_pct is not None:
            sets.append("final_functionality_pct = ?")
            params.append(final_pct)
        if task is not None:
            sets.append("task = ?")
            params.append(task)
        if not sets:
            return
        params.append(cycle_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE cycles SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()

    def get_cycle(self, cycle_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cycles WHERE id = ?", (cycle_id,)
            ).fetchone()
            return dict(row) if row else None

    # --- Iterações --------------------------------------------------------
    def log_iteration(
        self,
        cycle_id: int,
        iteration_number: int,
        phase: str,
        component: str,
        input_summary: str = "",
        output_summary: str = "",
        full_output: str = "",
        duration_seconds: float = 0.0,
    ) -> int:
        """Regista um passo de um componente numa fase/iteração."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO iterations
                   (cycle_id, iteration_number, phase, component,
                    input_summary, output_summary, full_output,
                    duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (cycle_id, iteration_number, phase, component,
                 input_summary, output_summary, full_output,
                 duration_seconds),
            )
            self._conn.commit()
            return cur.lastrowid

    # --- Relatórios -------------------------------------------------------
    def log_report(
        self,
        cycle_id: int,
        iteration_number: int,
        functionality_pct: float,
        failures: str = "",
        improvements: str = "",
    ) -> int:
        """Regista um relatório de funcionalidade (CORTEX ou CEREBELLUM)."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO reports
                   (cycle_id, iteration_number, functionality_pct,
                    failures, improvements)
                   VALUES (?, ?, ?, ?, ?)""",
                (cycle_id, iteration_number, functionality_pct,
                 failures, improvements),
            )
            self._conn.commit()
            return cur.lastrowid

    # --- Decisões ---------------------------------------------------------
    def log_decision(
        self,
        cycle_id: int,
        iteration_number: int,
        component: str,
        decision_text: str,
    ) -> int:
        """Regista uma decisão relevante (texto conciso, uma frase).

        Usado pelo CORTEX para o registo estruturado do raciocínio de
        distribuição — permite auditar e depurar por que motivo cada
        NEURON foi (ou não) chamado numa ronda.
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO decisions
                   (cycle_id, iteration_number, component, decision_text)
                   VALUES (?, ?, ?, ?)""",
                (cycle_id, iteration_number, component, decision_text),
            )
            self._conn.commit()
            return cur.lastrowid

    # --- Exportação para fine-tuning -------------------------------------
    def export_to_jsonl(
        self,
        cycle_id: int,
        component: Optional[str] = None,
        output_path: str = "datasets/export.jsonl",
    ) -> int:
        """Filtra iterações por cycle_id e, opcionalmente, por componente.

        Gera JSONL pronto a usar como dataset de fine-tuning. Cada linha é
        um par prompt/completion derivado do input/output registado.
        Devolve o número de registos exportados.
        """
        query = (
            "SELECT * FROM iterations WHERE cycle_id = ?"
        )
        params: list = [cycle_id]
        if component:
            query += " AND component = ?"
            params.append(component)
        query += " ORDER BY iteration_number, id"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for row in rows:
                record = {
                    "cycle_id": row["cycle_id"],
                    "iteration": row["iteration_number"],
                    "phase": row["phase"],
                    "component": row["component"],
                    "prompt": row["input_summary"] or "",
                    "completion": row["full_output"] or row["output_summary"] or "",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ======================================================================
    # HIPPOCAMPUS — camada de apoio de ML
    # ======================================================================
    def record_ml_sample(
        self,
        consumer: str,
        cycle_id: Optional[int],
        features: dict,
        label: Optional[float],
    ) -> int:
        """Guarda uma amostra de treino (features JSON + label).

        Acumula histórico operacional mesmo com ML_ENABLED=false — é este
        volume que, mais tarde, permite treinar o HIPPOCAMPUS.
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO ml_training_data
                   (consumer, cycle_id, features, label)
                   VALUES (?, ?, ?, ?)""",
                (consumer, cycle_id, json.dumps(features, ensure_ascii=False), label),
            )
            self._conn.commit()
            return cur.lastrowid

    def count_ml_samples(self, consumer: str, labelled_only: bool = True) -> int:
        """Conta amostras disponíveis para um consumidor."""
        query = "SELECT COUNT(*) AS n FROM ml_training_data WHERE consumer = ?"
        if labelled_only:
            query += " AND label IS NOT NULL"
        with self._lock:
            row = self._conn.execute(query, (consumer,)).fetchone()
            return row["n"] if row else 0

    def get_ml_samples(self, consumer: str, labelled_only: bool = True) -> list:
        """Devolve as amostras de treino de um consumidor."""
        query = "SELECT * FROM ml_training_data WHERE consumer = ?"
        if labelled_only:
            query += " AND label IS NOT NULL"
        query += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(query, (consumer,)).fetchall()
        out = []
        for row in rows:
            try:
                features = json.loads(row["features"])
            except (ValueError, TypeError):
                continue
            out.append(
                {
                    "id": row["id"],
                    "cycle_id": row["cycle_id"],
                    "features": features,
                    "label": row["label"],
                    "created_at": row["created_at"],
                }
            )
        return out

    def register_ml_model(
        self,
        consumer: str,
        model_path: str,
        validation_metric: float,
        training_samples_count: int,
        activate: bool = False,
    ) -> int:
        """Regista uma versão de modelo treinado.

        Se `activate` for True, desactiva as versões anteriores desse
        consumidor — a promoção é sempre por comparação explícita, feita
        pelo ml_pipeline, nunca por omissão.
        """
        with self._lock:
            if activate:
                self._conn.execute(
                    "UPDATE ml_model_versions SET is_active = 0 WHERE consumer = ?",
                    (consumer,),
                )
            cur = self._conn.execute(
                """INSERT INTO ml_model_versions
                   (consumer, model_path, validation_metric,
                    is_active, training_samples_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (consumer, model_path, validation_metric,
                 1 if activate else 0, training_samples_count),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_active_ml_model(self, consumer: str) -> Optional[dict]:
        """Devolve o modelo activo de um consumidor, ou None (cold start)."""
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM ml_model_versions
                   WHERE consumer = ? AND is_active = 1
                   ORDER BY id DESC LIMIT 1""",
                (consumer,),
            ).fetchone()
            return dict(row) if row else None

    def list_ml_models(self, consumer: Optional[str] = None) -> list:
        """Lista as versões de modelo registadas (para o comando ml-status)."""
        query = "SELECT * FROM ml_model_versions"
        params: list = []
        if consumer:
            query += " WHERE consumer = ?"
            params.append(consumer)
        query += " ORDER BY consumer, id DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def log_ml_prediction(
        self,
        cycle_id: int,
        iteration_number: int,
        consumer: str,
        prediction: Optional[float],
        llm_final_decision: str = "",
    ) -> int:
        """Regista uma previsão do HIPPOCAMPUS e a decisão final do LLM.

        Permite medir a concordância ML vs LLM ao longo do tempo — base para
        ML_RETRAIN_DEVIATION_THRESHOLD.
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO ml_predictions_log
                   (cycle_id, iteration_number, consumer, prediction,
                    llm_final_decision)
                   VALUES (?, ?, ?, ?, ?)""",
                (cycle_id, iteration_number, consumer, prediction,
                 llm_final_decision),
            )
            self._conn.commit()
            return cur.lastrowid

    def ml_prediction_deviation(self, consumer: str, limit: int = 50) -> Optional[float]:
        """Desvio médio (em pontos) entre previsão do ML e decisão do LLM.

        Usa as últimas `limit` previsões cuja decisão do LLM foi registada
        como valor numérico. Devolve None se não houver dados comparáveis.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT prediction, llm_final_decision
                   FROM ml_predictions_log
                   WHERE consumer = ? AND prediction IS NOT NULL
                   ORDER BY id DESC LIMIT ?""",
                (consumer, limit),
            ).fetchall()
        deltas = []
        for row in rows:
            try:
                actual = float(row["llm_final_decision"])
            except (TypeError, ValueError):
                continue
            deltas.append(abs(float(row["prediction"]) - actual))
        return sum(deltas) / len(deltas) if deltas else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
