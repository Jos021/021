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

    def close(self) -> None:
        with self._lock:
            self._conn.close()
