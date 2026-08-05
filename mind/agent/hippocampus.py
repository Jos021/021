"""HIPPOCAMPUS — camada autónoma de apoio de Machine Learning.

Segue a metáfora cerebral: o hipocampo é a região da memória e da
aprendizagem. É um serviço consultado on-demand, treinado sobre o histórico
operacional da SYNAPSE DB — NÃO está embutido no CORTEX nem no CEREBELLUM.

Princípios inegociáveis
-----------------------
1. Sempre consultivo — o LLM mantém a palavra final sobre aprovação.
2. Assimetria de segurança — pode apoiar auto-REJEIÇÃO, NUNCA auto-APROVAÇÃO.
3. Tudo local — treina e corre localmente, sem excepção.
4. Cold-start seguro — sem dados suficientes (ML_MIN_TRAINING_SAMPLES),
   consult() devolve None e o sistema funciona como se o HIPPOCAMPUS não
   existisse. NUNCA lança erro por falta de modelo.
5. Promoção por comparação — modelo novo só substitui o activo se for igual
   ou melhor num conjunto de validação separado (ver ml_pipeline.py).

O HIPPOCAMPUS NÃO ajusta parâmetros de controlo do agente (MAX_ITERATIONS,
thresholds) — excluído deliberadamente por ser arriscado com modelos
imaturos. Só apoia o raciocínio nos momentos de trabalho cognitivo pesado.
"""

import os

from .ml_features import vectorize

# Consumidores válidos e a respectiva secção do ml_config.yaml.
CONSUMERS = {
    "cortex": "cortex_support",
    "cerebellum": "cerebellum_support",
}


def ml_enabled() -> bool:
    """True só se a camada estiver explicitamente activada no .env."""
    return os.getenv("ML_ENABLED", "false").lower() == "true"


class Hippocampus:
    """Serviço de apoio de ML, consultado on-demand pelo CORTEX e CEREBELLUM.

    Carrega o modelo activo de cada consumidor. Se não houver modelo
    (cold start), consult() devolve None sem erro.
    """

    def __init__(self, db, config: dict = None, models_dir: str = "models/hippocampus"):
        self.db = db
        self.config = config or {}
        self.models_dir = models_dir
        self.min_samples = int(os.getenv("ML_MIN_TRAINING_SAMPLES", "100"))
        self.auto_reject_confidence = float(
            os.getenv("ML_AUTO_REJECT_CONFIDENCE", "0.9")
        )
        self._cache: dict = {}   # {consumer: (model_path, modelo carregado)}

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def consult(self, consumer: str, features: dict) -> dict | None:
        """consumer: 'cortex' ou 'cerebellum'.

        Devolve previsão/sugestão, ou None se não houver modelo activo.
        NUNCA lança excepção por ausência de modelo — qualquer falha
        (dependência em falta, ficheiro corrompido, features inesperadas)
        degrada silenciosamente para None.
        """
        try:
            return self._consult_inner(consumer, features)
        except Exception:
            # Princípio 4: nunca deixar a camada de ML derrubar o ciclo.
            return None

    def _consult_inner(self, consumer: str, features: dict) -> dict | None:
        if not ml_enabled():
            return None
        if consumer not in CONSUMERS:
            return None
        # Cold-start: sem volume mínimo de histórico não há consulta.
        if self.db.count_ml_samples(consumer) < self.min_samples:
            return None

        model = self._load_model(consumer)
        if model is None:
            return None

        section = self.config.get(CONSUMERS[consumer], {}) or {}
        feature_names = section.get("features", sorted(features))
        vector = vectorize(features, feature_names)
        if not vector:
            return None

        record = self.db.get_active_ml_model(consumer) or {}

        if consumer == "cerebellum":
            return self._consult_cerebellum(model, vector, section, record)
        return self._consult_cortex(model, vector, record)

    def _consult_cortex(self, model, vector: list, record: dict) -> dict | None:
        """Apoio ao CORTEX: que abordagem funcionou em tarefas semelhantes.

        Regressão sobre a % de funcionalidade histórica. Puramente
        informativo — nunca decide nada.
        """
        predicted = float(model.predict([vector])[0])
        predicted = max(0.0, min(100.0, predicted))
        return {
            "consumer": "cortex",
            "prediction": predicted,
            "confidence": None,
            "auto_reject": False,          # o CORTEX nunca auto-rejeita
            "model_path": record.get("model_path"),
            "suggestion": (
                "Tarefas semelhantes no histórico atingiram cerca de "
                f"{predicted:.0f}% de funcionalidade. "
                + (
                    "Histórico favorável — abordagem semelhante tende a passar."
                    if predicted >= 80
                    else "Histórico difícil — vale reforçar validação e testes "
                         "nas zonas críticas."
                )
            ),
        }

    def _consult_cerebellum(
        self, model, vector: list, section: dict, record: dict
    ) -> dict | None:
        """Apoio ao CEREBELLUM: zonas de risco e probabilidade de passar.

        Classificação binária. `prediction` é a probabilidade de APROVAR;
        a confiança de falha (1 - prediction) é o que pode desencadear
        auto-REJEIÇÃO. Nunca o inverso.
        """
        prob_pass = self._predict_proba(model, vector)
        if prob_pass is None:
            return None
        prob_fail = 1.0 - prob_pass

        # O threshold do ml_config tem precedência sobre o do .env, se existir.
        threshold = float(
            section.get("auto_reject_confidence", self.auto_reject_confidence)
        )
        auto_reject = prob_fail >= threshold

        return {
            "consumer": "cerebellum",
            "prediction": prob_pass,
            "confidence": prob_fail,
            "auto_reject": auto_reject,
            "threshold": threshold,
            "model_path": record.get("model_path"),
            "reason": (
                f"Padrão de falha conhecido detectado com confiança "
                f"{prob_fail:.2f} >= {threshold:.2f}."
                if auto_reject
                else f"Probabilidade histórica de passar: {prob_pass:.2f}."
            ),
            "suggestion": (
                "Zonas com padrões de risco conhecidos — auditar com atenção."
                if prob_fail >= 0.5
                else "Sem padrões de risco relevantes no histórico."
            ),
        }

    @staticmethod
    def _predict_proba(model, vector: list):
        """Probabilidade da classe positiva (aprovar), se o modelo a expuser."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([vector])[0]
            # A classe positiva é a última coluna nos estimadores sklearn/xgboost.
            return float(proba[-1])
        if hasattr(model, "predict"):
            return float(max(0.0, min(1.0, model.predict([vector])[0])))
        return None

    # ------------------------------------------------------------------
    # Sandbox evolutiva — recuperação de testes de ciclos anteriores
    # ------------------------------------------------------------------
    def recommend_tests(
        self,
        task_embedding: list,
        limit: int = 8,
        min_similarity: float = 0.75,
    ) -> list:
        """Testes mais relevantes de ciclos anteriores APROVADOS.

        Consulta a test_library por similaridade de embedding. Se
        ML_ENABLED=false, se não houver modelo activo, ou se task_embedding
        for None, delega directamente em db.get_tests_by_embedding() com o
        embedding determinístico — a consulta por embedding funciona
        independentemente de ML_ENABLED, e é isso que permite a biblioteca
        crescer desde o primeiro ciclo, antes de haver camada de ML.

        CAMINHO FUTURO: quando o HIPPOCAMPUS tiver modelo de ML activo com
        histórico suficiente, esta chamada passará a incluir também um
        ranking por qualidade histórica (taxa de pass/fail em tarefas
        semelhantes), e não apenas por similaridade de embedding. Os campos
        times_used/times_passed/times_failed da test_library existem
        precisamente para alimentar esse ranking quando chegar a altura.

        Nunca lança excepção. Se algo falhar, devolve lista vazia.
        """
        try:
            if not task_embedding:
                return []
            return self.db.get_tests_by_embedding(
                task_embedding, min_similarity, limit
            )
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Carregamento de modelos
    # ------------------------------------------------------------------
    def _load_model(self, consumer: str):
        """Carrega (com cache) o modelo activo do consumidor, ou None."""
        record = self.db.get_active_ml_model(consumer)
        if not record:
            return None
        path = record.get("model_path")
        if not path or not os.path.exists(path):
            return None

        cached_path, cached_model = self._cache.get(consumer, (None, None))
        if cached_path == path and cached_model is not None:
            return cached_model

        model = _load_from_disk(path)
        if model is not None:
            self._cache[consumer] = (path, model)
        return model

    def invalidate_cache(self, consumer: str = None) -> None:
        """Esquece o modelo em cache (usado após uma promoção)."""
        if consumer:
            self._cache.pop(consumer, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Registo de amostras e previsões
    # ------------------------------------------------------------------
    def record_sample(
        self, consumer: str, cycle_id: int, features: dict, label: float = None
    ) -> None:
        """Acumula uma amostra de treino. Silencioso em caso de falha.

        Corre mesmo com ML_ENABLED=false: é assim que o histórico necessário
        ao primeiro treino se acumula sem a camada estar activa.
        """
        try:
            self.db.record_ml_sample(consumer, cycle_id, features, label)
        except Exception:
            pass

    def log_prediction(
        self,
        cycle_id: int,
        iteration: int,
        consumer: str,
        prediction: float,
        llm_final_decision: str = "",
    ) -> None:
        """Regista a previsão e a decisão do LLM (concordância ML vs LLM)."""
        try:
            self.db.log_ml_prediction(
                cycle_id, iteration, consumer, prediction, llm_final_decision
            )
        except Exception:
            pass


def _load_from_disk(path: str):
    """Carrega um modelo serializado (joblib, com fallback para pickle)."""
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        pass
    try:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def load_ml_config(path: str = "config/ml_config.yaml") -> dict:
    """Lê o ml_config.yaml. Devolve {} se não existir ou for inválido."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}
