"""Pipeline de treino do HIPPOCAMPUS — treino, validação e promoção.

Job periódico (APScheduler, ML_TRAINING_INTERVAL_HOURS) ou manual via
`python main.py ml-train --force`:

  1. Verificar amostras suficientes desde o último treino
  2. Exportar ml_training_data (split treino/validação)
  3. Treinar modelo (scikit-learn / xgboost, conforme config/ml_config.yaml)
  4. Avaliar em validação
  5. SE métrica >= modelo activo: promover; SENÃO: manter o activo
  6. Tudo local — nada sai do hardware do utilizador

A promoção é SEMPRE por comparação explícita, nunca por omissão: um modelo
novo pior que o activo é registado (para histórico) mas não é activado.
"""

import os
import time
from datetime import datetime

from .hippocampus import CONSUMERS, load_ml_config

# Fracção reservada para o conjunto de validação separado.
VALIDATION_SPLIT = 0.25


class MLPipeline:
    """Treina, valida e promove os modelos do HIPPOCAMPUS."""

    def __init__(
        self,
        db,
        config: dict = None,
        models_dir: str = "models/hippocampus",
        hippocampus=None,
    ):
        self.db = db
        self.config = config if config is not None else load_ml_config()
        self.models_dir = models_dir
        self.hippocampus = hippocampus
        self.min_samples = int(os.getenv("ML_MIN_TRAINING_SAMPLES", "100"))
        self.interval_hours = int(os.getenv("ML_TRAINING_INTERVAL_HOURS", "24"))
        self.deviation_threshold = float(
            os.getenv("ML_RETRAIN_DEVIATION_THRESHOLD", "20")
        )
        os.makedirs(models_dir, exist_ok=True)
        self._scheduler = None

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def train(self, consumer: str, force: bool = False) -> dict:
        """Treina um consumidor. Devolve um relatório estruturado.

        Nunca lança: qualquer falha (dependências em falta, dados
        insuficientes) devolve um relatório com `trained=False` e a razão.
        """
        if consumer not in CONSUMERS:
            return {"consumer": consumer, "trained": False,
                    "reason": "consumidor desconhecido"}

        samples = self.db.get_ml_samples(consumer)
        n = len(samples)

        # 1. Amostras suficientes?
        if n < self.min_samples and not force:
            return {
                "consumer": consumer, "trained": False, "samples": n,
                "reason": f"amostras insuficientes ({n} < {self.min_samples})",
            }
        if n < 4:
            return {
                "consumer": consumer, "trained": False, "samples": n,
                "reason": f"amostras insuficientes para split treino/validação ({n})",
            }

        section = self.config.get(CONSUMERS[consumer], {}) or {}
        feature_names = section.get("features", [])

        # 2. Split treino/validação (separado, determinístico).
        try:
            X, y = self._build_matrix(samples, feature_names, consumer)
        except Exception as exc:
            return {"consumer": consumer, "trained": False,
                    "reason": f"falha a montar features: {exc}"}
        if not X:
            return {"consumer": consumer, "trained": False,
                    "reason": "sem features utilizáveis"}

        split = max(1, int(len(X) * (1 - VALIDATION_SPLIT)))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        if not X_val:
            return {"consumer": consumer, "trained": False,
                    "reason": "conjunto de validação vazio"}

        # 3. Treinar.
        try:
            model = self._build_model(consumer, section)
            if model is None:
                return {"consumer": consumer, "trained": False,
                        "reason": "dependências de ML não instaladas "
                                  "(ver requirements-ml.txt)"}
            model.fit(X_train, y_train)
        except Exception as exc:
            return {"consumer": consumer, "trained": False,
                    "reason": f"falha no treino: {exc}"}

        # 4. Avaliar no conjunto de validação separado.
        try:
            metric = self._evaluate(consumer, model, X_val, y_val)
        except Exception as exc:
            return {"consumer": consumer, "trained": False,
                    "reason": f"falha na validação: {exc}"}

        # 5. Promoção por comparação — nunca por omissão.
        active = self.db.get_active_ml_model(consumer)
        active_metric = (active or {}).get("validation_metric")
        promote = active_metric is None or metric >= float(active_metric)

        path = self._save_model(consumer, model)
        if path is None:
            return {"consumer": consumer, "trained": False,
                    "reason": "falha a serializar o modelo"}

        self.db.register_ml_model(
            consumer=consumer,
            model_path=path,
            validation_metric=metric,
            training_samples_count=len(X_train),
            activate=promote,
        )
        if promote and self.hippocampus is not None:
            self.hippocampus.invalidate_cache(consumer)

        return {
            "consumer": consumer,
            "trained": True,
            "samples": n,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "metric": metric,
            "previous_metric": active_metric,
            "promoted": promote,
            "model_path": path,
            "reason": (
                "promovido: métrica igual ou melhor que o modelo activo"
                if promote
                else "mantido o modelo activo: métrica inferior"
            ),
        }

    def train_all(self, force: bool = False) -> list:
        """Treina todos os consumidores."""
        return [self.train(c, force=force) for c in CONSUMERS]

    # ------------------------------------------------------------------
    # Construção de dados e modelos
    # ------------------------------------------------------------------
    def _build_matrix(self, samples: list, feature_names: list, consumer: str):
        """Constrói X (features) e y (labels) a partir das amostras."""
        from .ml_features import vectorize

        X, y = [], []
        for sample in samples:
            features = sample["features"]
            names = feature_names or sorted(features)
            vector = vectorize(features, names)
            if not vector:
                continue
            label = sample["label"]
            if label is None:
                continue
            if consumer == "cerebellum":
                # Classificação: aprovado (1) vs reprovado (0).
                threshold = float(os.getenv("APPROVAL_THRESHOLD", "98"))
                label = 1 if float(label) >= threshold else 0
            X.append(vector)
            y.append(label)

        # Todos os vectores têm de ter o mesmo comprimento (embeddings podem
        # variar se a dependência mudou entre execuções) — normalizamos.
        if X:
            width = min(len(v) for v in X)
            X = [v[:width] for v in X]
        return X, y

    def _build_model(self, consumer: str, section: dict):
        """Instancia o estimador conforme o ml_config.yaml.

        cortex_support     -> regressão
        cerebellum_support -> classificação (expõe predict_proba)
        """
        model_type = (section.get("model_type") or "random_forest").lower()
        n_estimators = int(section.get("n_estimators", 100))
        max_depth = int(section.get("max_depth", 10))
        is_classifier = consumer == "cerebellum"

        if model_type == "xgboost":
            try:
                import xgboost as xgb

                cls = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
                return cls(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    verbosity=0,
                )
            except Exception:
                # Sem xgboost, cai para a floresta do scikit-learn.
                model_type = "random_forest"

        if model_type == "random_forest":
            try:
                from sklearn.ensemble import (
                    RandomForestClassifier,
                    RandomForestRegressor,
                )

                cls = RandomForestClassifier if is_classifier else RandomForestRegressor
                return cls(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                )
            except Exception:
                return None
        return None

    def _evaluate(self, consumer: str, model, X_val: list, y_val: list) -> float:
        """Métrica de validação: acurácia (classificação) ou R² (regressão)."""
        preds = model.predict(X_val)
        if consumer == "cerebellum":
            correct = sum(
                1 for p, actual in zip(preds, y_val) if int(round(float(p))) == int(actual)
            )
            return float(correct) / len(y_val)
        # Regressão: R² calculado à mão (evita depender de sklearn.metrics).
        mean = sum(y_val) / len(y_val)
        ss_tot = sum((actual - mean) ** 2 for actual in y_val)
        ss_res = sum((actual - float(p)) ** 2 for p, actual in zip(preds, y_val))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    def _save_model(self, consumer: str, model):
        """Serializa o modelo versionado em models/hippocampus/."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.models_dir, f"{consumer}_{stamp}.joblib")
        try:
            import joblib

            joblib.dump(model, path)
            return path
        except Exception:
            pass
        try:
            import pickle

            path = path.replace(".joblib", ".pkl")
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
            return path
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Estado e exportação
    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Estado dos modelos, amostras e concordância ML vs LLM."""
        from .hippocampus import ml_enabled

        out = {
            "ml_enabled": ml_enabled(),
            "min_training_samples": self.min_samples,
            "training_interval_hours": self.interval_hours,
            "deviation_threshold": self.deviation_threshold,
            "consumers": {},
        }
        for consumer in CONSUMERS:
            active = self.db.get_active_ml_model(consumer)
            samples = self.db.count_ml_samples(consumer)
            deviation = self.db.ml_prediction_deviation(consumer)
            out["consumers"][consumer] = {
                "samples": samples,
                "cold_start": samples < self.min_samples,
                "active_model": (active or {}).get("model_path"),
                "validation_metric": (active or {}).get("validation_metric"),
                "trained_at": (active or {}).get("trained_at"),
                "versions": len(self.db.list_ml_models(consumer)),
                "mean_deviation_vs_llm": deviation,
                "needs_retrain": (
                    deviation is not None and deviation > self.deviation_threshold
                ),
            }
        return out

    def export(self, consumer: str, output_path: str) -> int:
        """Exporta as amostras de um consumidor para CSV. Devolve nº de linhas."""
        import csv

        from .ml_features import vectorize

        samples = self.db.get_ml_samples(consumer, labelled_only=False)
        if not samples:
            return 0
        section = self.config.get(CONSUMERS.get(consumer, ""), {}) or {}
        feature_names = section.get("features", []) or sorted(samples[0]["features"])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        rows = 0
        with open(output_path, "w", encoding="utf-8", newline="") as fh:
            writer = None
            for sample in samples:
                vector = vectorize(sample["features"], feature_names)
                if writer is None:
                    header = (
                        ["cycle_id"]
                        + [f"f{i}" for i in range(len(vector))]
                        + ["label"]
                    )
                    writer = csv.writer(fh)
                    writer.writerow(header)
                writer.writerow(
                    [sample["cycle_id"]] + vector + [sample["label"]]
                )
                rows += 1
        return rows

    # ------------------------------------------------------------------
    # Agendamento
    # ------------------------------------------------------------------
    def start_scheduler(self) -> None:
        """Arranca o job periódico de treino (só se ML_ENABLED=true)."""
        from .hippocampus import ml_enabled

        if not ml_enabled():
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                self.train_all,
                "interval",
                hours=self.interval_hours,
                id="hippocampus_training",
                replace_existing=True,
            )
            self._scheduler.start()
        except Exception:
            self._scheduler = None

    def stop_scheduler(self) -> None:
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
