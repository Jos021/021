"""Pipeline de treino do HIPPOCAMPUS — promoção por comparação.

O princípio central: um modelo novo só substitui o activo se for igual ou
melhor num conjunto de validação separado. Nunca por omissão.
"""

import pytest

from agent.hippocampus import Hippocampus
from agent.ml_features import extract_cerebellum_features, extract_cortex_features
from agent.ml_pipeline import MLPipeline

tem_sklearn = True
try:
    import sklearn  # noqa: F401
except ImportError:
    tem_sklearn = False

precisa_sklearn = pytest.mark.skipif(
    not tem_sklearn, reason="scikit-learn não instalado (requirements-ml.txt)"
)

CONFIG = {
    "cortex_support": {
        "model_type": "random_forest", "n_estimators": 10, "max_depth": 5,
        "features": ["task_embedding", "task_keywords", "history_success_rate"],
    },
    "cerebellum_support": {
        "model_type": "random_forest", "n_estimators": 10, "max_depth": 5,
        "features": ["code_complexity", "test_coverage", "failure_patterns",
                     "history_approval_rate"],
        "auto_reject_confidence": 0.9,
    },
}

CODIGO_MAU = "\n".join(
    f"if x{j}:\n    while y{j}:\n        try:\n            pass\n"
    f"        except: pass" for j in range(6)
)
CODIGO_BOM = "def f():\n    return 1\n"


@pytest.fixture
def hippo(db, tmp_path):
    return Hippocampus(db, CONFIG, str(tmp_path / "modelos"))


@pytest.fixture
def pipeline(db, hippo):
    return MLPipeline(db, CONFIG, hippo.models_dir, hippo)


def _historico_separavel(hippo, db, cycle_id, n=40):
    for i in range(n):
        facil = i % 2 == 0
        hippo.record_sample(
            "cortex", cycle_id,
            extract_cortex_features(
                "parser json" if facil else "scanner de rede", db),
            99.0 if facil else 40.0)
        hippo.record_sample(
            "cerebellum", cycle_id,
            extract_cerebellum_features(
                CODIGO_BOM if facil else CODIGO_MAU,
                "success=True\n" * 3 if facil else "success=False\nerror\n", db),
            99.0 if facil else 40.0)


# --- Guardas antes de treinar --------------------------------------------
def test_nao_treina_abaixo_do_minimo(pipeline, hippo, db, cycle_id, monkeypatch):
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "100")
    pipeline.min_samples = 100
    _historico_separavel(hippo, db, cycle_id, n=5)
    resultado = pipeline.train("cortex")
    assert resultado["trained"] is False
    assert "insuficientes" in resultado["reason"]


def test_consumidor_desconhecido_e_recusado(pipeline):
    resultado = pipeline.train("hipotalamo")
    assert resultado["trained"] is False
    assert "desconhecido" in resultado["reason"]


def test_sem_amostras_nenhumas_nao_rebenta(pipeline):
    resultado = pipeline.train("cortex", force=True)
    assert resultado["trained"] is False


# --- Promoção por comparação ---------------------------------------------
@precisa_sklearn
def test_primeiro_modelo_e_sempre_promovido(pipeline, hippo, db, cycle_id):
    pipeline.min_samples = 10
    _historico_separavel(hippo, db, cycle_id)
    resultado = pipeline.train("cerebellum")
    assert resultado["trained"] and resultado["promoted"]
    assert db.get_active_ml_model("cerebellum") is not None


@precisa_sklearn
def test_modelo_pior_nao_substitui_o_activo(pipeline, hippo, db, cycle_id):
    """O coração do princípio 5: promoção nunca acontece por omissão."""
    import random

    pipeline.min_samples = 10
    _historico_separavel(hippo, db, cycle_id)
    primeiro = pipeline.train("cerebellum")
    assert primeiro["promoted"]
    activo_antes = db.get_active_ml_model("cerebellum")["model_path"]

    # Injecta ruído puro: labels aleatórias sem relação com as features.
    random.seed(1)
    for _ in range(80):
        hippo.record_sample(
            "cerebellum", cycle_id,
            extract_cerebellum_features(
                random.choice([CODIGO_BOM, CODIGO_MAU]),
                random.choice(["success=True\n", "success=False\nerror\n"]), db),
            random.choice([99.0, 20.0]))

    segundo = pipeline.train("cerebellum")
    assert segundo["trained"]
    if not segundo["promoted"]:
        assert segundo["metric"] < primeiro["metric"]
        assert db.get_active_ml_model("cerebellum")["model_path"] == activo_antes, \
            "um modelo pior NUNCA pode tornar-se o activo"


@precisa_sklearn
def test_modelo_nao_promovido_fica_registado_no_historico(
    pipeline, hippo, db, cycle_id
):
    pipeline.min_samples = 10
    _historico_separavel(hippo, db, cycle_id)
    pipeline.train("cerebellum")
    pipeline.train("cerebellum")
    assert len(db.list_ml_models("cerebellum")) == 2, \
        "todas as versões ficam no histórico, promovidas ou não"


@precisa_sklearn
def test_so_existe_um_modelo_activo_por_consumidor(pipeline, hippo, db, cycle_id):
    pipeline.min_samples = 10
    _historico_separavel(hippo, db, cycle_id)
    pipeline.train("cortex")
    pipeline.train("cortex")
    activos = [m for m in db.list_ml_models("cortex") if m["is_active"]]
    assert len(activos) <= 1


# --- Estado e exportação --------------------------------------------------
def test_status_reporta_cold_start(pipeline, db, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "false")
    estado = pipeline.status()
    assert estado["ml_enabled"] is False
    for consumidor in ("cortex", "cerebellum"):
        assert estado["consumers"][consumidor]["cold_start"] is True
        assert estado["consumers"][consumidor]["active_model"] is None


def test_status_sinaliza_desvio_excessivo(pipeline, db, cycle_id):
    """Desvio médio acima do limiar recomenda retreino."""
    pipeline.deviation_threshold = 5.0
    for _ in range(5):
        db.log_ml_prediction(cycle_id, 1, "cerebellum", 0.9, "50.0")
    estado = pipeline.status()
    assert estado["consumers"]["cerebellum"]["needs_retrain"] is True


def test_desvio_ignora_decisoes_nao_numericas(db, cycle_id):
    db.log_ml_prediction(cycle_id, 1, "cortex", 0.9, "aprovado")
    assert db.ml_prediction_deviation("cortex") is None


def test_export_csv(pipeline, hippo, db, cycle_id, tmp_path):
    _historico_separavel(hippo, db, cycle_id, n=6)
    destino = str(tmp_path / "cortex.csv")
    linhas = pipeline.export("cortex", destino)
    assert linhas == 6
    conteudo = open(destino, encoding="utf-8").read().splitlines()
    assert conteudo[0].startswith("cycle_id,f0")
    assert len(conteudo) == 7  # cabeçalho + 6 amostras


def test_export_sem_amostras_devolve_zero(pipeline, tmp_path):
    assert pipeline.export("cortex", str(tmp_path / "vazio.csv")) == 0


def test_scheduler_nao_arranca_com_ml_desligado(pipeline, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "false")
    pipeline.start_scheduler()
    assert pipeline._scheduler is None
    pipeline.stop_scheduler()  # não pode rebentar
