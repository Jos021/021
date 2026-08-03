"""HIPPOCAMPUS — princípios inegociáveis da camada de apoio de ML.

Os testes que exigem treino real são saltados se o scikit-learn não estiver
instalado (vem em requirements-ml.txt). Os princípios de segurança —
cold-start seguro e assimetria — são testados sempre, porque é precisamente
sem dependências e sem modelo que têm de se aguentar.
"""

import pytest

from agent.hippocampus import Hippocampus, ml_enabled
from agent.ml_features import (
    code_complexity,
    extract_cerebellum_features,
    extract_cortex_features,
    failure_patterns,
    task_embedding,
    vectorize,
)
# Importado com outro nome: `test_coverage` seria recolhido pelo pytest como
# se fosse um teste, e não é — é a função de extracção de features.
from agent.ml_features import test_coverage as cobertura_de_testes

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
TESTES_MAUS = "success=False\nTraceback\nerror\ntimeout\n"
CODIGO_BOM = "def f():\n    return 1\n"
TESTES_BONS = "success=True\n" * 3


@pytest.fixture
def hippo(db, tmp_path):
    return Hippocampus(db, CONFIG, str(tmp_path / "modelos"))


# ======================================================================
# Princípio 4 — cold-start seguro (nunca lança, devolve None)
# ======================================================================
def test_desligado_devolve_none(hippo, db, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "false")
    assert hippo.consult("cortex", extract_cortex_features("t", db)) is None
    assert hippo.consult("cerebellum", {"code_complexity": 1}) is None


def test_ml_enabled_le_o_ambiente(monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    assert ml_enabled() is True
    monkeypatch.setenv("ML_ENABLED", "false")
    assert ml_enabled() is False


def test_cold_start_sem_amostras_devolve_none(hippo, db, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "100")
    assert hippo.consult("cortex", extract_cortex_features("t", db)) is None


def test_amostras_suficientes_mas_sem_modelo_devolve_none(
    hippo, db, cycle_id, monkeypatch
):
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "2")
    for _ in range(5):
        hippo.record_sample("cortex", cycle_id, {"a": 1.0}, 90.0)
    assert hippo.consult("cortex", {"a": 1.0}) is None


@pytest.mark.parametrize("features", [None, {}, {"lixo": object()}])
def test_features_invalidas_nunca_lancam(hippo, monkeypatch, features):
    monkeypatch.setenv("ML_ENABLED", "true")
    assert hippo.consult("cortex", features) is None


def test_consumidor_desconhecido_devolve_none(hippo, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    assert hippo.consult("hipotalamo", {}) is None


def test_modelo_em_falta_no_disco_devolve_none(hippo, db, cycle_id, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "1")
    hippo.record_sample("cortex", cycle_id, {"a": 1.0}, 90.0)
    db.register_ml_model("cortex", "/caminho/que/nao/existe.joblib", 1.0, 10,
                         activate=True)
    assert hippo.consult("cortex", {"a": 1.0}) is None


# ======================================================================
# Princípio — o histórico acumula mesmo com a camada desligada
# ======================================================================
def test_amostras_acumulam_com_ml_desligado(hippo, db, cycle_id, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "false")
    hippo.record_sample("cortex", cycle_id, {"a": 1.0}, 95.0)
    assert db.count_ml_samples("cortex") == 1


def test_amostra_sem_label_nao_conta_para_treino(hippo, db, cycle_id):
    hippo.record_sample("cortex", cycle_id, {"a": 1.0}, None)
    assert db.count_ml_samples("cortex", labelled_only=True) == 0
    assert db.count_ml_samples("cortex", labelled_only=False) == 1


# ======================================================================
# Princípio 2 — assimetria de segurança
# ======================================================================
@precisa_sklearn
def test_cerebellum_auto_rejeita_padrao_de_falha(hippo, db, cycle_id, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "10")
    _treinar(hippo, db, cycle_id)

    resultado = hippo.consult(
        "cerebellum", extract_cerebellum_features(CODIGO_MAU, TESTES_MAUS, db)
    )
    assert resultado is not None
    assert resultado["auto_reject"] is True
    assert resultado["confidence"] >= 0.9
    assert resultado["reason"], "a razão da rejeição tem de vir preenchida"


@precisa_sklearn
def test_codigo_bom_nunca_e_auto_rejeitado(hippo, db, cycle_id, monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "10")
    _treinar(hippo, db, cycle_id)

    resultado = hippo.consult(
        "cerebellum", extract_cerebellum_features(CODIGO_BOM, TESTES_BONS, db)
    )
    assert resultado is not None
    assert resultado["auto_reject"] is False


@precisa_sklearn
def test_cortex_nunca_auto_rejeita(hippo, db, cycle_id, monkeypatch):
    """Não existe auto-aprovação nem auto-rejeição do lado do CORTEX."""
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "10")
    _treinar(hippo, db, cycle_id)

    resultado = hippo.consult("cortex", extract_cortex_features("tarefa", db))
    assert resultado is not None
    assert resultado["auto_reject"] is False


@precisa_sklearn
def test_nenhuma_consulta_devolve_auto_aprovacao(hippo, db, cycle_id, monkeypatch):
    """Nenhum caminho do HIPPOCAMPUS pode aprovar seja o que for."""
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "10")
    _treinar(hippo, db, cycle_id)

    for consumidor, features in [
        ("cortex", extract_cortex_features("tarefa", db)),
        ("cerebellum", extract_cerebellum_features(CODIGO_BOM, TESTES_BONS, db)),
    ]:
        resultado = hippo.consult(consumidor, features)
        assert "auto_approve" not in resultado
        assert "auto_aprovar" not in resultado


@precisa_sklearn
def test_threshold_do_config_tem_precedencia(db, cycle_id, tmp_path, monkeypatch):
    """auto_reject_confidence do ml_config sobrepõe-se ao valor do .env."""
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MIN_TRAINING_SAMPLES", "10")
    monkeypatch.setenv("ML_AUTO_REJECT_CONFIDENCE", "0.1")

    config = {**CONFIG, "cerebellum_support": {
        **CONFIG["cerebellum_support"], "auto_reject_confidence": 0.99}}
    hippo = Hippocampus(db, config, str(tmp_path / "m"))
    hippo.auto_reject_confidence = 0.1   # o do .env, que o config deve vencer
    _treinar(hippo, db, cycle_id)

    resultado = hippo.consult(
        "cerebellum", extract_cerebellum_features(CODIGO_BOM, TESTES_BONS, db)
    )
    assert resultado["threshold"] == 0.99


def _treinar(hippo, db, cycle_id):
    """Gera histórico sintético separável e treina os dois consumidores."""
    from agent.ml_pipeline import MLPipeline

    # O Hippocampus lê ML_MIN_TRAINING_SAMPLES na construção, e a fixture é
    # criada antes do monkeypatch do teste — por isso ajustamo-lo aqui.
    hippo.min_samples = 10

    for i in range(40):
        facil = i % 2 == 0
        tarefa = "parser json simples" if facil else "scanner de rede com sockets"
        codigo = CODIGO_BOM if facil else CODIGO_MAU
        testes = TESTES_BONS if facil else TESTES_MAUS
        pct = 99.0 if facil else 40.0
        hippo.record_sample("cortex", cycle_id,
                            extract_cortex_features(tarefa, db), pct)
        hippo.record_sample("cerebellum", cycle_id,
                            extract_cerebellum_features(codigo, testes, db), pct)

    pipeline = MLPipeline(db, hippo.config, hippo.models_dir, hippo)
    for resultado in pipeline.train_all():
        assert resultado["trained"], resultado.get("reason")


# ======================================================================
# Extracção de features — fallbacks determinísticos
# ======================================================================
def test_embedding_e_deterministico():
    assert task_embedding("cria um parser") == task_embedding("cria um parser")


def test_embedding_distingue_tarefas():
    assert task_embedding("parser json") != task_embedding("scanner de rede")


def test_complexidade_cresce_com_controlo_de_fluxo():
    assert code_complexity(CODIGO_MAU) > code_complexity(CODIGO_BOM)


def test_complexidade_de_codigo_vazio_e_zero():
    assert code_complexity("") == 0.0


def test_cobertura_de_testes():
    assert cobertura_de_testes("success=True\nsuccess=True") == 1.0
    assert cobertura_de_testes("success=False\nsuccess=True") == 0.5
    assert cobertura_de_testes("") == 0.0


def test_padroes_de_falha_sao_contados():
    assert failure_patterns(TESTES_MAUS) > 0
    assert failure_patterns(TESTES_BONS) == 0


def test_vectorize_e_estavel_entre_chamadas():
    features = {"a": {"z": 1.0, "b": 2.0}, "c": [1.0, 2.0], "d": 3.0}
    nomes = ["a", "c", "d"]
    assert vectorize(features, nomes) == vectorize(features, nomes)
    # 'b' antes de 'z' (ordem alfabética), depois a lista, depois o escalar.
    assert vectorize(features, nomes) == [2.0, 1.0, 1.0, 2.0, 3.0]


def test_vectorize_ignora_tipos_inesperados():
    assert vectorize({"a": "texto"}, ["a"]) == [0.0]
    assert vectorize({}, ["ausente"]) == [0.0]
