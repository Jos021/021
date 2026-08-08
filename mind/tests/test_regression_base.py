"""Regressão da base: com ML_ENABLED=false, o MIND funciona como antes.

A camada HIPPOCAMPUS é uma extensão. Desligada, o sistema tem de se
comportar exactamente como a base — e o histórico de treino tem de
continuar a acumular, porque é esse volume que torna possível activá-la
mais tarde.
"""

import os

import pytest

from agent.graph import MindGraph
from agent.hippocampus import Hippocampus
from agent.state import new_state
from tests.conftest import RouterFalso

CONFIG_ML = {
    "cortex_support": {"model_type": "random_forest", "n_estimators": 10,
                       "max_depth": 5,
                       "features": ["task_embedding", "task_keywords",
                                    "history_success_rate"]},
    "cerebellum_support": {"model_type": "random_forest", "n_estimators": 10,
                           "max_depth": 5,
                           "features": ["code_complexity", "test_coverage",
                                        "failure_patterns",
                                        "history_approval_rate"]},
}


@pytest.fixture
def hippo(db, tmp_path):
    return Hippocampus(db, CONFIG_ML, str(tmp_path / "modelos"))


@pytest.fixture(autouse=True)
def ml_desligado(monkeypatch):
    monkeypatch.setenv("ML_ENABLED", "false")
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "3")


def test_ciclo_aprova_com_ml_desligado(db, cycle_id, com_modelos, hippo):
    grafo = MindGraph(RouterFalso(), db, {}, None, hippo)
    final = grafo.run(new_state("somar dois números", cycle_id))
    assert final["status"] == "approved"
    assert final["functionality_pct"] >= 98


def test_codigo_final_sem_marcadores_com_ml_desligado(db, cycle_id,
                                                     com_modelos, hippo):
    grafo = MindGraph(RouterFalso(), db, {}, None, hippo)
    final = grafo.run(new_state("somar", cycle_id))
    assert "[NEURON_" not in final["final_code"]
    assert final["final_code"].strip()


def test_output_compilado_com_ml_desligado(db, cycle_id, com_modelos, hippo):
    grafo = MindGraph(RouterFalso(), db, {}, None, hippo)
    grafo.run(new_state("somar", cycle_id))
    destino = os.path.join(grafo.workspace, "output", "resultado_final.txt")
    assert os.path.exists(destino)


def test_historico_de_treino_continua_a_acumular(db, cycle_id, com_modelos,
                                                 hippo):
    """A razão de ser: sem acumular agora, nunca haveria dados para activar."""
    grafo = MindGraph(RouterFalso(), db, {}, None, hippo)
    grafo.run(new_state("somar", cycle_id))
    assert db.count_ml_samples("cortex") > 0
    assert db.count_ml_samples("cerebellum") > 0


def test_nenhuma_consulta_ao_hippocampus_com_ml_desligado(db, cycle_id,
                                                         com_modelos, hippo):
    consultas = []
    original = hippo.consult

    def espiar(consumer, features):
        resultado = original(consumer, features)
        consultas.append((consumer, resultado))
        return resultado

    hippo.consult = espiar
    grafo = MindGraph(RouterFalso(), db, {}, None, hippo)
    grafo.run(new_state("somar", cycle_id))
    assert all(r is None for _, r in consultas), \
        "com ML_ENABLED=false nenhuma consulta pode devolver resultado"


def test_ciclo_sem_hippocampus_e_equivalente(db, com_modelos):
    """Passar hippocampus=None dá o mesmo resultado que tê-lo desligado."""
    c1 = db.create_cycle("t1")
    sem = MindGraph(RouterFalso(), db, {}, None, None).run(new_state("somar", c1))
    c2 = db.create_cycle("t2")
    com = MindGraph(RouterFalso(), db, {}, None,
                    Hippocampus(db, CONFIG_ML, "/tmp/nao-usado")).run(
                        new_state("somar", c2))
    assert sem["status"] == com["status"] == "approved"
    assert sem["final_code"] == com["final_code"]


def test_tabelas_ml_existem_mesmo_desligado(db):
    """As tabelas são criadas sempre: ter tabelas vazias não custa nada."""
    for tabela in ("ml_training_data", "ml_model_versions",
                   "ml_predictions_log"):
        assert db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (tabela,)).fetchone() is not None
