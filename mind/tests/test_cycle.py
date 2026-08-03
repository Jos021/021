"""Ciclo completo do MIND, ponta-a-ponta, com router simulado.

Cobre o fluxo das três fases, a decisão por validação cruzada, a regra de
divergência, o rollback, o limite de iterações e a compilação final.
"""

import os

import pytest

from agent.cerebellum import Cerebellum
from agent.cortex import Cortex
from agent.graph import MindGraph
from agent.state import new_state
from tests.conftest import RouterFalso


@pytest.fixture
def grafo(router, db, com_modelos, monkeypatch):
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "3")
    monkeypatch.setenv("NEURON_TIMEOUT_SECONDS", "15")
    return MindGraph(router, db, {}, None)


# --- Estado inicial -------------------------------------------------------
def test_estado_inicial_esta_limpo():
    estado = new_state("uma tarefa", 7)
    assert estado["task"] == "uma tarefa"
    assert estado["cycle_id"] == 7
    assert estado["iteration"] == 0
    assert estado["status"] == "in_progress"
    assert estado["functionality_pct"] == 0.0
    assert estado["markers"] == {}


# --- Ciclo que aprova -----------------------------------------------------
def test_ciclo_completo_aprova(grafo, db, cycle_id):
    final = grafo.run(new_state("somar dois números", cycle_id))
    assert final["status"] == "approved"
    assert final["functionality_pct"] >= 98


def test_marcadores_removidos_do_codigo_final(grafo, db, cycle_id):
    final = grafo.run(new_state("somar", cycle_id))
    assert "[NEURON_" not in final["final_code"]
    assert final["final_code"].strip(), "o código final não pode ficar vazio"


def test_marcadores_mantidos_durante_o_loop(router, db, cycle_id, com_modelos):
    """Regra de retenção: só se removem na compilação final."""
    cortex = Cortex(router, db, {}, None)
    estado = new_state("t", cycle_id)
    cortex.create(estado)
    cortex.annotate_markers(estado)
    assert "[NEURON_" in estado["base_code"]

    estado["neuron_outputs"] = {"neuron_1": "# [NEURON_1]\ndef x(): pass"}
    estado["contract_violations"] = []
    cortex.organize(estado)
    assert "[NEURON_" in estado["organized_code"], \
        "os marcadores mantêm-se durante as Fases 2 e 3"


def test_output_compilado_em_workspace_output(grafo, db, cycle_id):
    grafo.run(new_state("somar", cycle_id))
    destino = os.path.join(grafo.workspace, "output", "resultado_final.txt")
    assert os.path.exists(destino)
    assert open(destino, encoding="utf-8").read().strip()


def test_ciclo_aprovado_e_registado_na_db(grafo, db, cycle_id):
    grafo.run(new_state("somar", cycle_id))
    ciclo = db.get_cycle(cycle_id)
    assert ciclo["status"] == "approved"
    assert ciclo["final_functionality_pct"] >= 98


# --- Ciclo que reprova ----------------------------------------------------
def test_limite_de_iteracoes_leva_a_needs_human(db, cycle_id, com_modelos,
                                                monkeypatch):
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "2")
    router = RouterFalso(respostas={
        "cortex": lambda p: ("# [NEURON_1:python]\npass" if "Anota" in p
                             else "PCT: 40\nfalta muito"),
        "cerebellum": lambda p: "PCT: 40\nneuron_1: implementa mesmo a função",
    })
    grafo = MindGraph(router, db, {}, None)
    final = grafo.run(new_state("tarefa difícil", cycle_id))

    assert final["status"] == "needs_human"
    assert db.get_cycle(cycle_id)["status"] == "needs_human"


def test_melhorias_sao_atribuidas_por_neuron(router, db, cycle_id, com_modelos):
    cerebellum = Cerebellum(
        RouterFalso(respostas={
            "cerebellum": "PCT: 50\nneuron_2: corrige o parsing\n"
                          "neuron_5: acrescenta validação"}),
        db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, test_results="success=False", markers={},
                  cortex_test_report="PCT: 50")
    cerebellum.compare_and_decide(estado)
    assert set(estado["improvements"]) == {"neuron_2", "neuron_5"}


# --- Validação cruzada e regra de divergência -----------------------------
def test_percentagem_e_media_dos_dois_relatorios(db, cycle_id, com_modelos):
    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": "PCT: 80"}), db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, cortex_test_report="PCT: 90",
                  test_results="x", markers={})
    cerebellum.compare_and_decide(estado)
    assert estado["functionality_pct"] == 85.0


def test_divergencia_forca_terceira_ronda(db, cycle_id, com_modelos, monkeypatch):
    monkeypatch.setenv("DIVERGENCE_THRESHOLD", "15")
    chamadas = []

    def resposta_cerebellum(prompt):
        chamadas.append(prompt)
        if "reconcilia" in prompt.lower():
            return "PCT: 70"
        return "PCT: 50"

    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": resposta_cerebellum}), db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, cortex_test_report="PCT: 95",
                  test_results="x", markers={})
    cerebellum.compare_and_decide(estado)

    assert any("reconcilia" in c.lower() for c in chamadas), \
        "diferença de 45pp tem de desencadear a 3.ª ronda"
    assert estado["functionality_pct"] == 70.0


def test_sem_divergencia_nao_ha_terceira_ronda(db, cycle_id, com_modelos):
    chamadas = []

    def resposta(prompt):
        chamadas.append(prompt)
        return "PCT: 92"

    cerebellum = Cerebellum(RouterFalso(respostas={"cerebellum": resposta}),
                            db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, cortex_test_report="PCT: 90",
                  test_results="x", markers={})
    cerebellum.compare_and_decide(estado)
    assert not any("reconcilia" in c.lower() for c in chamadas)


def test_divergencia_e_registada(db, cycle_id, com_modelos):
    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": "PCT: 40"}), db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, cortex_test_report="PCT: 95",
                  test_results="x", markers={})
    cerebellum.compare_and_decide(estado)
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id=?", (cycle_id,)
    ).fetchall()
    assert any("Divergência" in d["decision_text"] for d in decisoes)


# --- Marcadores órfãos ----------------------------------------------------
def test_marcador_orfao_penaliza_a_percentagem(db, cycle_id, com_modelos):
    """Nunca se compila com marcadores por preencher — conta como falha."""
    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": "PCT: 100"}), db, None)
    estado = new_state("t", cycle_id)
    estado.update(
        iteration=1, cortex_test_report="PCT: 100", test_results="x",
        markers={"neuron_1": {"language": "python"},
                 "neuron_2": {"language": "python"}},
        organized_code="# [NEURON_1]\ndef feito(): return 1\n# [NEURON_2]\npass",
    )
    cerebellum.compare_and_decide(estado)
    assert estado["functionality_pct"] < 100, \
        "um marcador por preencher tem de baixar a percentagem"


# --- Rollback -------------------------------------------------------------
def test_rollback_retoma_a_melhor_versao(router, db, cycle_id, com_modelos,
                                         monkeypatch):
    monkeypatch.setenv("ENABLE_ROLLBACK", "true")
    cortex = Cortex(router, db, {}, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=2, best_pct_so_far=90.0,
                  best_code_so_far="# a melhor versão",
                  functionality_pct=40.0, organized_code="# versão pior")
    cortex.distribute_improvements(estado)
    assert estado["organized_code"] == "# a melhor versão"


def test_rollback_desligado_mantem_a_versao_actual(router, db, cycle_id,
                                                   com_modelos, monkeypatch):
    monkeypatch.setenv("ENABLE_ROLLBACK", "false")
    cortex = Cortex(router, db, {}, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=2, best_pct_so_far=90.0,
                  best_code_so_far="# a melhor versão",
                  functionality_pct=40.0, organized_code="# versão pior")
    cortex.distribute_improvements(estado)
    assert estado["organized_code"] == "# versão pior"


def test_melhor_versao_e_guardada(db, cycle_id, com_modelos):
    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": "PCT: 75"}), db, None)
    estado = new_state("t", cycle_id)
    estado.update(iteration=1, cortex_test_report="PCT: 75",
                  test_results="x", markers={}, organized_code="# versão A")
    cerebellum.compare_and_decide(estado)
    assert estado["best_pct_so_far"] == 75.0
    assert estado["best_code_so_far"] == "# versão A"


# --- Grafo LangGraph ------------------------------------------------------
def test_langgraph_compila(grafo):
    compilado = grafo.build_langgraph()
    assert compilado is not None


# --- Degradação sem modelos configurados ----------------------------------
def test_sem_modelos_o_ciclo_corre_e_termina(router, db, cycle_id, monkeypatch):
    """Campos _MODEL vazios são o estado por omissão da especificação."""
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "1")
    grafo = MindGraph(router, db, {}, None)
    final = grafo.run(new_state("t", cycle_id))
    assert final["status"] == "needs_human"
    assert router.chamadas == [], "sem modelo configurado não se chama nada"
