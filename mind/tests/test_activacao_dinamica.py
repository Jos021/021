"""Activação dinâmica de NEURONS.

Regra: na 1.ª passagem correm TODOS os NEURONS disponíveis; a partir da 2.ª
iteração só correm os que o CORTEX designou explicitamente para aplicar uma
melhoria nessa ronda. ENABLE_NEURON_N define se o NEURON EXISTE, não se
corre numa iteração específica.
"""

import asyncio

import pytest

from agent.cortex import Cortex
from agent.neurons import build_neurons, run_neurons_parallel


@pytest.fixture
def cortex(router, db):
    return Cortex(router, db, {}, None)


def _estado(cycle_id, **extra):
    base = {
        "cycle_id": cycle_id,
        "iteration": 1,
        "task": "t",
        "base_code": "",
        "markers": {f"neuron_{n}": {"language": "python"} for n in (1, 2, 3)},
        "improvements": {},
        "active_neurons": [],
        "neuron_outputs": {},
    }
    base.update(extra)
    return base


def test_primeira_passagem_corre_todos(cortex, cycle_id):
    estado = _estado(cycle_id)
    cortex.distribute(estado)
    assert sorted(estado["active_neurons"]) == ["neuron_1", "neuron_2", "neuron_3"]


def test_segunda_iteracao_corre_so_os_visados(cortex, cycle_id):
    estado = _estado(cycle_id, iteration=2,
                     improvements={"neuron_2": "corrige o parsing"})
    cortex.select_neurons_for_improvement(estado)
    assert estado["active_neurons"] == ["neuron_2"]


def test_neuron_sem_melhoria_nao_e_chamado(cortex, cycle_id):
    estado = _estado(cycle_id, iteration=2,
                     improvements={"neuron_1": "x", "neuron_3": "y"})
    cortex.select_neurons_for_improvement(estado)
    assert "neuron_2" not in estado["active_neurons"]


def test_sem_melhorias_nenhum_neuron_corre(cortex, cycle_id):
    estado = _estado(cycle_id, iteration=2, improvements={})
    cortex.select_neurons_for_improvement(estado)
    assert estado["active_neurons"] == []


def test_enable_neuron_define_existencia_nao_activacao(
    cortex, cycle_id, monkeypatch
):
    """ENABLE_NEURON_2=false retira-o do sistema, mesmo com marcador."""
    monkeypatch.setenv("ENABLE_NEURON_2", "false")
    estado = _estado(cycle_id)
    cortex.distribute(estado)
    assert "neuron_2" not in estado["active_neurons"]
    assert "neuron_1" in estado["active_neurons"]


def test_neuron_desactivado_nao_e_construido(router, db, monkeypatch):
    monkeypatch.setenv("ENABLE_NEURON_5", "false")
    neurons = build_neurons(router, db, {})
    assert "neuron_5" not in neurons
    assert "neuron_1" in neurons


def test_decisao_de_distribuicao_e_registada(cortex, db, cycle_id):
    """Registo estruturado do raciocínio de distribuição."""
    cortex.distribute(_estado(cycle_id))
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert any("1ª passagem" in d["decision_text"] for d in decisoes)


# --- Execução paralela e circuit breaker ---------------------------------
def test_so_os_neurons_activos_sao_invocados(router, db, cycle_id, com_modelos):
    neurons = build_neurons(router, db, {})
    estado = _estado(cycle_id)
    asyncio.run(run_neurons_parallel(
        neurons, ["neuron_2"], estado, {"neuron_2": "melhora"}, timeout=10
    ))
    assert router.chamadas_async == ["neuron_2"], \
        "nenhum NEURON fora da lista de activos pode ser chamado"


def test_circuit_breaker_corta_neuron_lento(db, cycle_id, com_modelos):
    """Um NEURON que excede o timeout não bloqueia os restantes."""

    class RouterLento:
        async def agenerate(self, prompt, model, endpoint, system="",
                            component="", timeout=120):
            if component == "neuron_1":
                await asyncio.sleep(5)
            return f"# [NEURON_{component.split('_')[-1]}]\npass"

        def generate(self, *a, **k):
            return ""

    neurons = build_neurons(RouterLento(), db, {})
    estado = _estado(cycle_id)
    saidas = asyncio.run(run_neurons_parallel(
        neurons, ["neuron_1", "neuron_2"], estado, {}, timeout=0.3
    ))
    assert "NEURON_ERRO" in saidas["neuron_1"]
    assert "[NEURON_2]" in saidas["neuron_2"], \
        "o NEURON rápido tem de completar apesar do lento ter sido cortado"
