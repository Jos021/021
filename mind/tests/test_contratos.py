"""Validação de contrato de interface entre NEURONS.

A especificação define três verificações. As duas primeiras estão
implementadas de forma activa; a terceira é hoje heurística — o teste
`test_alteracao_fora_do_ambito_nao_e_detectada` documenta essa lacuna de
forma explícita, para que deixe de passar quando ela for fechada.
"""

import pytest

from agent.cerebellum import Cerebellum
from agent.cortex import Cortex


@pytest.fixture
def cortex(router, db):
    return Cortex(router, db, {}, None)


def _estado(cycle_id, outputs):
    return {
        "cycle_id": cycle_id,
        "iteration": 1,
        "task": "t",
        "base_code": "# [NEURON_1]\npass\n# [NEURON_2]\npass",
        "markers": {"neuron_1": {"language": "python"},
                    "neuron_2": {"language": "python"}},
        "neuron_outputs": outputs,
        "contract_violations": [],
        "improvements": {},
        "active_neurons": ["neuron_1", "neuron_2"],
    }


def test_resposta_valida_nao_viola(cortex, cycle_id):
    estado = _estado(cycle_id, {"neuron_1": "# [NEURON_1]\ndef ok(): return 1"})
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == []


def test_marcador_proprio_ausente_viola(cortex, cycle_id):
    """Verificação 1: tem de conter o próprio marcador."""
    estado = _estado(cycle_id, {"neuron_1": "def sem_marcador(): pass"})
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]


def test_marcador_alheio_viola(cortex, cycle_id):
    """Verificação 2: a resposta do NEURON_2 não pode conter [NEURON_3]."""
    estado = _estado(
        cycle_id, {"neuron_2": "# [NEURON_2]\n# [NEURON_3]\ndef mau(): pass"}
    )
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_2"]


def test_erro_do_neuron_conta_como_violacao(cortex, cycle_id):
    estado = _estado(cycle_id, {"neuron_1": "[NEURON_ERRO] timeout"})
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]


def test_resposta_vazia_conta_como_violacao(cortex, cycle_id):
    estado = _estado(cycle_id, {"neuron_1": ""})
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]


def test_apenas_o_violador_e_marcado(cortex, cycle_id):
    """O ciclo volta à Fase 2 só para quem violou — não para todos."""
    estado = _estado(cycle_id, {
        "neuron_1": "# [NEURON_1]\ndef bom(): pass",
        "neuron_2": "# [NEURON_2]\n# [NEURON_1]\ndef mau(): pass",
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_2"]


def test_violacao_e_registada_na_synapse_db(cortex, db, cycle_id):
    estado = _estado(cycle_id, {"neuron_1": "sem marcador"})
    cortex.validate_contracts(estado)
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert any("violou contrato" in d["decision_text"] for d in decisoes)


def test_seccao_do_violador_nao_entra_no_codigo_organizado(cortex, cycle_id):
    estado = _estado(cycle_id, {
        "neuron_1": "# [NEURON_1]\ndef legitimo(): pass",
        "neuron_2": "# [NEURON_2]\n# [NEURON_1]\ndef intruso(): pass",
    })
    cortex.validate_contracts(estado)
    cortex.organize(estado)
    assert "legitimo" in estado["organized_code"]
    assert "intruso" not in estado["organized_code"]


# --- Reprovação pelo CEREBELLUM ------------------------------------------
def test_cerebellum_reprova_com_justificacao_especifica(router, db, cycle_id):
    cerebellum = Cerebellum(router, db, None)
    estado = {"cycle_id": cycle_id, "iteration": 1,
              "contract_violations": ["neuron_3"], "active_neurons": [],
              "improvements": {}}
    cerebellum.reject_contract(estado)

    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert any(
        "NEURON_3 violou contrato de interface" in d["decision_text"]
        for d in decisoes
    ), "a justificação tem de nomear o NEURON violador"


def test_reprovacao_atribui_melhoria_so_ao_violador(router, db, cycle_id):
    cerebellum = Cerebellum(router, db, None)
    estado = {"cycle_id": cycle_id, "iteration": 1,
              "contract_violations": ["neuron_2"], "active_neurons": [],
              "improvements": {}}
    cerebellum.reject_contract(estado)
    assert list(estado["improvements"]) == ["neuron_2"]
    assert estado["status"] == "in_progress"


@pytest.mark.xfail(
    reason="Verificação 3 do contrato ainda é heurística: valida-se o "
           "marcador próprio e a ausência de alheios, mas não se faz diff "
           "do código fora da secção atribuída. Ver relatório, secção 4.2.",
    strict=True,
)
def test_alteracao_fora_do_ambito_nao_e_detectada(cortex, cycle_id):
    """Um NEURON que devolve a sua secção correcta MAIS código fora do âmbito.

    O contrato deveria apanhar isto (verificação 3). Hoje não apanha — este
    teste falha de propósito e passará a verde quando a lacuna for fechada.
    """
    estado = _estado(cycle_id, {
        "neuron_1": (
            "# [NEURON_1]\n"
            "def a_minha_parte(): return 1\n"
            "\n"
            "def funcao_de_outra_seccao_que_eu_nao_devia_tocar(): return 2\n"
        ),
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]
