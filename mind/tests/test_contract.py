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


# ==========================================================================
# Verificação 3 — diff real das secções alheias (autoritativa)
# ==========================================================================
# A heurística acima é só o primeiro filtro. O diff compara o conteúdo das
# secções que não pertencem ao NEURON, antes e depois de integrar a resposta.

BASE_COM_PREAMBULO = (
    "IMPORTS = ['os', 'sys']\n"
    "TIMEOUT = 30\n"
    "# [NEURON_1:python]\n"
    "def autenticar(): pass\n"
    "# [NEURON_2:python]\n"
    "def cifrar(): pass\n"
)


def _estado_com_fotografia(cortex, cycle_id, outputs):
    """Estado que passou pela distribuição, logo com fotografia tirada."""
    estado = {
        "cycle_id": cycle_id, "iteration": 1, "task": "t",
        "base_code": BASE_COM_PREAMBULO,
        "markers": {"neuron_1": {"language": "python"},
                    "neuron_2": {"language": "python"}},
        "neuron_outputs": {}, "contract_violations": [], "improvements": {},
        "active_neurons": ["neuron_1", "neuron_2"],
    }
    cortex._snapshot_foreign_sections(estado, BASE_COM_PREAMBULO)
    estado["neuron_outputs"] = outputs
    return estado


def test_diff_apanha_alteracao_no_preambulo(cortex, cycle_id):
    """O caso que a heurística NÃO apanha.

    O NEURON devolve a sua secção correctamente, com o seu marcador e sem
    marcadores alheios — passa o filtro heurístico — mas altera o preâmbulo
    partilhado. O diff real apanha-o.
    """
    resposta = (
        "IMPORTS = ['os', 'sys', 'requests']\n"   # <-- mexeu no preâmbulo
        "TIMEOUT = 30\n"
        "# [NEURON_1:python]\n"
        "def autenticar(): return True\n"
    )
    # Confirma primeiro que a heurística sozinha deixaria passar:
    from agent.cortex import find_other_markers, marker_present
    assert marker_present(resposta, "neuron_1")
    assert find_other_markers(resposta, "neuron_1") == []

    estado = _estado_com_fotografia(cortex, cycle_id, {"neuron_1": resposta})
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]


def test_diff_aceita_resposta_dentro_do_ambito(cortex, cycle_id):
    """Implementar só a sua secção nunca pode ser violação."""
    estado = _estado_com_fotografia(cortex, cycle_id, {
        "neuron_1": "# [NEURON_1:python]\ndef autenticar(): return True",
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == []


def test_diff_aceita_resposta_que_repete_o_preambulo_intacto(cortex, cycle_id):
    """Devolver o preâmbulo sem o alterar é legítimo."""
    estado = _estado_com_fotografia(cortex, cycle_id, {
        "neuron_1": (
            "IMPORTS = ['os', 'sys']\n"
            "TIMEOUT = 30\n"
            "# [NEURON_1:python]\n"
            "def autenticar(): return True\n"
        ),
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == []


def test_diff_ignora_whitespace_final(cortex, cycle_id):
    """Espaços no fim das linhas não são alteração de código."""
    estado = _estado_com_fotografia(cortex, cycle_id, {
        "neuron_1": (
            "IMPORTS = ['os', 'sys']   \n"
            "TIMEOUT = 30\t\n"
            "# [NEURON_1:python]\n"
            "def autenticar(): return True\n"
        ),
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == []


def test_diff_apanha_alteracao_de_seccao_alheia(cortex, cycle_id):
    """Reescrever a secção de outro NEURON é violação."""
    estado = _estado_com_fotografia(cortex, cycle_id, {
        "neuron_1": (
            "# [NEURON_1:python]\n"
            "def autenticar(): return True\n"
            "# [NEURON_2:python]\n"
            "def cifrar(): return 'roubei esta secção'\n"
        ),
    })
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == ["neuron_1"]


def test_razao_da_violacao_nomeia_o_ambito(cortex, db, cycle_id):
    estado = _estado_com_fotografia(cortex, cycle_id, {
        "neuron_1": "IMPORTS = ['alterado']\n# [NEURON_1:python]\ndef a(): pass",
    })
    cortex.validate_contracts(estado)
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert any("fora do seu âmbito" in d["decision_text"] for d in decisoes)


def test_sem_fotografia_o_diff_nao_inventa_violacoes(cortex, cycle_id):
    """Estado construído à mão, sem passar pela distribuição.

    Sem referência, o diff abstém-se e só a heurística se aplica — não pode
    reprovar por falta de informação.
    """
    estado = _estado(cycle_id, {"neuron_1": "# [NEURON_1]\ndef ok(): pass"})
    assert "foreign_sections" not in estado
    cortex.validate_contracts(estado)
    assert estado["contract_violations"] == []


# --- Funções de partição usadas pelo diff --------------------------------
def test_split_by_markers_separa_preambulo_e_seccoes():
    from agent.cortex import PREAMBULO, split_by_markers

    seccoes = split_by_markers(BASE_COM_PREAMBULO)
    assert "IMPORTS" in seccoes[PREAMBULO]
    assert "autenticar" in seccoes["neuron_1"]
    assert "cifrar" in seccoes["neuron_2"]


def test_foreign_sections_exclui_o_proprio():
    from agent.cortex import PREAMBULO, foreign_sections

    alheias = foreign_sections(BASE_COM_PREAMBULO, "neuron_1")
    assert set(alheias) == {PREAMBULO, "neuron_2"}


def test_diff_foreign_sections_detecta_ausencia_e_surgimento():
    from agent.cortex import diff_foreign_sections

    assert diff_foreign_sections({"a": "x"}, {"a": "x"}) == []
    assert diff_foreign_sections({"a": "x"}, {"a": "y"}) == ["a"]
    assert diff_foreign_sections({"a": "x"}, {}) == ["a"]
    assert diff_foreign_sections({}, {"b": "z"}) == ["b"]
