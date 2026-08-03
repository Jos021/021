"""Mecanismo de marcação no código.

Cobre a regra de retenção (marcadores mantidos durante todo o loop, só
removidos após aprovação) e a leitura da anotação de linguagem, que é o que
alimenta o sandbox multi-linguagem.
"""

import pytest

from agent.cortex import (
    extract_section,
    find_other_markers,
    marker_present,
    parse_markers,
)
from agent.sandbox import get_language_for_section

CODIGO = """# [NEURON_1:python] — implementar esta função
def autenticar_utilizador(user):
    pass

// [NEURON_4:rust] — implementar esta função
fn gerar_payload() { }
"""


def test_extrai_marcadores_com_linguagem():
    assert parse_markers(CODIGO) == {
        "neuron_1": {"language": "python"},
        "neuron_4": {"language": "rust"},
    }


def test_marcador_sem_linguagem_assume_python():
    assert parse_markers("# [NEURON_2]\ndef x(): pass") == {
        "neuron_2": {"language": "python"}
    }


@pytest.mark.parametrize("prefixo", ["#", "//"])
def test_reconhece_ambos_os_estilos_de_comentario(prefixo):
    assert parse_markers(f"{prefixo} [NEURON_3:go]\ncode") == {
        "neuron_3": {"language": "go"}
    }


def test_codigo_sem_marcadores():
    assert parse_markers("def x():\n    return 1") == {}


def test_marker_present_distingue_o_proprio():
    assert marker_present(CODIGO, "neuron_1")
    assert marker_present(CODIGO, "neuron_4")
    assert not marker_present(CODIGO, "neuron_2")


def test_find_other_markers_ignora_o_proprio():
    assert find_other_markers(CODIGO, "neuron_1") == ["neuron_4"]
    assert find_other_markers(CODIGO, "neuron_4") == ["neuron_1"]
    assert find_other_markers("# [NEURON_1]\nx", "neuron_1") == []


def test_extract_section_delimita_ate_ao_marcador_seguinte():
    seccao = extract_section(CODIGO, "neuron_1")
    assert "autenticar_utilizador" in seccao
    assert "gerar_payload" not in seccao, "não pode invadir a secção seguinte"


def test_extract_section_ultima_seccao_vai_ate_ao_fim():
    assert "gerar_payload" in extract_section(CODIGO, "neuron_4")


def test_extract_section_de_neuron_inexistente_e_vazia():
    assert extract_section(CODIGO, "neuron_6") == ""


# --- Ligação ao sandbox: a linguagem vem do marcador, nunca do conteúdo ---
def test_linguagem_lida_do_marcador():
    markers = parse_markers(CODIGO)
    assert get_language_for_section("neuron_4", markers) == "rust"


def test_linguagem_por_omissao_e_python():
    assert get_language_for_section("neuron_9", parse_markers(CODIGO)) == "python"


# --- Marcadores órfãos ----------------------------------------------------
# Um marcador sem substituição no código organizado é falha de
# funcionalidade: nunca se compila um resultado final com marcadores por
# preencher, e a percentagem tem de reflectir isso.
def test_marcador_orfao_e_detectavel_pela_seccao_vazia():
    codigo = "# [NEURON_1:python]\ndef feito(): return 1\n# [NEURON_2:python]\npass"
    assert extract_section(codigo, "neuron_1")
    assert extract_section(codigo, "neuron_2") == "pass", \
        "uma secção que ficou em 'pass' é um marcador por preencher"


def test_marcador_declarado_sem_seccao_no_codigo():
    """O marcador existe nos markers mas não no código organizado."""
    markers = parse_markers("# [NEURON_1]\ndef x(): pass")
    assert "neuron_1" in markers
    assert extract_section("# [NEURON_1]\ndef x(): pass", "neuron_3") == ""


def test_penalizacao_de_marcadores_orfaos(db, cycle_id, com_modelos):
    """A percentagem final tem de baixar quando sobram marcadores por preencher."""
    from agent.cerebellum import Cerebellum
    from tests.conftest import RouterFalso

    router = RouterFalso(respostas={"cerebellum": RouterFalso.avaliacao(100)})
    cerebellum = Cerebellum(router, db, None)
    estado = {
        "cycle_id": cycle_id, "iteration": 1, "task": "t",
        "cortex_test_report": RouterFalso.avaliacao(100),
        "test_results": "x", "active_neurons": [],
        "markers": {"neuron_1": {"language": "python"},
                    "neuron_2": {"language": "python"}},
        "organized_code": ("# [NEURON_1]\ndef feito(): return 1\n"
                           "# [NEURON_2]\npass"),
    }
    cerebellum.compare_and_decide(estado)
    assert estado["functionality_pct"] < 100


def test_todos_preenchidos_nao_sao_penalizados(db, cycle_id, com_modelos):
    from agent.cerebellum import Cerebellum
    from tests.conftest import RouterFalso

    router = RouterFalso(respostas={"cerebellum": RouterFalso.avaliacao(100)})
    cerebellum = Cerebellum(router, db, None)
    estado = {
        "cycle_id": cycle_id, "iteration": 1, "task": "t",
        "cortex_test_report": RouterFalso.avaliacao(100),
        "test_results": "x", "active_neurons": [],
        "markers": {"neuron_1": {"language": "python"}},
        "organized_code": "# [NEURON_1]\ndef feito(): return 1",
    }
    cerebellum.compare_and_decide(estado)
    assert estado["functionality_pct"] == 100


def test_linguagem_nao_e_inferida_do_conteudo():
    """Uma docstring que fala de 'def ' não pode alterar a linguagem.

    É precisamente a fragilidade que a especificação manda evitar: detecção
    por matching de strings sobre o código.
    """
    codigo = '// [NEURON_2:rust]\nfn f() { /* def isto nao e python */ }'
    markers = parse_markers(codigo)
    assert get_language_for_section("neuron_2", markers) == "rust"
