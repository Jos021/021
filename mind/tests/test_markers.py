"""Mecanismo de marcação no código.

Cobre a regra de retenção (marcadores mantidos durante todo o loop, só
removidos após aprovação) e a leitura da anotação de linguagem, que é o que
alimenta o sandbox multi-linguagem.
"""

import pytest

from agent.cortex import (
    extract_section,
    find_other_markers,
    limpar_codigo_modelo,
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


# ==========================================================================
# Extracção de código de respostas de modelo (markdown + prosa)
# ==========================================================================
# Modelos reais quase sempre embrulham o código em ```fences``` e prosa. Sem
# limpar isso, ia para a sandbox como erro de sintaxe e reprovava todo ciclo.
# Encontrado num ensaio adversário que imita um 7B indisciplinado.
def test_extrai_codigo_de_fence_com_prosa():
    prosa = (
        "Claro! Aqui está a implementação:\n\n"
        "```python\n# [NEURON_1:python]\ndef f(x):\n    return x is not None\n```\n\n"
        "Espero que ajude!"
    )
    limpo = limpar_codigo_modelo(prosa)
    assert limpo == "# [NEURON_1:python]\ndef f(x):\n    return x is not None"
    assert "Claro!" not in limpo and "Espero" not in limpo
    assert "```" not in limpo


def test_marcador_sobrevive_a_limpeza():
    """O contrato de interface depende de o marcador não se perder."""
    out = "```python\n# [NEURON_3:python]\npass\n```"
    assert marker_present(limpar_codigo_modelo(out), "neuron_3")


def test_sem_fences_devolve_intacto():
    """Sem cercas não se corta nada — preâmbulo legítimo não se perde."""
    codigo = "import os\n# [NEURON_1:python]\ndef f(): pass"
    assert limpar_codigo_modelo(codigo) == codigo


def test_marcadores_de_erro_passam_intactos():
    """[NEURON_ERRO]/[CORTEX_ERRO] não têm cercas e não podem ser mexidos."""
    assert limpar_codigo_modelo("[NEURON_ERRO] timeout") == "[NEURON_ERRO] timeout"


def test_varios_blocos_sao_concatenados():
    out = "```python\n# [NEURON_1:python]\na = 1\n```\ntexto\n```python\nb = 2\n```"
    limpo = limpar_codigo_modelo(out)
    assert "a = 1" in limpo and "b = 2" in limpo and "texto" not in limpo


def test_vazio_nao_rebenta():
    assert limpar_codigo_modelo("") == ""
    assert limpar_codigo_modelo(None) is None


# ==========================================================================
# Normalização de marcadores de modelos reais
# ==========================================================================
# Um 7B não reproduz o formato exacto `# [NEURON_N:python]`. Escreve
# [neuron_1] minúsculo e sem #. Encontrado no primeiro piloto real: o modelo
# devolveu "[neuron_1] descrição\ndef soma..." e o MIND ficou com markers=[].
def test_normaliza_marcador_minusculo_sem_prefixo():
    from agent.cortex import limpar_codigo_modelo
    real = "[neuron_1] Função de soma.\ndef soma(a, b):\n    return a + b"
    limpo = limpar_codigo_modelo(real)
    assert parse_markers(limpo) == {"neuron_1": {"language": "python"}}
    import ast
    ast.parse(limpo)   # o marcador normalizado é comentário -> compila


def test_marcador_sem_prefixo_ganha_comentario():
    from agent.cortex import normalizar_marcadores
    assert normalizar_marcadores("[NEURON_2:rust]\ncode").startswith("# [NEURON_2:rust]")


def test_marcador_ja_canonico_nao_muda():
    from agent.cortex import normalizar_marcadores
    canon = "# [NEURON_3:python]\nx = 1"
    assert normalizar_marcadores(canon) == canon


def test_normalizacao_preserva_linguagem_e_caixa():
    from agent.cortex import normalizar_marcadores
    markers = parse_markers(normalizar_marcadores("[Neuron_4:GO]\ncode"))
    assert markers == {"neuron_4": {"language": "go"}}


def test_marcador_de_erro_nao_e_normalizado():
    from agent.cortex import normalizar_marcadores
    # Sem dígito, [NEURON_ERRO] não é um marcador de secção — não se toca.
    assert normalizar_marcadores("[NEURON_ERRO] x") == "[NEURON_ERRO] x"


# ==========================================================================
# Marcadores alheios no output de um NEURON (rótulos de sub-passos)
# ==========================================================================
# Encontrado no primeiro piloto real: o NEURON_2 devolveu código VÁLIDO mas
# usou [NEURON_3:...] como rótulo de sub-passo no próprio corpo. A heurística
# reprovava código bom; a prova é o diff autoritativo, não a heurística.
def test_marcador_alheio_no_output_e_rebaixado():
    from agent.cortex import manter_apenas_marcador_proprio
    out = ("# [NEURON_2:python]\ndef soma(a, b):\n"
           "    # [NEURON_3:condicional]\n    return a + b")
    limpo = manter_apenas_marcador_proprio(out, "neuron_2")
    assert find_other_markers(limpo, "neuron_2") == []
    assert marker_present(limpo, "neuron_2"), "o marcador próprio mantém-se"
    import ast
    ast.parse(limpo)


def test_marcador_proprio_repetido_sobrevive():
    from agent.cortex import manter_apenas_marcador_proprio
    out = "# [NEURON_1:python]\nx = 1\n# [NEURON_1:python]\ny = 2"
    limpo = manter_apenas_marcador_proprio(out, "neuron_1")
    assert marker_present(limpo, "neuron_1")


def test_sem_alheios_fica_igual():
    from agent.cortex import manter_apenas_marcador_proprio
    out = "# [NEURON_4:rust]\nfn f() {}"
    assert manter_apenas_marcador_proprio(out, "neuron_4") == out
