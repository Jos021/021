"""Esquema JSON dos relatórios de avaliação, e o recurso ao regex.

O JSON é o formato pedido. O regex mantém-se apenas para modelos que não
respeitem o contrato — e nesse caso fica registado um aviso na SYNAPSE DB,
que é informação útil para o piloto com modelos reais.
"""

import json

import pytest

from agent.report_schema import (
    INSTRUCAO_JSON,
    parse_relatorio,
    registar_conformidade,
)

BEM_FORMADO = json.dumps({
    "functionality_pct": 87,
    "failures": ["o parser rebenta com entrada vazia"],
    "improvements": {"neuron_2": "validar entrada", "neuron_5": "tratar erro"},
    "auto_reject": False,
})


# --- Caminho principal: JSON bem formado ---------------------------------
def test_json_bem_formado_e_lido_directamente():
    r = parse_relatorio(BEM_FORMADO)
    assert r.via == "json"
    assert r.formato_respeitado
    assert r.functionality_pct == 87.0
    assert r.failures == ["o parser rebenta com entrada vazia"]
    assert r.improvements == {"neuron_2": "validar entrada",
                              "neuron_5": "tratar erro"}
    assert r.auto_reject is False


def test_percentagem_e_numero_nao_texto():
    assert isinstance(parse_relatorio(BEM_FORMADO).functionality_pct, float)


def test_auto_reject_verdadeiro_e_lido():
    r = parse_relatorio(json.dumps({"functionality_pct": 10,
                                    "auto_reject": True}))
    assert r.auto_reject is True


@pytest.mark.parametrize("valor,esperado", [(150, 100.0), (-20, 0.0),
                                            (87.5, 87.5), ("91", 91.0)])
def test_percentagem_e_limitada_ao_intervalo(valor, esperado):
    r = parse_relatorio(json.dumps({"functionality_pct": valor}))
    assert r.functionality_pct == esperado


def test_json_embrulhado_em_bloco_de_codigo():
    """Modelos embrulham JSON em ```json — é lido, mas conta como desvio."""
    r = parse_relatorio(f"Aqui está:\n```json\n{BEM_FORMADO}\n```\n")
    assert r.functionality_pct == 87.0
    assert r.via == "json_embrulhado"
    assert not r.formato_respeitado, "embrulhar é um desvio ao formato pedido"


def test_campos_ausentes_usam_omissoes_seguras():
    r = parse_relatorio(json.dumps({"functionality_pct": 50}))
    assert r.failures == []
    assert r.improvements == {}
    assert r.auto_reject is False


def test_tipos_errados_nao_rebentam():
    r = parse_relatorio(json.dumps({
        "functionality_pct": 60, "failures": "uma falha só",
        "improvements": ["isto devia ser um objecto"],
    }))
    assert r.failures == ["uma falha só"]
    assert r.improvements == {}


# --- Recurso: regex sobre texto livre ------------------------------------
def test_json_malformado_cai_no_regex():
    r = parse_relatorio('{"functionality_pct": 87, "failures": [')
    assert r.via == "regex"
    assert not r.formato_respeitado


def test_texto_livre_com_pct_rotulada():
    r = parse_relatorio("PCT: 73\nneuron_2: corrige o parsing")
    assert r.via == "regex"
    assert r.functionality_pct == 73.0
    assert r.improvements == {"neuron_2": "corrige o parsing"}


def test_texto_livre_com_percentagem():
    assert parse_relatorio("Está a 64% de funcionalidade.").functionality_pct == 64.0


def test_json_valido_sem_a_percentagem_cai_no_regex():
    """O campo essencial falta: o formato não foi respeitado."""
    r = parse_relatorio('{"failures": ["x"], "improvements": {}}')
    assert r.via == "regex"


def test_json_que_nao_e_objecto_cai_no_regex():
    assert parse_relatorio("[1, 2, 3]").via == "regex"


@pytest.mark.parametrize("texto", ["", None, "sem números nenhuns"])
def test_entradas_degeneradas_nao_rebentam(texto):
    r = parse_relatorio(texto)
    assert r.functionality_pct == 0.0


# --- Registo do desvio na SYNAPSE DB -------------------------------------
def test_desvio_de_formato_e_registado(db, cycle_id):
    registar_conformidade(db, cycle_id, 1, "cerebellum",
                               parse_relatorio("PCT: 50"))
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert any("não respeitou o esquema JSON" in d["decision_text"]
               for d in decisoes)


def test_formato_respeitado_tambem_e_registado(db, cycle_id):
    """Os dois desfechos ficam registados — sem isso não há denominador.

    Registar só os desvios obrigava quem media a conformidade a inventar um
    denominador, e o que se usava (o total de chamadas ao modelo) incluía
    chamadas que nem sequer pediam JSON.
    """
    registar_conformidade(db, cycle_id, 1, "cerebellum",
                          parse_relatorio(BEM_FORMADO))
    decisoes = db._conn.execute(
        "SELECT decision_text FROM decisions WHERE cycle_id = ?", (cycle_id,)
    ).fetchall()
    assert len(decisoes) == 1
    assert "Esquema JSON respeitado" in decisoes[0]["decision_text"]
    assert "não respeitou" not in decisoes[0]["decision_text"]


def test_os_dois_marcadores_sao_distinguiveis_por_consulta(db, cycle_id):
    """A população inteira e os desvios têm de sair de consultas separadas."""
    registar_conformidade(db, cycle_id, 1, "cortex",
                          parse_relatorio(BEM_FORMADO))
    registar_conformidade(db, cycle_id, 1, "cortex",
                          parse_relatorio("PCT: 50"))
    total = db._conn.execute(
        "SELECT COUNT(*) AS n FROM decisions WHERE cycle_id = ? "
        "AND decision_text LIKE '%esquema JSON%'", (cycle_id,)
    ).fetchone()["n"]
    desvios = db._conn.execute(
        "SELECT COUNT(*) AS n FROM decisions WHERE cycle_id = ? "
        "AND decision_text LIKE '%não respeitou o esquema JSON%'", (cycle_id,)
    ).fetchone()["n"]
    assert total == 2, "a consulta da população tem de apanhar os dois"
    assert desvios == 1, "a consulta dos desvios só pode apanhar um"


def test_instrucao_json_descreve_todos_os_campos():
    for campo in ("functionality_pct", "failures", "improvements",
                  "auto_reject"):
        assert campo in INSTRUCAO_JSON


# --- Integração com o CEREBELLUM -----------------------------------------
def test_cerebellum_le_json_do_modelo(db, cycle_id, com_modelos):
    from agent.cerebellum import Cerebellum
    from tests.conftest import RouterFalso

    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": BEM_FORMADO}), db, None)
    estado = {"cycle_id": cycle_id, "iteration": 1, "task": "t",
              "cortex_test_report": BEM_FORMADO, "test_results": "x",
              "markers": {}, "organized_code": "", "active_neurons": []}
    cerebellum.compare_and_decide(estado)
    assert estado["functionality_pct"] == 87.0
    assert estado["improvements"] == {"neuron_2": "validar entrada",
                                      "neuron_5": "tratar erro"}


def test_auto_reject_do_modelo_nunca_aprova(db, cycle_id, com_modelos):
    """auto_reject só pode reprovar — a assimetria vale também aqui."""
    from agent.cerebellum import Cerebellum
    from tests.conftest import RouterFalso

    resposta = json.dumps({"functionality_pct": 100, "auto_reject": True})
    cerebellum = Cerebellum(
        RouterFalso(respostas={"cerebellum": resposta}), db, None)
    estado = {"cycle_id": cycle_id, "iteration": 1, "task": "t",
              "cortex_test_report": resposta, "test_results": "x",
              "markers": {}, "organized_code": "", "active_neurons": []}
    cerebellum.compare_and_decide(estado)
    assert estado["status"] != "approved"
    assert estado["functionality_pct"] < 98
