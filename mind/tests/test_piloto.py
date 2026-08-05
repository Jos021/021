"""Piloto com modelos reais — verificação e medição.

O piloto é o que falta para validar o MIND. Estes testes garantem que a
ferramenta em si funciona: que apanha endpoints em baixo antes de se gastar
um ciclo, que mede o que diz medir, e que uma tarefa que rebenta não
interrompe as seguintes.
"""

import httpx
import pytest

from agent.piloto import (
    ResultadoTarefa,
    carregar_tarefas,
    correr_piloto,
    exportar_csv,
    resumir,
    verificar_componentes,
)


class _Resposta:
    def __init__(self, texto="OK", status=200):
        self._texto = texto
        self.status_code = status

    def json(self):
        return {"choices": [{"message": {"content": self._texto}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("erro", request=None, response=self)


# ==========================================================================
# Verificação de ligação
# ==========================================================================
def test_sem_modelos_configurados_nada_e_chamado(db, monkeypatch):
    """Campos _MODEL vazios: reporta, não tenta ligar."""
    chamou = []
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: chamou.append(1) or _Resposta())
    diagnosticos = verificar_componentes(db)
    assert diagnosticos
    assert all(not d.configurado for d in diagnosticos)
    assert all(d.estado == "sem modelo" for d in diagnosticos)
    assert not chamou


def test_endpoint_a_responder(db, com_modelos, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta("OK"))
    diagnosticos = verificar_componentes(db)
    assert all(d.respondeu for d in diagnosticos)
    assert all(d.estado == "ok" for d in diagnosticos)
    assert all(d.latencia_s >= 0 for d in diagnosticos)


def test_endpoint_em_baixo_e_reportado(db, com_modelos, monkeypatch):
    """É o caso que justifica o comando existir."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(
        httpx.ConnectError("instância em baixo")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    diagnosticos = verificar_componentes(db)
    assert all(not d.respondeu for d in diagnosticos)
    assert all(d.erro for d in diagnosticos), "a razão tem de ser reportada"
    assert all(d.estado == "falhou" for d in diagnosticos)


def test_resposta_vazia_conta_como_falha(db, com_modelos, monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta(""))
    d = verificar_componentes(db)[0]
    assert not d.respondeu
    assert "vazia" in d.erro


def test_token_errado_e_reportado_sem_repetir(db, com_modelos, monkeypatch):
    """Um 401 não melhora com insistência: uma tentativa por componente."""
    tentativas = []

    def post(*a, **k):
        tentativas.append(1)
        return _Resposta(status=401)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    diagnosticos = verificar_componentes(db)
    assert all(not d.respondeu for d in diagnosticos)
    assert len(tentativas) == len(diagnosticos), \
        "sem retry: exactamente uma tentativa por componente"


def test_neuron_desactivado_nao_e_verificado(db, com_modelos, monkeypatch):
    monkeypatch.setenv("ENABLE_NEURON_3", "false")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resposta())
    nomes = [d.componente for d in verificar_componentes(db)]
    assert "neuron_3" not in nomes
    assert "neuron_1" in nomes


# ==========================================================================
# Tarefas de referência
# ==========================================================================
def test_carregar_tarefas_do_projecto():
    import os

    caminho = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "piloto_tarefas.yaml",
    )
    tarefas = carregar_tarefas(caminho)
    assert len(tarefas) >= 5, "o critério de selecção pede 5-10 tarefas"
    assert all(isinstance(t, str) and t.strip() for t in tarefas)


def test_ficheiro_inexistente_devolve_vazio():
    assert carregar_tarefas("/caminho/que/nao/existe.yaml") == []


def test_ficheiro_invalido_devolve_vazio(tmp_path):
    mau = tmp_path / "mau.yaml"
    mau.write_text("isto: [não fecha", encoding="utf-8")
    assert carregar_tarefas(str(mau)) == []


# ==========================================================================
# Execução e medição
# ==========================================================================
class _GrafoFalso:
    def __init__(self, resultado=None, rebentar=False):
        self.resultado = resultado or {
            "status": "approved", "functionality_pct": 99.0, "iteration": 2,
        }
        self.rebentar = rebentar

    def run(self, state):
        if self.rebentar:
            raise RuntimeError("o ciclo rebentou")
        return {**state, **self.resultado}


def test_piloto_mede_cada_tarefa(db):
    resultados = correr_piloto(
        ["tarefa A", "tarefa B"], db, lambda: _GrafoFalso())
    assert len(resultados) == 2
    assert all(r.status == "approved" for r in resultados)
    assert all(r.functionality_pct == 99.0 for r in resultados)
    assert all(r.duracao_s >= 0 for r in resultados)
    assert all(r.cycle_id for r in resultados)


def test_tarefa_que_rebenta_nao_interrompe_as_seguintes(db):
    """Um piloto que pára na primeira falha não mede nada."""
    grafos = iter([_GrafoFalso(rebentar=True), _GrafoFalso()])
    resultados = correr_piloto(["má", "boa"], db, lambda: next(grafos))
    assert len(resultados) == 2
    assert resultados[0].status == "erro" and resultados[0].erro
    assert resultados[1].status == "approved"


def test_max_tarefas_limita(db):
    resultados = correr_piloto(
        ["a", "b", "c", "d"], db, lambda: _GrafoFalso(), max_tarefas=2)
    assert len(resultados) == 2


def test_cada_tarefa_tem_o_seu_ciclo(db):
    resultados = correr_piloto(["a", "b"], db, lambda: _GrafoFalso())
    assert resultados[0].cycle_id != resultados[1].cycle_id


def test_conformidade_json_e_medida(db):
    """Os desvios já são registados pelo report_schema; aqui só se contam."""
    from agent.report_schema import parse_relatorio, registar_desvio_de_formato

    def grafo_com_desvio():
        class G:
            def run(self, state):
                cid = state["cycle_id"]
                db.log_iteration(cid, 1, "3", "cerebellum")
                db.log_iteration(cid, 1, "3", "cortex")
                # Um dos dois não respeitou o formato.
                registar_desvio_de_formato(
                    db, cid, 1, "cerebellum", parse_relatorio("PCT: 50"))
                return {**state, "status": "approved"}
        return G()

    r = correr_piloto(["t"], db, grafo_com_desvio)[0]
    assert r.chamadas_modelo == 2
    assert r.desvios_formato == 1
    assert r.conformidade_json == 50.0


def test_conformidade_none_sem_chamadas(db):
    r = correr_piloto(["t"], db, lambda: _GrafoFalso())[0]
    assert r.conformidade_json is None


def test_testes_da_sandbox_sao_contados(db):
    grafo = _GrafoFalso(resultado={
        "status": "approved", "functionality_pct": 99.0, "iteration": 1,
        "generated_tests": [{"a": 1}, {"b": 2}, {"c": 3}],
        "test_breakdown": {"level_1": {"passed": 5, "total": 8},
                           "level_2": {"passed": 2, "total": 4}},
    })
    r = correr_piloto(["t"], db, lambda: grafo)[0]
    assert r.testes_gerados == 3
    assert r.testes_executados == 12
    assert r.testes_passados == 7


# ==========================================================================
# Resumo e exportação
# ==========================================================================
def test_resumo_agrega():
    resultados = [
        ResultadoTarefa("a", 1, status="approved", functionality_pct=99.0,
                        iteracoes=2, duracao_s=10.0, chamadas_modelo=10,
                        desvios_formato=1),
        ResultadoTarefa("b", 2, status="needs_human", functionality_pct=40.0,
                        iteracoes=10, duracao_s=30.0, chamadas_modelo=10,
                        desvios_formato=3),
    ]
    r = resumir(resultados)
    assert r["tarefas"] == 2
    assert r["aprovadas"] == 1
    assert r["taxa_aprovacao"] == 50.0
    assert r["duracao_media_s"] == 20.0
    assert r["duracao_max_s"] == 30.0
    assert r["iteracoes_media"] == 6.0
    assert r["conformidade_json_media"] == 80.0   # média de 90% e 70%


def test_resumo_vazio_nao_rebenta():
    assert resumir([]) == {"tarefas": 0}


def test_resumo_ignora_duracao_de_tarefas_com_erro():
    resultados = [
        ResultadoTarefa("a", 1, status="approved", duracao_s=10.0),
        ResultadoTarefa("b", 2, status="erro", duracao_s=0.1, erro="rebentou"),
    ]
    r = resumir(resultados)
    assert r["erros"] == 1
    assert r["duracao_media_s"] == 10.0


def test_taxa_de_testes_passados():
    resultados = [ResultadoTarefa("a", 1, testes_executados=10,
                                  testes_passados=7)]
    assert resumir(resultados)["taxa_testes_passados"] == 70.0


def test_exportar_csv(tmp_path):
    destino = str(tmp_path / "sub" / "piloto.csv")
    resultados = [ResultadoTarefa("tarefa A", 1, status="approved",
                                  functionality_pct=99.0)]
    assert exportar_csv(resultados, destino) == 1
    linhas = open(destino, encoding="utf-8").read().splitlines()
    assert linhas[0].startswith("tarefa,cycle_id,status")
    assert "tarefa A" in linhas[1]
    assert len(linhas) == 2


def test_exportar_csv_vazio(tmp_path):
    destino = str(tmp_path / "vazio.csv")
    assert exportar_csv([], destino) == 0
    assert open(destino, encoding="utf-8").read().strip().startswith("tarefa,")
