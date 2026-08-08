"""Sandbox de testes evolutiva — biblioteca que cresce + rigor crescente.

Cobre a lista obrigatória da Parte 9 da especificação: infraestrutura,
geração, execução, lógica de níveis, herança entre ciclos, HIPPOCAMPUS, e
— criticamente — a regressão da base com SANDBOX_TESTS_ENABLED=false.
"""

import json
import uuid

import pytest

from agent.graph import MindGraph
from agent.hippocampus import Hippocampus
from agent.ml_features import EMBEDDING_DIM, cosine_similarity, embed_task
from agent.report_schema import parse_relatorio
from agent.state import new_state
from agent.test_generator import TestGenerator, sandbox_tests_enabled
from agent.test_runner import (
    TestRunner,
    calcular_percentagem,
    nivel_completo,
    seleccionar_testes,
)
from tests.conftest import RouterFalso


@pytest.fixture
def ligada(monkeypatch):
    """Activa a sandbox evolutiva para o teste."""
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "true")


def _teste(nivel=1, categoria="basic", codigo="assert True",
           esperado="pass", alvo="neuron_1", linguagem="python",
           test_id=None, cycle_id=None):
    return {
        "test_id": test_id or str(uuid.uuid4()),
        "cycle_id": cycle_id,
        "task_summary": "tarefa de teste",
        "task_embedding": None,
        "neuron_target": alvo,
        "language": linguagem,
        "level": nivel,
        "category": categoria,
        "description": f"teste de nível {nivel}",
        "code": codigo,
        "expected_outcome": esperado,
    }


# ==========================================================================
# Infraestrutura
# ==========================================================================
@pytest.mark.parametrize("tabela", ["test_library", "test_results"])
def test_tabelas_criadas(db, tabela):
    assert db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (tabela,)).fetchone() is not None


def test_save_test_com_id_duplicado_nao_lanca(db, cycle_id):
    t = _teste(cycle_id=cycle_id)
    db.save_test(t)
    db.save_test(t)   # INSERT OR IGNORE
    assert len(db.get_tests_for_cycle(cycle_id)) == 1


def test_record_test_result_incrementa_passed(db, cycle_id):
    t = _teste(cycle_id=cycle_id)
    db.save_test(t)
    db.record_test_result({"test_id": t["test_id"], "cycle_id": cycle_id,
                           "iteration_number": 1, "outcome": "pass",
                           "output": "", "duration_seconds": 0.1})
    guardado = db.get_tests_for_cycle(cycle_id)[0]
    assert guardado["times_passed"] == 1
    assert guardado["times_failed"] == 0
    assert guardado["times_used"] == 1


@pytest.mark.parametrize("outcome", ["fail", "error", "timeout"])
def test_record_test_result_incrementa_failed(db, cycle_id, outcome):
    """error e timeout contam como falha: não passaram."""
    t = _teste(cycle_id=cycle_id)
    db.save_test(t)
    db.record_test_result({"test_id": t["test_id"], "cycle_id": cycle_id,
                           "iteration_number": 1, "outcome": outcome})
    assert db.get_tests_for_cycle(cycle_id)[0]["times_failed"] == 1


def test_get_tests_for_cycle_filtra_por_nivel(db, cycle_id):
    for nivel in (1, 2, 3):
        db.save_test(_teste(nivel=nivel, cycle_id=cycle_id))
    assert len(db.get_tests_for_cycle(cycle_id)) == 3
    assert len(db.get_tests_for_cycle(cycle_id, level=2)) == 1


def test_get_tests_by_embedding_com_none_devolve_vazio(db):
    assert db.get_tests_by_embedding(None) == []


def test_get_tests_by_embedding_sem_testes_devolve_vazio(db):
    assert db.get_tests_by_embedding([0.1] * EMBEDDING_DIM) == []


def test_mark_tests_permanent_incrementa_times_used(db, cycle_id):
    t = _teste(cycle_id=cycle_id)
    db.save_test(t)
    db.mark_tests_permanent([t["test_id"]])
    assert db.get_tests_for_cycle(cycle_id)[0]["times_used"] == 1


def test_mark_tests_permanent_com_lista_vazia_nao_lanca(db):
    db.mark_tests_permanent([])
    db.mark_tests_permanent(["id-que-nao-existe"])


# --- Embedding e similaridade --------------------------------------------
def test_embed_task_tem_dimensao_fixa():
    assert len(embed_task("cria um validador de email")) == EMBEDDING_DIM


def test_embed_task_e_deterministico():
    assert embed_task("parser json") == embed_task("parser json")


def test_embed_task_distingue_tarefas():
    assert embed_task("parser json") != embed_task("scanner de rede")


def test_embed_task_vazia_devolve_none():
    assert embed_task("") is None


def test_embed_task_funciona_sem_sentence_transformers(monkeypatch):
    """O fallback determinístico é o caso inicial (ML_ENABLED=false)."""
    import agent.ml_features as mf

    monkeypatch.setattr(mf, "_sentence_model", None)
    monkeypatch.setattr(mf, "_sentence_model_tried", True)
    vector = mf.embed_task("tarefa qualquer")
    assert vector is not None and len(vector) == EMBEDDING_DIM


@pytest.mark.parametrize("a,b", [(None, [1.0]), ([1.0], None), ([], [1.0]),
                                 ([0.0, 0.0], [0.0, 0.0])])
def test_cosine_similarity_casos_degenerados(a, b):
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_vectores_iguais():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_ortogonais():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# ==========================================================================
# Geração de testes
# ==========================================================================
def _router_com_testes(por_nivel=2):
    """Router que devolve um array JSON de testes válido."""
    def resposta(prompt):
        nivel = 1
        for n in (1, 2, 3):
            if f"NÍVEL {n}" in prompt:
                nivel = n
        categoria = {1: "basic", 2: "edge", 3: "error"}[nivel]
        return json.dumps([
            {"neuron_target": "neuron_1", "language": "python",
             "level": nivel, "category": categoria,
             "description": f"teste {i} de nível {nivel}",
             "code": "assert True", "expected_outcome": "pass"}
            for i in range(por_nivel)
        ])
    return RouterFalso(respostas={"cerebellum": resposta})


def test_generate_devolve_testes_nos_tres_niveis(db, cycle_id, com_modelos,
                                                 ligada):
    gerador = TestGenerator(_router_com_testes(), db, None)
    testes = gerador.generate("validar email", "# [NEURON_1:python]\npass",
                              {"neuron_1": {"language": "python"}}, cycle_id)
    assert testes
    categorias = {t["category"] for t in testes}
    assert categorias == {"basic", "edge", "error"}
    assert {t["level"] for t in testes} == {1, 2, 3}


def test_generate_guarda_na_biblioteca(db, cycle_id, com_modelos, ligada):
    gerador = TestGenerator(_router_com_testes(), db, None)
    gerador.generate("validar email", "código", {}, cycle_id)
    assert len(db.get_tests_for_cycle(cycle_id)) == 6   # 2 por nível


def test_generate_com_json_invalido_devolve_vazio(db, cycle_id, com_modelos,
                                                  ligada):
    """Se o modelo não respeitar o formato, o ciclo continua sem testes."""
    gerador = TestGenerator(
        RouterFalso(respostas={"cerebellum": "isto não é JSON nenhum"}),
        db, None)
    assert gerador.generate("t", "c", {}, cycle_id) == []


def test_generate_sem_modelo_devolve_vazio(db, cycle_id, ligada):
    assert TestGenerator(RouterFalso(), db, None).generate(
        "t", "c", {}, cycle_id) == []


def test_generate_nunca_lanca(db, cycle_id, com_modelos, ligada):
    class RouterQueRebenta:
        def generate(self, *a, **k):
            raise RuntimeError("rebentou")

    assert TestGenerator(RouterQueRebenta(), db, None).generate(
        "t", "c", {}, cycle_id) == []


def test_generate_le_json_embrulhado(db, cycle_id, com_modelos, ligada):
    array = json.dumps([{"neuron_target": "all", "language": "python",
                         "level": 1, "category": "basic",
                         "description": "d", "code": "assert True",
                         "expected_outcome": "pass"}])
    gerador = TestGenerator(
        RouterFalso(respostas={"cerebellum": f"Aqui vão:\n```json\n{array}\n```"}),
        db, None)
    assert gerador.generate("t", "c", {}, cycle_id)


def test_testes_herdados_sao_incluidos(db, com_modelos, ligada, tmp_path):
    """Testes de um ciclo aprovado entram na lista devolvida."""
    antigo = db.create_cycle("validar endereço de email")
    embedding = embed_task("validar endereço de email")
    db.save_test({**_teste(cycle_id=antigo), "task_embedding": embedding,
                  "task_summary": "validar endereço de email"})
    db.update_cycle(antigo, status="approved", final_pct=99.0)

    novo = db.create_cycle("validar endereço de email")
    hippo = Hippocampus(db, {}, str(tmp_path / "m"))
    gerador = TestGenerator(_router_com_testes(), db, hippo)
    testes = gerador.generate("validar endereço de email", "c", {}, novo)
    assert any(t.get("times_used") is not None for t in testes), \
        "os testes herdados têm de aparecer na lista devolvida"


# ==========================================================================
# Execução de testes
# ==========================================================================
def test_run_tests_teste_que_passa(db, cycle_id):
    r = TestRunner(db).run_tests([_teste(codigo="assert 1 + 1 == 2")],
                                 cycle_id, 1)
    assert r["level_1"]["passed"] == 1
    assert r["results"][0]["outcome"] == "pass"


def test_run_tests_teste_que_falha(db, cycle_id):
    r = TestRunner(db).run_tests([_teste(codigo="assert 1 == 2")], cycle_id, 1)
    assert r["level_1"]["passed"] == 0
    assert r["results"][0]["outcome"] == "fail"


def test_teste_de_falha_esperada_que_falha_conta_como_passado(db, cycle_id):
    """expected_outcome='fail': o código DEVE rebentar."""
    r = TestRunner(db).run_tests(
        [_teste(codigo="raise ValueError('esperado')", esperado="fail")],
        cycle_id, 1)
    assert r["results"][0]["outcome"] == "pass"


def test_teste_de_falha_esperada_que_passa_conta_como_falhado(db, cycle_id):
    r = TestRunner(db).run_tests(
        [_teste(codigo="pass", esperado="fail")], cycle_id, 1)
    assert r["results"][0]["outcome"] == "fail"


def test_timeout_nao_bloqueia_o_ciclo(db, cycle_id, monkeypatch):
    monkeypatch.setenv("MUNDJI_SANDBOX_TIMEOUT", "1")
    r = TestRunner(db).run_tests(
        [_teste(codigo="import time; time.sleep(5)"),
         _teste(codigo="assert True")], cycle_id, 1)
    outcomes = [x["outcome"] for x in r["results"]]
    assert "timeout" in outcomes
    assert "pass" in outcomes, "o teste seguinte tem de correr na mesma"


def test_resultados_registados_na_test_results(db, cycle_id):
    t = _teste(cycle_id=cycle_id, codigo="assert True")
    db.save_test(t)
    TestRunner(db).run_tests([t], cycle_id, 1)
    linhas = db._conn.execute(
        "SELECT * FROM test_results WHERE cycle_id = ?", (cycle_id,)).fetchall()
    assert len(linhas) == 1
    assert linhas[0]["outcome"] == "pass"
    assert linhas[0]["duration_seconds"] is not None


def test_lista_vazia_devolve_breakdown_zerado(db, cycle_id):
    r = TestRunner(db).run_tests([], cycle_id, 1)
    assert all(r[f"level_{n}"]["total"] == 0 for n in (1, 2, 3))


# ==========================================================================
# Lógica de níveis
# ==========================================================================
@pytest.mark.parametrize("iteracao,esperado", [(1, 1), (2, 2), (3, 3),
                                               (4, 3), (10, 3)])
def test_nivel_da_iteracao(iteracao, esperado):
    assert min(max(1, iteracao), 3) == esperado


def test_acumulacao_activa_corre_niveis_anteriores(monkeypatch):
    monkeypatch.setenv("SANDBOX_ACCUMULATE_LEVELS", "true")
    todos = [_teste(nivel=n) for n in (1, 2, 3)]
    assert {t["level"] for t in seleccionar_testes(todos, 3)} == {1, 2, 3}


def test_acumulacao_desligada_corre_so_o_nivel(monkeypatch):
    monkeypatch.setenv("SANDBOX_ACCUMULATE_LEVELS", "false")
    todos = [_teste(nivel=n) for n in (1, 2, 3)]
    assert {t["level"] for t in seleccionar_testes(todos, 3)} == {3}


def test_iteracao_1_nunca_corre_niveis_superiores(monkeypatch):
    monkeypatch.setenv("SANDBOX_ACCUMULATE_LEVELS", "true")
    todos = [_teste(nivel=n) for n in (1, 2, 3)]
    assert {t["level"] for t in seleccionar_testes(todos, 1)} == {1}


# --- Fórmula da percentagem ----------------------------------------------
def _bd(p1=(0, 0), p2=(0, 0), p3=(0, 0)):
    return {f"level_{i+1}": {"passed": p[0], "total": p[1]}
            for i, p in enumerate((p1, p2, p3))}


def test_nivel_1_completo_da_33():
    assert calcular_percentagem(_bd(p1=(8, 8))) == 33.0


def test_niveis_1_e_2_completos_dao_66():
    assert calcular_percentagem(_bd(p1=(8, 8), p2=(8, 8))) == 66.0


def test_tres_niveis_completos_dao_99():
    assert calcular_percentagem(_bd(p1=(8, 8), p2=(8, 8), p3=(8, 8))) == 99.0


def test_com_qualidade_do_relatorio_da_100():
    assert calcular_percentagem(
        _bd(p1=(8, 8), p2=(8, 8), p3=(8, 8)), pct_relatorio=1.0) == 100.0


def test_percentagem_parcial_e_proporcional():
    # metade do nível 1 = 16.5
    assert calcular_percentagem(_bd(p1=(4, 8))) == 16.5


def test_sem_testes_a_percentagem_e_zero():
    assert calcular_percentagem({}) == 0.0


def test_nivel_sem_testes_nao_contribui():
    """Código que nunca foi confrontado com testes de limite não chega a 66."""
    assert calcular_percentagem(_bd(p1=(8, 8), p2=(0, 0))) == 33.0


def test_nivel_completo_reconhece():
    assert nivel_completo(_bd(p1=(8, 8)), 1) is True
    assert nivel_completo(_bd(p1=(7, 8)), 1) is False
    assert nivel_completo({}, 1) is False


# ==========================================================================
# Herança entre ciclos
# ==========================================================================
def test_testes_de_ciclo_aprovado_sao_herdados(db):
    aprovado = db.create_cycle("validar email de utilizador")
    emb = embed_task("validar email de utilizador")
    db.save_test({**_teste(cycle_id=aprovado), "task_embedding": emb})
    db.update_cycle(aprovado, status="approved", final_pct=99.0)

    herdados = db.get_tests_by_embedding(emb, min_similarity=0.5)
    assert len(herdados) == 1


def test_testes_de_ciclo_reprovado_nao_sao_herdados(db):
    """Herdar de um ciclo que nunca passou seria propagar critérios por provar."""
    reprovado = db.create_cycle("validar email de utilizador")
    emb = embed_task("validar email de utilizador")
    db.save_test({**_teste(cycle_id=reprovado), "task_embedding": emb})
    db.update_cycle(reprovado, status="needs_human", final_pct=40.0)

    assert db.get_tests_by_embedding(emb, min_similarity=0.5) == []


def test_tarefa_dissimilar_nao_herda(db):
    aprovado = db.create_cycle("validar email")
    db.save_test({**_teste(cycle_id=aprovado),
                  "task_embedding": embed_task("validar email")})
    db.update_cycle(aprovado, status="approved", final_pct=99.0)

    outro = embed_task("compilar um kernel de sistema operativo em assembly")
    assert db.get_tests_by_embedding(outro, min_similarity=0.95) == []


def test_limite_de_testes_herdados_e_respeitado(db):
    aprovado = db.create_cycle("tarefa comum")
    emb = embed_task("tarefa comum")
    for _ in range(10):
        db.save_test({**_teste(cycle_id=aprovado), "task_embedding": emb})
    db.update_cycle(aprovado, status="approved", final_pct=99.0)
    assert len(db.get_tests_by_embedding(emb, min_similarity=0.5, limit=3)) == 3


# ==========================================================================
# HIPPOCAMPUS
# ==========================================================================
def test_recommend_tests_com_ml_desligado_funciona(db, tmp_path, monkeypatch):
    """A consulta por embedding é independente de ML_ENABLED."""
    monkeypatch.setenv("ML_ENABLED", "false")
    aprovado = db.create_cycle("tarefa x")
    emb = embed_task("tarefa x")
    db.save_test({**_teste(cycle_id=aprovado), "task_embedding": emb})
    db.update_cycle(aprovado, status="approved", final_pct=99.0)

    hippo = Hippocampus(db, {}, str(tmp_path / "m"))
    assert len(hippo.recommend_tests(emb, min_similarity=0.5)) == 1


def test_recommend_tests_com_embedding_none(db, tmp_path):
    hippo = Hippocampus(db, {}, str(tmp_path / "m"))
    assert hippo.recommend_tests(None) == []


def test_recommend_tests_nunca_lanca(db, tmp_path):
    hippo = Hippocampus(db, {}, str(tmp_path / "m"))
    hippo.db = None    # força falha interna
    assert hippo.recommend_tests([0.1] * EMBEDDING_DIM) == []


# ==========================================================================
# Schema JSON estendido (retrocompatível)
# ==========================================================================
def test_relatorio_sem_campos_novos_usa_defaults():
    r = parse_relatorio(json.dumps({"functionality_pct": 87}))
    assert r.test_breakdown == {"level_1": None, "level_2": None,
                                "level_3": None}
    assert r.new_tests_generated == 0
    assert r.inherited_tests_used == 0
    assert r.tests_to_persist == []


def test_relatorio_com_campos_novos():
    r = parse_relatorio(json.dumps({
        "functionality_pct": 87,
        "test_breakdown": {"level_1": {"passed": 8, "total": 8, "pct": 100}},
        "new_tests_generated": 24, "inherited_tests_used": 12,
        "tests_to_persist": ["uuid1", "uuid2"],
    }))
    assert r.test_breakdown["level_1"]["passed"] == 8
    assert r.test_breakdown["level_2"] is None   # preenchido com o default
    assert r.new_tests_generated == 24
    assert r.inherited_tests_used == 12
    assert r.tests_to_persist == ["uuid1", "uuid2"]


def test_campos_novos_com_tipos_errados_nao_rebentam():
    r = parse_relatorio(json.dumps({
        "functionality_pct": 50, "test_breakdown": "isto devia ser objecto",
        "new_tests_generated": "muitos", "tests_to_persist": "uuid",
    }))
    assert r.test_breakdown == {"level_1": None, "level_2": None,
                                "level_3": None}
    assert r.new_tests_generated == 0
    assert r.tests_to_persist == []


# ==========================================================================
# REGRESSÃO DA BASE (crítico)
# ==========================================================================
def test_desligada_por_defeito():
    assert sandbox_tests_enabled() is False


def test_ciclo_completo_corre_como_antes(db, cycle_id, com_modelos,
                                         monkeypatch):
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "false")
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "3")
    final = MindGraph(RouterFalso(), db, {}, None).run(
        new_state("somar dois números", cycle_id))
    assert final["status"] == "approved"
    assert final["functionality_pct"] >= 98
    assert "[NEURON_" not in final["final_code"]


def test_nenhuma_chamada_a_geracao_quando_desligada(db, cycle_id, com_modelos,
                                                    monkeypatch):
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "false")
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "2")
    MindGraph(RouterFalso(), db, {}, None).run(new_state("somar", cycle_id))
    assert db._conn.execute(
        "SELECT COUNT(*) AS n FROM test_library").fetchone()["n"] == 0
    assert db._conn.execute(
        "SELECT COUNT(*) AS n FROM test_results").fetchone()["n"] == 0


def test_percentagem_continua_estimada_pelo_modelo(db, cycle_id, com_modelos,
                                                   monkeypatch):
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "false")
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "2")
    router = RouterFalso(respostas={"cerebellum": RouterFalso.avaliacao(99),
                                    "cortex": RouterFalso.avaliacao(99)})
    final = MindGraph(router, db, {}, None).run(new_state("somar", cycle_id))
    assert final["functionality_pct"] == 99.0, \
        "desligada, a percentagem é a estimativa do modelo"


def test_breakdown_fica_vazio_quando_desligada(db, cycle_id, com_modelos,
                                               monkeypatch):
    monkeypatch.setenv("SANDBOX_TESTS_ENABLED", "false")
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "2")
    final = MindGraph(RouterFalso(), db, {}, None).run(
        new_state("somar", cycle_id))
    assert not final.get("test_breakdown")


# ==========================================================================
# Integração com a extensão ligada
# ==========================================================================
def test_ciclo_com_sandbox_ligada_mede_a_percentagem(db, cycle_id,
                                                    com_modelos, ligada,
                                                    monkeypatch):
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "1")

    def resposta_cerebellum(prompt):
        if "NÍVEL" in prompt:
            nivel = next(n for n in (1, 2, 3) if f"NÍVEL {n}" in prompt)
            return json.dumps([{
                "neuron_target": "neuron_1", "language": "python",
                "level": nivel, "category": {1: "basic", 2: "edge",
                                             3: "error"}[nivel],
                "description": f"t{nivel}", "code": "assert True",
                "expected_outcome": "pass"}])
        return RouterFalso.avaliacao(99)

    router = RouterFalso(respostas={"cerebellum": resposta_cerebellum})
    final = MindGraph(router, db, {}, None).run(new_state("somar", cycle_id))

    # Iteração 1 -> só nível 1 -> 33% + 1% de qualidade de relatório.
    assert final["current_test_level"] == 1
    assert final["functionality_pct"] < 98, \
        "com testes reais, um nível 1 sozinho não pode aprovar"
    assert db._conn.execute(
        "SELECT COUNT(*) AS n FROM test_results").fetchone()["n"] >= 1


def test_testes_persistidos_na_aprovacao(db, cycle_id):
    """mark_tests_permanent é chamado no nó de aprovação."""
    t = _teste(cycle_id=cycle_id)
    db.save_test(t)
    antes = db.get_tests_for_cycle(cycle_id)[0]["times_used"]
    db.mark_tests_permanent([t["test_id"]])
    assert db.get_tests_for_cycle(cycle_id)[0]["times_used"] == antes + 1
