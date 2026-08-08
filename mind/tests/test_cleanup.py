"""Limpeza real do histórico git temporário (requisito 20).

Após aprovação, o histórico do workspace fica só com os commits permanentes
(functionality_pct >= GIT_PERMANENT_THRESHOLD) mais o commit final aprovado.
Os objectos dos commits descartados deixam de ser alcançáveis.

Se o ciclo terminar em needs_human, a limpeza NÃO corre e o histórico
completo mantém-se — é essa a rede de segurança para intervenção manual.
"""

import os

import pytest

from agent.graph import TAG_FINAL, TAG_PERMANENTE, GitVersioner, MindGraph
from agent.state import new_state
from tests.conftest import RouterFalso

git = pytest.importorskip("git", reason="GitPython não instalado")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("MUNDJI_WORKSPACE", str(ws))
    monkeypatch.setenv("GIT_PERMANENT_THRESHOLD", "70")
    return str(ws)


@pytest.fixture
def versioner(workspace):
    v = GitVersioner(workspace)
    assert v.repo is not None, "o repositório git tem de ser inicializado"
    return v


def _escrever(workspace, texto):
    with open(os.path.join(workspace, "codigo.py"), "w", encoding="utf-8") as fh:
        fh.write(texto)


def _historico(versioner):
    return [c.message.strip()
            for c in versioner.repo.iter_commits(
                versioner.repo.active_branch.name)]


def _objecto_existe(versioner, sha):
    """True se o objecto ainda existir fisicamente no repositório."""
    try:
        versioner.repo.git.cat_file("-e", f"{sha}^{{commit}}")
        return True
    except Exception:
        return False


# --- Marcação de commits permanentes -------------------------------------
def test_commit_acima_do_threshold_recebe_tag_permanente(versioner, workspace):
    _escrever(workspace, "v1")
    versioner.commit_iteration(1, 85.0)
    assert f"{TAG_PERMANENTE}1" in [t.name for t in versioner.repo.tags]


def test_commit_abaixo_do_threshold_nao_recebe_tag(versioner, workspace):
    _escrever(workspace, "v1")
    versioner.commit_iteration(1, 40.0)
    assert f"{TAG_PERMANENTE}1" not in [t.name for t in versioner.repo.tags]


def test_commit_final_recebe_tag(versioner, workspace):
    _escrever(workspace, "v1")
    versioner.commit_iteration(1, 40.0)
    _escrever(workspace, "final")
    versioner.commit_final(2, 99.0)
    assert TAG_FINAL in [t.name for t in versioner.repo.tags]


# --- Limpeza real ---------------------------------------------------------
@pytest.fixture
def historico_misto(versioner, workspace):
    """Cinco iterações: duas permanentes, três temporárias, mais o final."""
    shas = {}
    for iteracao, pct in [(1, 30.0), (2, 75.0), (3, 50.0), (4, 88.0), (5, 60.0)]:
        _escrever(workspace, f"versao {iteracao} — pct {pct}")
        shas[iteracao] = versioner.commit_iteration(iteracao, pct)
    _escrever(workspace, "output final aprovado")
    shas["final"] = versioner.commit_final(6, 99.0)
    return shas


def test_limpeza_preserva_apenas_permanentes_e_final(versioner, historico_misto):
    antes = _historico(versioner)
    assert len(antes) == 6

    relatorio = versioner.cleanup_temporary()
    assert relatorio["executed"], relatorio.get("reason")

    depois = _historico(versioner)
    assert len(depois) == 3, f"esperado 2 permanentes + 1 final, obtido: {depois}"
    assert any("aprovado" in m for m in depois)
    # As iterações permanentes (75% e 88%) ficam; as temporárias saem.
    assert any("iteracao 2" in m for m in depois)
    assert any("iteracao 4" in m for m in depois)
    for temporaria in ("iteracao 1", "iteracao 3", "iteracao 5"):
        assert not any(temporaria in m for m in depois), \
            f"{temporaria} era temporária e devia ter sido removida"


def test_relatorio_da_limpeza_e_coerente(versioner, historico_misto):
    relatorio = versioner.cleanup_temporary()
    assert relatorio["commits_antes"] == 6
    assert relatorio["commits_depois"] == 3
    assert relatorio["descartados"] == 3
    assert relatorio["preservados"] == 3


def test_objectos_temporarios_deixam_de_ser_alcancaveis(versioner,
                                                        historico_misto):
    """Não basta sair do log — os objectos têm de ser podados."""
    temporarios = [historico_misto[i] for i in (1, 3, 5)]
    for sha in temporarios:
        assert _objecto_existe(versioner, sha), "pré-condição: existiam antes"

    versioner.cleanup_temporary()

    ainda_existem = [sha for sha in temporarios if _objecto_existe(versioner, sha)]
    assert not ainda_existem, \
        f"objectos temporários ainda alcançáveis após gc: {ainda_existem}"


def test_tags_apontam_para_os_commits_novos(versioner, historico_misto):
    versioner.cleanup_temporary()
    nomes = [t.name for t in versioner.repo.tags]
    assert TAG_FINAL in nomes

    shas_do_historico = {c.hexsha for c in versioner.repo.iter_commits(
        versioner.repo.active_branch.name)}
    for tag in versioner.repo.tags:
        assert tag.commit.hexsha in shas_do_historico, \
            f"a tag {tag.name} aponta para fora do histórico reconstruído"


def test_conteudo_final_sobrevive_a_limpeza(versioner, workspace,
                                            historico_misto):
    versioner.cleanup_temporary()
    with open(os.path.join(workspace, "codigo.py"), encoding="utf-8") as fh:
        assert fh.read() == "output final aprovado"


def test_limpeza_sem_commits_permanentes_nao_corre(versioner, workspace):
    """Sem nada permanente, não há histórico a reconstruir."""
    for iteracao in (1, 2):
        _escrever(workspace, f"v{iteracao}")
        versioner.commit_iteration(iteracao, 30.0)
    relatorio = versioner.cleanup_temporary()
    assert relatorio["executed"] is False
    assert len(_historico(versioner)) == 2, "nada pode ser destruído"


def test_limpeza_sem_temporarios_nao_faz_nada(versioner, workspace):
    _escrever(workspace, "v1")
    versioner.commit_iteration(1, 90.0)
    relatorio = versioner.cleanup_temporary()
    assert relatorio["executed"] is False
    assert len(_historico(versioner)) == 1


def test_limpeza_sem_repositorio_devolve_relatorio(tmp_path):
    v = GitVersioner(str(tmp_path / "sem-git"))
    v.repo = None
    relatorio = v.cleanup_temporary()
    assert relatorio["executed"] is False


# --- needs_human: o histórico completo mantém-se --------------------------
def test_needs_human_nao_limpa_o_historico(db, cycle_id, com_modelos,
                                           workspace, monkeypatch):
    """Sem aprovação, a limpeza não corre — o histórico fica todo."""
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "3")
    router = RouterFalso(respostas={
        "cortex": lambda p: ("# [NEURON_1:python]\npass" if "Anota" in p
                             else "PCT: 40\nfalta muito"),
        "cerebellum": lambda p: "PCT: 40\nneuron_1: implementa a função",
    })
    grafo = MindGraph(router, db, {}, None)
    final = grafo.run(new_state("tarefa difícil", cycle_id))
    assert final["status"] == "needs_human"

    historico = _historico(grafo.git)
    assert len(historico) >= 3, \
        "em needs_human o histórico completo tem de ser mantido"
    assert not any("aprovado" in m for m in historico)


def test_ciclo_aprovado_limpa_o_historico(db, cycle_id, com_modelos,
                                          workspace, monkeypatch):
    """O caminho completo: aprovação seguida de limpeza efectiva."""
    monkeypatch.setenv("MUNDJI_MAX_ITERATIONS", "3")
    monkeypatch.setenv("GIT_PERMANENT_THRESHOLD", "70")
    grafo = MindGraph(RouterFalso(), db, {}, None)
    final = grafo.run(new_state("somar dois números", cycle_id))
    assert final["status"] == "approved"

    historico = _historico(grafo.git)
    assert any("aprovado" in m for m in historico)
