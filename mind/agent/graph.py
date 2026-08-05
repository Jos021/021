"""Grafo LangGraph do MIND.

Liga CORTEX, CEREBELLUM e NEURONS num ciclo iterativo de três fases até
atingir >=APPROVAL_THRESHOLD de funcionalidade ou esgotar MAX_ITERATIONS
(-> needs_human, com o histórico git mantido).

Fluxo:
  FASE 1: cortex_create -> cerebellum_evaluate_f1 -> cortex_refine
          -> cortex_annotate_markers -> cortex_distribute (todos os Neurons)
  FASE 2: [neurons activos] (asyncio.gather, retry + circuit breaker)
          -> cortex_validate_contracts
             SE violação: cerebellum_reject -> volta Fase 2 (só violador)
             SE ok: cortex_organize -> cerebellum_audit
  FASE 3: cortex_test -> cortex_report -> cerebellum_compare_and_decide
          REGRA DE DIVERGÊNCIA: diff > threshold -> 3ª ronda (no cerebellum)
          SE >=98: cortex_sanitize -> cortex_approve (limpa git, compila,
                   remove marcadores) -> FIM
          SE <98:  cortex_select_neurons_for_improvement
                   -> cortex_distribute_improvements -> volta à Fase 2
                   (SE ENABLE_ROLLBACK e iteração pior que a melhor: parte
                    da melhor)

Comunicação CORTEX <-> CEREBELLUM: estado do grafo (AgentState), directo,
em memória, sem latência de rede. CORTEX -> NEURONS: asyncio.gather().
Sem Redis (caminho de migração futuro para múltiplas GPUs/máquinas).
"""

import asyncio
import os

from .cerebellum import Cerebellum
from .cortex import Cortex
from .neurons import build_neurons, run_neurons_parallel

try:
    from langgraph.graph import END, StateGraph
    from .state import AgentState
    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - langgraph opcional em dev
    _HAS_LANGGRAPH = False


TAG_PERMANENTE = "permanent_iter_"
TAG_FINAL = "final_approved"
BRANCH_LIMPEZA = "mind-limpeza-temporaria"


class GitVersioner:
    """Versionamento git local do workspace/ (sem remote).

    Commit automático por iteração. Retenção permanente para iterações com
    functionality_pct >= GIT_PERMANENT_THRESHOLD, marcadas com a tag
    permanent_iter_N. Limpeza real do histórico temporário só após aprovação
    a 98%. Se needs_human, o histórico completo mantém-se intacto.
    """

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.permanent_threshold = float(os.getenv("GIT_PERMANENT_THRESHOLD", "70"))
        self.repo = None
        self._init_repo()

    def _init_repo(self) -> None:
        try:
            import git
        except Exception:
            return
        os.makedirs(self.workspace, exist_ok=True)
        try:
            self.repo = git.Repo(self.workspace)
        except Exception:
            try:
                self.repo = git.Repo.init(self.workspace)
                # Config local mínima para permitir commits sem identidade global.
                with self.repo.config_writer() as cw:
                    cw.set_value("user", "name", "MIND")
                    cw.set_value("user", "email", "mind@mundji.local")
            except Exception:
                self.repo = None

    def commit_iteration(self, iteration: int, pct: float) -> str:
        """Commit local da iteração; tag permanent_iter_N se >= threshold.

        Commita mesmo quando a árvore não mudou: o requisito é um commit por
        iteração, e o registo da percentagem atingida tem valor de auditoria
        por si só — é o que permite ler a trajectória do ciclo no histórico.
        Os commits abaixo do threshold são temporários e serão removidos na
        limpeza pós-aprovação.
        """
        if not self.repo:
            return ""
        try:
            self.repo.git.add(A=True)
            commit = self.repo.index.commit(
                f"iteracao {iteration} — funcionalidade {pct:.1f}%"
            )
            if pct >= self.permanent_threshold:
                tag = f"{TAG_PERMANENTE}{iteration}"
                if tag not in [t.name for t in self.repo.tags]:
                    self.repo.create_tag(tag, ref=commit.hexsha)
            return commit.hexsha
        except Exception:
            return ""

    def commit_final(self, iteration: int, pct: float) -> str:
        """Commit do output aprovado, marcado com a tag final_approved."""
        if not self.repo:
            return ""
        try:
            self.repo.git.add(A=True)
            if self.repo.is_dirty(untracked_files=True):
                commit = self.repo.index.commit(
                    f"aprovado — funcionalidade {pct:.1f}% (iteracao {iteration})"
                )
            else:
                commit = self.repo.head.commit
            if TAG_FINAL not in [t.name for t in self.repo.tags]:
                self.repo.create_tag(TAG_FINAL, ref=commit.hexsha)
            return commit.hexsha
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Limpeza real do histórico temporário
    # ------------------------------------------------------------------
    def cleanup_temporary(self) -> dict:
        """Após aprovação: reconstrói o histórico só com os permanentes.

        Só corre depois da aprovação — se o ciclo terminar em needs_human,
        esta função não é chamada e o histórico completo mantém-se.

        Passos, com git a sério:
          1. Identificar os commits permanentes (tags permanent_iter_N) mais
             o commit final aprovado, por ordem cronológica.
          2. Criar um branch órfão e reconstruir o histórico só com esses.
          3. Substituir o branch de trabalho pelo novo (git branch -M).
          4. Expirar o reflog e correr git gc --prune=now, para que os
             objectos dos commits descartados deixem de existir fisicamente.

        Devolve um relatório do que foi feito. Nota sobre o método de
        reconstrução: tenta-se git cherry-pick de cada commit, como manda a
        especificação; se um cherry-pick entrar em conflito (o que pode
        acontecer ao replicar diffs de uma história a que faltam commits
        intermédios), recorre-se a git commit-tree para preservar a árvore
        exacta desse commit. A segunda via é plumbing de git, não um atalho:
        preserva o snapshot aprovado em vez de um diff recalculado.
        """
        if not self.repo:
            return {"executed": False, "reason": "sem repositório git"}

        g = self.repo.git
        try:
            preservar = self._commits_a_preservar()
            if not preservar:
                return {"executed": False,
                        "reason": "nenhum commit permanente a preservar"}

            branch = self.repo.active_branch.name
            todos = [c.hexsha for c in self.repo.iter_commits(branch)]
            manter = {sha for sha, _ in preservar}
            descartados = [sha for sha in todos if sha not in manter]

            if not descartados:
                return {"executed": False, "reason": "nada temporário a limpar",
                        "preservados": len(preservar)}

            metodo = self._reconstruir(g, preservar, branch)
            self._reescrever_tags(g, preservar)
            self._podar(g)

            restantes = [c.hexsha for c in self.repo.iter_commits(branch)]
            return {
                "executed": True,
                "branch": branch,
                "preservados": len(preservar),
                "descartados": len(descartados),
                "commits_antes": len(todos),
                "commits_depois": len(restantes),
                "metodo": metodo,
            }
        except Exception as exc:
            # Limpeza falhada nunca pode destruir o trabalho: aborta e deixa
            # o histórico como estava.
            try:
                g.cherry_pick("--abort")
            except Exception:
                pass
            return {"executed": False, "reason": f"falha na limpeza: {exc}"}

    def _commits_a_preservar(self) -> list:
        """[(sha, mensagem)] dos permanentes + final, por ordem cronológica."""
        alvos = {}
        for tag in self.repo.tags:
            if tag.name.startswith(TAG_PERMANENTE) or tag.name == TAG_FINAL:
                alvos[tag.commit.hexsha] = tag.commit.message.strip()
        if not alvos:
            return []
        # Ordem cronológica real do histórico, não a ordem das tags.
        ordenados = []
        for commit in self.repo.iter_commits(self.repo.active_branch.name,
                                             reverse=True):
            if commit.hexsha in alvos:
                ordenados.append((commit.hexsha, alvos[commit.hexsha]))
        return ordenados

    def _reconstruir(self, g, preservar: list, branch: str) -> str:
        """Constrói o histórico novo num branch órfão. Devolve o método usado."""
        metodos = set()
        g.checkout("--orphan", BRANCH_LIMPEZA)
        g.reset("--hard")

        # O primeiro commit não pode ser cherry-picked: não há sobre o que
        # replicar um diff. Materializa-se a sua árvore directamente.
        sha, msg = preservar[0]
        novo = g.commit_tree(f"{sha}^{{tree}}", "-m", msg)
        g.update_ref(f"refs/heads/{BRANCH_LIMPEZA}", novo)
        g.reset("--hard", novo)
        metodos.add("commit-tree")

        for sha, msg in preservar[1:]:
            try:
                g.cherry_pick(sha)
                metodos.add("cherry-pick")
            except Exception:
                # Conflito: abortar e preservar a árvore exacta do commit.
                try:
                    g.cherry_pick("--abort")
                except Exception:
                    pass
                pai = g.rev_parse("HEAD")
                novo = g.commit_tree(f"{sha}^{{tree}}", "-p", pai, "-m", msg)
                g.update_ref(f"refs/heads/{BRANCH_LIMPEZA}", novo)
                g.reset("--hard", novo)
                metodos.add("commit-tree")

        g.branch("-M", branch)
        return "+".join(sorted(metodos))

    def _reescrever_tags(self, g, preservar: list) -> None:
        """Aponta as tags para os commits novos e apaga as antigas.

        Sem isto, as tags antigas manteriam os commits originais alcançáveis
        e, através dos seus pais, também os temporários — o gc não podaria
        nada e a limpeza seria só aparente.
        """
        for tag in list(self.repo.tags):
            if tag.name.startswith(TAG_PERMANENTE) or tag.name == TAG_FINAL:
                g.tag("-d", tag.name)

        novos = list(self.repo.iter_commits(self.repo.active_branch.name,
                                            reverse=True))
        for (_, msg), commit in zip(preservar, novos):
            if msg.startswith("aprovado"):
                nome = TAG_FINAL
            else:
                # "iteracao N — ..." -> permanent_iter_N
                partes = msg.split()
                nome = (f"{TAG_PERMANENTE}{partes[1]}"
                        if len(partes) > 1 and partes[0] == "iteracao"
                        else None)
            if nome and nome not in [t.name for t in self.repo.tags]:
                self.repo.create_tag(nome, ref=commit.hexsha)

    @staticmethod
    def _podar(g) -> None:
        """Torna os objectos descartados fisicamente inalcançáveis."""
        g.reflog("expire", "--expire=now", "--all")
        g.gc("--prune=now")

class MindGraph:
    """Orquestrador do ciclo do MIND.

    IMPLEMENTAÇÃO ÚNICA: o ciclo corre sobre o grafo LangGraph compilado.
    Não existe nenhuma segunda implementação executável do fluxo.

    A decisão foi tomada com base num teste técnico que verificou, em
    execução real e todas ao mesmo tempo, que o LangGraph exprime as cinco
    decisões do MIND sem distorcer a lógica: rollback para a melhor versão,
    3ª ronda por divergência, activação dinâmica de NEURONS, circuit breaker
    individual e limite de iterações com saída para needs_human.
    """

    def __init__(self, router, db, specialties, console=None, hippocampus=None):
        self.db = db
        self.console = console
        # HIPPOCAMPUS é opcional e sempre consultivo. Com ML_ENABLED=false
        # (ou hippocampus=None) o ciclo corre exactamente como a base.
        self.hippocampus = hippocampus
        self.cortex = Cortex(router, db, specialties, console, hippocampus)
        self.cerebellum = Cerebellum(router, db, console, hippocampus)
        self.neurons = build_neurons(router, db, specialties)
        self.max_iterations = int(os.getenv("MUNDJI_MAX_ITERATIONS", "10"))
        self.neuron_timeout = float(os.getenv("NEURON_TIMEOUT_SECONDS", "60"))
        # Sandbox evolutiva: quantas iterações se insiste no mesmo nível.
        self.max_iter_por_nivel = int(
            os.getenv("SANDBOX_MAX_ITER_PER_LEVEL", "3")
        )
        self._iters_no_nivel = 0
        self._nivel_observado = None
        workspace = os.getenv("MUNDJI_WORKSPACE", "./workspace")
        self.git = GitVersioner(workspace)
        self.workspace = workspace
        self._app = None

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def run(self, state: dict) -> dict:
        """Executa o ciclo completo sobre o grafo compilado."""
        app = self._app or self.build_langgraph()
        self._app = app
        # Cada iteração atravessa cerca de uma dezena de nós; o limite de
        # recursão do LangGraph tem de acomodar o pior caso do ciclo.
        limite = max(60, self.max_iterations * 20 + 60)
        final = app.invoke(state, config={"recursion_limit": limite})
        return dict(final)

    # ------------------------------------------------------------------
    # Nós que pertencem ao ciclo (e não a um agente em particular)
    # ------------------------------------------------------------------
    def _node_neurons_run(self, state: dict) -> dict:
        """Corre os NEURONS activos em paralelo, com circuit breaker."""
        outputs = asyncio.run(
            run_neurons_parallel(
                self.neurons,
                state.get("active_neurons", []),
                state,
                state.get("improvements", {}),
                self.neuron_timeout,
            )
        )
        state["neuron_outputs"] = outputs
        return state

    def _node_nova_iteracao(self, state: dict) -> dict:
        """Incrementa o contador de iterações do ciclo."""
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def _node_so_o_violador(self, state: dict) -> dict:
        """Após reprovação por contrato, só o NEURON violador volta à Fase 2."""
        state["active_neurons"] = list(state.get("improvements", {}).keys())
        return state

    def _node_versionar(self, state: dict) -> dict:
        """Escreve o código da iteração no workspace e commita-o.

        O código tem de estar em disco ANTES do commit, senão não há nada
        para versionar e o histórico — que serve o rollback e a intervenção
        manual em needs_human — ficaria vazio.
        """
        self._write_iteration_snapshot(state)
        self.git.commit_iteration(
            state.get("iteration", 0), state.get("functionality_pct", 0.0)
        )
        return state

    def _node_melhorar(self, state: dict) -> dict:
        """Selecciona os NEURONS visados e aplica o rollback, se for caso."""
        state = self.cortex.select_neurons_for_improvement(state)
        return self.cortex.distribute_improvements(state)

    def _node_aprovar(self, state: dict) -> dict:
        """Sanitiza, compila o output final e limpa o histórico temporário."""
        state = self.cortex.sanitize(state)
        state = self.cortex.approve(state)
        pct = state.get("functionality_pct", 0.0)
        # A ordem importa: o output final é escrito e commitado antes da
        # limpeza, para que o commit aprovado o contenha e seja o topo do
        # histórico preservado.
        self._compile_output(state)
        self.git.commit_final(state.get("iteration", 0), pct)
        relatorio = self.git.cleanup_temporary()
        if self.console and relatorio.get("executed"):
            self.console.print(
                f"[bold cyan]MIND[/] Histórico git limpo: "
                f"{relatorio['commits_antes']} -> {relatorio['commits_depois']} "
                f"commits ({relatorio['descartados']} temporários removidos)."
            )
        # Sandbox evolutiva: os testes usados num ciclo aprovado passam a
        # estar disponíveis para ciclos futuros com tarefas semelhantes.
        self.db.mark_tests_permanent(state.get("tests_to_persist") or [])
        self.db.update_cycle(state["cycle_id"], status="approved", final_pct=pct)
        return state

    def _node_needs_human(self, state: dict) -> dict:
        """Limite de iterações atingido: histórico git mantido na íntegra."""
        state["status"] = "needs_human"
        self.db.update_cycle(
            state["cycle_id"], status="needs_human",
            final_pct=state.get("functionality_pct", 0.0),
        )
        if self.console:
            self.console.print(
                "[bold yellow]MIND[/] Limite de iterações atingido — "
                "needs_human. Histórico git mantido."
            )
        return state

    # ------------------------------------------------------------------
    # Decisões condicionais do grafo
    # ------------------------------------------------------------------
    def _rota_contrato(self, state: dict) -> str:
        return "reprovar" if state.get("contract_violations") else "seguir"

    def _rota_apos_reprovacao(self, state: dict) -> str:
        if state.get("iteration", 0) >= self.max_iterations:
            return "desistir"
        return "repetir"

    def _rota_decisao(self, state: dict) -> str:
        if state.get("status") == "approved":
            return "aprovar"
        if state.get("iteration", 0) >= self.max_iterations:
            return "desistir"
        if self._nivel_estagnado(state):
            return "desistir"
        return "melhorar"

    def _nivel_estagnado(self, state: dict) -> bool:
        """Limite por nível da sandbox evolutiva.

        Se ao fim de SANDBOX_MAX_ITER_PER_LEVEL iterações o nível actual não
        foi completamente superado, passa-se a needs_human com a razão
        registada, em vez de insistir indefinidamente no mesmo nível. O
        histórico git mantém-se intacto, como em qualquer needs_human.
        """
        from .test_generator import sandbox_tests_enabled
        from .test_runner import nivel_completo

        if not sandbox_tests_enabled():
            return False
        breakdown = state.get("test_breakdown") or {}
        if not breakdown:
            return False

        nivel = state.get("current_test_level", 1)
        if nivel_completo(breakdown, nivel):
            self._iters_no_nivel = 0
            self._nivel_observado = nivel
            return False

        if getattr(self, "_nivel_observado", None) != nivel:
            self._nivel_observado = nivel
            self._iters_no_nivel = 0
        self._iters_no_nivel = getattr(self, "_iters_no_nivel", 0) + 1

        if self._iters_no_nivel >= self.max_iter_por_nivel:
            razao = (
                f"Nível {nivel} não superado ao fim de "
                f"{self._iters_no_nivel} iterações — needs_human."
            )
            self.db.log_decision(
                state["cycle_id"], state.get("iteration", 0), "cortex", razao
            )
            if self.console:
                self.console.print(f"[bold yellow]MIND[/] {razao}")
            return True
        return False

    # ------------------------------------------------------------------
    # Construção do grafo
    # ------------------------------------------------------------------
    def build_langgraph(self):
        """Constrói e compila o StateGraph — a única forma de executar o ciclo.

        Fase 1: cortex_create -> cerebellum_evaluate_f1 -> cortex_refine
                -> cortex_annotate_markers -> cortex_distribute
        Fase 2: neurons_run (asyncio.gather + circuit breaker)
                -> cortex_validate_contracts
                   -> violação: cerebellum_reject -> volta à Fase 2 só para
                      o violador
                   -> ok: cortex_organize -> cerebellum_audit
        Fase 3: cortex_test -> cortex_report -> cerebellum_compare_and_decide
                -> aprovar | melhorar (volta à Fase 2) | needs_human
        """
        if not _HAS_LANGGRAPH:
            raise RuntimeError(
                "LangGraph não está instalado (pip install -r requirements.txt)."
            )

        g = StateGraph(AgentState)

        # --- Fase 1 ---
        g.add_node("cortex_create", self.cortex.create)
        g.add_node("cerebellum_evaluate_f1", self.cerebellum.evaluate_f1)
        g.add_node("cortex_refine", self.cortex.refine)
        g.add_node("cortex_annotate_markers", self.cortex.annotate_markers)
        g.add_node("cortex_distribute", self.cortex.distribute)
        # --- Ciclo ---
        g.add_node("nova_iteracao", self._node_nova_iteracao)
        g.add_node("neurons_run", self._node_neurons_run)
        g.add_node("cortex_validate_contracts", self.cortex.validate_contracts)
        g.add_node("cerebellum_reject", self.cerebellum.reject_contract)
        g.add_node("versionar_reprovacao", self._node_versionar)
        g.add_node("so_o_violador", self._node_so_o_violador)
        g.add_node("cortex_organize", self.cortex.organize)
        g.add_node("cerebellum_audit", self.cerebellum.audit)
        # --- Fase 3 ---
        g.add_node("cortex_test", self.cortex.test)
        g.add_node("cortex_report", self.cortex.report)
        g.add_node("cerebellum_compare_and_decide",
                   self.cerebellum.compare_and_decide)
        g.add_node("versionar_iteracao", self._node_versionar)
        # --- Saídas ---
        g.add_node("cortex_melhorar", self._node_melhorar)
        g.add_node("cortex_aprovar", self._node_aprovar)
        g.add_node("needs_human", self._node_needs_human)

        g.set_entry_point("cortex_create")
        g.add_edge("cortex_create", "cerebellum_evaluate_f1")
        g.add_edge("cerebellum_evaluate_f1", "cortex_refine")
        g.add_edge("cortex_refine", "cortex_annotate_markers")
        g.add_edge("cortex_annotate_markers", "cortex_distribute")
        g.add_edge("cortex_distribute", "nova_iteracao")
        g.add_edge("nova_iteracao", "neurons_run")
        g.add_edge("neurons_run", "cortex_validate_contracts")

        g.add_conditional_edges(
            "cortex_validate_contracts", self._rota_contrato,
            {"reprovar": "cerebellum_reject", "seguir": "cortex_organize"},
        )
        g.add_edge("cerebellum_reject", "versionar_reprovacao")
        g.add_conditional_edges(
            "versionar_reprovacao", self._rota_apos_reprovacao,
            {"desistir": "needs_human", "repetir": "so_o_violador"},
        )
        g.add_edge("so_o_violador", "nova_iteracao")

        g.add_edge("cortex_organize", "cerebellum_audit")
        g.add_edge("cerebellum_audit", "cortex_test")
        g.add_edge("cortex_test", "cortex_report")
        g.add_edge("cortex_report", "cerebellum_compare_and_decide")
        g.add_edge("cerebellum_compare_and_decide", "versionar_iteracao")

        g.add_conditional_edges(
            "versionar_iteracao", self._rota_decisao,
            {"aprovar": "cortex_aprovar", "desistir": "needs_human",
             "melhorar": "cortex_melhorar"},
        )
        g.add_edge("cortex_melhorar", "nova_iteracao")
        g.add_edge("cortex_aprovar", END)
        g.add_edge("needs_human", END)

        return g.compile()

    # ------------------------------------------------------------------
    # Escrita no workspace
    # ------------------------------------------------------------------
    def _write_iteration_snapshot(self, state: dict) -> None:
        """Escreve o código da iteração no workspace, para o git o versionar."""
        os.makedirs(self.workspace, exist_ok=True)
        caminho = os.path.join(self.workspace, "trabalho_em_curso.txt")
        with open(caminho, "w", encoding="utf-8") as fh:
            fh.write(state.get("organized_code", "") or "")

    def _compile_output(self, state: dict) -> None:
        """Escreve a cópia final aprovada em workspace/output/.

        ------------------------------------------------------------------
        DECISÃO CONSCIENTE DE NÃO-FAZER: saída multi-ficheiro
        ------------------------------------------------------------------
        A compilação final produz um único ficheiro. Isto é uma decisão
        tomada, não uma lacuna por resolver: o impacto é baixo enquanto as
        tarefas couberem num ficheiro, e a alternativa acrescenta estrutura
        que ainda não é precisa.

        Evolução natural, quando for preciso gerar projectos com mais do que
        um ficheiro: fazer os marcadores transportarem também o ficheiro de
        destino, no formato [NEURON_2:python:auth/tokens.py]. O parser de
        marcadores passaria a extrair três campos em vez de dois, e esta
        função escreveria cada secção no seu caminho. O resto do sistema —
        contrato de interface, sandbox por linguagem, activação dinâmica —
        não precisaria de mudar.
        """
        out_dir = os.path.join(self.workspace, "output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "resultado_final.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(state.get("final_code", ""))
        if self.console:
            self.console.print(
                f"[bold green]MIND[/] Cópia final compilada em {path}"
            )
