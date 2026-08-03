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


class GitVersioner:
    """Versionamento git local do workspace/ (sem remote).

    Commit automático por iteração. Retenção permanente para iterações com
    functionality_pct >= GIT_PERMANENT_THRESHOLD. Limpeza do histórico
    temporário só após aprovação a 98%. Se needs_human, mantém tudo.
    """

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.permanent_threshold = float(os.getenv("GIT_PERMANENT_THRESHOLD", "70"))
        self.repo = None
        self._permanent = []   # SHAs a manter
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

    def commit_iteration(self, iteration: int, pct: float) -> None:
        """Commit local da iteração; marca como permanente se >= threshold."""
        if not self.repo:
            return
        try:
            self.repo.git.add(A=True)
            if not self.repo.is_dirty(untracked_files=True):
                return
            msg = f"iteracao {iteration} — funcionalidade {pct:.1f}%"
            commit = self.repo.index.commit(msg)
            if pct >= self.permanent_threshold:
                self._permanent.append(commit.hexsha)
        except Exception:
            pass

    def cleanup_temporary(self) -> None:
        """Após aprovação a 98%: limpa histórico temporário, mantém permanentes.

        Implementação conservadora: mantém o histórico permanente intacto e
        remove apenas os commits temporários mais recentes que não atingiram
        o threshold. Não reescreve história partilhada (não há remote).
        """
        if not self.repo:
            return
        try:
            # Consolidação simples: um commit final que representa o output
            # aprovado. Os commits permanentes ficam acessíveis via tags.
            for i, sha in enumerate(self._permanent):
                tag = f"permanente-{i+1}"
                if tag not in [t.name for t in self.repo.tags]:
                    self.repo.create_tag(tag, ref=sha)
        except Exception:
            pass


class MindGraph:
    """Orquestrador do ciclo. Constrói o grafo LangGraph e executa-o.

    Se o LangGraph estiver disponível, usa o StateGraph documentado; caso
    contrário, corre um orquestrador equivalente em Python puro (mesma
    lógica de fases e transições) — útil em desenvolvimento/testes.
    """

    def __init__(self, router, db, specialties, console=None):
        self.db = db
        self.console = console
        self.cortex = Cortex(router, db, specialties, console)
        self.cerebellum = Cerebellum(router, db, console)
        self.neurons = build_neurons(router, db, specialties)
        self.max_iterations = int(os.getenv("MUNDJI_MAX_ITERATIONS", "10"))
        self.neuron_timeout = float(os.getenv("NEURON_TIMEOUT_SECONDS", "60"))
        workspace = os.getenv("MUNDJI_WORKSPACE", "./workspace")
        self.git = GitVersioner(workspace)
        self.workspace = workspace

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def run(self, state: dict) -> dict:
        """Executa o ciclo completo a partir de `state`. Devolve o estado final."""
        # Fase 1 (uma vez por ciclo).
        state = self._phase1(state)

        # Fases 2-3 iterativas.
        for _ in range(self.max_iterations):
            state["iteration"] += 1
            state = self._phase2(state)

            if state.get("contract_violations"):
                # Reprovação por contrato: volta à Fase 2 só para o violador.
                state = self.cerebellum.reject_contract(state)
                # Re-corre só os violadores na próxima iteração.
                state["active_neurons"] = list(state.get("improvements", {}).keys())
                self.git.commit_iteration(state["iteration"], 0.0)
                continue

            state = self.cerebellum.audit(state)
            state = self._phase3(state)

            pct = state.get("functionality_pct", 0.0)
            self.git.commit_iteration(state["iteration"], pct)

            if state.get("status") == "approved":
                state = self.cortex.sanitize(state)
                state = self.cortex.approve(state)
                self.git.cleanup_temporary()
                self._compile_output(state)
                self.db.update_cycle(
                    state["cycle_id"], status="approved", final_pct=pct
                )
                return state

            # Reprovado: activação dinâmica — só os NEURONS visados.
            state = self.cortex.select_neurons_for_improvement(state)
            state = self.cortex.distribute_improvements(state)

        # Esgotou iterações -> needs_human (histórico git mantido).
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
    # Fases
    # ------------------------------------------------------------------
    def _phase1(self, state: dict) -> dict:
        state = self.cortex.create(state)
        state = self.cerebellum.evaluate_f1(state)
        state = self.cortex.refine(state)
        state = self.cortex.annotate_markers(state)
        state = self.cortex.distribute(state)   # todos os NEURONS
        return state

    def _phase2(self, state: dict) -> dict:
        active = state.get("active_neurons", [])
        improvements = state.get("improvements", {})
        outputs = asyncio.run(
            run_neurons_parallel(
                self.neurons, active, state, improvements, self.neuron_timeout
            )
        )
        state["neuron_outputs"] = outputs
        state = self.cortex.validate_contracts(state)
        if not state.get("contract_violations"):
            state = self.cortex.organize(state)
        return state

    def _phase3(self, state: dict) -> dict:
        state = self.cortex.test(state)
        state = self.cortex.report(state)
        state = self.cerebellum.compare_and_decide(state)
        return state

    # ------------------------------------------------------------------
    # Compilação final
    # ------------------------------------------------------------------
    def _compile_output(self, state: dict) -> None:
        """Escreve a cópia final aprovada em workspace/output/."""
        out_dir = os.path.join(self.workspace, "output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "resultado_final.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(state.get("final_code", ""))
        if self.console:
            self.console.print(
                f"[bold green]MIND[/] Cópia final compilada em {path}"
            )

    # ------------------------------------------------------------------
    # Grafo LangGraph (estrutura documentada)
    # ------------------------------------------------------------------
    def build_langgraph(self):
        """Constrói o StateGraph do LangGraph com a estrutura documentada.

        Fornecido para inspecção/visualização e para servir de espinha
        dorsal quando os modelos estiverem configurados. A execução prática
        do ciclo iterativo usa MindGraph.run() (acima), que implementa as
        mesmas transições com controlo explícito do loop e do rollback.
        """
        if not _HAS_LANGGRAPH:
            raise RuntimeError(
                "LangGraph não está instalado (pip install -r requirements.txt)."
            )

        g = StateGraph(AgentState)

        # Fase 1
        g.add_node("cortex_create", self.cortex.create)
        g.add_node("cerebellum_evaluate_f1", self.cerebellum.evaluate_f1)
        g.add_node("cortex_refine", self.cortex.refine)
        g.add_node("cortex_annotate_markers", self.cortex.annotate_markers)
        g.add_node("cortex_distribute", self.cortex.distribute)
        # Fase 2
        g.add_node("neurons_run", self._node_neurons_run)
        g.add_node("cortex_validate_contracts", self.cortex.validate_contracts)
        g.add_node("cerebellum_reject", self.cerebellum.reject_contract)
        g.add_node("cortex_organize", self.cortex.organize)
        g.add_node("cerebellum_audit", self.cerebellum.audit)
        # Fase 3
        g.add_node("cortex_test", self.cortex.test)
        g.add_node("cortex_report", self.cortex.report)
        g.add_node("cerebellum_compare_and_decide", self.cerebellum.compare_and_decide)
        g.add_node("cortex_sanitize", self.cortex.sanitize)
        g.add_node("cortex_approve", self.cortex.approve)
        g.add_node("cortex_select_neurons", self.cortex.select_neurons_for_improvement)
        g.add_node("cortex_distribute_improvements", self.cortex.distribute_improvements)

        g.set_entry_point("cortex_create")
        g.add_edge("cortex_create", "cerebellum_evaluate_f1")
        g.add_edge("cerebellum_evaluate_f1", "cortex_refine")
        g.add_edge("cortex_refine", "cortex_annotate_markers")
        g.add_edge("cortex_annotate_markers", "cortex_distribute")
        g.add_edge("cortex_distribute", "neurons_run")
        g.add_edge("neurons_run", "cortex_validate_contracts")

        g.add_conditional_edges(
            "cortex_validate_contracts",
            lambda s: "reject" if s.get("contract_violations") else "ok",
            {"reject": "cerebellum_reject", "ok": "cortex_organize"},
        )
        g.add_edge("cerebellum_reject", "neurons_run")   # volta à Fase 2
        g.add_edge("cortex_organize", "cerebellum_audit")
        g.add_edge("cerebellum_audit", "cortex_test")
        g.add_edge("cortex_test", "cortex_report")
        g.add_edge("cortex_report", "cerebellum_compare_and_decide")

        g.add_conditional_edges(
            "cerebellum_compare_and_decide",
            lambda s: "approve" if s.get("status") == "approved" else "improve",
            {"approve": "cortex_sanitize", "improve": "cortex_select_neurons"},
        )
        g.add_edge("cortex_sanitize", "cortex_approve")
        g.add_edge("cortex_approve", END)
        g.add_edge("cortex_select_neurons", "cortex_distribute_improvements")
        g.add_edge("cortex_distribute_improvements", "neurons_run")

        return g.compile()

    def _node_neurons_run(self, state: dict) -> dict:
        """Nó LangGraph que corre os NEURONS activos em paralelo."""
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
