#!/usr/bin/env python3
"""TESTE TÉCNICO — o LangGraph exprime o fluxo real do MIND?

Verifica se as CINCO decisões abaixo se exprimem no grafo compilado, todas
ao mesmo tempo, sem distorcer a lógica:

  1. rollback: uma iteração pior volta a partir da melhor versão anterior
  2. regra de divergência: >15pp entre relatórios força 3ª ronda
  3. activação dinâmica: só os Neurons visados correm a partir da 2ª iteração
  4. circuit breaker: timeout individual por Neuron sem bloquear o ciclo
  5. limite de iterações com saída para needs_human

Cada uma é observada em execução real, não por inspecção do código.
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.update(
    MUNDJI_WORKSPACE=tempfile.mkdtemp(prefix="mind_spike_ws_"),
    MUNDJI_MAX_ITERATIONS="4", NEURON_TIMEOUT_SECONDS="1",
    DIVERGENCE_THRESHOLD="15", ENABLE_ROLLBACK="true",
    APPROVAL_THRESHOLD="98", CORTEX_MODEL="fake", CEREBELLUM_MODEL="fake",
    ML_ENABLED="false",
)
for n in range(1, 7):
    os.environ[f"NEURON_{n}_MODEL"] = "fake"

from langgraph.errors import GraphRecursionError            # noqa: E402
from langgraph.graph import END, StateGraph                 # noqa: E402

from agent.cerebellum import Cerebellum                     # noqa: E402
from agent.cortex import Cortex                             # noqa: E402
from agent.database import SynapseDB                        # noqa: E402
from agent.neurons import build_neurons, run_neurons_parallel  # noqa: E402
from agent.state import AgentState, new_state               # noqa: E402

REGISTO = {
    "neurons_por_iteracao": [],
    "terceira_ronda": False,
    "rollback": False,
    "breaker_cortou": False,
    "saida": None,
}


class RouterEnsaio:
    """Router que força divergência na 1ª ronda e degradação na 2ª.

    Serve para provocar, numa só execução, a 3ª ronda por divergência e o
    rollback por iteração pior.
    """

    def __init__(self):
        self.ronda = 0
        self.breaker_ja_disparou = False

    def generate(self, prompt, model, endpoint, system="", component="",
                 timeout=120):
        if component == "cortex":
            if "Anota" in prompt:
                return ("# [NEURON_1:python]\nprint(1)\n"
                        "# [NEURON_2:python]\nprint(2)")
            if "Aprimora" in prompt:
                return "# base"
            if "relatório" in prompt.lower():
                return "PCT: 90"          # CORTEX diz 90
            return "Lógica.\n===CODIGO===\n# base"
        if component == "cerebellum":
            if "reconcilia" in prompt.lower():
                REGISTO["terceira_ronda"] = True
                # A 1ª reconciliação fixa a melhor versão em 60; as seguintes
                # degradam, para que haja uma iteração pior que a melhor.
                return "PCT: 60" if self.ronda <= 1 else "PCT: 20"
            if "independente" in prompt:
                self.ronda += 1
                if self.ronda == 1:
                    return "PCT: 40\nneuron_1: melhora"   # 90 vs 40 = 50pp
                # Iterações seguintes: pior que a melhor -> força rollback
                return "PCT: 20\nneuron_1: melhora outra vez"
            return "auditoria ok"
        return ""

    async def agenerate(self, prompt, model, endpoint, system="",
                        component="", timeout=120):
        n = component.split("_")[-1]
        # O neuron_2 só estoira o tempo na PRIMEIRA vez: exercita o circuit
        # breaker sem bloquear o ciclo para sempre (um NEURON que falha o
        # contrato em todas as rondas nunca deixaria o ciclo chegar à Fase 3,
        # e não é isso que se quer medir aqui).
        if n == "2" and not self.breaker_ja_disparou:
            self.breaker_ja_disparou = True
            await asyncio.sleep(5)
        return f"# [NEURON_{n}:python]\nprint({n})"


def construir_grafo(cortex, cerebellum, neurons, max_iter, breaker):
    """Exprime o fluxo completo do MIND como StateGraph do LangGraph."""

    def neurons_run(state):
        activos = state.get("active_neurons", [])
        REGISTO["neurons_por_iteracao"].append(list(activos))
        saidas = asyncio.run(run_neurons_parallel(
            neurons, activos, state, state.get("improvements", {}), breaker))
        for nid, saida in saidas.items():
            if "NEURON_ERRO" in saida:
                REGISTO["breaker_cortou"] = True
        state["neuron_outputs"] = saidas
        return state

    def marcar_iteracao(state):
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def rollback_e_distribuir(state):
        cortex.select_neurons_for_improvement(state)
        cortex.distribute_improvements(state)
        # O rollback deteta-se pelo registo na SYNAPSE DB, não por comparar
        # o código: se a melhor versão for idêntica à actual, a substituição
        # acontece na mesma mas é invisível a olho nu.
        linha = cortex.db._conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE cycle_id=? "
            "AND decision_text LIKE 'Rollback:%'", (state["cycle_id"],)
        ).fetchone()
        if linha and linha["n"]:
            REGISTO["rollback"] = True
        return state

    def so_o_violador(state):
        """Após reprovação por contrato, só o NEURON violador volta à Fase 2."""
        state["active_neurons"] = list(state.get("improvements", {}).keys())
        return state

    def desistir(state):
        state["status"] = "needs_human"
        REGISTO["saida"] = "needs_human"
        return state

    def aprovar(state):
        cortex.sanitize(state)
        cortex.approve(state)
        REGISTO["saida"] = "approved"
        return state

    g = StateGraph(AgentState)
    g.add_node("cortex_create", cortex.create)
    g.add_node("cerebellum_evaluate_f1", cerebellum.evaluate_f1)
    g.add_node("cortex_refine", cortex.refine)
    g.add_node("cortex_annotate_markers", cortex.annotate_markers)
    g.add_node("cortex_distribute", cortex.distribute)
    g.add_node("marcar_iteracao", marcar_iteracao)
    g.add_node("neurons_run", neurons_run)
    g.add_node("cortex_validate_contracts", cortex.validate_contracts)
    g.add_node("cerebellum_reject", cerebellum.reject_contract)
    g.add_node("so_o_violador", so_o_violador)
    g.add_node("cortex_organize", cortex.organize)
    g.add_node("cerebellum_audit", cerebellum.audit)
    g.add_node("cortex_test", cortex.test)
    g.add_node("cortex_report", cortex.report)
    g.add_node("cerebellum_decide", cerebellum.compare_and_decide)
    g.add_node("cortex_sanitize_approve", aprovar)
    g.add_node("cortex_improve", rollback_e_distribuir)
    g.add_node("needs_human", desistir)

    g.set_entry_point("cortex_create")
    g.add_edge("cortex_create", "cerebellum_evaluate_f1")
    g.add_edge("cerebellum_evaluate_f1", "cortex_refine")
    g.add_edge("cortex_refine", "cortex_annotate_markers")
    g.add_edge("cortex_annotate_markers", "cortex_distribute")
    g.add_edge("cortex_distribute", "marcar_iteracao")
    g.add_edge("marcar_iteracao", "neurons_run")
    g.add_edge("neurons_run", "cortex_validate_contracts")

    g.add_conditional_edges(
        "cortex_validate_contracts",
        lambda s: "reject" if s.get("contract_violations") else "ok",
        {"reject": "cerebellum_reject", "ok": "cortex_organize"})
    g.add_conditional_edges(
        "cerebellum_reject",
        lambda s: ("desistir" if s.get("iteration", 0) >= max_iter
                   else "repetir"),
        {"desistir": "needs_human", "repetir": "so_o_violador"})
    g.add_edge("so_o_violador", "marcar_iteracao")
    g.add_edge("cortex_organize", "cerebellum_audit")
    g.add_edge("cerebellum_audit", "cortex_test")
    g.add_edge("cortex_test", "cortex_report")
    g.add_edge("cortex_report", "cerebellum_decide")

    def decidir(s):
        if s.get("status") == "approved":
            return "aprovar"
        if s.get("iteration", 0) >= max_iter:
            return "desistir"
        return "melhorar"

    g.add_conditional_edges(
        "cerebellum_decide", decidir,
        {"aprovar": "cortex_sanitize_approve", "desistir": "needs_human",
         "melhorar": "cortex_improve"})
    g.add_edge("cortex_improve", "marcar_iteracao")
    g.add_edge("cortex_sanitize_approve", END)
    g.add_edge("needs_human", END)
    return g.compile()


def main():
    db = SynapseDB(os.path.join(tempfile.mkdtemp(prefix="mind_spike_db_"), "spike.db"))
    cid = db.create_cycle("spike langgraph")
    router = RouterEnsaio()
    cortex = Cortex(router, db, {}, None)
    cerebellum = Cerebellum(router, db, None)
    neurons = build_neurons(router, db, {})

    app = construir_grafo(cortex, cerebellum, neurons, max_iter=4, breaker=1.0)
    estado = new_state("tarefa de ensaio", cid)

    try:
        final = app.invoke(estado, config={"recursion_limit": 200})
        erro = None
    except GraphRecursionError as exc:
        final, erro = None, f"GraphRecursionError: {exc}"

    print("=" * 68)
    print("RESULTADO DO TESTE TÉCNICO — LangGraph")
    print("=" * 68)
    if erro:
        print("FALHOU:", erro)
        db.close()
        return 1

    verificacoes = [
        ("1. rollback (iteração pior parte da melhor)", REGISTO["rollback"]),
        ("2. regra de divergência força 3ª ronda", REGISTO["terceira_ronda"]),
        ("3. activação dinâmica (só visados a partir da 2ª)",
         len(REGISTO["neurons_por_iteracao"]) >= 2
         and len(REGISTO["neurons_por_iteracao"][0]) == 2
         and all(len(r) == 1 for r in REGISTO["neurons_por_iteracao"][1:])),
        ("4. circuit breaker cortou sem bloquear o ciclo",
         REGISTO["breaker_cortou"]),
        ("5. limite de iterações -> needs_human",
         REGISTO["saida"] == "needs_human"),
    ]
    for nome, ok in verificacoes:
        print(f"  [{'OK ' if ok else 'NAO'}] {nome}")

    print(f"\n  neurons por iteração: {REGISTO['neurons_por_iteracao']}")
    print(f"  estado final: {final.get('status')}")
    print(f"  iterações: {final.get('iteration')}")

    todas = all(ok for _, ok in verificacoes)
    print("\n" + ("VEREDICTO: o LangGraph exprime o fluxo completo."
                  if todas else
                  "VEREDICTO: o LangGraph NAO exprime tudo sem contorções."))
    db.close()
    return 0 if todas else 2


if __name__ == "__main__":
    sys.exit(main())
