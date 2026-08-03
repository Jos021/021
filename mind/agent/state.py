"""Estado partilhado do grafo LangGraph do MIND.

Toda a comunicação CORTEX <-> CEREBELLUM passa por este estado, em memória,
sem latência de rede. Os NEURONS recebem cópias do que precisam via os nós
do grafo — nunca leem/escrevem o estado directamente entre si.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Estado do ciclo de geração de código.

    Campos preenchidos ao longo das três fases. `total=False` permite que
    o estado inicial arranque só com `task` e vá sendo completado.
    """

    # --- Entrada ---------------------------------------------------------
    task: str                        # descrição da tarefa a implementar

    # --- Fase 1: criação e refinamento -----------------------------------
    base_logic: str                  # lógica base definida pelo CORTEX
    base_code: str                   # código base (com marcadores [NEURON_N])
    markers: dict                    # {neuron_id: {"language": str}}
    cerebellum_feedback_f1: str      # feedback do CEREBELLUM na Fase 1

    # --- Fase 2: desenvolvimento -----------------------------------------
    active_neurons: list             # quais NEURONS correm nesta iteração
    neuron_outputs: dict             # {neuron_id: código devolvido}
    contract_violations: list        # lista de neuron_ids que violaram contrato
    # Fotografia das secções alheias por NEURON, tirada ANTES de distribuir.
    # É a referência do diff real na validação de contrato de interface.
    foreign_sections: dict           # {neuron_id: {secção: conteúdo}}
    contract_baseline: str           # código a que essa fotografia se refere
    organized_code: str              # código reunido/organizado pelo CORTEX

    # --- Fase 3: testes, avaliação e decisão -----------------------------
    cortex_test_report: str          # relatório independente do CORTEX
    cerebellum_report: str           # relatório independente do CEREBELLUM
    test_results: str                # saída bruta da sandbox
    functionality_pct: float         # % de funcionalidade (validação cruzada)
    improvements: dict               # {neuron_id: melhoria atribuída}

    # --- Controlo do ciclo -----------------------------------------------
    iteration: int                   # número da iteração actual
    cycle_id: int                    # id do ciclo na SYNAPSE DB
    status: str                      # in_progress | approved | needs_human

    # --- Rollback / melhor versão ----------------------------------------
    best_pct_so_far: float
    best_code_so_far: str

    # --- Saída final -----------------------------------------------------
    final_code: str                  # código final (sem marcadores, sanitizado)


def new_state(task: str, cycle_id: int) -> AgentState:
    """Cria um estado inicial limpo para um novo ciclo."""
    return AgentState(
        task=task,
        base_logic="",
        base_code="",
        markers={},
        cerebellum_feedback_f1="",
        active_neurons=[],
        neuron_outputs={},
        contract_violations=[],
        foreign_sections={},
        contract_baseline="",
        organized_code="",
        cortex_test_report="",
        cerebellum_report="",
        test_results="",
        functionality_pct=0.0,
        improvements={},
        iteration=0,
        cycle_id=cycle_id,
        status="in_progress",
        best_pct_so_far=0.0,
        best_code_so_far="",
        final_code="",
    )
