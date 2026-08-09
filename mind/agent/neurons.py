"""NEURON 1 a 6 — unidades de execução.

SEM PERSONA — só produzem código, sem comunicação narrativa.

Princípio estrutural (aplica-se mesmo com especialidades vazias):
  - Cada NEURON recebe SEMPRE o código base completo e a lógica/sintaxe já
    aprimoradas pelo CORTEX — garante alinhamento com o resto do sistema.
  - Apesar de receber tudo, cada NEURON desenvolve APENAS a parte do seu
    marcador [NEURON_N] — nunca escreve/altera código fora do seu âmbito.
  - Ao terminar, devolve o resultado ao CORTEX respeitando o contrato de
    interface (o output deve conter o próprio marcador e nenhum alheio).
  - Na Fase 3, só é chamado se o CORTEX lhe atribuir uma melhoria (activação
    dinâmica).

Comunicação CORTEX -> NEURONS via asyncio.gather() (ver agent/graph.py),
com retry+backoff (model_router) e circuit breaker por NEURON (timeout).
"""

import asyncio
import time

from .model_router import ModelError, ModelRouter, component_config

NEURON_SYSTEM = (
    "És um NEURON do MIND, uma unidade de execução sem personalidade. "
    "Recebes o código base completo e a lógica aprimorada, mas só implementas "
    "a secção do TEU marcador [NEURON_N]. Nunca escreves nem alteras código "
    "fora do teu âmbito, nunca introduzes marcadores de outros NEURONS. "
    "Devolves apenas código, com o teu marcador presente. Comentários em "
    "português europeu."
)


class Neuron:
    """Uma unidade de execução especializada."""

    def __init__(self, neuron_id: str, router: ModelRouter, db, specialty=None):
        self.neuron_id = neuron_id                  # 'neuron_1' ... 'neuron_6'
        self.n = neuron_id.split("_")[-1]
        self.router = router
        self.db = db
        self.specialty = specialty or {}
        self.endpoint, self.model, self.enabled = component_config(neuron_id)

    def _build_prompt(self, state: dict, improvement: str = "") -> str:
        """Constrói o prompt: contexto completo + âmbito restrito ao marcador."""
        spec = ""
        if self.specialty:
            spec = f"Especialidade: {self.specialty}\n"
        improve = f"\nMelhoria atribuída nesta ronda:\n{improvement}\n" if improvement else ""
        return (
            f"{spec}"
            "Lógica aprimorada:\n" + state.get("base_logic", "") + "\n\n"
            "Código base completo (para alinhamento — NÃO alterar fora do teu "
            f"marcador [NEURON_{self.n}]):\n"
            + (state.get("organized_code") or state.get("base_code", "")) + "\n"
            + improve +
            f"\nImplementa APENAS a secção [NEURON_{self.n}]. Devolve só essa "
            "secção, começando pelo teu marcador."
        )

    async def run(self, state: dict, improvement: str = "", timeout: float = 60.0) -> str:
        """Corre o NEURON de forma assíncrona (para asyncio.gather)."""
        t0 = time.time()
        if not self.model:
            # Sem modelo configurado: devolve o próprio marcador com um stub,
            # mantendo o contrato válido (marcador presente, sem alheios).
            return f"# [NEURON_{self.n}]\n# (sem modelo configurado)\npass"
        prompt = self._build_prompt(state, improvement)
        try:
            out = await self.router.agenerate(
                prompt=prompt,
                model=self.model,
                endpoint=self.endpoint,
                system=NEURON_SYSTEM,
                component=self.neuron_id,
                timeout=timeout,
                cycle_id=state.get("cycle_id"),
                iteration=state.get("iteration", 0),
            )
        except ModelError as exc:
            out = f"[NEURON_ERRO] {exc.message}"
        else:
            # Modelos reais embrulham o código em markdown e prosa. Extrair
            # o código cercado aqui, à entrada, garante que a validação de
            # contrato, a organização e a sandbox vêem código limpo. Sem
            # isto, um NEURON que respondesse "Aqui está: ```py ...```"
            # reprovava sempre por erro de sintaxe na sandbox. Marcadores
            # de erro internos não passam por aqui (só o ramo de sucesso).
            from .cortex import limpar_codigo_modelo

            out = limpar_codigo_modelo(out)
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "2", self.neuron_id,
            input_summary=(improvement or "implementar secção")[:200],
            output_summary="código da secção",
            full_output=out, duration_seconds=time.time() - t0,
        )
        return out


def build_neurons(router: ModelRouter, db, specialties: dict) -> dict:
    """Cria os 6 NEURONS, respeitando ENABLE_NEURON_N (existência)."""
    neurons = {}
    for n in range(1, 7):
        nid = f"neuron_{n}"
        neuron = Neuron(nid, router, db, specialty=(specialties or {}).get(nid))
        if neuron.enabled:
            neurons[nid] = neuron
    return neurons


async def run_neurons_parallel(
    neurons: dict,
    active_ids: list,
    state: dict,
    improvements: dict = None,
    timeout: float = 60.0,
) -> dict:
    """Dispara os NEURONS activos "em simultâneo" via asyncio.gather().

    Circuit breaker por NEURON: cada tarefa tem timeout individual
    (NEURON_TIMEOUT_SECONDS). Se um NEURON estoirar o tempo, regista-se erro
    e segue-se em frente sem bloquear os restantes.

    Nota sobre Redis: não é usado. Se os NEURONS partilham uma GPU local, a
    fila é natural e uma message queue não traz paralelismo real. asyncio
    chega e sobra nesta fase; Redis fica como caminho de migração futuro
    (múltiplas GPUs/máquinas).
    """
    improvements = improvements or {}
    tasks = {}
    for nid in active_ids:
        neuron = neurons.get(nid)
        if not neuron:
            continue
        tasks[nid] = asyncio.create_task(
            _run_with_breaker(neuron, state, improvements.get(nid, ""), timeout)
        )

    outputs = {}
    for nid, task in tasks.items():
        outputs[nid] = await task
    return outputs


async def _run_with_breaker(neuron: Neuron, state: dict, improvement: str, timeout: float) -> str:
    """Circuit breaker individual: aborta o NEURON se exceder o timeout total."""
    try:
        return await asyncio.wait_for(
            neuron.run(state, improvement, timeout=timeout), timeout=timeout
        )
    except asyncio.TimeoutError:
        neuron.db.log_decision(
            state["cycle_id"], state["iteration"], "cortex",
            f"Circuit breaker: {neuron.neuron_id} excedeu {timeout}s — cortado.",
        )
        return f"[NEURON_ERRO] {neuron.neuron_id} excedeu o timeout ({timeout}s)."
