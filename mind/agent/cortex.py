"""CORTEX — orquestrador central do MIND.

Único componente com persona (JARVIS — ver agent/persona.py). Conduz as
três fases do ciclo:

  FASE 1 — Criação: cria lógica/sintaxe/código base, recebe feedback do
           CEREBELLUM, aprimora, anota marcadores [NEURON_N] e distribui.
  FASE 2 — Desenvolvimento: valida contratos de interface, reúne e organiza
           o código dos NEURONS.
  FASE 3 — Testes: executa a sandbox, gera relatório próprio, e recebe do
           CEREBELLUM a decisão final (validação cruzada).

Divisão de responsabilidades: o CORTEX executa os testes, mas NÃO calcula
sozinho a percentagem de funcionalidade nem decide sozinho a aprovação —
isso é feito em conjunto com o CEREBELLUM através de validação cruzada.
"""

import re
import time

from .model_router import ModelError, ModelRouter, component_config
from .persona import cortex_system_prompt
from .sandbox import get_language_for_section, run_section
from .sanitizer import sanitize_output

# Marcador: # [NEURON_1] ou # [NEURON_1:python] ou // [NEURON_4:rust]
MARKER_RE = re.compile(
    r"(?:#|//)\s*\[NEURON_(\d+)(?::([a-zA-Z0-9#+]+))?\]"
)


def parse_markers(code: str) -> dict:
    """Extrai os marcadores do código -> {neuron_id: {'language': str}}."""
    markers: dict = {}
    for match in MARKER_RE.finditer(code):
        n = match.group(1)
        lang = (match.group(2) or "python").lower()
        markers[f"neuron_{n}"] = {"language": lang}
    return markers


def marker_present(code: str, neuron_id: str) -> bool:
    """True se o código contém o marcador do NEURON indicado."""
    n = neuron_id.split("_")[-1]
    return re.search(rf"(?:#|//)\s*\[NEURON_{n}(?::[a-zA-Z0-9#+]+)?\]", code) is not None


def find_other_markers(code: str, neuron_id: str) -> list:
    """Devolve marcadores de OUTROS NEURONS presentes no código."""
    own = neuron_id.split("_")[-1]
    found = []
    for match in MARKER_RE.finditer(code):
        if match.group(1) != own:
            found.append(f"neuron_{match.group(1)}")
    return sorted(set(found))


def extract_section(code: str, neuron_id: str) -> str:
    """Extrai o bloco de texto que segue o marcador de um NEURON.

    A secção vai do marcador até ao próximo marcador (de qualquer NEURON)
    ou até ao fim do ficheiro.
    """
    n = neuron_id.split("_")[-1]
    start_re = re.compile(rf"(?:#|//)\s*\[NEURON_{n}(?::[a-zA-Z0-9#+]+)?\]")
    m = start_re.search(code)
    if not m:
        return ""
    rest = code[m.end():]
    nxt = MARKER_RE.search(rest)
    return (rest[: nxt.start()] if nxt else rest).strip()


class Cortex:
    """Orquestrador com persona JARVIS."""

    def __init__(self, router: ModelRouter, db, specialties: dict, console=None):
        self.router = router
        self.db = db
        self.specialties = specialties or {}
        self.console = console
        self.endpoint, self.model, _ = component_config("cortex")
        self.system = cortex_system_prompt()

    # --- Utilitário de geração -------------------------------------------
    def _generate(self, prompt: str, timeout: float = 120.0) -> str:
        """Gera texto com a persona JARVIS. Degrada com aviso se sem modelo."""
        if not self.model:
            # Sem modelo configurado (campos _MODEL vazios por defeito).
            # Não inventamos código — devolvemos marcador de indisponibilidade
            # para que o ciclo prossiga de forma determinística e auditável.
            return ""
        try:
            return self.router.generate(
                prompt=prompt,
                model=self.model,
                endpoint=self.endpoint,
                system=self.system,
                component="cortex",
                timeout=timeout,
            )
        except ModelError as exc:
            return f"[CORTEX_ERRO] {exc.message}"

    def _log(self, msg: str) -> None:
        if self.console:
            self.console.print(f"[bold cyan]CORTEX[/] {msg}")

    # ================================================================
    # FASE 1 — Criação
    # ================================================================
    def create(self, state: dict) -> dict:
        """Cria lógica, sintaxe e código base da tarefa recebida."""
        t0 = time.time()
        task = state["task"]
        prompt = (
            "Tarefa a implementar:\n"
            f"{task}\n\n"
            "Define a LÓGICA base (passos e estrutura) e o CÓDIGO base "
            "inicial. Responde em duas secções separadas por '===CODIGO==='."
        )
        out = self._generate(prompt)
        if "===CODIGO===" in out:
            logic, code = out.split("===CODIGO===", 1)
        else:
            logic, code = out, out
        state["base_logic"] = logic.strip()
        state["base_code"] = code.strip()
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "1", "cortex",
            input_summary=task[:200], output_summary="lógica+código base",
            full_output=out, duration_seconds=time.time() - t0,
        )
        self._log("Fase 1: lógica e código base criados.")
        return state

    def refine(self, state: dict) -> dict:
        """Aprimora o código base com o feedback do CEREBELLUM."""
        t0 = time.time()
        feedback = state.get("cerebellum_feedback_f1", "")
        prompt = (
            "Lógica base:\n" + state["base_logic"] + "\n\n"
            "Código base:\n" + state["base_code"] + "\n\n"
            "Feedback do CEREBELLUM:\n" + feedback + "\n\n"
            "Aprimora o código base tendo em conta o feedback. Devolve só "
            "o código base aprimorado."
        )
        out = self._generate(prompt)
        if out and not out.startswith("[CORTEX_ERRO]"):
            state["base_code"] = out.strip()
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "1", "cortex",
            input_summary="feedback f1", output_summary="código aprimorado",
            full_output=out, duration_seconds=time.time() - t0,
        )
        self._log("Fase 1: código base aprimorado com o feedback.")
        return state

    def annotate_markers(self, state: dict) -> dict:
        """Anota o código base com marcadores [NEURON_N] por secção.

        Se o modelo já tiver produzido marcadores, aproveita-os; caso
        contrário pede-os explicitamente. Cada marcador pode especificar a
        linguagem: [NEURON_N:python], [NEURON_N:rust], etc.
        """
        t0 = time.time()
        code = state["base_code"]
        if not parse_markers(code):
            enabled = self._enabled_neurons()
            prompt = (
                "Código base:\n" + code + "\n\n"
                "Anota cada secção com um comentário-marcador do NEURON "
                "responsável, no formato [NEURON_N] ou [NEURON_N:linguagem]. "
                f"NEURONS disponíveis: {', '.join(enabled)}. "
                "Devolve só o código anotado."
            )
            out = self._generate(prompt)
            if out and not out.startswith("[CORTEX_ERRO]"):
                code = out.strip()
                state["base_code"] = code
        state["markers"] = parse_markers(code)
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "1", "cortex",
            input_summary="anotar marcadores",
            output_summary=f"marcadores: {list(state['markers'])}",
            full_output=code, duration_seconds=time.time() - t0,
        )
        self._log(f"Fase 1: marcadores anotados -> {list(state['markers'])}")
        return state

    def distribute(self, state: dict) -> dict:
        """1ª passagem: TODOS os NEURONS com especialidade/marcador correm.

        Activação dinâmica: na primeira passagem não há selecção — distribui
        o código base a todos os NEURONS disponíveis que tenham marcador.
        """
        markers = state.get("markers", {})
        enabled = set(self._enabled_neurons())
        active = [nid for nid in markers if nid in enabled]
        state["active_neurons"] = active
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cortex",
            f"1ª passagem: distribuído a todos os NEURONS activos {active}.",
        )
        self._log(f"Fase 1->2: distribuído a todos os NEURONS: {active}")
        return state

    # ================================================================
    # FASE 2 — Desenvolvimento
    # ================================================================
    def validate_contracts(self, state: dict) -> dict:
        """Valida o contrato de interface de cada resposta de NEURON.

        1. Contém pelo menos uma ocorrência do próprio marcador [NEURON_N].
        2. Não contém marcadores de OUTROS NEURONS.
        3. Não altera código fora da secção atribuída (heurística: a resposta
           não deve introduzir marcadores alheios nem exceder o âmbito).
        """
        violations = []
        for neuron_id, output in state.get("neuron_outputs", {}).items():
            if not output or output.startswith("[NEURON_ERRO]"):
                violations.append(neuron_id)
                self.db.log_decision(
                    state["cycle_id"], state["iteration"], "cortex",
                    f"{neuron_id} não respondeu ou devolveu erro.",
                )
                continue
            reasons = []
            if not marker_present(output, neuron_id):
                reasons.append("marcador próprio ausente")
            others = find_other_markers(output, neuron_id)
            if others:
                reasons.append(f"contém marcadores alheios {others}")
            if reasons:
                violations.append(neuron_id)
                self.db.log_iteration(
                    state["cycle_id"], state["iteration"], "2", "cortex",
                    input_summary=f"validar contrato {neuron_id}",
                    output_summary="VIOLAÇÃO: " + "; ".join(reasons),
                    full_output=output[:2000],
                )
                self.db.log_decision(
                    state["cycle_id"], state["iteration"], "cortex",
                    f"{neuron_id} violou contrato de interface: "
                    + "; ".join(reasons) + ".",
                )
        state["contract_violations"] = violations
        if violations:
            self._log(f"Fase 2: violações de contrato -> {violations}")
        else:
            self._log("Fase 2: todos os NEURONS respeitaram o contrato.")
        return state

    def organize(self, state: dict) -> dict:
        """Reúne e organiza o código dos NEURONS na estrutura marcada.

        Substitui cada secção marcada pelo output do respectivo NEURON,
        mantendo os marcadores (só são removidos após aprovação final).
        Marcadores órfãos (NEURON sem output nesta ronda) mantêm o código
        base anterior — nunca se compila resultado final com marcadores por
        preencher (isso é validado em cortex_approve).
        """
        t0 = time.time()
        organized = state.get("organized_code") or state["base_code"]
        for neuron_id, output in state.get("neuron_outputs", {}).items():
            if neuron_id in state.get("contract_violations", []):
                continue  # secção do violador não entra
            organized = self._replace_section(organized, neuron_id, output)
        state["organized_code"] = organized
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "2", "cortex",
            input_summary="organizar código",
            output_summary="código organizado", full_output=organized,
            duration_seconds=time.time() - t0,
        )
        self._log("Fase 2: código organizado a partir dos NEURONS.")
        return state

    def _replace_section(self, code: str, neuron_id: str, output: str) -> str:
        """Substitui a secção do NEURON pelo seu output, mantendo o marcador.

        Se o output do NEURON já incluir o marcador (contrato válido), usamos
        o output tal e qual para a secção; caso contrário anexamos ao marcador.
        """
        n = neuron_id.split("_")[-1]
        marker_re = re.compile(
            rf"((?:#|//)\s*\[NEURON_{n}(?::[a-zA-Z0-9#+]+)?\][^\n]*\n)"
        )
        m = marker_re.search(code)
        if not m:
            return code
        # Fim da secção = próximo marcador ou fim do ficheiro.
        after = code[m.end():]
        nxt = MARKER_RE.search(after)
        section_end = m.end() + (nxt.start() if nxt else len(after))
        # O output do NEURON substitui o corpo da secção (marcador incluído).
        new_body = output.strip() + "\n"
        return code[: m.start()] + new_body + code[section_end:]

    # ================================================================
    # FASE 3 — Testes, avaliação e decisão
    # ================================================================
    def test(self, state: dict) -> dict:
        """Executa os testes na sandbox multi-linguagem, secção a secção.

        A linguagem vem sempre do marcador (nunca heurística). Gera a saída
        bruta agregada em state['test_results'].
        """
        t0 = time.time()
        code = state["organized_code"]
        markers = state.get("markers", {})
        results = []
        for neuron_id in markers:
            section = extract_section(code, neuron_id)
            if not section:
                continue
            lang = get_language_for_section(neuron_id, markers)
            res = run_section(section, lang)
            results.append(
                f"[{neuron_id}:{lang}] success={res.success} "
                f"rc={res.returncode} timeout={res.timed_out}\n"
                f"stdout: {res.stdout[:500]}\n"
                f"stderr: {res.stderr[:500]}"
            )
        # Se não houver secções extraíveis, tenta correr tudo como python.
        if not results:
            res = run_section(code, "python")
            results.append(
                f"[full:python] success={res.success} rc={res.returncode}\n"
                f"stdout: {res.stdout[:500]}\nstderr: {res.stderr[:500]}"
            )
        state["test_results"] = "\n---\n".join(results)
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "3", "cortex",
            input_summary="executar sandbox",
            output_summary="resultados de testes",
            full_output=state["test_results"], duration_seconds=time.time() - t0,
        )
        self._log("Fase 3: testes executados na sandbox.")
        return state

    def report(self, state: dict) -> dict:
        """Gera o relatório INDEPENDENTE do CORTEX sobre os testes."""
        t0 = time.time()
        prompt = (
            "Resultados dos testes na sandbox:\n"
            + state["test_results"] + "\n\n"
            "Gera o TEU relatório independente: estima a percentagem de "
            "funcionalidade (0-100), lista falhas e melhorias necessárias. "
            "Começa a resposta com 'PCT: <numero>'."
        )
        out = self._generate(prompt)
        state["cortex_test_report"] = out
        pct = _extract_pct(out)
        self.db.log_report(
            state["cycle_id"], state["iteration"], pct,
            failures="(ver relatório cortex)", improvements="",
        )
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "3", "cortex",
            input_summary="relatório cortex",
            output_summary=f"pct estimada={pct}", full_output=out,
            duration_seconds=time.time() - t0,
        )
        self._log(f"Fase 3: relatório do CORTEX gerado (pct~{pct}).")
        return state

    # ================================================================
    # Refinamento (a partir da 2ª iteração) — activação dinâmica
    # ================================================================
    def select_neurons_for_improvement(self, state: dict) -> dict:
        """Selecciona só os NEURONS visados por melhoria nesta ronda.

        A partir da 2ª iteração, só correm os NEURONS que o CORTEX designou
        explicitamente. Um NEURON sem melhoria atribuída NÃO é chamado.
        """
        improvements = state.get("improvements", {})
        enabled = set(self._enabled_neurons())
        targeted = [nid for nid in improvements if nid in enabled]
        state["active_neurons"] = targeted
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cortex",
            f"Refinamento: só correm os NEURONS visados {targeted}.",
        )
        self._log(f"Refinamento: NEURONS seleccionados -> {targeted}")
        return state

    def distribute_improvements(self, state: dict) -> dict:
        """Distribui as melhorias atribuídas apenas aos NEURONS visados."""
        # Rollback: se activo e a iteração anterior foi pior que a melhor,
        # parte-se da melhor versão conhecida.
        import os as _os

        if _os.getenv("ENABLE_ROLLBACK", "true").lower() == "true":
            if (
                state.get("best_code_so_far")
                and state.get("functionality_pct", 0)
                < state.get("best_pct_so_far", 0)
            ):
                state["organized_code"] = state["best_code_so_far"]
                self.db.log_decision(
                    state["cycle_id"], state["iteration"], "cortex",
                    "Rollback: retomado a partir da melhor versão conhecida.",
                )
                self._log("Rollback: retomada a melhor versão até agora.")
        self._log("Refinamento: melhorias distribuídas aos NEURONS visados.")
        return state

    # ================================================================
    # Aprovação e compilação final
    # ================================================================
    def sanitize(self, state: dict) -> dict:
        """Aplica o filtro de sanitização antes da compilação final."""
        code = state["organized_code"]
        state["final_code"] = sanitize_output(code)
        self._log("Aprovação: output sanitizado.")
        return state

    def approve(self, state: dict) -> dict:
        """Remove marcadores e prepara a cópia final.

        Nunca compila com marcadores por preencher: se sobrarem marcadores
        órfãos, é falha de funcionalidade (o CEREBELLUM reflecte-a na %).
        A remoção dos marcadores acontece uma única vez, aqui.
        """
        code = state.get("final_code") or state["organized_code"]
        # Remover TODOS os marcadores [NEURON_N] (linha completa do comentário).
        cleaned = re.sub(
            r"(?m)^[ \t]*(?:#|//)\s*\[NEURON_\d+(?::[a-zA-Z0-9#+]+)?\][^\n]*\n?",
            "",
            code,
        )
        # Também remover marcadores inline residuais.
        cleaned = MARKER_RE.sub("", cleaned)
        state["final_code"] = cleaned.strip() + "\n"
        state["status"] = "approved"
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cortex",
            f"Aprovado a {state.get('functionality_pct', 0):.1f}% — "
            "marcadores removidos e cópia final compilada.",
        )
        self._log(
            f"Aprovado a {state.get('functionality_pct', 0):.1f}%. "
            "Cópia final pronta."
        )
        return state

    # --- Helpers ---------------------------------------------------------
    def _enabled_neurons(self) -> list:
        """NEURONS que EXISTEM no sistema (ENABLE_NEURON_N=true)."""
        out = []
        for n in range(1, 7):
            _, _, enabled = component_config(f"neuron_{n}")
            if enabled:
                out.append(f"neuron_{n}")
        return out


def _extract_pct(text: str) -> float:
    """Extrai a percentagem de um relatório ('PCT: 87' ou '87%')."""
    if not text:
        return 0.0
    m = re.search(r"PCT:\s*([\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"([\d.]+)\s*%", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0
