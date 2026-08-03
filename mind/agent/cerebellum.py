"""CEREBELLUM — revisor e auditor técnico.

SEM PERSONA — output técnico, directo, estruturado. Nunca narra.

  FASE 1: recebe lógica/sintaxe/código base, avalia e devolve melhorias.
  FASE 2: audita o código organizado; se o CORTEX reportou violação de
          contrato, reprova o ciclo nesse ponto sem avançar à Fase 3.
  FASE 3: acompanha os testes de forma independente, gera o seu próprio
          relatório, compara os dois (validação cruzada), calcula a % final
          e decide aprovado (>=98%) ou reprovado (<98%).

Regra de divergência: se |pct_cortex - pct_cerebellum| > DIVERGENCE_THRESHOLD
(default 15pp), força-se uma 3ª ronda de verificação automática antes de
qualquer decisão.
"""

import os
import re
import time

from .model_router import ModelError, ModelRouter, component_config

# CEREBELLUM não tem persona: system prompt técnico e neutro.
CEREBELLUM_SYSTEM = (
    "És o CEREBELLUM, revisor e auditor técnico do MIND. Sem personalidade. "
    "Produzes apenas output técnico, directo e estruturado. Avalias lógica, "
    "sintaxe e funcionalidade, calculas percentagens e atribuis melhorias a "
    "NEURONS específicos (nunca genéricas). Responde sempre em português "
    "europeu, conciso."
)


class Cerebellum:
    """Revisor/auditor com validação cruzada."""

    def __init__(self, router: ModelRouter, db, console=None):
        self.router = router
        self.db = db
        self.console = console
        self.endpoint, self.model, _ = component_config("cerebellum")
        self.system = CEREBELLUM_SYSTEM
        self.approval_threshold = float(os.getenv("APPROVAL_THRESHOLD", "98"))
        self.divergence_threshold = float(os.getenv("DIVERGENCE_THRESHOLD", "15"))

    def _generate(self, prompt: str, timeout: float = 120.0) -> str:
        if not self.model:
            return ""
        try:
            return self.router.generate(
                prompt=prompt,
                model=self.model,
                endpoint=self.endpoint,
                system=self.system,
                component="cerebellum",
                timeout=timeout,
            )
        except ModelError as exc:
            return f"[CEREBELLUM_ERRO] {exc.message}"

    def _log(self, msg: str) -> None:
        if self.console:
            self.console.print(f"[bold magenta]CEREBELLUM[/] {msg}")

    # ================================================================
    # FASE 1
    # ================================================================
    def evaluate_f1(self, state: dict) -> dict:
        """Avalia lógica/sintaxe/código base e devolve melhorias ao CORTEX."""
        t0 = time.time()
        prompt = (
            "Lógica base:\n" + state.get("base_logic", "") + "\n\n"
            "Código base:\n" + state.get("base_code", "") + "\n\n"
            "Avalia lógica, sintaxe e código base. Aponta problemas e "
            "melhorias concretas. Output estruturado."
        )
        out = self._generate(prompt)
        state["cerebellum_feedback_f1"] = out
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "1", "cerebellum",
            input_summary="avaliar base", output_summary="feedback f1",
            full_output=out, duration_seconds=time.time() - t0,
        )
        self._log("Fase 1: avaliação do código base concluída.")
        return state

    # ================================================================
    # FASE 2
    # ================================================================
    def audit(self, state: dict) -> dict:
        """Audita o código organizado (lógica + funcionalidade)."""
        t0 = time.time()
        prompt = (
            "Código organizado (com marcadores [NEURON_N]):\n"
            + state.get("organized_code", "") + "\n\n"
            "Faz auditoria: verifica se cada NEURON ficou no seu âmbito, "
            "coerência lógica e cobertura funcional. Output estruturado."
        )
        out = self._generate(prompt)
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "2", "cerebellum",
            input_summary="auditar código", output_summary="auditoria",
            full_output=out, duration_seconds=time.time() - t0,
        )
        self._log("Fase 2: auditoria do código organizado concluída.")
        return state

    def reject_contract(self, state: dict) -> dict:
        """Reprova o ciclo por violação de contrato (não avança à Fase 3)."""
        violations = state.get("contract_violations", [])
        for neuron_id in violations:
            n = neuron_id.split("_")[-1]
            justification = f"NEURON_{n} violou contrato de interface"
            self.db.log_decision(
                state["cycle_id"], state["iteration"], "cerebellum",
                justification + ".",
            )
            self._log(f"Fase 2: reprovado — {justification}.")
        # O ciclo volta à Fase 2 apenas para os NEURONS violadores.
        state["improvements"] = {
            nid: "Reimplementar respeitando o contrato de interface: manter "
                 "apenas o próprio marcador e não tocar noutras secções."
            for nid in violations
        }
        state["status"] = "in_progress"
        return state

    # ================================================================
    # FASE 3
    # ================================================================
    def compare_and_decide(self, state: dict) -> dict:
        """Gera relatório próprio, compara com o do CORTEX e decide.

        Calcula a percentagem final por validação cruzada. Aplica a regra de
        divergência (3ª ronda) e a penalização por marcadores órfãos.
        """
        t0 = time.time()
        prompt = (
            "Resultados dos testes (independente do relatório do CORTEX):\n"
            + state.get("test_results", "") + "\n\n"
            "Gera o TEU relatório independente. Estima a percentagem de "
            "funcionalidade (0-100) e atribui melhorias a NEURONS ESPECÍFICOS "
            "(formato 'neuron_N: <melhoria>'). Começa com 'PCT: <numero>'."
        )
        cere_report = self._generate(prompt)
        state["cerebellum_report"] = cere_report

        pct_cortex = _extract_pct(state.get("cortex_test_report", ""))
        pct_cere = _extract_pct(cere_report)

        # Regra de divergência: força 3ª ronda antes de decidir.
        divergence = abs(pct_cortex - pct_cere)
        if divergence > self.divergence_threshold:
            self.db.log_decision(
                state["cycle_id"], state["iteration"], "cerebellum",
                f"Divergência {divergence:.1f}pp > {self.divergence_threshold} "
                "— 3ª ronda de verificação automática.",
            )
            self._log(
                f"Divergência {divergence:.1f}pp — 3ª ronda de verificação."
            )
            third = self._generate(
                "Terceira verificação. Reconcilia as duas estimativas "
                f"({pct_cortex} vs {pct_cere}). Começa com 'PCT: <numero>'.\n\n"
                + state.get("test_results", "")
            )
            pct_third = _extract_pct(third)
            final_pct = pct_third if pct_third > 0 else (pct_cortex + pct_cere) / 2
        else:
            # Validação cruzada normal: média das duas estimativas.
            final_pct = (pct_cortex + pct_cere) / 2 if (pct_cortex or pct_cere) else 0.0

        # Penalização por marcadores órfãos por preencher no código organizado.
        final_pct = self._penalize_orphan_markers(state, final_pct)

        state["functionality_pct"] = final_pct
        improvements = self._extract_improvements(cere_report)
        state["improvements"] = improvements

        # Guardar melhor versão para rollback.
        if final_pct >= state.get("best_pct_so_far", 0.0):
            state["best_pct_so_far"] = final_pct
            state["best_code_so_far"] = state.get("organized_code", "")

        approved = final_pct >= self.approval_threshold
        decision = "aprovado" if approved else "reprovado"
        state["status"] = "approved" if approved else "in_progress"

        self.db.log_report(
            state["cycle_id"], state["iteration"], final_pct,
            failures="(ver relatório cerebellum)",
            improvements="; ".join(f"{k}:{v}" for k, v in improvements.items()),
        )
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cerebellum",
            f"{decision.capitalize()} a {final_pct:.1f}% "
            f"(cortex={pct_cortex:.0f}, cerebellum={pct_cere:.0f}).",
        )
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "3", "cerebellum",
            input_summary="comparar e decidir",
            output_summary=f"{decision} {final_pct:.1f}%",
            full_output=cere_report, duration_seconds=time.time() - t0,
        )
        self._log(
            f"Fase 3: {decision} a {final_pct:.1f}% "
            f"(cortex={pct_cortex:.0f} / cerebellum={pct_cere:.0f})."
        )
        return state

    def _penalize_orphan_markers(self, state: dict, pct: float) -> float:
        """Reduz a % se sobrarem marcadores órfãos por preencher.

        Um marcador cuja secção continua vazia/por implementar é falha de
        funcionalidade e deve reflectir-se na percentagem.
        """
        from .cortex import extract_section

        code = state.get("organized_code", "")
        markers = state.get("markers", {})
        orphan = 0
        for neuron_id in markers:
            section = extract_section(code, neuron_id)
            if not section or "pass" == section.strip() or "TODO" in section:
                orphan += 1
        if orphan and markers:
            penalty = (orphan / len(markers)) * 100.0
            adjusted = max(0.0, pct - penalty)
            if adjusted < pct:
                self.db.log_decision(
                    state["cycle_id"], state["iteration"], "cerebellum",
                    f"{orphan} marcador(es) órfão(s) — penalização de "
                    f"{penalty:.0f}pp aplicada.",
                )
            return adjusted
        return pct

    @staticmethod
    def _extract_improvements(report: str) -> dict:
        """Extrai melhorias atribuídas por NEURON ('neuron_N: texto')."""
        improvements: dict = {}
        if not report:
            return improvements
        for match in re.finditer(
            r"neuron_(\d+)\s*:\s*(.+)", report, re.IGNORECASE
        ):
            improvements[f"neuron_{match.group(1)}"] = match.group(2).strip()
        return improvements


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
