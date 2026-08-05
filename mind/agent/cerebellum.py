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
from .report_schema import (
    INSTRUCAO_JSON,
    parse_relatorio,
    registar_desvio_de_formato,
)

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

    def __init__(self, router: ModelRouter, db, console=None, hippocampus=None):
        self.router = router
        self.db = db
        self.console = console
        # Apoio de ML opcional — ver agent/hippocampus.py. Sempre consultivo,
        # com uma única excepção assimétrica: pode apoiar auto-REJEIÇÃO.
        self.hippocampus = hippocampus
        self.endpoint, self.model, _ = component_config("cerebellum")
        self.system = CEREBELLUM_SYSTEM
        self.approval_threshold = float(os.getenv("APPROVAL_THRESHOLD", "98"))
        self.divergence_threshold = float(os.getenv("DIVERGENCE_THRESHOLD", "15"))

    def _consult_hippocampus(self, state: dict) -> dict | None:
        """Consulta o HIPPOCAMPUS sobre zonas de risco e probabilidade.

        Devolve None se a camada estiver desligada, sem modelo activo ou em
        cold start — nesse caso o CEREBELLUM funciona exactamente como a base.
        """
        if self.hippocampus is None:
            return None
        from .ml_features import extract_cerebellum_features

        features = extract_cerebellum_features(
            state.get("organized_code", ""), state.get("test_results", ""), self.db
        )
        return self.hippocampus.consult("cerebellum", features)

    def _generate(self, prompt: str, timeout: float = 120.0,
                  state: dict = None) -> str:
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
                cycle_id=(state or {}).get("cycle_id"),
                iteration=(state or {}).get("iteration", 0),
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
        out = self._generate(prompt, state=state)
        state["cerebellum_feedback_f1"] = out
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "1", "cerebellum",
            input_summary="avaliar base", output_summary="feedback f1",
            full_output=out, duration_seconds=time.time() - t0,
        )
        self._log("Fase 1: avaliação do código base concluída.")
        return self._gerar_testes(state)

    def _gerar_testes(self, state: dict) -> dict:
        """Gera os testes dos 3 níveis na transição Fase 1 -> Fase 2.

        É o CEREBELLUM que gera os testes pelos quais o CORTEX vai ser
        avaliado — separação deliberada. Com SANDBOX_TESTS_ENABLED=false não
        se faz chamada nenhuma e o estado fica exactamente como antes.
        """
        from .test_generator import TestGenerator, sandbox_tests_enabled

        if not sandbox_tests_enabled():
            return state

        gerador = TestGenerator(self.router, self.db, self.hippocampus)
        testes = gerador.generate(
            task=state.get("task", ""),
            base_code=state.get("base_code", ""),
            markers=state.get("markers", {}),
            cycle_id=state["cycle_id"],
        )
        herdados = [t for t in testes if t.get("times_used") is not None]
        novos = [t for t in testes if t.get("times_used") is None]
        state["generated_tests"] = testes
        state["inherited_tests"] = herdados

        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cerebellum",
            f"Gerados {len(novos)} testes novos e herdados {len(herdados)} "
            "de ciclos aprovados anteriores.",
        )
        self._log(
            f"Fase 1: {len(novos)} testes gerados, {len(herdados)} herdados."
        )
        return state

    # ================================================================
    # FASE 2
    # ================================================================
    def audit(self, state: dict) -> dict:
        """Audita o código organizado (lógica + funcionalidade)."""
        t0 = time.time()
        # Consulta ao HIPPOCAMPUS: zonas com padrões de risco conhecidos.
        hint = ""
        ml = self._consult_hippocampus(state)
        if ml:
            hint = (
                "\n\nApoio do HIPPOCAMPUS (histórico local, consultivo):\n"
                + ml.get("suggestion", "") + "\n"
            )
            self._log(
                f"HIPPOCAMPUS: risco={ml.get('confidence', 0):.2f} "
                f"(prob. passar {ml.get('prediction', 0):.2f})."
            )
        prompt = (
            "Código organizado (com marcadores [NEURON_N]):\n"
            + state.get("organized_code", "") + "\n\n"
            "Faz auditoria: verifica se cada NEURON ficou no seu âmbito, "
            "coerência lógica e cobertura funcional. Output estruturado."
            + hint
        )
        out = self._generate(prompt, state=state)
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

        # --- Auto-rejeição do HIPPOCAMPUS (única excepção assimétrica) ----
        # Se um padrão de falha conhecido for detectado com confiança
        # >= ML_AUTO_REJECT_CONFIDENCE, reprova-se já, sem esperar pela
        # análise completa do LLM. NUNCA existe o inverso (auto-aprovação).
        ml = self._consult_hippocampus(state)
        if ml and ml.get("auto_reject"):
            return self._auto_reject(state, ml, t0)

        prompt = (
            "Resultados dos testes (independente do relatório do CORTEX):\n"
            + state.get("test_results", "") + "\n\n"
            "Gera a TUA avaliação independente: percentagem de funcionalidade, "
            "falhas encontradas, e melhorias atribuídas a NEURONS ESPECÍFICOS "
            "(nunca genéricas).\n\n" + INSTRUCAO_JSON
        )
        cere_report = self._generate(prompt, state=state)
        state["cerebellum_report"] = cere_report

        avaliacao = parse_relatorio(cere_report)
        state["_formato_respeitado"] = avaliacao.formato_respeitado
        registar_desvio_de_formato(
            self.db, state["cycle_id"], state["iteration"], "cerebellum",
            avaliacao,
        )
        pct_cere = avaliacao.functionality_pct

        avaliacao_cortex = parse_relatorio(state.get("cortex_test_report", ""))
        registar_desvio_de_formato(
            self.db, state["cycle_id"], state["iteration"], "cortex",
            avaliacao_cortex,
        )
        pct_cortex = avaliacao_cortex.functionality_pct

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
            prompt_terceira = (
                "Terceira verificação. Reconcilia as duas estimativas "
                f"({pct_cortex} vs {pct_cere}) a partir dos resultados dos "
                "testes.\n\n" + state.get("test_results", "") + "\n\n"
                + INSTRUCAO_JSON
            )
            third = self._generate(prompt_terceira, state=state)
            reconciliacao = parse_relatorio(third)
            pct_third = reconciliacao.functionality_pct
            final_pct = pct_third if pct_third > 0 else (pct_cortex + pct_cere) / 2
        else:
            # Validação cruzada normal: média das duas estimativas.
            final_pct = (pct_cortex + pct_cere) / 2 if (pct_cortex or pct_cere) else 0.0

        # Sandbox evolutiva: com testes reais, a percentagem deixa de ser uma
        # estimativa do modelo e passa a ser medida sobre resultados.
        final_pct = self._pct_sobre_testes_reais(state, final_pct)

        # Penalização por marcadores órfãos por preencher no código organizado.
        final_pct = self._penalize_orphan_markers(state, final_pct)

        state["functionality_pct"] = final_pct
        # O modelo pode indicar testes a persistir; junta-se aos que o runner
        # executou, sem duplicar.
        if avaliacao.tests_to_persist:
            ja = set(state.get("tests_to_persist") or [])
            state["tests_to_persist"] = list(
                ja | set(avaliacao.tests_to_persist)
            )
        improvements = avaliacao.improvements
        state["improvements"] = improvements

        # auto_reject vindo do modelo só pode reprovar, nunca aprovar: a
        # assimetria de segurança aplica-se aqui tal como no HIPPOCAMPUS.
        if avaliacao.auto_reject:
            final_pct = min(final_pct, self.approval_threshold - 1)
            state["functionality_pct"] = final_pct

        # Guardar melhor versão para rollback.
        if final_pct >= state.get("best_pct_so_far", 0.0):
            state["best_pct_so_far"] = final_pct
            state["best_code_so_far"] = state.get("organized_code", "")

        approved = final_pct >= self.approval_threshold
        decision = "aprovado" if approved else "reprovado"
        state["status"] = "approved" if approved else "in_progress"

        self.db.log_report(
            state["cycle_id"], state["iteration"], final_pct,
            failures="; ".join(avaliacao.failures) or "(nenhuma reportada)",
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
        self._record_ml_outcome(state, final_pct, ml)
        return state

    # ------------------------------------------------------------------
    # HIPPOCAMPUS — auto-rejeição e acumulação de histórico
    # ------------------------------------------------------------------
    def _auto_reject(self, state: dict, ml: dict, t0: float) -> dict:
        """Reprova o ciclo por padrão de falha conhecido de alta confiança.

        Assimetria de segurança: esta é a ÚNICA decisão que o HIPPOCAMPUS
        pode apoiar sozinho, e é sempre no sentido restritivo. A razão fica
        sempre registada.
        """
        reason = ml.get("reason", "padrão de falha conhecido")
        state["functionality_pct"] = 0.0
        state["status"] = "in_progress"
        state["cerebellum_report"] = f"[AUTO-REJEIÇÃO HIPPOCAMPUS] {reason}"
        # Sem melhorias atribuídas pelo LLM, todos os NEURONS activos são
        # revisitados na ronda seguinte.
        state["improvements"] = {
            nid: "Rever: o histórico indica padrão de falha nesta zona."
            for nid in state.get("active_neurons", [])
        }
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cerebellum",
            f"Auto-rejeição apoiada pelo HIPPOCAMPUS: {reason}",
        )
        self.db.log_iteration(
            state["cycle_id"], state["iteration"], "3", "cerebellum",
            input_summary="auto-rejeição hippocampus",
            output_summary="reprovado sem análise completa do LLM",
            full_output=reason, duration_seconds=time.time() - t0,
        )
        self._log(f"Fase 3: AUTO-REJEIÇÃO (HIPPOCAMPUS) — {reason}")
        self._record_ml_outcome(state, 0.0, ml)
        return state

    def _record_ml_outcome(self, state: dict, final_pct: float, ml: dict = None) -> None:
        """Acumula histórico de treino e a concordância ML vs LLM.

        Corre mesmo com ML_ENABLED=false — é assim que o volume necessário
        ao primeiro treino se acumula antes de a camada ser activada.
        """
        if self.hippocampus is None:
            return
        from .ml_features import (
            extract_cerebellum_features,
            extract_cortex_features,
        )

        cycle_id = state.get("cycle_id")
        # Amostra do CORTEX: tarefa -> % de funcionalidade atingida.
        self.hippocampus.record_sample(
            "cortex", cycle_id,
            extract_cortex_features(state.get("task", ""), self.db),
            final_pct,
        )
        # Amostra do CEREBELLUM: código/testes -> % de funcionalidade.
        self.hippocampus.record_sample(
            "cerebellum", cycle_id,
            extract_cerebellum_features(
                state.get("organized_code", ""),
                state.get("test_results", ""),
                self.db,
            ),
            final_pct,
        )
        if ml:
            self.hippocampus.log_prediction(
                cycle_id, state.get("iteration", 0), "cerebellum",
                ml.get("prediction"), str(final_pct),
            )

    def _pct_sobre_testes_reais(self, state: dict, pct_estimada: float) -> float:
        """Substitui a estimativa do modelo pela medição sobre testes reais.

        Fórmula: nível 1 completo = 33%, nível 2 = 66% acumulado, nível 3 =
        99%, mais 1% para a qualidade do relatório — que é a única parte que
        continua a ser um juízo do CEREBELLUM, e por isso vale 1% e não mais.

        Se a sandbox evolutiva estiver desligada, ou se não houver breakdown
        (nenhum teste correu), devolve a estimativa como antes.
        """
        from .test_generator import sandbox_tests_enabled
        from .test_runner import calcular_percentagem

        if not sandbox_tests_enabled():
            return pct_estimada
        breakdown = state.get("test_breakdown") or {}
        if not any((breakdown.get(f"level_{n}") or {}).get("total")
                   for n in (1, 2, 3)):
            return pct_estimada

        # A qualidade do relatório vale 1%: atribui-se por o modelo ter
        # respeitado o esquema JSON, que é o que se consegue verificar.
        qualidade = 1.0 if state.get("_formato_respeitado") else 0.0
        medida = calcular_percentagem(breakdown, qualidade)
        self.db.log_decision(
            state["cycle_id"], state["iteration"], "cerebellum",
            f"Percentagem medida sobre testes reais: {medida:.1f}% "
            f"(estimativa do modelo era {pct_estimada:.1f}%).",
        )
        return medida

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
