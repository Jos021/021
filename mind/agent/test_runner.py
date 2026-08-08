"""Execução de testes na sandbox — sandbox evolutiva.

Encapsula o agent/sandbox.py existente para executar testes individuais em
vez de código de produção. O sandbox.py NÃO é modificado: continua a ser o
mesmo runner multi-linguagem, com a mesma linguagem lida do marcador e o
mesmo isolamento — só recebe código de teste em vez de código de aplicação.

Cada teste corre com o timeout de MUNDJI_SANDBOX_TIMEOUT. Se exceder,
outcome = 'timeout'. Se a sandbox lançar, outcome = 'error'. O ciclo nunca
falha por causa de um teste individual.
"""

import os
import time

from .sandbox import run_section

# Peso de cada nível na percentagem final. Três níveis completos dão 99%; o
# 1% restante é a qualidade do relatório, avaliada pelo CEREBELLUM.
PESO_NIVEL = 33.0
PESO_RELATORIO = 1.0


class TestRunner:
    """Executa testes na sandbox e agrega o resultado por nível."""

    # O nome começa por 'Test' (exigido pela especificação), mas isto
    # não é uma classe de testes do pytest — evita a recolha indevida.
    __test__ = False


    def __init__(self, db):
        self.db = db

    def run_tests(self, tests: list, cycle_id: int,
                  iteration_number: int) -> dict:
        """Executa cada teste e devolve o breakdown por nível.

        Formato devolvido:
            {"level_1": {"passed": N, "total": N, "pct": N},
             "level_2": {...}, "level_3": {...},
             "results": [...]}
        """
        breakdown = {
            f"level_{n}": {"passed": 0, "total": 0, "pct": 0.0}
            for n in (1, 2, 3)
        }
        resultados = []

        for teste in tests or []:
            resultado = self._correr_um(teste, cycle_id, iteration_number)
            resultados.append(resultado)
            chave = f"level_{teste.get('level', 1)}"
            if chave not in breakdown:
                continue
            breakdown[chave]["total"] += 1
            if resultado["outcome"] == "pass":
                breakdown[chave]["passed"] += 1

        for chave, dados in breakdown.items():
            dados["pct"] = (
                round(dados["passed"] / dados["total"] * 100, 1)
                if dados["total"] else 0.0
            )

        breakdown["results"] = resultados
        return breakdown

    # ------------------------------------------------------------------
    def _correr_um(self, teste: dict, cycle_id: int, iteracao: int) -> dict:
        """Corre um teste isolado e regista o resultado na test_results."""
        t0 = time.time()
        esperado = teste.get("expected_outcome", "pass")
        try:
            res = run_section(teste.get("code", ""),
                              teste.get("language", "python"))
            if res.timed_out:
                outcome = "timeout"
            elif res.success:
                # Um teste de falha esperada que passa é, ele próprio, falha.
                outcome = "pass" if esperado == "pass" else "fail"
            else:
                # Código que rebenta é sucesso quando a falha era esperada.
                outcome = "pass" if esperado == "fail" else "fail"
            saida = (res.stdout or "") + (res.stderr or "")
        except Exception as exc:
            outcome = "error"
            saida = f"excepção da sandbox: {exc}"

        resultado = {
            "test_id": teste.get("test_id"),
            "cycle_id": cycle_id,
            "iteration_number": iteracao,
            "level": teste.get("level", 1),
            "outcome": outcome,
            "output": saida[:2000],
            "duration_seconds": time.time() - t0,
        }
        try:
            self.db.record_test_result(resultado)
        except Exception:
            pass  # o registo nunca pode derrubar a execução
        return resultado


def seleccionar_testes(tests: list, nivel_actual: int) -> list:
    """Testes a executar nesta iteração, conforme a política de acumulação.

    Com SANDBOX_ACCUMULATE_LEVELS=true (default), corre-se o nível actual e
    todos os anteriores: não basta não regredir, tem de progredir. Com
    false, corre-se apenas o nível actual.
    """
    acumular = os.getenv("SANDBOX_ACCUMULATE_LEVELS", "true").lower() == "true"
    if acumular:
        return [t for t in tests or [] if t.get("level", 1) <= nivel_actual]
    return [t for t in tests or [] if t.get("level", 1) == nivel_actual]


def calcular_percentagem(breakdown: dict, pct_relatorio: float = 0.0) -> float:
    """Percentagem de funcionalidade medida sobre resultados reais.

        nível 1 completo = 33%   |   nível 2 completo = 66% acumulado
        nível 3 completo = 99%   |   qualidade do relatório = +1%

    Parcial é proporcional: pct_nivel = (passados / total) * 33.

    Um nível sem testes contribui 0 — o que significa que um código que
    ainda não foi confrontado com testes de limite não pode ser aprovado,
    e o número reflecte isso em vez de o esconder.
    """
    total = 0.0
    for n in (1, 2, 3):
        dados = (breakdown or {}).get(f"level_{n}") or {}
        if dados.get("total"):
            total += (dados["passed"] / dados["total"]) * PESO_NIVEL
    total += max(0.0, min(PESO_RELATORIO, pct_relatorio))
    return round(min(100.0, total), 1)


def nivel_completo(breakdown: dict, nivel: int) -> bool:
    """True se todos os testes desse nível passaram (e existiam)."""
    dados = (breakdown or {}).get(f"level_{nivel}") or {}
    return bool(dados.get("total")) and dados["passed"] == dados["total"]
