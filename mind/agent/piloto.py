"""Piloto com modelos reais — verificação e medição.

O MIND está inteiro mas nunca correu com um LLM verdadeiro. Este módulo é a
ponte: verifica que os endpoints respondem antes de se gastar um ciclo, e
corre um conjunto de tarefas de referência medindo o que só se pode medir
com modelos reais.

O que o piloto responde, e a base não consegue:
  - quanto demora um ciclo, e quantas iterações precisa
  - que percentagem das respostas respeita o esquema JSON dos relatórios
  - se NEURON_TIMEOUT_SECONDS e MUNDJI_SANDBOX_TIMEOUT são realistas
  - com a sandbox evolutiva ligada, se os modelos geram testes executáveis

Nada aqui inventa resultados: se um endpoint não responde, diz-se que não
responde.
"""

import os
import time
from dataclasses import dataclass, field

from .model_router import ModelError, ModelRouter, component_config

COMPONENTES = ["cortex", "cerebellum"] + [f"neuron_{n}" for n in range(1, 7)]

# Prompt mínimo para a verificação: barato em tokens, e a resposta esperada
# é verificável sem ambiguidade.
PROMPT_VERIFICACAO = "Responde apenas com a palavra: OK"


@dataclass
class Diagnostico:
    """Resultado da verificação de um componente."""

    componente: str
    configurado: bool
    endpoint: str = ""
    modelo: str = ""
    respondeu: bool = False
    latencia_s: float = 0.0
    erro: str = ""
    amostra: str = ""

    @property
    def estado(self) -> str:
        if not self.configurado:
            return "sem modelo"
        return "ok" if self.respondeu else "falhou"


@dataclass
class ResultadoTarefa:
    """Medições de um ciclo do piloto."""

    tarefa: str
    cycle_id: int
    status: str = ""
    functionality_pct: float = 0.0
    iteracoes: int = 0
    duracao_s: float = 0.0
    relatorios_json: int = 0
    desvios_formato: int = 0
    testes_gerados: int = 0
    testes_executados: int = 0
    testes_passados: int = 0
    erro: str = ""

    @property
    def conformidade_json(self) -> float | None:
        """Percentagem de respostas que respeitaram o esquema JSON."""
        if not self.relatorios_json:
            return None
        ok = max(0, self.relatorios_json - self.desvios_formato)
        return round(ok / self.relatorios_json * 100, 1)


def verificar_componentes(db=None, timeout: float = 30.0) -> list:
    """Testa a ligação a cada componente configurado.

    Não corre nenhum ciclo: só confirma que o endpoint responde e devolve
    texto. É a primeira coisa a fazer com uma instância nova — um URL errado
    ou um token em falta descobre-se aqui em segundos, não a meio de um
    ciclo.
    """
    router = ModelRouter(db=db)
    diagnosticos = []
    for componente in COMPONENTES:
        endpoint, modelo, activo = component_config(componente)
        if not activo:
            continue
        d = Diagnostico(componente=componente, configurado=bool(modelo),
                        endpoint=endpoint, modelo=modelo)
        if not modelo:
            diagnosticos.append(d)
            continue
        t0 = time.time()
        try:
            resposta = router.generate(
                prompt=PROMPT_VERIFICACAO, model=modelo, endpoint=endpoint,
                component=componente, timeout=timeout,
            )
            d.respondeu = bool((resposta or "").strip())
            d.amostra = (resposta or "").strip()[:80]
            if not d.respondeu:
                d.erro = "resposta vazia"
        except ModelError as exc:
            d.erro = exc.message
        except Exception as exc:                     # noqa: BLE001
            d.erro = f"{type(exc).__name__}: {exc}"
        d.latencia_s = round(time.time() - t0, 2)
        diagnosticos.append(d)
    return diagnosticos


def carregar_tarefas(caminho: str) -> list:
    """Lê as tarefas de referência do piloto (YAML). [] se não existir."""
    try:
        import yaml

        with open(caminho, "r", encoding="utf-8") as fh:
            dados = yaml.safe_load(fh) or {}
        tarefas = dados.get("tarefas") or []
        return [str(t) for t in tarefas if str(t).strip()]
    except Exception:
        return []


def correr_piloto(tarefas: list, db, construir_grafo, console=None,
                  max_tarefas: int = 0) -> list:
    """Corre cada tarefa como um ciclo completo e mede o resultado.

    `construir_grafo` é uma função sem argumentos que devolve um MindGraph
    novo — recebida assim para o piloto não depender da montagem concreta,
    que vive no main.py.

    Uma tarefa que rebente não interrompe as seguintes: fica registada com
    o erro e passa-se à frente. Um piloto que pára na primeira falha não
    mede nada.
    """
    from .state import new_state

    if max_tarefas:
        tarefas = tarefas[:max_tarefas]

    resultados = []
    for i, tarefa in enumerate(tarefas, start=1):
        if console:
            console.print(
                f"[bold cyan]PILOTO[/] ({i}/{len(tarefas)}) {tarefa[:70]}"
            )
        cycle_id = db.create_cycle(tarefa)
        resultado = ResultadoTarefa(tarefa=tarefa, cycle_id=cycle_id)
        t0 = time.time()
        try:
            final = construir_grafo().run(new_state(tarefa, cycle_id))
            resultado.status = final.get("status", "")
            resultado.functionality_pct = final.get("functionality_pct", 0.0)
            resultado.iteracoes = final.get("iteration", 0)
            resultado.testes_gerados = len(final.get("generated_tests") or [])
            breakdown = final.get("test_breakdown") or {}
            for n in (1, 2, 3):
                nivel = breakdown.get(f"level_{n}") or {}
                resultado.testes_executados += nivel.get("total", 0)
                resultado.testes_passados += nivel.get("passed", 0)
        except Exception as exc:                     # noqa: BLE001
            resultado.erro = f"{type(exc).__name__}: {exc}"
            resultado.status = "erro"
        resultado.duracao_s = round(time.time() - t0, 1)
        _medir_conformidade(db, cycle_id, resultado)
        resultados.append(resultado)
    return resultados


def _medir_conformidade(db, cycle_id: int, resultado: ResultadoTarefa) -> None:
    """Conta respostas com contrato de formato, e quantas o quebraram.

    O denominador é o número de respostas que DEVIAM ser JSON — não o total
    de chamadas ao modelo. A diferença não é académica: num ciclo, a maioria
    das chamadas ao CORTEX gera código, anota marcadores ou aprimora, e não
    tem esquema nenhum para respeitar. Contá-las diluía os desvios e fazia um
    modelo que nunca produz JSON válido parecer 78% conforme.

    O report_schema regista os dois desfechos em cada leitura de relatório;
    aqui só se contam.
    """
    try:
        total = db._conn.execute(                    # noqa: SLF001
            "SELECT COUNT(*) AS n FROM decisions WHERE cycle_id = ? "
            "AND decision_text LIKE '%esquema JSON%'", (cycle_id,)
        ).fetchone()
        resultado.relatorios_json = total["n"] if total else 0

        desvios = db._conn.execute(                  # noqa: SLF001
            "SELECT COUNT(*) AS n FROM decisions WHERE cycle_id = ? "
            "AND decision_text LIKE '%não respeitou o esquema JSON%'",
            (cycle_id,)
        ).fetchone()
        resultado.desvios_formato = desvios["n"] if desvios else 0
    except Exception:
        pass


def conformidade_por_componente(db) -> dict:
    """Conformidade com o esquema JSON acumulada, por componente.

    Lida directamente da SYNAPSE DB, sobre todo o histórico — não depende de
    ter corrido o piloto. É o que o runbook do piloto manda consultar no
    fim da primeira passagem, e é o número que diz se um modelo cumpre o
    contrato de formato.

    "respostas" conta apenas as que deviam ser JSON, não todas as chamadas
    ao modelo — ver _medir_conformidade para a razão.

    Devolve {componente: {"respostas": N, "desvios": N, "pct": X}}.
    """
    resultado = {}
    try:
        for componente in ("cortex", "cerebellum"):
            chamadas = db._conn.execute(          # noqa: SLF001
                "SELECT COUNT(*) AS n FROM decisions WHERE component = ? "
                "AND decision_text LIKE '%esquema JSON%'",
                (componente,)
            ).fetchone()
            desvios = db._conn.execute(           # noqa: SLF001
                "SELECT COUNT(*) AS n FROM decisions WHERE component = ? "
                "AND decision_text LIKE '%não respeitou o esquema JSON%'",
                (componente,)
            ).fetchone()
            n = chamadas["n"] if chamadas else 0
            d = desvios["n"] if desvios else 0
            resultado[componente] = {
                "respostas": n,
                "desvios": d,
                "pct": round(max(0, n - d) / n * 100, 1) if n else None,
            }
    except Exception:
        return {}
    return resultado


def exportar_csv(resultados: list, caminho: str) -> int:
    """Escreve as medições em CSV. Devolve o número de linhas."""
    import csv

    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    colunas = ["tarefa", "cycle_id", "status", "functionality_pct",
               "iteracoes", "duracao_s", "relatorios_json",
               "desvios_formato", "conformidade_json", "testes_gerados",
               "testes_executados", "testes_passados", "erro"]
    with open(caminho, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(colunas)
        for r in resultados:
            escritor.writerow([
                r.tarefa, r.cycle_id, r.status, r.functionality_pct,
                r.iteracoes, r.duracao_s, r.relatorios_json,
                r.desvios_formato,
                "" if r.conformidade_json is None else r.conformidade_json,
                r.testes_gerados, r.testes_executados, r.testes_passados,
                r.erro,
            ])
    return len(resultados)


def resumir(resultados: list) -> dict:
    """Agrega as medições do piloto num resumo legível."""
    if not resultados:
        return {"tarefas": 0}

    aprovados = [r for r in resultados if r.status == "approved"]
    duracoes = [r.duracao_s for r in resultados if not r.erro]
    conformidades = [r.conformidade_json for r in resultados
                     if r.conformidade_json is not None]
    executados = sum(r.testes_executados for r in resultados)
    passados = sum(r.testes_passados for r in resultados)

    return {
        "tarefas": len(resultados),
        "aprovadas": len(aprovados),
        "taxa_aprovacao": round(len(aprovados) / len(resultados) * 100, 1),
        "erros": sum(1 for r in resultados if r.erro),
        "duracao_media_s": round(sum(duracoes) / len(duracoes), 1)
        if duracoes else None,
        "duracao_max_s": max(duracoes) if duracoes else None,
        "iteracoes_media": round(
            sum(r.iteracoes for r in resultados) / len(resultados), 1),
        "conformidade_json_media": round(
            sum(conformidades) / len(conformidades), 1)
        if conformidades else None,
        "testes_gerados": sum(r.testes_gerados for r in resultados),
        "testes_executados": executados,
        "taxa_testes_passados": round(passados / executados * 100, 1)
        if executados else None,
    }
