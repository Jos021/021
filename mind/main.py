#!/usr/bin/env python3
"""MIND — Muñdji Intelligent Neural Developer (CLI).

Uso:
    # Correr uma tarefa (argumento principal):
    python main.py "cria um sistema de autenticação JWT com refresh tokens"

    # Exportar iterações para dataset de fine-tuning (JSONL):
    python main.py export --cycle-id 5 --component neuron_1 \\
        --output ./datasets/neuron_1.jsonl

    # Intervenção manual: actualiza a tarefa de um ciclo; o CORTEX continua
    # a partir do estado actual na iteração seguinte, sem recomeçar:
    python main.py intervene --cycle-id 5 --new-task "nova descrição"

    # HIPPOCAMPUS (camada de apoio de ML — ver agent/hippocampus.py):
    python main.py ml-status                                  # estado dos modelos
    python main.py ml-train --force                           # força treino
    python main.py ml-export --consumer cortex --output ./datasets/cortex.csv

Interface: apenas CLI nesta fase. A GUI é decisão futura, fora do escopo.
A separação entre a lógica do MIND e qualquer interface já é limpa por
natureza — o CORTEX/CEREBELLUM/NEURONS não sabem que interface os chama.
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

try:
    from rich.console import Console
except Exception:  # pragma: no cover
    Console = None

from agent.backup import BackupManager
from agent.database import SynapseDB
from agent.graph import MindGraph
from agent.hippocampus import Hippocampus, load_ml_config, ml_enabled
from agent.ml_pipeline import MLPipeline
from agent.model_router import ModelRouter
from agent.state import new_state

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "synapse.db")
SPECIALTIES_PATH = os.path.join(BASE_DIR, "config", "neuron_specialties.yaml")
ML_CONFIG_PATH = os.path.join(BASE_DIR, "config", "ml_config.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models", "hippocampus")


def _console():
    return Console() if Console else None


def _load_specialties() -> dict:
    """Lê config/neuron_specialties.yaml (pode estar vazio)."""
    try:
        with open(SPECIALTIES_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            return {k: v for k, v in data.items() if v}
    except FileNotFoundError:
        return {}


def _build_hippocampus(db) -> Hippocampus:
    """Instancia o HIPPOCAMPUS (camada de apoio de ML).

    É sempre instanciado: com ML_ENABLED=false, consult() devolve None e o
    componente limita-se a acumular histórico para o treino futuro.
    """
    return Hippocampus(db, load_ml_config(ML_CONFIG_PATH), MODELS_DIR)


def _build_pipeline(db, hippocampus=None) -> MLPipeline:
    return MLPipeline(
        db, load_ml_config(ML_CONFIG_PATH), MODELS_DIR, hippocampus
    )


def _build_graph(db, console) -> MindGraph:
    router = ModelRouter(db=db)
    specialties = _load_specialties()
    return MindGraph(router, db, specialties, console, _build_hippocampus(db))


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
def cmd_run(task: str) -> None:
    """Corre um ciclo completo do MIND para a tarefa dada."""
    console = _console()
    db = SynapseDB(DB_PATH)
    backup = BackupManager(DB_PATH, os.path.join(BASE_DIR, "backups"))
    backup.start()
    # Treino periódico do HIPPOCAMPUS — só arranca se ML_ENABLED=true.
    pipeline = _build_pipeline(db)
    pipeline.start_scheduler()
    try:
        cycle_id = db.create_cycle(task)
        if console:
            console.print(
                f"[bold cyan]MIND[/] Ciclo {cycle_id} iniciado.\n"
                f"[dim]Tarefa:[/] {task}"
            )
        graph = _build_graph(db, console)
        state = new_state(task, cycle_id)
        final = graph.run(state)
        _report_result(console, final)
    finally:
        pipeline.stop_scheduler()
        backup.stop()
        db.close()


def cmd_export(cycle_id: int, component: str, output: str) -> None:
    """Exporta iterações filtradas para JSONL."""
    console = _console()
    db = SynapseDB(DB_PATH)
    try:
        n = db.export_to_jsonl(cycle_id, component, output)
        msg = (
            f"Exportadas {n} iterações do ciclo {cycle_id}"
            + (f" (componente {component})" if component else "")
            + f" para {output}."
        )
        if console:
            console.print(f"[bold green]MIND[/] {msg}")
        else:
            print(msg)
    finally:
        db.close()


def cmd_intervene(cycle_id: int, new_task: str) -> None:
    """Intervenção manual: actualiza a tarefa de um ciclo existente.

    O CORTEX continua a partir do estado actual na iteração seguinte, sem
    recomeçar. Aqui actualizamos a tarefa e o estado do ciclo na SYNAPSE DB;
    a retoma efectiva usa esse novo estado.
    """
    console = _console()
    db = SynapseDB(DB_PATH)
    try:
        cycle = db.get_cycle(cycle_id)
        if not cycle:
            msg = f"Ciclo {cycle_id} não existe."
            (console.print(f"[bold red]MIND[/] {msg}") if console else print(msg))
            sys.exit(1)
        db.update_cycle(cycle_id, status="in_progress", task=new_task)
        db.log_decision(
            cycle_id, 0, "cortex",
            f"Intervenção manual: tarefa actualizada para '{new_task[:80]}'.",
        )
        # Retomar o ciclo a partir do estado actual com a nova tarefa.
        graph = _build_graph(db, console)
        state = new_state(new_task, cycle_id)
        if console:
            console.print(
                f"[bold cyan]MIND[/] Intervenção no ciclo {cycle_id}. "
                "A retomar com a nova tarefa."
            )
        final = graph.run(state)
        _report_result(console, final)
    finally:
        db.close()


# --------------------------------------------------------------------------
# Comandos da extensão HIPPOCAMPUS
# --------------------------------------------------------------------------
def cmd_ml_status() -> None:
    """Mostra o estado dos modelos do HIPPOCAMPUS."""
    console = _console()
    db = SynapseDB(DB_PATH)
    try:
        status = _build_pipeline(db).status()
        lines = [
            f"ML_ENABLED: {status['ml_enabled']}",
            f"Amostras mínimas para treino: {status['min_training_samples']}",
            f"Intervalo de treino: {status['training_interval_hours']}h",
            f"Limiar de desvio para retreino: {status['deviation_threshold']}",
            "",
        ]
        for consumer, info in status["consumers"].items():
            # Sem parênteses rectos: o rich interpretá-los-ia como marcação.
            lines.append(f"* {consumer.upper()}")
            lines.append(
                f"  amostras: {info['samples']}"
                + ("  (COLD START — consult() devolve None)"
                   if info["cold_start"] else "")
            )
            lines.append(f"  modelo activo: {info['active_model'] or '— nenhum —'}")
            metric = info["validation_metric"]
            lines.append(
                f"  métrica de validação: "
                f"{metric if metric is None else round(float(metric), 4)}"
            )
            lines.append(f"  treinado em: {info['trained_at'] or '—'}")
            lines.append(f"  versões registadas: {info['versions']}")
            dev = info["mean_deviation_vs_llm"]
            lines.append(
                "  desvio médio vs LLM: "
                + ("—" if dev is None else f"{dev:.2f}")
                + ("  [RETREINO RECOMENDADO]" if info["needs_retrain"] else "")
            )
            lines.append("")
        text = "\n".join(lines)
        if console:
            console.print(f"[bold cyan]HIPPOCAMPUS[/]\n{text}")
        else:
            print(text)
    finally:
        db.close()


def cmd_ml_train(force: bool, consumer: str = None) -> None:
    """Treina os modelos do HIPPOCAMPUS (com promoção por comparação)."""
    console = _console()
    db = SynapseDB(DB_PATH)
    try:
        pipeline = _build_pipeline(db, _build_hippocampus(db))
        results = (
            [pipeline.train(consumer, force=force)] if consumer
            else pipeline.train_all(force=force)
        )
        for res in results:
            if not res.get("trained"):
                msg = f"{res['consumer']}: não treinado — {res.get('reason')}"
                (console.print(f"[yellow]{msg}[/]") if console else print(msg))
                continue
            promo = "PROMOVIDO" if res["promoted"] else "não promovido"
            prev = res.get("previous_metric")
            msg = (
                f"{res['consumer']}: treinado com {res['train_size']} amostras "
                f"(validação: {res['val_size']}) — métrica {res['metric']:.4f}"
                + (f" vs activo {float(prev):.4f}" if prev is not None else "")
                + f" -> {promo}"
            )
            (console.print(f"[green]{msg}[/]") if console else print(msg))
    finally:
        db.close()


def cmd_ml_export(consumer: str, output: str) -> None:
    """Exporta as amostras de treino de um consumidor para CSV."""
    console = _console()
    db = SynapseDB(DB_PATH)
    try:
        n = _build_pipeline(db).export(consumer, output)
        msg = f"Exportadas {n} amostras de '{consumer}' para {output}."
        (console.print(f"[bold green]HIPPOCAMPUS[/] {msg}") if console else print(msg))
    finally:
        db.close()


def _report_result(console, state: dict) -> None:
    status = state.get("status")
    pct = state.get("functionality_pct", 0.0)
    if console:
        if status == "approved":
            console.print(
                f"[bold green]MIND[/] Ciclo aprovado a {pct:.1f}%. "
                "Output em workspace/output/."
            )
        elif status == "needs_human":
            console.print(
                f"[bold yellow]MIND[/] needs_human a {pct:.1f}%. "
                "Histórico git mantido para intervenção manual."
            )
        else:
            console.print(f"[bold]MIND[/] Ciclo terminou com estado: {status}.")
    else:
        print(f"MIND: estado={status} pct={pct:.1f}")


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mind",
        description="MIND — Muñdji Intelligent Neural Developer (CLI).",
    )
    sub = parser.add_subparsers(dest="command")

    # export
    p_export = sub.add_parser("export", help="Exportar iterações para JSONL.")
    p_export.add_argument("--cycle-id", type=int, required=True)
    p_export.add_argument("--component", type=str, default=None,
                          help="Ex: cortex, cerebellum, neuron_1 ...")
    p_export.add_argument("--output", type=str, default="datasets/export.jsonl")

    # intervene
    p_int = sub.add_parser("intervene", help="Intervenção manual num ciclo.")
    p_int.add_argument("--cycle-id", type=int, required=True)
    p_int.add_argument("--new-task", type=str, required=True)

    # --- HIPPOCAMPUS (camada de apoio de ML) -----------------------------
    sub.add_parser("ml-status", help="Estado dos modelos do HIPPOCAMPUS.")

    p_train = sub.add_parser("ml-train", help="Treina os modelos do HIPPOCAMPUS.")
    p_train.add_argument("--force", action="store_true",
                         help="Treina mesmo abaixo de ML_MIN_TRAINING_SAMPLES.")
    p_train.add_argument("--consumer", type=str, default=None,
                         help="cortex ou cerebellum (default: ambos).")

    p_mlexp = sub.add_parser("ml-export", help="Exporta amostras de treino (CSV).")
    p_mlexp.add_argument("--consumer", type=str, required=True)
    p_mlexp.add_argument("--output", type=str, default="datasets/ml_export.csv")

    # tarefa principal (argumento posicional livre)
    parser.add_argument("task", nargs="?", help="Descrição da tarefa a gerar.")
    return parser


def main(argv=None) -> None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    argv = list(sys.argv[1:] if argv is None else argv)

    # A tarefa é o argumento principal e é texto livre — não pode colidir com
    # os subcomandos. Só encaminhamos para o parser de subcomandos quando o
    # primeiro token é explicitamente 'export' ou 'intervene' (ou ajuda).
    parser = build_parser()
    subcommands = (
        "export", "intervene", "ml-status", "ml-train", "ml-export",
        "-h", "--help",
    )
    if argv and argv[0] in subcommands:
        args = parser.parse_args(argv)
        if args.command == "export":
            cmd_export(args.cycle_id, args.component, args.output)
        elif args.command == "intervene":
            cmd_intervene(args.cycle_id, args.new_task)
        elif args.command == "ml-status":
            cmd_ml_status()
        elif args.command == "ml-train":
            cmd_ml_train(args.force, args.consumer)
        elif args.command == "ml-export":
            cmd_ml_export(args.consumer, args.output)
        else:
            parser.print_help()
            sys.exit(1)
        return

    if argv:
        # Tudo o resto é tratado como a descrição da tarefa.
        cmd_run(" ".join(argv))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
