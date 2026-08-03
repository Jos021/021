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
from agent.model_router import ModelRouter
from agent.state import new_state

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "synapse.db")
SPECIALTIES_PATH = os.path.join(BASE_DIR, "config", "neuron_specialties.yaml")


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


def _build_graph(db, console) -> MindGraph:
    router = ModelRouter(db=db)
    specialties = _load_specialties()
    return MindGraph(router, db, specialties, console)


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
def cmd_run(task: str) -> None:
    """Corre um ciclo completo do MIND para a tarefa dada."""
    console = _console()
    db = SynapseDB(DB_PATH)
    backup = BackupManager(DB_PATH, os.path.join(BASE_DIR, "backups"))
    backup.start()
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
    if argv and argv[0] in ("export", "intervene", "-h", "--help"):
        args = parser.parse_args(argv)
        if args.command == "export":
            cmd_export(args.cycle_id, args.component, args.output)
        elif args.command == "intervene":
            cmd_intervene(args.cycle_id, args.new_task)
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
