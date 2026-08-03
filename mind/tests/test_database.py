"""SYNAPSE DB — base de dados única, local, em modo WAL."""

import json
import os

import pytest


def test_modo_wal_activo(db):
    """Modo WAL é obrigatório desde o início."""
    modo = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert modo.lower() == "wal"


def test_foreign_keys_activas(db):
    assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


@pytest.mark.parametrize("tabela", [
    "cycles", "iterations", "reports", "decisions",
    "ml_training_data", "ml_model_versions", "ml_predictions_log",
])
def test_tabela_existe(db, tabela):
    linha = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (tabela,),
    ).fetchone()
    assert linha is not None


def test_ciclo_arranca_em_progresso(db):
    cid = db.create_cycle("uma tarefa")
    ciclo = db.get_cycle(cid)
    assert ciclo["status"] == "in_progress"
    assert ciclo["task"] == "uma tarefa"
    assert ciclo["final_functionality_pct"] is None


def test_actualizar_ciclo(db, cycle_id):
    db.update_cycle(cycle_id, status="approved", final_pct=99.5)
    ciclo = db.get_cycle(cycle_id)
    assert ciclo["status"] == "approved"
    assert ciclo["final_functionality_pct"] == 99.5


def test_intervencao_actualiza_a_tarefa(db, cycle_id):
    db.update_cycle(cycle_id, task="nova descrição")
    assert db.get_cycle(cycle_id)["task"] == "nova descrição"


def test_ciclo_inexistente_devolve_none(db):
    assert db.get_cycle(9999) is None


def test_foreign_key_impede_iteracao_orfa(db):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.log_iteration(9999, 1, "1", "cortex")


def test_registo_de_iteracao_e_decisao(db, cycle_id):
    db.log_iteration(cycle_id, 1, "1", "cortex", "entrada", "saida", "completo", 1.5)
    db.log_decision(cycle_id, 1, "cortex", "distribuí a todos.")
    it = db._conn.execute(
        "SELECT * FROM iterations WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    assert it["component"] == "cortex" and it["duration_seconds"] == 1.5
    dec = db._conn.execute(
        "SELECT * FROM decisions WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    assert dec["decision_text"] == "distribuí a todos."


def test_registo_de_relatorio(db, cycle_id):
    db.log_report(cycle_id, 1, 87.5, "falhas", "melhorias")
    rel = db._conn.execute(
        "SELECT * FROM reports WHERE cycle_id=?", (cycle_id,)
    ).fetchone()
    assert rel["functionality_pct"] == 87.5


# --- Exportação JSONL para fine-tuning -----------------------------------
def test_export_jsonl_filtra_por_ciclo(db, tmp_path):
    c1, c2 = db.create_cycle("a"), db.create_cycle("b")
    db.log_iteration(c1, 1, "1", "cortex", "in1", "out1", "full1")
    db.log_iteration(c2, 1, "1", "cortex", "in2", "out2", "full2")
    destino = str(tmp_path / "export.jsonl")

    assert db.export_to_jsonl(c1, output_path=destino) == 1
    registos = [json.loads(l) for l in open(destino, encoding="utf-8")]
    assert registos[0]["cycle_id"] == c1
    assert registos[0]["prompt"] == "in1"
    assert registos[0]["completion"] == "full1"


def test_export_jsonl_filtra_por_componente(db, tmp_path):
    cid = db.create_cycle("a")
    db.log_iteration(cid, 1, "2", "neuron_1", "i", "o", "f")
    db.log_iteration(cid, 1, "2", "neuron_2", "i", "o", "f")
    destino = str(tmp_path / "n1.jsonl")

    assert db.export_to_jsonl(cid, component="neuron_1", output_path=destino) == 1
    registo = json.loads(open(destino, encoding="utf-8").readline())
    assert registo["component"] == "neuron_1"


def test_export_cria_o_directorio(db, cycle_id, tmp_path):
    destino = str(tmp_path / "novo" / "sub" / "e.jsonl")
    db.export_to_jsonl(cycle_id, output_path=destino)
    assert os.path.exists(destino)


def test_export_de_ciclo_vazio_produz_ficheiro_vazio(db, cycle_id, tmp_path):
    destino = str(tmp_path / "vazio.jsonl")
    assert db.export_to_jsonl(cycle_id, output_path=destino) == 0
    assert open(destino, encoding="utf-8").read() == ""
