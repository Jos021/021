"""Backup da SYNAPSE DB.

Estratégia:
  - Contínuo: modo WAL do SQLite (garantido em database.py)
  - Corrente: job periódico (APScheduler) a cada BACKUP_INTERVAL_MINUTES
  - Rotação: BACKUP_RETENTION_HOURS + BACKUP_DAILY_RETENTION_DAYS
  - Off-site (opcional): SEMPRE local/rede privada — NUNCA cloud externa
  - Corre em thread/processo separado

Princípio inviolável: nada sai do hardware ou rede do utilizador. O caminho
off-site é um caminho de sistema de ficheiros (ex: NAS na rede privada),
nunca um endpoint de nuvem externa.
"""

import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler


class BackupManager:
    """Gere o backup periódico e a rotação de cópias da SYNAPSE DB."""

    def __init__(
        self,
        db_path: str = "synapse.db",
        backup_dir: str = "backups",
    ):
        self.db_path = db_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

        self.interval_minutes = int(os.getenv("BACKUP_INTERVAL_MINUTES", "15"))
        self.retention_hours = int(os.getenv("BACKUP_RETENTION_HOURS", "24"))
        self.daily_retention_days = int(
            os.getenv("BACKUP_DAILY_RETENTION_DAYS", "30")
        )
        # Off-site opcional — SEMPRE um caminho local/rede privada.
        self.offsite_path = os.getenv("BACKUP_OFFSITE_PATH", "").strip()

        self._scheduler = BackgroundScheduler()

    def start(self) -> None:
        """Arranca o scheduler numa thread separada."""
        self._scheduler.add_job(
            self.run_backup,
            "interval",
            minutes=self.interval_minutes,
            id="synapse_backup",
            replace_existing=True,
        )
        self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def run_backup(self) -> str:
        """Faz uma cópia consistente da base (API de backup do SQLite)."""
        if not os.path.exists(self.db_path):
            return ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(self.backup_dir, f"synapse_{ts}.db")

        # Backup consistente mesmo com WAL activo, via API nativa do SQLite.
        src = sqlite3.connect(self.db_path)
        dst = sqlite3.connect(dest)
        try:
            with dst:
                src.backup(dst)
        finally:
            src.close()
            dst.close()

        # Off-site opcional (sempre local/rede privada).
        if self.offsite_path:
            os.makedirs(self.offsite_path, exist_ok=True)
            shutil.copy2(dest, os.path.join(self.offsite_path, os.path.basename(dest)))

        self._rotate()
        return dest

    def _rotate(self) -> None:
        """Aplica a política de retenção.

        Mantém todas as cópias dentro de BACKUP_RETENTION_HOURS. Para cópias
        mais antigas, mantém uma por dia até BACKUP_DAILY_RETENTION_DAYS;
        o resto é removido.
        """
        now = datetime.now()
        hourly_cutoff = now - timedelta(hours=self.retention_hours)
        daily_cutoff = now - timedelta(days=self.daily_retention_days)

        backups = []
        for name in os.listdir(self.backup_dir):
            if not (name.startswith("synapse_") and name.endswith(".db")):
                continue
            path = os.path.join(self.backup_dir, name)
            try:
                stamp = name[len("synapse_"):-len(".db")]
                when = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            backups.append((when, path))

        backups.sort(reverse=True)
        kept_days = set()
        for when, path in backups:
            if when >= hourly_cutoff:
                continue  # dentro da janela horária -> manter tudo
            if when < daily_cutoff:
                self._safe_remove(path)  # demasiado antigo -> remover
                continue
            day_key = when.strftime("%Y%m%d")
            if day_key in kept_days:
                self._safe_remove(path)  # já há uma cópia deste dia
            else:
                kept_days.add(day_key)

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass
