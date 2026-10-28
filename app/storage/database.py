"""SQLite lifecycle and transactions, using Python's standard library."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from app.storage.errors import DatabaseNotInitialized, StorageError

SCHEMA_VERSION = 1
SCHEMA = (
    """CREATE TABLE users (
        id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL, email TEXT NOT NULL,
        mfa_enabled INTEGER NOT NULL CHECK(mfa_enabled IN (0,1)))""",
    """CREATE TABLE payments (
        id TEXT PRIMARY KEY NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
        amount_cents INTEGER NOT NULL CHECK(typeof(amount_cents)='integer' AND amount_cents>0),
        currency TEXT NOT NULL CHECK(currency='AUD'),
        status TEXT NOT NULL CHECK(status IN ('pending','completed','failed','refunded')),
        created_at TEXT NOT NULL)""",
    """CREATE TABLE tickets (
        id TEXT PRIMARY KEY NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
        subject TEXT NOT NULL, message TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('open','in_progress','resolved')),
        created_at TEXT NOT NULL, UNIQUE(id,user_id))""",
    """CREATE TABLE complaints (
        id TEXT PRIMARY KEY NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
        ticket_id TEXT NOT NULL, description TEXT NOT NULL CHECK(length(trim(description))>0),
        status TEXT NOT NULL CHECK(status IN ('open','investigating','resolved','closed')),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(typeof(version)='integer' AND version>=1),
        FOREIGN KEY(ticket_id,user_id) REFERENCES tickets(id,user_id))""",
    "CREATE INDEX complaints_owner ON complaints(user_id)",
    "CREATE TABLE seed_metadata (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT NOT NULL)",
)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()

    def _open(self, create: bool = False) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5)
        else:
            if not self.path.is_file():
                raise DatabaseNotInitialized('Initialize the local database first.')
            # mode=rw prevents a lookup from silently creating a missing database.
            connection = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        return connection

    def initialize(self) -> bool:
        connection = self._open(create=True)
        try:
            connection.execute('BEGIN IMMEDIATE')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version == SCHEMA_VERSION:
                connection.commit()
                return False
            if version != 0 or connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone():
                raise StorageError('Unrecognized database schema; no tables were replaced.')
            for statement in SCHEMA:
                connection.execute(statement)
            connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, write: bool = False):
        connection = self._open()
        try:
            if connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
                raise DatabaseNotInitialized('Database schema is not supported.')
            connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
