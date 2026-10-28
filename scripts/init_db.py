"""Explicit local database initialization; never resets or overwrites records."""
import argparse
import sqlite3
from pathlib import Path
from pydantic import ValidationError
from app.config import Settings
from app.storage.database import Database
from app.storage.errors import StorageError
from app.storage.seed import seed_database
from scripts.validate_data import validate_dataset


def main():
    parser = argparse.ArgumentParser(description='Initialize and seed the local demo SQLite database.')
    parser.add_argument('--db', type=Path, help='Override the configured database path.')
    args = parser.parse_args()
    try:
        dataset = validate_dataset()
        database = Database(args.db or Settings().database_path)
        created = database.initialize()
        seeded = seed_database(database, dataset)
    except StorageError as error:
        raise SystemExit(str(error)) from None
    except (sqlite3.Error, ValidationError, OSError, ValueError):
        raise SystemExit('Database initialization failed; check fixtures, configuration, and permissions.') from None
    print(f'Database ready. Schema created: {created}. Fixtures imported: {seeded}.')


if __name__ == '__main__':
    main()
