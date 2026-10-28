"""One-time, atomic fixture import; reruns do not overwrite edited records."""
import hashlib
import json
from app.schemas.domain import Dataset
from app.storage.database import Database
from app.storage.errors import SeedConflict


def seed_database(database: Database, dataset: Dataset) -> bool:
    dataset = Dataset.model_validate(dataset.model_dump())
    records = {name: [r.model_dump(mode='json') for r in getattr(dataset, name)]
               for name in ('users', 'payments', 'tickets', 'complaints')}
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    with database.transaction(write=True) as connection:
        previous = connection.execute('SELECT fingerprint FROM seed_metadata WHERE id=1').fetchone()
        if previous:
            if previous['fingerprint'] != fingerprint:
                raise SeedConflict('Fixtures changed; existing database records were not overwritten.')
            return False
        if any(connection.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone() for table in records):
            raise SeedConflict('Database already contains unseeded records; import stopped.')
        connection.executemany('INSERT INTO users VALUES(?,?,?,?)', [
            (u.id, u.name, u.email, int(u.mfa_enabled)) for u in dataset.users])
        connection.executemany('INSERT INTO payments VALUES(?,?,?,?,?,?)', [
            (p.id, p.user_id, p.amount_cents, p.currency, p.status, p.created_at.isoformat())
            for p in dataset.payments])
        connection.executemany('INSERT INTO tickets VALUES(?,?,?,?,?,?)', [
            (t.id, t.user_id, t.subject, t.message, t.status, t.created_at.isoformat())
            for t in dataset.tickets])
        connection.executemany('INSERT INTO complaints VALUES(?,?,?,?,?,?,?,?)', [
            (c.id, c.user_id, c.ticket_id, c.description, c.status, c.created_at.isoformat(),
             c.created_at.isoformat(), 1) for c in dataset.complaints])
        connection.execute('INSERT INTO seed_metadata VALUES(1,?)', (fingerprint,))
    return True
