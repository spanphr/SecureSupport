"""Parameterized SQL and owner-scoped transactional record operations."""
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.domain import Payment, Ticket
from app.schemas.tools import (
    AccountStatus,
    ComplaintRecord,
    CreateComplaint,
    CreateTicket,
    DeleteComplaint,
    DeletedComplaint,
    UpdateComplaint,
)
from app.storage.database import Database
from app.storage.errors import RecordNotFound, VersionConflict

NOT_FOUND = 'Record not found for this user.'


def _owned_row(connection, table: str, identifier: str, user_id: str):
    if table not in ('payments', 'tickets', 'complaints'):
        raise ValueError('Unsupported table.')
    row = connection.execute(
        f'SELECT * FROM {table} WHERE id=? AND user_id=?', (identifier, user_id)
    ).fetchone()
    if row is None:
        # Missing and another user's record receive the same message.
        raise RecordNotFound(NOT_FOUND)
    return dict(row)


class SupportRepository:
    def __init__(self, database: Database):
        self.database = database

    def get_account(self, user_id: str) -> AccountStatus:
        """Minimal account view: deliberately excludes name and email."""
        with self.database.transaction() as connection:
            row = connection.execute(
                'SELECT id, mfa_enabled FROM users WHERE id=?', (user_id,)
            ).fetchone()
            if row is None:
                # Missing and another user's record are indistinguishable to the caller.
                raise RecordNotFound(NOT_FOUND)
            return AccountStatus(user_id=row['id'], mfa_enabled=bool(row['mfa_enabled']))

    def create_ticket(self, user_id: str, request: CreateTicket) -> Ticket:
        with self.database.transaction(write=True) as connection:
            if connection.execute('SELECT 1 FROM users WHERE id=?', (user_id,)).fetchone() is None:
                raise RecordNotFound(NOT_FOUND)
            now = datetime.now(timezone.utc).isoformat()
            identifier = 'tkt_' + uuid4().hex
            connection.execute(
                'INSERT INTO tickets VALUES(?,?,?,?,?,?)',
                (identifier, user_id, request.subject, request.message, 'open', now))
            return Ticket.model_validate(_owned_row(connection, 'tickets', identifier, user_id))

    def get_payment(self, user_id: str, identifier: str) -> Payment:
        with self.database.transaction() as connection:
            return Payment.model_validate(_owned_row(connection, 'payments', identifier, user_id))

    def get_ticket(self, user_id: str, identifier: str) -> Ticket:
        with self.database.transaction() as connection:
            return Ticket.model_validate(_owned_row(connection, 'tickets', identifier, user_id))

    def get_complaint(self, user_id: str, identifier: str) -> ComplaintRecord:
        with self.database.transaction() as connection:
            return ComplaintRecord.model_validate(_owned_row(connection, 'complaints', identifier, user_id))

    def list_complaints(self, user_id: str) -> list[ComplaintRecord]:
        with self.database.transaction() as connection:
            if connection.execute('SELECT 1 FROM users WHERE id=?', (user_id,)).fetchone() is None:
                raise RecordNotFound(NOT_FOUND)
            return [ComplaintRecord.model_validate(dict(row)) for row in connection.execute(
                'SELECT * FROM complaints WHERE user_id=? ORDER BY created_at,id', (user_id,))]

    def create_complaint(self, user_id: str, request: CreateComplaint) -> ComplaintRecord:
        with self.database.transaction(write=True) as connection:
            _owned_row(connection, 'tickets', request.ticket_id, user_id)
            now = datetime.now(timezone.utc).isoformat()
            identifier = 'cmp_' + uuid4().hex
            connection.execute('INSERT INTO complaints VALUES(?,?,?,?,?,?,?,?)',
                               (identifier, user_id, request.ticket_id, request.description,
                                'open', now, now, 1))
            return ComplaintRecord.model_validate(_owned_row(connection, 'complaints', identifier, user_id))

    def update_complaint(self, user_id: str, request: UpdateComplaint) -> ComplaintRecord:
        with self.database.transaction(write=True) as connection:
            row = _owned_row(connection, 'complaints', request.complaint_id, user_id)
            if row['version'] != request.expected_version:
                raise VersionConflict('Complaint changed; read its latest version before retrying.')
            description = row['description'] if request.description is None else request.description
            status = row['status'] if request.status is None else request.status
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                'UPDATE complaints SET description=?,status=?,updated_at=?,version=version+1 '
                'WHERE id=? AND user_id=? AND version=?',
                (description, status, now, request.complaint_id, user_id, request.expected_version))
            return ComplaintRecord.model_validate(_owned_row(connection, 'complaints', request.complaint_id, user_id))

    def delete_complaint(self, user_id: str, request: DeleteComplaint) -> DeletedComplaint:
        with self.database.transaction(write=True) as connection:
            row = _owned_row(connection, 'complaints', request.complaint_id, user_id)
            if row['version'] != request.expected_version:
                raise VersionConflict('Complaint changed; read its latest version before retrying.')
            connection.execute('DELETE FROM complaints WHERE id=? AND user_id=? AND version=?',
                               (request.complaint_id, user_id, request.expected_version))
            return DeletedComplaint(complaint_id=request.complaint_id)
