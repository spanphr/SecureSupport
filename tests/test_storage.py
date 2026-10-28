import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from app.schemas.tools import CreateComplaint, DeleteComplaint, RecordLookup, ToolContext, UpdateComplaint
from app.storage.database import Database
from app.storage.errors import DatabaseNotInitialized, RecordNotFound, SeedConflict, StorageError, VersionConflict
from app.storage.repository import SupportRepository
from app.storage.seed import seed_database
from app.tools.support import SupportTools
from scripts.validate_data import validate_dataset


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'support.sqlite3'
        self.database = Database(self.path)
        self.database.initialize()
        self.dataset = validate_dataset()
        seed_database(self.database, self.dataset)
        self.tools = SupportTools(SupportRepository(self.database))
        self.alex = ToolContext(user_id='usr_001')
        self.sam = ToolContext(user_id='usr_002')

    def test_crud_persists_across_connections(self):
        created = self.tools.create_complaint(self.alex, CreateComplaint(ticket_id='tkt_001', description='Payment concern'))
        self.assertEqual(created.version, 1)
        reopened = SupportTools(SupportRepository(Database(self.path)))
        read = reopened.get_complaint(self.alex, RecordLookup(record_id=created.id))
        self.assertEqual(read.description, 'Payment concern')
        updated = reopened.update_complaint(self.alex, UpdateComplaint(
            complaint_id=created.id, expected_version=1, description='Updated concern', status='investigating'))
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.created_at, created.created_at)
        self.assertEqual(updated.status, 'investigating')
        self.assertEqual(updated.description, 'Updated concern')
        deleted = reopened.delete_complaint(self.alex, DeleteComplaint(complaint_id=created.id, expected_version=2))
        self.assertTrue(deleted.deleted)
        with self.assertRaises(RecordNotFound):
            reopened.get_complaint(self.alex, RecordLookup(record_id=created.id))

    def test_cross_user_reads_and_unknown_ids_have_identical_errors(self):
        for method, foreign in [(self.tools.get_payment,'pay_002'), (self.tools.get_ticket,'tkt_002'), (self.tools.get_complaint,'cmp_001')]:
            messages = []
            for identifier in (foreign, 'does-not-exist'):
                with self.assertRaises(RecordNotFound) as caught:
                    method(self.alex, RecordLookup(record_id=identifier))
                messages.append(str(caught.exception))
            self.assertEqual(messages[0], messages[1])
        self.assertEqual(self.tools.get_payment(self.alex, RecordLookup(record_id='pay_001')).amount_cents, 2500)

    def test_cross_user_writes_are_denied_and_leave_data_unchanged(self):
        with self.assertRaises(RecordNotFound):
            self.tools.create_complaint(self.alex, CreateComplaint(ticket_id='tkt_002', description='Wrong owner'))
        with self.assertRaises(RecordNotFound):
            self.tools.update_complaint(self.alex, UpdateComplaint(complaint_id='cmp_001', expected_version=1, status='closed'))
        with self.assertRaises(RecordNotFound):
            self.tools.delete_complaint(self.alex, DeleteComplaint(complaint_id='cmp_001', expected_version=1))
        original = self.tools.get_complaint(self.sam, RecordLookup(record_id='cmp_001'))
        self.assertEqual(original.status, 'open')
        self.assertEqual(original.version, 1)

    def test_owner_scoped_list_and_unknown_user(self):
        self.assertEqual(self.tools.list_complaints(self.alex), [])
        self.assertEqual([c.id for c in self.tools.list_complaints(self.sam)], ['cmp_001'])
        with self.assertRaises(RecordNotFound):
            self.tools.list_complaints(ToolContext(user_id='missing'))

    def test_stale_updates_and_deletes_do_not_overwrite(self):
        first = self.tools.get_complaint(self.sam, RecordLookup(record_id='cmp_001'))
        self.tools.update_complaint(self.sam, UpdateComplaint(complaint_id=first.id, expected_version=first.version, status='resolved'))
        with self.assertRaises(VersionConflict):
            self.tools.update_complaint(self.sam, UpdateComplaint(complaint_id=first.id, expected_version=first.version, status='open'))
        with self.assertRaises(VersionConflict):
            self.tools.delete_complaint(self.sam, DeleteComplaint(complaint_id=first.id, expected_version=first.version))
        latest = self.tools.get_complaint(self.sam, RecordLookup(record_id=first.id))
        self.assertEqual(latest.status, 'resolved')
        self.assertEqual(latest.version, 2)

    def test_values_are_bound_and_sql_like_text_is_literal(self):
        injection = "' OR 1=1 --"
        with self.assertRaises(RecordNotFound):
            self.tools.get_payment(self.alex, RecordLookup(record_id=injection))
        with self.assertRaises(RecordNotFound):
            self.tools.get_payment(ToolContext(user_id=injection), RecordLookup(record_id='pay_002'))
        description = "Complaint'); DROP TABLE users; --"
        created = self.tools.create_complaint(self.alex, CreateComplaint(ticket_id='tkt_001', description=description))
        self.assertEqual(created.description, description)
        self.assertEqual(self.tools.get_payment(self.alex, RecordLookup(record_id='pay_001')).id, 'pay_001')

    def test_invalid_tool_arguments(self):
        for values in ({'ticket_id':'', 'description':'x'}, {'ticket_id':'tkt_001', 'description':'   '},
                       {'ticket_id':'tkt_001', 'description':'x'*4001},
                       {'ticket_id':'tkt_001', 'description':'x', 'user_id':'usr_002'}):
            with self.assertRaises(ValidationError):
                CreateComplaint(**values)
        for values in ({'complaint_id':'cmp_001','expected_version':1},
                       {'complaint_id':'cmp_001','expected_version':True,'status':'open'},
                       {'complaint_id':'cmp_001','expected_version':0,'status':'open'},
                       {'complaint_id':'cmp_001','expected_version':1,'status':'unknown'}):
            with self.assertRaises(ValidationError):
                UpdateComplaint(**values)

    def test_foreign_keys_enforce_owner_even_without_repository(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction(write=True) as connection:
                self.assertEqual(connection.execute('PRAGMA foreign_keys').fetchone()[0], 1)
                connection.execute('INSERT INTO complaints VALUES(?,?,?,?,?,?,?,?)',
                                   ('bad','usr_001','tkt_002','Mismatch','open','now','now',1))
        self.assertEqual(self.tools.list_complaints(self.alex), [])

    def test_failed_transaction_rolls_back_prior_writes(self):
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction(write=True) as connection:
                connection.execute("UPDATE complaints SET status='closed' WHERE id='cmp_001'")
                connection.execute("UPDATE payments SET amount_cents=-1 WHERE id='pay_001'")
        self.assertEqual(self.tools.get_complaint(self.sam, RecordLookup(record_id='cmp_001')).status, 'open')

    def test_seed_and_initialize_reruns_preserve_edits(self):
        self.tools.update_complaint(self.sam, UpdateComplaint(complaint_id='cmp_001', expected_version=1, description='Edited locally'))
        self.assertFalse(self.database.initialize())
        self.assertFalse(seed_database(self.database, self.dataset))
        self.assertEqual(self.tools.get_complaint(self.sam, RecordLookup(record_id='cmp_001')).description, 'Edited locally')
        self.tools.delete_complaint(self.sam, DeleteComplaint(complaint_id='cmp_001', expected_version=2))
        self.assertFalse(seed_database(self.database, self.dataset))
        self.assertEqual(self.tools.list_complaints(self.sam), [])

    def test_changed_fixtures_do_not_overwrite_database(self):
        changed = self.dataset.model_copy(deep=True)
        changed.users[0].name = 'Changed fixture'
        with self.assertRaises(SeedConflict):
            seed_database(self.database, changed)
        self.assertEqual(self.tools.get_payment(self.alex, RecordLookup(record_id='pay_001')).amount_cents, 2500)

    def test_seed_failure_rolls_back_all_insertions(self):
        empty = Database(Path(self.folder.name)/'empty.sqlite3')
        empty.initialize()
        invalid = self.dataset.model_copy(deep=True)
        invalid.complaints[0].ticket_id = 'tkt_001'
        with self.assertRaises(sqlite3.IntegrityError):
            seed_database(empty, invalid)
        with empty.transaction() as connection:
            for table in ('users','payments','tickets','complaints','seed_metadata'):
                self.assertEqual(connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0], 0)

    def test_missing_database_is_not_implicitly_created(self):
        path = Path(self.folder.name)/'missing.sqlite3'
        missing = SupportTools(SupportRepository(Database(path)))
        with self.assertRaises(DatabaseNotInitialized):
            missing.get_payment(self.alex, RecordLookup(record_id='pay_001'))
        self.assertFalse(path.exists())

    def test_unrecognized_database_is_not_replaced(self):
        path = Path(self.folder.name)/'unrelated.sqlite3'
        with sqlite3.connect(path) as connection:
            connection.execute('CREATE TABLE unrelated (id TEXT)')
        with self.assertRaises(StorageError):
            Database(path).initialize()
        with sqlite3.connect(path) as connection:
            self.assertIsNotNone(connection.execute("SELECT name FROM sqlite_master WHERE name='unrelated'").fetchone())


if __name__ == '__main__':
    unittest.main()
