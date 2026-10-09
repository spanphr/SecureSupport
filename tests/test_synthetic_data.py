import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from scripts.validate_data import DATA_PATH, validate_dataset


class SyntheticDataTests(unittest.TestCase):
    def validate_changed(self, change):
        payload = json.loads(DATA_PATH.read_text())
        change(payload)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text(json.dumps(payload))
            return validate_dataset(path)

    def test_fixture_and_guardrail_coverage(self):
        data = validate_dataset()
        self.assertEqual(len({case.category for case in data.failure_cases}), 6)
        self.assertGreater(len(data.knowledge_base), 0)

    def test_negative_payment_rejected(self):
        with self.assertRaises(ValidationError):
            self.validate_changed(lambda d: d['payments'][0].update(amount_cents=-1))

    def test_unknown_user_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown user'):
            self.validate_changed(lambda d: d['tickets'][0].update(user_id='missing'))

    def test_cross_user_complaint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.validate_changed(lambda d: d['complaints'][0].update(ticket_id='tkt_001'))

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate IDs'):
            self.validate_changed(lambda d: d['users'].append(d['users'][0].copy()))

    def test_real_email_rejected(self):
        with self.assertRaises(ValidationError):
            self.validate_changed(lambda d: d['users'][0].update(email='user@example.com'))


if __name__ == '__main__':
    unittest.main()
