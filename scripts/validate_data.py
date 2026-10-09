"""Validate fixtures without loading environment configuration or calling services."""
import json
from pathlib import Path

from app.schemas.domain import Dataset

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_support.json"


def validate_dataset(path: Path = DATA_PATH) -> Dataset:
    dataset = Dataset.model_validate(json.loads(path.read_text()))
    for field in Dataset.model_fields:
        records = getattr(dataset, field)
        identifiers = [record.id for record in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate IDs in {field}")
    user_ids = {user.id for user in dataset.users}
    tickets = {ticket.id: ticket for ticket in dataset.tickets}
    for record in [*dataset.payments, *dataset.tickets, *dataset.complaints]:
        if record.user_id not in user_ids:
            raise ValueError(f"Unknown user for {record.id}")
    for complaint in dataset.complaints:
        ticket = tickets.get(complaint.ticket_id)
        if ticket is None or ticket.user_id != complaint.user_id:
            raise ValueError(f"Invalid ticket ownership for {complaint.id}")
    return dataset


if __name__ == "__main__":
    data = validate_dataset()
    print("Synthetic dataset validated: " + ", ".join(
        f"{len(getattr(data, name))} {name}" for name in Dataset.model_fields
    ))
