"""Local demo CRUD CLI. Explicit --user is a demo identity, not authentication."""
import argparse
import json
import sqlite3
from pathlib import Path
from pydantic import ValidationError
from app.config import Settings
from app.schemas.tools import CreateComplaint, DeleteComplaint, RecordLookup, ToolContext, UpdateComplaint
from app.storage.database import Database
from app.storage.errors import StorageError
from app.storage.repository import SupportRepository
from app.tools.support import SupportTools


def main():
    parser = argparse.ArgumentParser(description='Run deterministic support tools on fictional local data.')
    parser.add_argument('--db', type=Path)
    parser.add_argument('--user', required=True, help='Demo user ID; not an authenticated session.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('get-payment', 'get-ticket', 'get-complaint'):
        sub = commands.add_parser(name)
        sub.add_argument('record_id')
    commands.add_parser('list-complaints')
    create = commands.add_parser('create-complaint')
    create.add_argument('--ticket', required=True)
    create.add_argument('--description', required=True)
    update = commands.add_parser('update-complaint')
    update.add_argument('complaint_id')
    update.add_argument('--version', required=True, type=int)
    update.add_argument('--description')
    update.add_argument('--status', choices=('open', 'investigating', 'resolved', 'closed'))
    delete = commands.add_parser('delete-complaint')
    delete.add_argument('complaint_id')
    delete.add_argument('--version', required=True, type=int)
    args = parser.parse_args()
    try:
        context = ToolContext(user_id=args.user)
        tools = SupportTools(SupportRepository(Database(args.db or Settings().database_path)))
        if args.command.startswith('get-'):
            method = {'get-payment': tools.get_payment, 'get-ticket': tools.get_ticket,
                      'get-complaint': tools.get_complaint}[args.command]
            result = method(context, RecordLookup(record_id=args.record_id))
        elif args.command == 'list-complaints':
            result = tools.list_complaints(context)
        elif args.command == 'create-complaint':
            result = tools.create_complaint(context, CreateComplaint(ticket_id=args.ticket, description=args.description))
        elif args.command == 'update-complaint':
            result = tools.update_complaint(context, UpdateComplaint(
                complaint_id=args.complaint_id, expected_version=args.version,
                description=args.description, status=args.status))
        else:
            result = tools.delete_complaint(context, DeleteComplaint(
                complaint_id=args.complaint_id, expected_version=args.version))
    except StorageError as error:
        raise SystemExit(str(error)) from None
    except ValidationError:
        raise SystemExit('Invalid tool arguments; check required fields and allowed values.') from None
    except (sqlite3.Error, OSError):
        raise SystemExit('Database operation failed; no completion can be confirmed.') from None
    data = [item.model_dump(mode='json') for item in result] if isinstance(result, list) else result.model_dump(mode='json')
    print(json.dumps(data, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
