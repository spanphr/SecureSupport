"""One-message CLI for the current agent stage."""
import argparse
from app.agent.factory import build_agent
from pydantic import ValidationError


def main():
    parser = argparse.ArgumentParser(description="Ask the SecureSupport demo agent one question.")
    parser.add_argument('message')
    args = parser.parse_args()
    try:
        response = build_agent().run(args.message)
    except ValidationError:
        raise SystemExit('Message must contain between 1 and 4000 characters.') from None
    except Exception:
        raise SystemExit('Agent setup failed. Check local API-key configuration and dependencies.') from None
    print(response.model_dump_json(indent=2))
    if response.status == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
