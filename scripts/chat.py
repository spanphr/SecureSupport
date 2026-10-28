"""One-turn CLI for the support pipeline."""

import argparse

from pydantic import ValidationError

from app.agent.factory import build_pipeline
from app.schemas.pipeline import PipelineRequest, SessionContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the SecureSupport agent one question.")
    parser.add_argument("message")
    parser.add_argument("--user", required=True,
                        help="Demo customer id from the session; not authentication.")
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm a pending state-changing action.")
    args = parser.parse_args()
    try:
        response = build_pipeline().run(
            PipelineRequest(message=args.message, confirm_write=args.confirm),
            SessionContext(customer_id=args.user),
        )
    except ValidationError:
        raise SystemExit("Message must contain between 1 and 4000 characters.") from None
    except Exception:
        raise SystemExit(
            "Agent setup failed. Check local API-key configuration and dependencies."
        ) from None
    print(response.model_dump_json(indent=2))
    if response.status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
