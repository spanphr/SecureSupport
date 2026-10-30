"""Run the browser demo of the support pipeline.

Scripted mode (default) is free, deterministic, and answers the four preset
questions through the real guardrails, tools, and escalation. `--live` hands the
same requests to the configured OpenAI model and Chroma index, which spends tokens
on your key.

    python -m scripts.serve              # http://127.0.0.1:8000
    python -m scripts.serve --live       # real model (paid)
    python -m scripts.serve --port 9000
"""

import argparse

import uvicorn

from app.api.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SecureSupport browser demo.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Keep loopback unless you intend to expose it.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--live", action="store_true",
                        help="Use the configured OpenAI model and index (spends tokens).")
    args = parser.parse_args()

    mode = "live" if args.live else "scripted"
    if mode == "live":
        print("LIVE MODE: requests use the configured chat model and retrieval index, "
              "spending tokens on your key.")
    else:
        print("SCRIPTED MODE: canned model output, real guardrails, tools, and escalation. "
              "Free and deterministic. Use --live for the real model.")
    print(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run(create_app(mode=mode), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
