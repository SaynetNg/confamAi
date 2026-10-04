#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N-ATLaS laptop client entry point (modular layout).

Usage:
    python -m n_atlas_client                    # chat
    python -m n_atlas_client --debug            # chat with developer output
    python -m n_atlas_client --test             # test suite
    python -m n_atlas_client --version
"""

import argparse
import platform
import sys

# Ensure multilingual text (Yoruba/Igbo diacritics) prints on all terminals.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="n_atlas_client",
        description="ConfamAI laptop client — talks to the remote Kaggle "
                    "server (Tesla T4) over an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show HTTP statuses, request durations and exception details "
             "(the API key is never shown)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run the test suite: connection, health, models, four languages, "
             "memory and language switching",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument(
        "--env",
        default=None,
        metavar="PATH",
        help="custom .env file path",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .config import ConfigError, load_config, validate

    cfg = load_config(env_file=args.env, debug=args.debug)

    if args.version:
        print("n_atlas_client %s (python %s, %s)" % (
            "1.0.0", platform.python_version(), platform.system()
        ))
        return 0

    try:
        validate(cfg)
    except ConfigError as exc:
        print(str(exc))
        return 2

    try:
        from openai import OpenAI  # noqa: F401

        import dotenv  # noqa: F401
    except Exception:
        print("Install the required packages first:\n"
              "    pip install openai python-dotenv")
        return 2

    if args.test:
        from .tests import run_tests

        return run_tests(cfg)

    from .api import NAtlasClient
    from .terminal import run_repl

    client = NAtlasClient(cfg, debug=args.debug)
    try:
        client.connect()
        client.initialize()
    except Exception as exc:
        print("Unable to reach the N-ATLaS server. Check the server or "
              "network connection.")
        if args.debug:
            import traceback

            traceback.print_exc()
        else:
            print("  (%s)" % exc)
        return 1

    try:
        return run_repl(client)
    except KeyboardInterrupt:
        print()
        print("Bye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())