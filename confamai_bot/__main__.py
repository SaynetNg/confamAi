"""Entry point.

    python -m confamai_bot                 serve the webhook
    python -m confamai_bot --status        is the server reachable and green
    python -m confamai_bot --test          the offline test suite
    python -m confamai_bot --ask "..."     one turn, no WhatsApp, for checking
    python -m confamai_bot --assay         a batch of real questions against
                                           the live server
    python -m confamai_bot --encode-ds PASSWORD   build a percent-encoded DSN and test it
    python -m confamai_bot --version

`--ask` and `--assay` exist because the bot's real dependency is a remote GPU
you do not control, and "it works on my machine" is not a claim anyone can make
about it without talking to the server. They exercise the exact same
`NAtlasClient.chat_with_retry` the WhatsApp path uses, so what they prove is the
thing that actually breaks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# The assay prints Yoruba and Igbo diacritics, and a Windows console defaults to
# cp1252, which raises UnicodeEncodeError on the first one — killing the run
# part-way through and losing every result after it. Same guard as
# n_atlas_client/main.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confamai_bot",
        description="Confam AI — WhatsApp bot for the N-ATLaS NAFDAC "
                    "verification API.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="bind address (default 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="bind port (default 8080)"
    )
    parser.add_argument(
        "--status", action="store_true", help="report server and greenbook readiness"
    )
    parser.add_argument(
        "--ask", metavar="TEXT", help="send one message and print the reply"
    )
    parser.add_argument(
        "--assay", action="store_true",
        help="run a batch of real questions against the live server",
    )
    parser.add_argument(
        "--assay-only", action="store_true",
        help="with --assay, skip the health probe",
    )
    parser.add_argument(
        "--encode-dsn", metavar="PASSWORD",
        help="build a correctly percent-encoded DSN from the raw Supabase "
             "password, using the host already in your .env",
    )
    parser.add_argument(
        "--test", action="store_true", help="run the offline test suite"
    )
    parser.add_argument("--env", default=None, metavar="PATH", help="custom .env path")
    parser.add_argument(
        "--debug", action="store_true", help="verbose output; never prints secrets"
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        print("confamai_bot %s (python %s)" % (__version__, sys.version.split()[0]))
        return 0

    if args.test:
        from .tests import run_tests

        return run_tests()

    from .config import ConfigError, load_config, validate

    cfg = load_config(env_file=args.env, debug=args.debug)

    if args.encode_dsn is not None:
        return _encode_dsn(cfg, args.encode_dsn)

    if args.status:
        return asyncio.run(_status(cfg))

    if args.ask:
        return asyncio.run(_ask(cfg, args.ask))

    if args.assay:
        return asyncio.run(_assay(cfg, skip_health=args.assay_only))

    # Serving needs Meta credentials. Everything above deliberately does not.
    try:
        validate(cfg, require_whatsapp=True)
    except ConfigError as exc:
        print(str(exc))
        return 2

    try:
        import uvicorn
    except ImportError:
        print("Install the dependencies first:\n"
              "    pip install -r confamai_bot/requirements.txt")
        return 2

    print("Confam AI WhatsApp bot %s" % __version__)
    for key, value in cfg.describe().items():
        print("  %-24s %s" % (key, value))
    print()
    print("Press Ctrl+C to stop.")
    print()

    uvicorn.run(
        "confamai_bot.webhook:app",
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


# ---------------------------------------------------------------------------
def _encode_dsn(cfg, raw_password: str) -> int:
    """Print a correctly percent-encoded DSN for a raw Supabase password.

    Supabase generates passwords containing `@ : / # ?`, all of which are
    structural in a postgres:// URL. Hand-editing one into .env means guessing
    the escaping, and a wrong guess surfaces as `missing '/' after host`, which
    reads like a host typo and sends an operator debugging the wrong half of
    the string.

    So: paste the raw password here and the encoded DSN comes out. The host,
    port and database are taken from whatever is already in .env, so only the
    one value you cannot easily type correctly is involved.

    The password is printed in clear text in the terminal, which is fine for a
    local machine and the reason this is a command rather than something that
    runs on a schedule.
    """
    from urllib.parse import quote

    if not cfg.session_dsn:
        print("No DSN in .env yet, so there is no host to build on.\n"
              "Put a best-guess DSN in confamai_bot/.env first, then re-run, or "
              "set WHATSAPP_SESSION_DSN with a placeholder password.")
        return 2

    import psycopg

    info = psycopg.conninfo.conninfo_to_dict(cfg.session_dsn)
    host = info.get("host", "")
    port = info.get("port", "5432")
    dbname = info.get("dbname", "postgres")
    user = info.get("user", "")

    encoded = quote(raw_password, safe="")
    print()
    print("Paste this into confamai_bot/.env as WHATSAPP_SESSION_DSN:")
    print()
    print("  postgresql://%s:%s@%s:%s/%s" % (user, encoded, host, port, dbname))
    print()
    if encoded != raw_password:
        escaped = sorted({ch for ch in raw_password if ch in "@:/#?%"})
        print("  encoded %s as %s — these break a postgres:// URL if left raw"
              % (
                  ", ".join(repr(c) for c in escaped),
                  ", ".join(quote(c, safe="") for c in escaped),
              ))
    else:
        print("  nothing needed encoding")
    print()

    # Prove it before they commit it to a file.
    candidate = "postgresql://%s:%s@%s:%s/%s" % (user, encoded, host, port, dbname)
    try:
        with psycopg.connect(candidate, connect_timeout=20) as conn:
            conn.execute("select 1")
        print("*** CONNECTED — this password is correct ***")
        print()
        print("Next: python -m confamai_bot --status")
        return 0
    except Exception as exc:
        first = str(exc).strip().splitlines()[0]
        print("still failing: %s" % first[:160])
        print()
        if "password authentication failed" in first.lower():
            print("  The encoding is now correct, so the password itself is")
            print("  wrong. Reset it in Supabase -> Settings -> Database, then")
            print("  re-run this command with the new value.")
        return 1


# ---------------------------------------------------------------------------
async def _status(cfg) -> int:
    from .natlas import NAtlasError, NAtlasClient

    print("Confam AI WhatsApp bot %s" % __version__)
    print()
    print("Configuration")
    for key, value in cfg.describe().items():
        print("  %-24s %s" % (key, value))

    print()
    print("N-ATLaS")
    print("  %-24s %s" % ("base url", cfg.base_url or "NOT SET"))
    async with NAtlasClient(cfg) as natlas:
        try:
            info = await natlas.health()
        except NAtlasError as exc:
            print("  %-24s unreachable (%s)" % ("status", exc))
            return 1

        print("  %-24s %s" % ("status", info.get("status")))
        print("  %-24s %s" % ("model", info.get("served_model_name")))
        print("  %-24s %s" % ("gpu", info.get("gpu")))
        print("  %-24s %s" % ("model loaded", info.get("model_loaded")))
        print("  %-24s %s" % ("uptime (s)", info.get("uptime_seconds")))
        print("  %-24s %s" % ("streaming", info.get("streaming")))
        print("  %-24s %s" % ("max new tokens", info.get("max_new_tokens")))

        nafdac = info.get("nafdac") or {}
        print()
        print("NAFDAC greenbook")
        print("  %-24s %s" % ("enabled", nafdac.get("enabled")))
        print("  %-24s %s" % ("ready", nafdac.get("ready")))
        if nafdac.get("reason"):
            print("  %-24s %s" % ("reason", nafdac.get("reason")))
        for source in nafdac.get("sources") or []:
            print(
                "  %-9s %-4s %s"
                % (source.get("name"), "ok" if source.get("usable") else "--",
                   source.get("detail"))
            )
        if not nafdac.get("ready"):
            print()
            print("  NOT READY. Replies will say verification is temporarily")
            print("  unavailable rather than answering from memory. See")
            print("  n_atlas_server/readme.md section on the greenbook.")
            return 1

    print()
    print("Ready. Run: python -m confamai_bot")
    return 0


async def _ask(cfg, text: str) -> int:
    from .natlas import NAtlasError, NAtlasClient
    from .render import clean

    if not cfg.base_url or not cfg.api_key:
        print("Set N_ATLAS_BASE_URL and N_ATLAS_API_KEY in .env first.")
        return 2

    async with NAtlasClient(cfg) as natlas:
        try:
            reply = await natlas.chat_with_retry(
                [{"role": "user", "content": text}]
            )
        except NAtlasError as exc:
            print("FAILED: %s" % exc)
            return 1

    print("---")
    print(clean(reply))
    print("---")
    return 0


async def _assay(cfg, skip_health: bool = False) -> int:
    """A real battery of questions against the live server.

    Every case here is one that has actually gone wrong in this project. The
    unregistered number and the "is it fake" pressure case come straight from
    the failures recorded in `NAFDAC_INSTRUCTION`
    (n_atlas_server/natlas_notebook.py:3642-3702): a verdict inverted three times
    in ten runs, and "just tell me yes or no" answered "Yes" twice out of three.
    """
    from .assay import run_assay

    return await run_assay(cfg, skip_health=skip_health)


if __name__ == "__main__":
    sys.exit(main())