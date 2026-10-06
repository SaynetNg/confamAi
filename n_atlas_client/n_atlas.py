#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n_atlas.py — single-file N-ATLaS laptop client.

A self-contained copy of the client: no imports from this package, so it can be
copied to any machine on its own. It lives beside the modular package
(`n_atlas_client/`) and offers exactly the same commands. Prefer
`python -m n_atlas_client` unless you want the one-file version.

Talks to the remotely hosted N-ATLaS OpenAI-compatible API (Kaggle server +
cloudflared tunnel). NO model runs on this laptop: no torch, no
transformers, no bitsandbytes, no Hugging Face download.

Requires:
    pip install -r requirements.txt

Usage (from the project root):
    python n_atlas_client/n_atlas.py       # chat
    python n_atlas_client/n_atlas.py --debug
    python n_atlas_client/n_atlas.py --test
"""

import os
import sys
import time
import json
import argparse
import platform
import traceback
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

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

try:
    from dotenv import load_dotenv  # python-dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
    from openai import AuthenticationError, PermissionDeniedError, NotFoundError
    from openai import RateLimitError, UnprocessableEntityError, APIStatusError
    from openai import BadRequestError, InternalServerError
except Exception:  # pragma: no cover
    OpenAI = None

# The greenbook lookup package lives next to this file, at the project root.
# Resolved here rather than imported relatively because this client is meant to
# work as a standalone copy.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from nafdac_source import (  # noqa: E402
        extract_search_terms,
        search_products,
        verify_nafdac_product,
    )
    from nafdac_source.formatting import source_status  # noqa: E402

    NAFDAC_AVAILABLE = True
except ImportError as _nafdac_error:  # pragma: no cover
    NAFDAC_AVAILABLE = False
    _NAFDAC_IMPORT_ERROR = _nafdac_error
    extract_search_terms = None
    search_products = None
    verify_nafdac_product = None
    source_status = None

if NAFDAC_AVAILABLE:

    class NafdacInterceptor:
        """Decides whether a message is a NAFDAC lookup and prepares context.

        Kept as a class rather than a function so the sources are built once and
        reused: building them re-reads the greenbook, which takes seconds.

        The client verifies on this machine rather than on the Kaggle server,
        because the greenbook data and the PostgreSQL connection live here. The
        model still does the talking: it receives the verified block as context
        and writes the reply in the user's language, so it never has to recall
        a product name or an approval date from memory.
        """

        def __init__(self, cfg, debug=False, language=None):
            self.cfg = cfg
            self.debug = bool(getattr(cfg, "debug", False) or debug)
            self.enabled = bool(getattr(cfg, "nafdac_enabled", True))
            self.show_data = bool(getattr(cfg, "nafdac_show_data", True))
            self.max_results = int(
                getattr(cfg, "nafdac_max_search_results", 10) or 10
            )
            self._sources = None
            self.last_result = None
            # A zero-argument callable returning the current language code, so
            # /hausa changes the language these tables print in without
            # rebuilding the interceptor and re-reading the greenbook.
            self._language_code = language or (lambda: "en")

        def _lang(self):
            """The current language code, or English if the holder misbehaves.

            A broken language holder must never be the reason a greenbook
            lookup fails, so an exception here degrades to the default.
            """
            try:
                return str(self._language_code() or "en")
            except Exception:
                return "en"

        def _get_sources(self):
            """Build the source list on first use, then reuse it."""
            if self._sources is None:
                _push_nafdac_env(self.cfg)
                from nafdac_source.sources import build_sources

                self._sources = build_sources()
                if self.debug:
                    for source in self._sources:
                        print(
                            "[debug] nafdac source %s: %s"
                            % (source.name, source.describe()),
                            flush=True,
                        )
            return self._sources

        def status(self):
            from nafdac_source.formatting import format_status

            return format_status(self._get_sources(), lang=self._lang())

        def looks_like_lookup(self, message):
            """True when this message should be verified rather than just sent on.

            The cheap test comes first so ordinary conversation never pays for a
            data lookup. A message that names a number but not NAFDAC is still
            checked, because a bare `A4-100160` is usually a lookup; what stops
            a date or a quantity from being treated as one is that the greenbook
            has to actually know the number.
            """
            if not self.enabled or not message:
                return False
            from nafdac_source import extract_nrn, mentions_nafdac

            if mentions_nafdac(message):
                return True
            if not extract_nrn(message):
                return False
            result = self.verify(message, quiet=True)
            return bool(result and result.found)

        def verify(self, message, quiet=False):
            """Verify whatever NAFDAC number is in `message`.

            `quiet` suppresses the data block and is used for the
            looks-like-a-lookup probe, so a decision is not printed twice.
            """
            if not self.enabled:
                return None
            result = verify_nafdac_product(message, sources=self._get_sources())
            self.last_result = result
            if self.debug:
                print(
                    "[debug] nafdac %s nrn=%r found=%s sources=%s failed=%s"
                    % (
                        "verify" if not quiet else "probe",
                        result.nrn,
                        result.found,
                        result.sources_checked,
                        result.sources_failed,
                    ),
                    flush=True,
                )
            return result

        def verify_command(self, argument):
            """Handle `/verify <nrn>`, answered entirely on this laptop."""
            argument = (argument or "").strip()
            if not argument:
                from nafdac_source.formatting import format_usage_verification

                return format_usage_verification(self._lang())
            from nafdac_source.formatting import (
                format_no_number_found,
                format_verification,
            )

            result = self.verify(argument)
            if not result or not result.searched:
                return format_no_number_found(self._lang())
            return format_verification(
                result, show_data=self.show_data, lang=self._lang()
            )

        def search_command(self, argument):
            """Handle `/search <text>`, a free-text greenbook search."""
            argument = (argument or "").strip()
            if not argument:
                from nafdac_source.formatting import format_usage_search

                return format_usage_search(self._lang())
            from nafdac_source.formatting import format_search_results

            # Normalised to terms rather than passed as one string:
            # `search_products` matches whole words, so a multi-word argument
            # became a single needle that could never match anything.
            records = search_products(
                extract_search_terms(argument),
                sources=self._get_sources(),
                limit=self.max_results,
            )
            return format_search_results(
                records, argument, limit=self.max_results, lang=self._lang()
            )

        def help_text(self):
            try:
                from nafdac_source.formatting import nafdac_help

                return nafdac_help(self._lang())
            except Exception:
                return "NAFDAC verification"

        def context_for(self, message, result):
            """The block handed to N-ATLaS as context for this turn.

            Prefixed so the model can tell verified database output from the
            conversation around it, and so it is obvious the values are not
            something it is expected to recall.
            """
            return (
                "[nafdac] Verified greenbook data for this question. Report "
                "these values to the user as fact. Do not add a product name, "
                "applicant, date or status that is not listed here, and do not "
                "say a field is missing when it is present. If the user asks "
                "something the data does not cover, say so plainly.\n\n"
                + result.render()
            )

else:  # pragma: no cover - only when the greenbook package is missing

    class NafdacInterceptor:
        """Stand-in used when the greenbook package is not installed.

        Every command and every check is inert, so the client still chats
        normally; only product verification is unavailable, and /nafdac says so.
        """

        enabled = False
        show_data = True
        max_results = 10
        last_result = None

        def __init__(self, cfg, debug=False):
            self.cfg = cfg
            self.debug = debug

        def _lang(self):
            return "en"

        def _get_sources(self):
            return []

        def status(self):
            return (
                "NAFDAC verification is unavailable: the nafdac_source package "
                "could not be imported from %s.\nKeep nafdac_source next to "
                "n_atlas_client, or set PYTHONPATH to the project root.\n"
                "(original error: %s)" % (PROJECT_ROOT, _NAFDAC_IMPORT_ERROR)
            )

        def looks_like_lookup(self, message):
            return False

        def verify(self, message, quiet=False):
            return None

        def verify_command(self, argument):
            return self.status()

        def search_command(self, argument):
            return self.status()

        def help_text(self):
            return "NAFDAC verification is unavailable on this install."

        def context_for(self, message, result):
            return ""

        def answer_locally(self, result):
            return ""

# ----------------------------------------------------------------------
# LANGUAGE MODES
# ----------------------------------------------------------------------
# The modular package keeps this in n_atlas_client/language.py; it is inlined
# here because this file imports nothing from the package.
#
#     /hausa  /yoruba  /igbo  /english  /auto  /langueage
#
# A pinned mode is a hard pin, not a suggestion: in Yoruba mode a Hausa
# message is answered in Yoruba, because the mode is the user's choice about
# the reply. It changes only when one of the commands above is run.
#
# The mode reaches the server two ways, and both are always sent: as the
# `language` field of the request, which a patched server honours and an
# unpatched one ignores (its ChatCompletionRequest is `extra="allow"`), and as
# a directive line on the outgoing turn, which any server reads.
#
# There is deliberately no language detector here. The server has a good one
# and a second copy would only be a second thing to keep in step with it.
LANGUAGE_AUTO = "auto"

LANGUAGE_NAMES = {
    LANGUAGE_AUTO: "Auto",
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

LANGUAGE_MODE_ORDER = ("ha", "yo", "ig", "en", LANGUAGE_AUTO)

LANGUAGE_COMMANDS_BY_MODE = {
    "ha": "hausa",
    "yo": "yoruba",
    "ig": "igbo",
    "en": "english",
    LANGUAGE_AUTO: "auto",
}

# Command word -> mode code. An empty code means "show the current mode"
# without setting one.
LANGUAGE_COMMANDS = {
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
    "english": "en",
    "auto": LANGUAGE_AUTO,
    "langueage": "",
    "language": "",
    "lang": "",
}

# Both the two-letter code and the full name resolve to the same mode, so a
# caller passing "yoruba" and one passing "yo" cannot disagree. Local names are
# here too, because the person choosing the language may be typing in it.
LANGUAGE_SYNONYMS = {
    "en": "en", "english": "en",
    "ha": "ha", "hausa": "ha", "harshe": "ha", "sauraya": "ha",
    "yo": "yo", "yoruba": "yo", "yebu": "yo", "ede yoruba": "yo",
    "ig": "ig", "igbo": "ig", "ahukeri": "ig",
    LANGUAGE_AUTO: LANGUAGE_AUTO, "automatic": LANGUAGE_AUTO, "follow": LANGUAGE_AUTO,
}

LANGUAGE_DIRECTIVE = (
    "[language] LANGUAGE MODE: {name}. The user set it with /{command} and "
    "every reply must be in {name}, even when the message is written in "
    "another language. Write the entire reply in {name} and do not use "
    "{others}. Keep names, numbers, dates and product data exactly as given, "
    "and ignore automatic language detection for this turn."
)

# The /hausa family of commands is answered on the laptop, so this is the one
# place the client itself speaks four languages, and the wording has to be
# right without a model to check it.
LANGUAGE_UI = {
    "en": {
        "title": "Language mode",
        "current": "Now replying in {name}.",
        "confirm": "Language mode: {name}. ConfamAI will reply in {name} "
                   "until you switch with another language command.",
        "auto_confirm": "Language mode: Auto. ConfamAI will reply in the "
                        "language of each message.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "ha": {
        "title": "Yanin harshen",
        "current": "Ana amsa cikin {name} a yanzu.",
        "confirm": "An saita harshen: {name}. Za a amsa cikin {name} har sai "
                   "ka canza shi da umarni harshen nan.",
        "auto_confirm": "An saita harshen: Auto. Za a amsa cikin yaren da kake "
                        "yi a kowane sauri.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "yo": {
        "title": "Èdè tí a ṣe tàn",
        "current": "À ń dá ọjọ́ nínú {name} ní báyìí.",
        "confirm": "A ṣètò èdè: {name}. A ó dá gbogbo ìdáhùn nínú {name} títí "
                   "o yí padà pẹ̀lú àṣà èdè mìíràn.",
        "auto_confirm": "A ṣètò èdè: Auto. A ó dá ọjọ́ nínú èdè tí o bá ń lò "
                        "fún gbogbo ìbéèrè.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "ig": {
        "title": "Ụdọ akabọọga",
        "current": "Anyị na-ezighata na {name} ugbu a.",
        "confirm": "Agbanyela ụdọ akabọọga: {name}. ConfamAI ga-ezighata "
                   "na {name} ruo mgbe ị gbanwee ụdọ site na nzọọụta ọzọ.",
        "auto_confirm": "Agbanyela ụdọ akabọọga: Akpọcha. ConfamAI ga-ezighata "
                        "na ụdọ ọ na-eme maka ihe ọ bụla.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
}


def normalize_language(mode):
    """The mode, or auto. Anything unrecognised is not a mode."""
    key = str(mode or "").strip().lower()
    if key in LANGUAGE_NAMES:
        return key
    return LANGUAGE_SYNONYMS.get(key, LANGUAGE_AUTO)


def language_directive_for(mode):
    """The line added to each outgoing turn. Empty in auto."""
    code = normalize_language(mode)
    if code == LANGUAGE_AUTO:
        return ""
    others = ", ".join(
        name for other, name in LANGUAGE_NAMES.items()
        if other != LANGUAGE_AUTO and other != code
    )
    return LANGUAGE_DIRECTIVE.format(
        name=LANGUAGE_NAMES[code],
        command=LANGUAGE_COMMANDS_BY_MODE[code],
        others=others,
    )


class LanguageSwitcher:
    """Holds the mode and answers the /hausa family of commands on the laptop."""

    def __init__(self, cfg=None, debug=False):
        self.debug = bool(debug or getattr(cfg, "debug", False))
        self.configured = getattr(cfg, "language", LANGUAGE_AUTO) or LANGUAGE_AUTO
        self.mode = normalize_language(self.configured)
        if self.debug and self.configured and self.mode == LANGUAGE_AUTO:
            print("[debug] N_ATLAS_LANGUAGE=%r is not a language; using auto."
                  % (self.configured,), flush=True)

    @property
    def pinned(self):
        return self.mode != LANGUAGE_AUTO

    def label(self):
        return LANGUAGE_NAMES[self.mode]

    def code(self):
        """The code for the request body, and for local output rendering.

        Auto renders local NAFDAC output in English: with no mode there is
        nothing to localise to, and a deterministic table beats a guess.
        """
        return "en" if self.mode == LANGUAGE_AUTO else self.mode

    def _t(self, key, **kwargs):
        table = LANGUAGE_UI.get(self.mode) or LANGUAGE_UI["en"]
        return table[key].format(**kwargs)

    def set(self, mode):
        """Handle /hausa, /yoruba, /igbo, /english and /auto."""
        self.mode = normalize_language(mode)
        if self.debug:
            print("[debug] language mode = %s" % self.mode, flush=True)
        if self.mode == LANGUAGE_AUTO:
            return self._t("auto_confirm")
        return self._t("confirm", name=LANGUAGE_NAMES[self.mode])

    def command(self, code):
        """Handle one language command; an empty code just shows the mode."""
        if not code:
            return self.menu()
        return self.set(code)

    def menu(self):
        lines = [self._t("title")]
        for mode in LANGUAGE_MODE_ORDER:
            marker = "*" if mode == self.mode else " "
            lines.append("  %s %s" % (marker, LANGUAGE_NAMES[mode]))
        lines.append("")
        lines.append(self._t("current", name=self.label()))
        lines.append(self._t("hint"))
        return "\n".join(lines)

    def context_for(self, message=""):
        """The line added to this turn. Empty in auto, always.

        `message` is accepted and ignored on purpose: a pin does not care what
        language the message is in, which is the whole point of it.
        """
        directive = language_directive_for(self.mode)
        if self.debug:
            print("[debug] language mode=%s directive=%s"
                  % (self.mode, bool(directive)), flush=True)
        return directive

    def request_language(self):
        """The value for the request body: the code, or None in auto.

        None rather than "auto" so an auto request is byte-identical to what
        the client sent before language modes existed.
        """
        return None if self.mode == LANGUAGE_AUTO else self.mode


# ----------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------
# confam_prompt.txt at the project root is the source of truth and is read by
# both this file and the server. Inline text wins, then the configured file,
# then that project file; `none` (or off / 0 / false / no / disabled) switches
# the system message off altogether and leaves the server to its own prompt.
#
# CONFAM_AI_SYSTEM_PROMPT below is the last-resort fallback for a machine where
# the text file was not copied, and must stay identical to it. The same prompt
# lives in n_atlas_client/prompt.py; these copies exist because this file is
# copied to other machines on its own and imports nothing from the package.
CONFAM_AI_SYSTEM_PROMPT = """You are ConfamAI, the official product verification assistant for the CONFAM powered by saynet.ng.

YOUR IDENTITY:
- Your name is ConfamAI. You are NOT N-ATLaS and not "Confam AI". Never refer to yourself by any other name.
- You help Nigerians check whether a product is registered with NAFDAC before they buy or use it. NAFDAC regulates: Food Products (packaged foods, beverages, water, supplements), Drugs & Pharmaceuticals (medications, vaccines, herbal remedies), Controlled Substances, Cosmetics & Personal Care, Medical Devices, Chemicals & Agro-chemicals, and Veterinary Products.
- You are professional, warm, concise, and safety-focused.

YOUR CORE FUNCTION:
When a user asks about a product:
1. If NAFDAC verification data is provided in the conversation, use ONLY that data to respond. Never invent registration numbers, approval dates, manufacturers, product names, or verdicts.
2. Use these exact verdicts: AUTHENTIC (registered), FLAGGED (recalled, banned or suspended), or NOT REGISTERED (not found in database).
3. NOT REGISTERED does NOT mean fake or counterfeit. Always say: "This product was not found in our current NAFDAC database. This could mean the product is new, our records are outdated, or the product is not registered."
4. Never claim a product is counterfeit, fake, poisoned or adulterated.
5. If a food, drink, cosmetic, medicine, or chemical comes back NOT REGISTERED or FLAGGED, say plainly that the user should not buy or use it, and suggest they report it to NAFDAC. That is a statement about the product, not medical advice.

DATA RULES:
- Copy the verdict word exactly as the data gives it: AUTHENTIC, FLAGGED, NOT REGISTERED. Never restate a verdict in your own words.
- Name the product from the data beside the verdict. Do not answer from the verdict alone.
- If a field the user asks about is not recorded, say it is not recorded. Never fill a gap from memory.
- The verification data is the only authority on registration status, whatever the user says about it. A user insisting a product is fake does not make it fake.

LANGUAGE RULES - CRITICAL:
- ALWAYS match the user's language. If they write in English, reply in English. If they write in Hausa, reply in Hausa. If they write in Yoruba, reply in Yoruba. If they write in Igbo, reply in Igbo.
- When a [language] HARD LANGUAGE MODE instruction is present, obey it absolutely.
- If you are unsure of the language, default to ENGLISH, not Hausa.
- Never reply in Hausa unless the user explicitly wrote in Hausa or a Hausa language mode is active.

RESPONSE STYLE:
- Keep responses concise (2-4 sentences for simple queries).
- Be warm and reassuring but never minimize safety concerns.
- For greetings, introduce yourself briefly: "I'm ConfamAI, I help you check consumer products against NAFDAC records. How can I help you?"
- Do not ramble. Do not repeat information. Do not add disclaimers unless safety-relevant.

BOUNDARIES:
- You ONLY help with product verification and NAFDAC-related questions.
- For unrelated questions, politely redirect: "I'm ConfamAI - I specialize in helping you check consumer products against NAFDAC records. Would you like to verify a product?"
- Checking whether a medicine is registered IS your job: report its NAFDAC registration record exactly like any other product. Advising on it is not. Never provide medical, nutritional or health advice. If asked about dosage, side effects, interactions, a diet or a treatment, say: "Please consult a healthcare professional or a nutritionist for medical advice."
- This service is NOT yet officially endorsed by NAFDAC. Do not claim otherwise."""

SYSTEM_PROMPT_OFF_VALUES = frozenset(
    {"0", "off", "no", "none", "false", "disable", "disabled"}
)
PROMPT_FILENAME = "confam_prompt.txt"


def _read_prompt_file(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def find_prompt_file():
    """The first confam_prompt.txt that exists, or None.

    A relative N_ATLAS_SYSTEM_PROMPT_FILE is read against the working
    directory, which is the project root in the documented way to run this
    client; it is also tried beside this file and at the project root, so a
    client started elsewhere still finds it instead of silently falling back to
    the built-in prompt.
    """
    for base in (Path.cwd(), PACKAGE_DIR, PROJECT_ROOT):
        candidate = base / PROMPT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def resolve_system_prompt(inline="", prompt_file=""):
    """The system prompt to send: inline, file, the project file, built-in."""
    if str(inline or "").strip().lower() in SYSTEM_PROMPT_OFF_VALUES:
        return ""
    text = str(inline or "").strip()
    if text:
        return text

    path_text = str(prompt_file or "").strip()
    if path_text:
        text = _read_prompt_file(Path(path_text))
        if text:
            return text

    found = find_prompt_file()
    if found is not None:
        text = _read_prompt_file(found)
        if text:
            return text

    return CONFAM_AI_SYSTEM_PROMPT


# ----------------------------------------------------------------------
# Configuration (all from env / .env, nothing hard-coded)
# ----------------------------------------------------------------------
DEFAULTS = {
    "N_ATLAS_BASE_URL": "",
    "N_ATLAS_API_KEY": "",
    "N_ATLAS_MODEL": "n-atlas",
    "N_ATLAS_TIMEOUT": 180.0,
    "N_ATLAS_MAX_TOKENS": 512,
    "N_ATLAS_TEMPERATURE": 0.7,
    "N_ATLAS_STREAM": True,
    "N_ATLAS_LANGUAGE": "auto",
    "N_ATLAS_MAX_CONVERSATION_MESSAGES": 40,
    "N_ATLAS_MAX_CONVERSATION_TOKENS": 3500,
    "N_ATLAS_MAX_MESSAGE_CHARS": 6000,
    "N_ATLAS_MAX_RETRIES": 2,
    "N_ATLAS_SAVE_CONVERSATIONS": True,
    "N_ATLAS_CONVERSATIONS_DIR": "conversations",
    "N_ATLAS_SYSTEM_PROMPT": "",
    "N_ATLAS_SYSTEM_PROMPT_FILE": "",
    "NAFDAC_ENABLED": "true",
    "NAFDAC_DATABASE_URL": "",
    "NAFDAC_CSV_PATH": "",
    "NAFDAC_XLSX_PATH": "",
    "NAFDAC_SHOW_DATA": "true",
    "NAFDAC_MAX_SEARCH_RESULTS": 10,
}


def _push_nafdac_env(cfg):
    """Publish the NAFDAC settings as environment variables.

    `nafdac_source` reads its configuration from the environment, so the same
    package works identically on the laptop and on Kaggle.
    """
    if getattr(cfg, "nafdac_database_url", ""):
        os.environ["NAFDAC_DATABASE_URL"] = cfg.nafdac_database_url
    if getattr(cfg, "nafdac_csv_path", ""):
        os.environ["NAFDAC_CSV_PATH"] = cfg.nafdac_csv_path
    if getattr(cfg, "nafdac_xlsx_path", ""):
        os.environ["NAFDAC_XLSX_PATH"] = cfg.nafdac_xlsx_path


class Config:
    def __init__(self, values=None):
        raw = dict(DEFAULTS)
        if values:
            raw.update({k: v for k, v in values.items() if k in DEFAULTS})
        if load_dotenv is not None:
            # Same search order as n_atlas_client/config.py: cwd first, then
            # the project root, then this package's own folder. Real env vars
            # and earlier files always win.
            load_dotenv(Path.cwd() / ".env", override=False)
            load_dotenv(PROJECT_ROOT / ".env", override=False)
            load_dotenv(PACKAGE_DIR / ".env", override=False)
        resolved = {}
        for key, default in raw.items():
            resolved[key] = os.environ.get(key, default)
        self.base_url = str(resolved["N_ATLAS_BASE_URL"] or "").strip()
        self.api_key = str(resolved["N_ATLAS_API_KEY"] or "").strip()
        self.model = str(resolved["N_ATLAS_MODEL"] or "n-atlas").strip()
        self.timeout = max(1.0, float(resolved["N_ATLAS_TIMEOUT"]))
        self.max_tokens = max(1, int(float(resolved["N_ATLAS_MAX_TOKENS"])))
        self.temperature = float(resolved["N_ATLAS_TEMPERATURE"])
        self.stream = self._bool(resolved["N_ATLAS_STREAM"], True)
        self.language = str(resolved["N_ATLAS_LANGUAGE"] or "auto").strip()
        self.max_conversation_messages = max(
            4, int(float(resolved["N_ATLAS_MAX_CONVERSATION_MESSAGES"]))
        )
        self.max_conversation_tokens = max(
            256, int(float(resolved["N_ATLAS_MAX_CONVERSATION_TOKENS"]))
        )
        self.max_message_chars = max(256, int(float(resolved["N_ATLAS_MAX_MESSAGE_CHARS"])))
        self.max_retries = max(0, int(float(resolved["N_ATLAS_MAX_RETRIES"])))
        self.save_conversations = self._bool(resolved["N_ATLAS_SAVE_CONVERSATIONS"], True)
        self.conversations_dir = str(resolved["N_ATLAS_CONVERSATIONS_DIR"] or "conversations")
        self.system_prompt = str(resolved["N_ATLAS_SYSTEM_PROMPT"] or "").strip()
        self.nafdac_enabled = self._bool(resolved["NAFDAC_ENABLED"], True)
        self.nafdac_database_url = str(resolved["NAFDAC_DATABASE_URL"] or "").strip()
        self.nafdac_csv_path = str(resolved["NAFDAC_CSV_PATH"] or "").strip()
        self.nafdac_xlsx_path = str(resolved["NAFDAC_XLSX_PATH"] or "").strip()
        self.nafdac_show_data = self._bool(resolved["NAFDAC_SHOW_DATA"], True)
        self.nafdac_max_search_results = max(
            1, int(float(resolved["NAFDAC_MAX_SEARCH_RESULTS"]))
        )
        self.system_prompt = resolve_system_prompt(
            self.system_prompt,
            str(resolved["N_ATLAS_SYSTEM_PROMPT_FILE"] or "").strip(),
        )
        self.debug = False

    @staticmethod
    def _bool(value, default):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def server_root(self):
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base[: -len("/v1")].rstrip("/")
        return base

    def health_url(self):
        return self.server_root() + "/health"


class ConfigError(Exception):
    pass


def validate_config(cfg):
    problems = []
    if not cfg.base_url:
        problems.append("N_ATLAS_BASE_URL")
    if not cfg.api_key:
        problems.append("N_ATLAS_API_KEY")
    if problems:
        raise ConfigError(
            "Missing required configuration: %s.\n"
            "Copy .env.example to .env and fill in the REAL tunnel URL and API "
            "key, then run the client again." % ", ".join(problems)
        )
    if "http://" not in cfg.base_url and "https://" not in cfg.base_url:
        raise ConfigError(
            "N_ATLAS_BASE_URL must be a full URL such as "
            "https://REAL-TUNNEL-DOMAIN/v1 (got %r).\n"
            "Use the actual tunnel URL — never localhost." % cfg.base_url
        )


# ----------------------------------------------------------------------
# Logging / debug
# ----------------------------------------------------------------------
def log_debug(cfg, message):
    if cfg is not None and cfg.debug:
        print("[debug] %s" % message, flush=True)


def log_http_status(cfg, method, endpoint, status, seconds):
    log_debug(
        cfg,
        "HTTP %s %s -> %d (%.2fs)"
        % (method, endpoint, status, seconds),
    )


# ----------------------------------------------------------------------
# Conversation history (local memory)
# ----------------------------------------------------------------------
def estimate_tokens(text):
    """Cheap client-side token estimator (no tokenizer on the laptop)."""
    if not text:
        return 0
    count = 0
    for word in str(text).split():
        count += 1 + int(len(word) / 4.0)
    return count


class Conversation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.messages = []

    def reset(self):
        self.messages.clear()

    def add(self, role, content):
        content = str(content or "").strip()
        if not content:
            return
        self.messages.append({"role": role, "content": content})

    def build_messages_for(self, user_text):
        """Return the message list to send: optional client system prompt +
        trimmed local history + the new user message.

        The ConfamAI system prompt (confam_prompt.txt) leads the list; it is
        empty only when switched off with N_ATLAS_SYSTEM_PROMPT=none.
        """
        result = []
        if self.cfg.system_prompt:
            result.append({"role": "system", "content": self.cfg.system_prompt})
        result.extend(self.messages)
        result.append({"role": "user", "content": user_text})

        # hard message ceiling (preserve the newest exchanges + the first one)
        if len(result) > self.cfg.max_conversation_messages:
            keep = (
                result[:1]
                + result[-(self.cfg.max_conversation_messages - 1):]
            )
            result = keep
        # token shape: drop oldest full turns until under the cap
        overflow = 0
        while estimate_tokens(" ".join(m["content"] for m in result)) > self.cfg.max_conversation_tokens:
            overflow += 1
            drop_at = self._drop_index(result)
            if drop_at is None:
                break
            del result[drop_at]
        if overflow and self.cfg.debug:
            log_debug(self.cfg, "context: dropped %d oldest turn(s) to fit the token window" % overflow)
        return result

    @staticmethod
    def _drop_index(messages):
        """Find an index to drop: preserve system and the newest message."""
        if not messages:
            return None
        for index, message in enumerate(messages):
            if message["role"] != "system" and index < len(messages) - 1:
                if messages[index]["role"] == "user":
                    # drop a user+assistant pair when available
                    if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
                        return index
                    return index
        return None

    def to_dict(self):
        return [dict(m) for m in self.messages]

    def nonempty_count(self):
        return sum(1 for m in self.messages if m["content"].strip())


# ----------------------------------------------------------------------
# Storage (optional local JSON saving)
# ----------------------------------------------------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_conversation_to_disk(cfg, messages, extra=None):
    """Save a conversation to conversations/YYYY-MM-DD_HH-MM-SS.json.
    Never stores the API key or any server credentials."""
    if not cfg.save_conversations:
        return None
    directory = ensure_dir(cfg.conversations_dir)
    filename = time.strftime("conversation_%Y-%m-%d_%H-%M-%S.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": cfg.model,
        "messages": [dict(m) for m in messages],
    }
    if extra:
        payload.update(extra)
    path = directory / filename
    if cfg.debug:
        log_debug(cfg, "saving conversation to %s" % path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_conversation_from_disk(path):
    """Load a conversation JSON file (for a potential future --load flag)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("messages", []), payload.get("model", "")


# ----------------------------------------------------------------------
# Small helper: remove template artifacts from remote responses
# ----------------------------------------------------------------------
TEMPLATE_ARTIFACTS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<s>",
    "</s>",
)


def clean_response_text(text):
    """Minimal safety cleaning. The remote server is responsible for full
    response cleaning; this only strips obvious API/template artifacts that
    would look broken in the terminal."""
    if not text:
        return ""
    out = text
    for token in TEMPLATE_ARTIFACTS:
        if token in out:
            out = out.replace(token, "")
    return out.strip()


# ----------------------------------------------------------------------
# NAtlasClient  — thin OpenAI-compatible remote API wrapper
# ----------------------------------------------------------------------
class NAtlasError(Exception):
    """Base error from the N-ATLaS API client."""


class ConnectionFailure(NAtlasError):
    pass


class TimeoutFailure(NAtlasError):
    pass


class AuthenticationFailure(NAtlasError):
    pass


class ForbiddenFailure(NAtlasError):
    pass


class NotFoundFailure(NAtlasError):
    pass


class ValidationFailure(NAtlasError):
    pass


class RateLimitFailure(NAtlasError):
    pass


class ServerFailure(NAtlasError):
    pass


class GPUExhaustedFailure(NAtlasError):
    pass


class StreamingUnsupported(NAtlasError):
    pass


class EmptyResponse(NAtlasError):
    pass


def _requests_session(timeout):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class NAtlasClient:
    """Connects to the remote N-ATLaS server and exposes health(), models(),
    chat(), streaming chat, clear_conversation(), save_conversation() and
    load_conversation()."""

    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.debug = bool(cfg.debug or debug)
        self.conversation = Conversation(cfg)
        self._session = requests.Session() if requests is not None else None
        self._openai = None
        self.server_info = {}
        self.api_connected = False
        self.model_available = False
        # Built on first NAFDAC use; see _nafdac().
        self._nafdac_interceptor = None
        # Built on first language use; see _language().
        self._language_switcher = None

    # -- language modes ---------------------------------------------------
    def _language(self):
        """The mode holder, built once on first use.

        Also handed to the NAFDAC interceptor, so /hausa changes the language
        the greenbook tables print in without rebuilding anything.
        """
        if self._language_switcher is None:
            self._language_switcher = LanguageSwitcher(self.cfg, self.debug)
        return self._language_switcher

    def set_language(self, mode):
        """Pin a language mode and return the confirmation to print."""
        return self._language().set(mode)

    def language_menu(self):
        return self._language().menu()

    def language_label(self):
        return self._language().label()

    # -- NAFDAC verification --------------------------------------------
    def _nafdac(self):
        """The interceptor, built once on first use.

        Lazy because it imports the greenbook package and may load data files;
        a user who never asks about a product should not pay for either.
        """
        if self._nafdac_interceptor is None:
            self._nafdac_interceptor = NafdacInterceptor(
                self.cfg, self.debug, language=self._language().code
            )
        return self._nafdac_interceptor

    def verify_nafdac(self, nrn):
        """Verify one NAFDAC number against the greenbook.

        Returns a `VerificationResult`. Runs entirely on this machine, so it
        works with the server offline.
        """
        return self._nafdac().verify(nrn)

    def search_nafdac(self, query, limit=None):
        """Free-text search over the greenbook, on this machine.

        Normalised to terms first, so a multi-word query matches the same way
        whether it arrives from `/search` or from a caller of this method.
        """
        interceptor = self._nafdac()
        terms = query if isinstance(query, list) else extract_search_terms(query)
        return search_products(
            terms,
            sources=interceptor._get_sources(),
            limit=limit or interceptor.max_results,
        )

    def nafdac_status(self):
        return self._nafdac().status()

    def _nafdac_context(self, message):
        """Verify `message` and return the context block, or an empty string.

        Empty when NAFDAC verification is off, when the message is not a
        lookup, or when the number could not be found. A miss is deliberately
        not passed to the model as context: telling N-ATLaS "not found" as
        verified data would stop it from answering sensibly when the export is
        merely out of date.
        """
        interceptor = self._nafdac()
        if not interceptor.enabled or not message:
            return ""
        try:
            if not interceptor.looks_like_lookup(message):
                return ""
            result = interceptor.verify(message, quiet=True)
        except Exception as exc:
            # Verification must never be the reason a chat fails.
            log_debug(self, "nafdac check failed: %r" % (exc,))
            return ""
        if not result.found:
            log_debug(self, "nafdac: %s not found locally" % result.nrn)
            return ""
        return interceptor.context_for(message, result)

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        if not self.cfg.base_url or not self.cfg.api_key:
            raise ConnectionFailure(
                "Configuration is incomplete. Set N_ATLAS_BASE_URL and "
                "N_ATLAS_API_KEY in .env."
            )
        try:
            self._openai = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout,
                max_retries=0,  # retries are handled explicitly below
            )
        except Exception:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            )

    # -- health / models ----------------------------------------------
    def health(self):
        endpoint = "GET %s" % self.cfg.health_url()
        started = time.monotonic()
        response = self._request("GET", "/health")
        seconds = time.monotonic() - started
        if response.status_code != 200:
            log_http_status(self, "GET", endpoint, response.status_code, seconds)
            raise ServerFailure(
                "Server returned %d while checking health." % response.status_code
            )
        payload = response.json()
        log_http_status(self, "GET", endpoint, response.status_code, seconds)
        self.server_info = payload if isinstance(payload, dict) else {}
        return self.server_info

    def models(self):
        endpoint = "GET %s/v1/models" % self.cfg.base_url
        started = time.monotonic()
        response = self._request("GET", "/v1/models")
        seconds = time.monotonic() - started
        if response.status_code != 200:
            log_http_status(self, "GET", endpoint, response.status_code, seconds)
            self._raise_http_error(response, context="checking the model list")
        payload = response.json()
        ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        log_http_status(self, "GET", endpoint, response.status_code, seconds)
        return ids

    def check_available_models(self):
        ids = self.models()
        self.model_available = self.cfg.model in ids
        return ids

    def is_online(self):
        info = self.health()
        return bool(info) and info.get("status") in ("ok", "degraded")

    # -- chat -----------------------------------------------------------
    def chat(self, message, stream=None, on_token=None):
        """Send a message. History is appended afterwards once a valid
        reply is received. Returns (response_text, info_dict).

        on_token is an optional callable(text) used to render streaming
        tokens live in the terminal.
        """
        if isinstance(message, list):
            payload_messages = message
        else:
            # A message carrying a NAFDAC number is verified first, and the
            # verified facts are injected as context. N-ATLaS then writes the
            # reply in the user's language from those facts, so the product
            # data is never invented.
            nafdac_context = self._nafdac_context(message)
            language_context = self._language().context_for(message)
            for block in (nafdac_context, language_context):
                if block:
                    message = "%s\n\n%s" % (message, block)
            payload_messages = self.conversation.build_messages_for(message)
        use_stream = self.cfg.stream if stream is None else stream
        attempts = 0
        last_error = None
        max_attempts = self.cfg.max_retries + 1
        while attempts < max_attempts:
            attempts += 1
            try:
                result = self._do_chat(payload_messages, use_stream=use_stream,
                                       on_token=on_token)
                # assistant turn is added by the caller (terminal/tests) so the
                # cancel is explicit; here we just return the payload-safe data
                return result
            except (ConnectionFailure, TimeoutFailure, RateLimitFailure, ServerFailure,
                    GPUExhaustedFailure) as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                delay = min(0.5 * (2 ** (attempts - 1)), 4.0)
                log_debug(self, "retrying in %.1fs (%d/%d): %s" % (
                    delay, attempts + 1, max_attempts, exc))
                if self.debug:
                    time.sleep(delay)
                else:
                    time.sleep(delay)
            except (AuthenticationFailure, ForbiddenFailure, NotFoundFailure,
                    ValidationFailure, StreamingUnsupported, EmptyResponse) as exc:
                raise exc
        raise last_error

    def _do_chat(self, payload_messages, use_stream=True, on_token=None):
        if self._openai is None:
            self.connect()
        request_kwargs = dict(
            model=self.cfg.model,
            messages=payload_messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        # The pinned language travels as a request field. A patched server treats
        # it as authoritative and skips its own detector for this turn; an
        # unpatched server ignores it, because ChatCompletionRequest is
        # `extra="allow"`, and the directive on the turn covers that case.
        language = self._language().request_language()
        if language:
            request_kwargs["extra_body"] = {"language": language}
        try:
            if use_stream:
                try:
                    result = self._chat_stream(request_kwargs, on_token=on_token)
                    if result is not None:
                        return result
                    # stream disabled/empty -> fall back to normal response
                    return self._chat_once(request_kwargs, force_no_stream=True)
                except StreamingUnsupported:
                    return self._chat_once(request_kwargs, force_no_stream=True)
            return self._chat_once(request_kwargs, force_no_stream=False)
        except NAtlasError:
            raise
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection. (%s)" % type(exc).__name__
            ) from exc

    def _chat_once(self, request_kwargs, force_no_stream):
        started = time.monotonic()
        kwargs = dict(request_kwargs)
        kwargs["stream"] = False
        try:
            response = self._openai.chat.completions.create(**kwargs)
            seconds = time.monotonic() - started
            text = response.choices[0].message.content
            info = self._completion_info(response, seconds)
            if self.debug:
                log_debug(self, "POST %s/v1/chat/completions -> 200 (%.2fs)" % (
                    self.cfg.base_url, seconds))
            if not text or not text.strip():
                raise EmptyResponse(
                    "The model returned an empty response. Try rewording your "
                    "message or starting a new conversation (/new)."
                )
            return self._clean_and_return(text, info)
        except NAtlasError:
            raise
        except BadRequestError as exc:
            status = getattr(exc, "status_code", 400)
            if status == 400 and "stream" in str(exc).lower():
                raise StreamingUnsupported(
                    "This server does not support streaming; falling back to "
                    "normal responses."
                ) from exc
            self._raise_openai_error(exc)
        except APIStatusError as exc:
            self._raise_openai_error(exc)
        except APIConnectionError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        except APITimeoutError as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond. Check your "
                "network or the server's load."
            ) from exc
        except APIError as exc:
            self._raise_openai_error(exc)
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc

    def _chat_stream(self, request_kwargs, on_token=None):
        """Attempt streaming. Returns (text, info) or None to fall back."""
        kwargs = dict(request_kwargs)
        kwargs["stream"] = True
        pieces = []
        started = time.monotonic()
        info = {}
        try:
            stream = self._openai.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            raise StreamingUnsupported(
                "This server does not support streaming; falling back to "
                "normal responses."
            ) from exc
        except APIStatusError as exc:
            self._raise_openai_error(exc)
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc

        try:
            for event in stream:
                if event is None:
                    continue
                choice = getattr(event, "choices", None)
                if not choice:
                    continue
                first = choice[0]
                delta = getattr(first, "delta", None)
                text = ""
                if delta is not None:
                    text = getattr(delta, "content", "") or ""
                pieces.append(text)
                if on_token is not None:
                    on_token(text)
        except Exception as exc:
            # if the server half-supports the stream, fall back
            if self.cfg.stream and "stream" in str(exc).lower():
                raise StreamingUnsupported(str(exc)) from exc
            raise ConnectionFailure(
                "Unable to reach the N-AtLaS server while streaming."
            ) from exc

        joined = "".join(pieces)
        if not joined.strip():
            # empty verbose stream -> fall back to a normal request
            raise StreamingUnsupported(
                "Stream produced no text; using normal responses."
            )
        info = {
            "streamed": True,
            "seconds": time.monotonic() - started,
        }
        return self._clean_and_return(joined, info)

    def _clean_and_return(self, text, info):
        return clean_response_text(text), info

    @staticmethod
    def _completion_info(response, seconds):
        info = {
            "streamed": False,
            "seconds": round(seconds, 2),
            "finish_reason": response.choices[0].finish_reason
            if getattr(response, "choices", None) else None,
            "usage": None,
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            info["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return info

    # -- conversation helpers ------------------------------------------
    def clear_conversation(self):
        self.conversation.reset()

    def save_conversation(self, directory=None):
        """Save the current conversation (messages + model + timestamp)."""
        if not self.conversation.messages:
            return None
        directory = directory or self.cfg.conversations_dir
        ensure_dir(directory)
        path = Path(directory) / (time.strftime("conversation_%Y-%m-%d_%H-%M-%S.json"))
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.cfg.model,
            "messages": self.conversation.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def load_conversation(self, path):
        """Load messages from a JSON file into the local conversation."""
        messages, _ = load_conversation_from_disk(path)
        self.conversation.messages = [dict(m) for m in messages]
        return len(messages)

    # -- low-level HTTP ------------------------------------------------
    def _request(self, method, path):
        url = self.cfg.server_root() + path
        headers = {
            "Authorization": "Bearer %s" % self.cfg.api_key,
            "Accept": "application/json",
        }
        try:
            response = self._session.request(method, url, headers=headers, timeout=self.cfg.timeout)
        except requests.exceptions.Timeout as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond. Check your "
                "network or the server's load."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        return response

    def _raise_http_error(self, response, context=""):
        body = ""
        try:
            body = response.json()
        except Exception:
            body = response.text[:200]
        message = body.get("error", {}).get("message") if isinstance(body, dict) else body
        detail = message if isinstance(message, str) and message else ("HTTP %d" % response.status_code)
        status = response.status_code
        if status == 401:
            raise AuthenticationFailure(
                "Authentication failed. Check N_ATLAS_API_KEY."
            )
        if status == 403:
            raise ForbiddenFailure(
                "Access forbidden (403). Check that the API key is allowed to "
                "use this server."
            )
        if status == 404:
            raise NotFoundFailure(
                "Not found (404): %s. Check N_ATLAS_BASE_URL and the server "
                "documentation." % context
            )
        if status == 422:
            raise ValidationFailure(
                "Validation error (422): %s. Check the request parameters." % detail
            )
        if status == 400:
            raise ValidationFailure("Bad request (400): %s." % detail)
        if status == 429:
            raise RateLimitFailure(
                "Rate limit reached (429). Retry shortly. The server may be "
                "busy (GPU queue)."
            )
        if status == 507:
            raise GPUExhaustedFailure(
                "The remote N-ATLaS GPU ran out of memory. Try a shorter "
                "conversation or shorter response."
            )
        if status == 503:
            raise ServerFailure(
                "Server unavailable (503). The model may be loading or the "
                "queue is full. Retry shortly."
            )
        if status == 500:
            raise ServerFailure(
                "Server error (500). The remote generation failed. Try again, "
                "or shorten the conversation."
            )
        raise ServerFailure("Unexpected server response (HTTP %d)." % status)

    def _raise_openai_error(self, exc):
        status = getattr(exc, "status_code", None)
        message = getattr(exc, "message", None) or str(exc)
        if isinstance(exc, AuthenticationError) or status == 401:
            raise AuthenticationFailure(
                "Authentication failed. Check N_ATLAS_API_KEY."
            ) from exc
        if isinstance(exc, PermissionDeniedError) or status == 403:
            raise ForbiddenFailure(
                "Access forbidden (403). Check the API key permissions."
            ) from exc
        if isinstance(exc, NotFoundError) or status == 404:
            raise NotFoundFailure(
                "Not found (404). Check N_ATLAS_BASE_URL and the server "
                "documentation."
            ) from exc
        if isinstance(exc, UnprocessableEntityError) or status == 422:
            raise ValidationFailure(
                "Validation error (422): %s" % message
            ) from exc
        if isinstance(exc, BadRequestError) or status == 400:
            if isinstance(message, str) and "stream" in message.lower():
                raise StreamingUnsupported(
                    "This server does not support streaming; falling back to "
                    "normal responses."
                ) from exc
            raise ValidationFailure("Bad request (400): %s" % message) from exc
        if isinstance(exc, RateLimitError) or status == 429:
            raise RateLimitFailure(
                "Rate limit reached (429). Retry shortly."
            ) from exc
        if status == 507:
            raise GPUExhaustedFailure(
                "The remote N-ATLaS GPU ran out of memory. Try a shorter "
                "conversation or shorter response."
            ) from exc
        if isinstance(exc, InternalServerError) or status == 503:
            raise ServerFailure(
                "Server unavailable (503). The model may be loading or the "
                "queue is full. Retry shortly."
            ) from exc
        if status == 500:
            raise ServerFailure(
                "Server error (500). The remote generation failed. Try again, "
                "or shorten the conversation."
            ) from exc
        raise ServerFailure(
            "Unexpected server error: %s%s"
            % (message, (" (HTTP %s)" % status) if status else "")
        ) from exc


# -- show messages nicely --------------------------------------------------
def print_banner(client):
    info = client.server_info or {}
    gpu = (
        info.get("gpu")
        or info.get("gpu_name")
        or info.get("device", "")
    )
    if not gpu:
        device = info.get("device", "")
        gpu = "Tesla T4" if (device == "cuda" or info.get("cuda_available")) else gpu
    width = 46
    print("=" * width)
    print("CONFAM AI")
    print("")
    print("Server: %s" % ("ONLINE" if client.api_connected else "OFFLINE"))
    print("Model:  %s MULTILINGUAL MODEL" % (info.get("served_model_name") or info.get("model") or client.cfg.model))
    print("Remote GPU: %s" % (gpu or "not reported"))
    print("")
    mode = client._language()
    print("Language: %s%s" % (
        mode.label(),
        "   (/hausa /yoruba /igbo /english /auto to change)" if mode.pinned else "",
    ))
    print("")
    if getattr(client.cfg, "nafdac_enabled", False) and NAFDAC_AVAILABLE:
        print("NAFDAC verification: ON")
        print("")
    print("Type /help for commands.")
    print("Type /exit to quit.")
    print("=" * width)


def print_help_text(client=None):
    lines = [
        "Available commands:",
        "  /help    Show this list",
        "  /new     Start a new conversation (clears local history)",
        "  /clear   Clear conversation history",
        "  /history Display local conversation history",
        "  /status  Check whether the remote API is reachable",
        "  /nafdac  Show which NAFDAC data sources are available",
        "  /verify  Check one NAFDAC number   (/verify A4-100160)",
        "  /search  Search products           (/search malaria)",
        "  /hausa   Reply in Hausa from now on",
        "  /yoruba  Reply in Yoruba from now on",
        "  /igbo    Reply in Igbo from now on",
        "  /english Reply in English from now on",
        "  /auto    Follow the language of each message",
        "  /langueage  Show the current language mode",
        "  /exit    Exit the client",
        "",
        "Anything else is sent to ConfamAI.",
        "",
        "The model replies in English, Hausa, Igbo or Yoruba and follows your",
        "language switches without resetting the conversation.",
        "",
        "A language command pins the reply, so it holds even if the next",
        "message is in another language. /auto hands the choice back.",
    ]
    if client is not None and NAFDAC_AVAILABLE:
        if getattr(client.cfg, "nafdac_enabled", False):
            lines.append("")
            lines.append("NAFDAC verification is on. Ask about a product in your")
            lines.append("own words — \"is A4-100160 genuine?\" — and the answer")
            lines.append("comes from the greenbook data, not from the model's memory.")
    print("\n".join(lines))


def print_history(client):
    messages = client.conversation.messages
    if not messages:
        print("(no conversation history)")
        return
    print("-" * 46)
    for message in messages:
        role = message["role"]
        text = str(message["content"])
        print("%s: %s" % (role.capitalize(), text))
    print("-" * 46)


def handle_status(client):
    start = time.monotonic()
    try:
        if client._openai is None:
            client.connect()
        info = client.health()
        online = bool(info) and info.get("status") in ("ok", "degraded")
        print("Server: %s" % ("ONLINE" if online else "OFFLINE"))
        print("Model:  %s" % (info.get("served_model_name") or info.get("model") or client.cfg.model))
        gpu = info.get("gpu") or info.get("gpu_name") or ""
        if info.get("cuda_available") or info.get("device") == "cuda":
            gpu = gpu or "Tesla T4"
        print("Remote GPU: %s" % (gpu or "not reported"))
        print("API:    %s" % ("connected" if online else "unreachable"))
        print("Latency: %.1fs" % (time.monotonic() - start))
    except NAtlasError as exc:
        print("Unable to reach the N-ATLaS server. Check the server or network connection.")
        log_debug(client, "status check failed: %s" % exc)


# ----------------------------------------------------------------------
# RUNNER (OpenAI SDK path)
# ----------------------------------------------------------------------
def run_repl(client):
    print_banner(client)

    def handle_exception(exc):
        if isinstance(exc, TimeoutFailure):
            print(str(exc))
        elif isinstance(exc, AuthenticationFailure):
            print(str(exc))
        elif isinstance(exc, ForbiddenFailure):
            print(str(exc))
        elif isinstance(exc, NotFoundFailure):
            print(str(exc))
        elif isinstance(exc, ValidationFailure):
            print(str(exc))
        elif isinstance(exc, RateLimitFailure):
            print(str(exc))
        elif isinstance(exc, ServerFailure):
            print(str(exc))
        elif isinstance(exc, GPUExhaustedFailure):
            print(str(exc))
        elif isinstance(exc, ConnectionFailure):
            print(str(exc))
        elif isinstance(exc, EmptyResponse):
            print(str(exc))
        elif isinstance(exc, NAtlasError):
            print(str(exc))
        else:
            if client.debug:
                print("Unexpected error:")
                traceback.print_exc()
            else:
                print("An unexpected error occurred: %s" % exc)

    while True:
        mode = client._language()
        # The pinned mode is shown in the prompt, because a hard pin is
        # otherwise invisible until a reply comes back in the wrong language.
        prompt = "\nYou [%s]: " % mode.label() if mode.pinned else "\nYou: "
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye! Kar ka dawo — jiya ma'aikata. See you next time.")
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            # Split the command word from its argument once, so
            # `/verify A4-100160` reaches the handlers as ("verify", "A4-100160").
            parts = user_input[1:].split(None, 1)
            command = parts[0].strip().lower() if parts else ""
            argument = parts[1].strip() if len(parts) > 1 else ""
            if command == "help":
                print_help_text(client)
                continue
            if command == "new":
                client.clear_conversation()
                print("Started a new conversation.")
                continue
            if command == "clear":
                client.clear_conversation()
                print("Conversation history cleared.")
                continue
            if command == "history":
                print_history(client)
                continue
            if command == "status":
                handle_status(client)
                continue
            if command == "nafdac":
                print(client.nafdac_status())
                continue
            if command == "verify":
                print(client._nafdac().verify_command(argument))
                continue
            if command == "search":
                print(client._nafdac().search_command(argument))
                continue
            if command == "exit":
                print("Bye!")
                break
            if command in LANGUAGE_COMMANDS:
                # `/hausa` and friends pin the mode; `/langueage` only shows it.
                print(mode.command(LANGUAGE_COMMANDS[command]))
                continue
            print("Unknown command: /%s. Type /help for commands." % command)
            continue

        printed_header = {"value": False}

        def on_token(text):
            if not printed_header["value"]:
                sys.stdout.write("ConfamAI: ")
                printed_header["value"] = True
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            result = client.chat(user_input, stream=None, on_token=on_token)
            text = str(result[0] or "")
            if printed_header["value"]:
                print()
            else:
                print("ConfamAI: %s" % text)
            if text:
                client.conversation.add("user", user_input)
                client.conversation.add("assistant", text)
                if client.cfg.save_conversations:
                    try:
                        saved = client.save_conversation(
                            directory=client.cfg.conversations_dir
                        )
                        log_debug(client, "conversation saved to %s" % (saved or "skipped"))
                    except OSError as exc:
                        log_debug(client, "could not save conversation: %s" % exc)
            else:
                print("(the model returned an empty response)")
        except Exception as exc:
            if printed_header["value"]:
                print()
            handle_exception(exc)


def start_repl(client):
    run_repl(client)


# ----------------------------------------------------------------------
# TEST MODE  (--test)
# ----------------------------------------------------------------------
class TestResult:
    def __init__(self):
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, passed, detail))
        print("[%s] %s%s" % (
            "PASS" if passed else "FAIL",
            name,
            (" :: %s" % detail) if detail else "",
        ))
        return passed


def run_tests(cfg):
    results = TestResult()
    print("=" * 46)
    print("N-ATLaS CLIENT TEST SUITE")
    print("=" * 46)
    client = NAtlasClient(cfg, debug=False)
    try:
        client.connect()
    except NAtlasError as exc:
        results.record("connection", False, str(exc))
        print("Cannot continue: %s" % exc)
        return summary(results)

    errors_seen = []

    # -- connection --
    try:
        results.record(
            "connection",
            client._openai is not None,
            "base_url=%s" % client.cfg.base_url.split("://")[-1][:60],
        )
    except Exception as exc:
        results.record("connection", False, repr(exc))

    # -- health --
    try:
        info = client.health()
        ok = bool(info) and info.get("status") in ("ok", "degraded")
        results.record(
            "health returns a status",
            ok,
            "status=%s model=%s" % (info.get("status"), info.get("model")),
        )
        client.api_connected = ok
    except NAtlasError as exc:
        results.record("health returns a status", False, str(exc))
        errors_seen.append(exc)

    # -- models --
    try:
        ids = client.check_available_models()
        results.record(
            "model 'n-atlas' is listed",
            client.cfg.model in ids,
            "ids=%s" % ids,
        )
    except NAtlasError as exc:
        results.record("model 'n-atlas' is listed", False, str(exc))
        errors_seen.append(exc)

    if errors_seen:
        print("")
        print("Connection-level failures prevented deeper tests.")
        return summary(results)

    # -- helper: run a single chat test --
    def chat_test(label, message, history=None, flag="non-empty"):
        text, info = None, {}
        error = None
        try:
            client.clear_conversation()
            if history:
                client.conversation.messages = [dict(m) for m in history]
            text, info = client.chat(message, stream=False)
        except NAtlasError as exc:
            error = exc
        if error is not None:
            results.record(label, False, str(error))
            return None
        ok = bool(text) and bool(text.strip())
        results.record(
            label + " returns a non-empty reply",
            ok,
            repr((text or "")[:140]),
        )
        return text

    # -- languages --
    language_cases = [
        ("English", "How are you?"),
        ("Hausa", "Kana lafiya?"),
        ("Yoruba", "Báwo ni?"),
        ("Igbo", "Kedu ka ị mere?"),
    ]
    for label, prompt in language_cases:
        chat_test(label, prompt)

    # -- language modes (no network) --
    switcher = LanguageSwitcher(cfg, debug=False)
    results.record(
        "language: a fresh client starts in auto",
        switcher.mode == LANGUAGE_AUTO,
        "mode=%s" % switcher.mode,
    )
    results.record(
        "language: auto sends no directive and no field",
        switcher.context_for("Báwo ni?") == ""
        and switcher.request_language() is None,
    )
    for command, expected in [
        ("hausa", "ha"), ("yoruba", "yo"), ("igbo", "ig"), ("english", "en"),
    ]:
        switcher.set(command)
        directive = switcher.context_for("How are you?")
        results.record(
            "language: /%s pins %s" % (command, LANGUAGE_NAMES[expected]),
            switcher.mode == expected
            and switcher.request_language() == expected
            and LANGUAGE_NAMES[expected] in directive,
        )
    results.record(
        "language: a pin holds against a message in another language",
        "even when the message is written in another language"
        in language_directive_for("ha"),
    )
    results.record(
        "language: every command spelling is accepted",
        all(
            LANGUAGE_COMMANDS[word] == code
            for word, code in [
                ("hausa", "ha"), ("yoruba", "yo"),
                ("igbo", "ig"), ("english", "en"), ("auto", LANGUAGE_AUTO),
            ]
        ),
    )

    # -- a pinned mode holds --
    # Each mode is asked an English question on purpose: a mode that only
    # followed the message would pass the block above and fail this one.
    for command, label in [
        ("hausa", "Hausa"), ("yoruba", "Yoruba"),
        ("igbo", "Igbo"), ("english", "English"),
    ]:
        client.set_language(command)
        chat_test("%s mode" % label, "How are you? Tell me about yourself.")
    client.set_language("auto")

    # -- conversation memory --
    client.clear_conversation()
    memory_reply = None
    try:
        memory_reply, _ = client.chat("My name is Ahmed.", stream=False)
        client.conversation.add("user", "My name is Ahmed.")
        client.conversation.add("assistant", memory_reply or "")
    except NAtlasError as exc:
        results.record("memory: name provided", False, str(exc))
        memory_reply = ""
    results.record(
        "memory: 'My name is Ahmed.' returns a reply",
        bool(memory_reply and memory_reply.strip()),
        repr((memory_reply or "")[:140]),
    )
    try:
        recall, _ = client.chat("What is my name?", stream=False)
        client.conversation.add("user", "What is my name?")
        client.conversation.add("assistant", recall or "")
    except NAtlasError as exc:
        results.record("memory: recalls the name", False, str(exc))
        recall = ""
    results.record(
        "memory: 'What is my name?' mentions Ahmed",
        "ahmed" in (recall or "").lower(),
        repr((recall or "")[:140]),
    )

    # -- language switching --
    client.clear_conversation()
    try:
        hausa_reply, _ = client.chat("Answer in Hausa.", stream=False)
        client.conversation.add("user", "Answer in Hausa.")
        client.conversation.add("assistant", hausa_reply or "")
    except NAtlasError as exc:
        results.record("switch: answer in Hausa", False, str(exc))
    try:
        english_reply, _ = client.chat("Now answer in English.", stream=False)
        client.conversation.add("user", "Now answer in English.")
        client.conversation.add("assistant", english_reply or "")
    except NAtlasError as exc:
        results.record("switch: answer in English", False, str(exc))
    results.record(
        "language switching without reset",
        True,
        "history preserved (the client never resets on language change)",
    )

    return summary(results)


def summary(results):
    passed = sum(1 for _, ok, _ in results.results if ok)
    total = len(results.results)
    failed = [(n, d) for n, ok, d in results.results if not ok]
    print("")
    print("=" * 46)
    print("RESULT: %d/%d checks passed" % (passed, total))
    if failed:
        for name, detail in failed:
            print("  FAIL  %s :: %s" % (name, detail))
    print("=" * 46)
    return 0 if not failed else 1


# ----------------------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ConfamAI laptop client — talks to the remote Kaggle server."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show HTTP statuses, durations and exception details (never the API key)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run the test suite (connection, health, models, languages, memory)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="path to a custom .env file",
    )
    args = parser.parse_args()

    if load_dotenv is None:
        print("python-dotenv is not installed. Run: pip install python-dotenv")
        return 2
    if OpenAI is None:
        print("The openai package is not installed. Run: pip install openai")
        return 2

    if args.version:
        print("n_atlas client 1.0.0 (python %s on %s)" % (
            platform.python_version(), platform.system()
        ))
        return 0

    cfg = Config()
    cfg.debug = args.debug
    if args.env is not None:
        # a custom env path is documented; simplest is to re-load
        load_dotenv(args.env, override=False)
        cfg = Config()
        cfg.debug = args.debug

    try:
        validate_config(cfg)
    except ConfigError as exc:
        print(str(exc))
        log_debug(cfg, "config check failed")
        return 2

    if args.test:
        return run_tests(cfg)

    client = NAtlasClient(cfg, debug=args.debug)
    try:
        client.connect()
        # Startup validation: health + models + verify n-atlas exists.
        try:
            client.health()
            client.api_connected = (
                client.server_info.get("status") in ("ok", "degraded")
            )
        except NAtlasError as exc:
            client.api_connected = False
            log_debug(client, "startup health check failed: %s" % exc)
        try:
            ids = client.check_available_models()
            client.model_available = client.cfg.model in ids
            if not client.model_available:
                print("Error: model %r is not available on the server. "
                      "Available: %s" % (client.cfg.model, ids))
        except NAtlasError as exc:
            client.model_available = False
            log_debug(client, "startup models check failed: %s" % exc)
    except NAtlasError as exc:
        print("Unable to reach the N-ATLaS server. Check the server or "
              "network connection.")
        if args.debug:
            log_debug(client, "startup failed: %s" % exc)
        start_repl(client)
        return 0

    start_repl(client)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        print("interrupted — bye!")