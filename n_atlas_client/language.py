"""Language modes for the N-ATLaS laptop client.

    /hausa     every reply is in Hausa
    /yoruba    every reply is in Yoruba
    /igbo      every reply is in Igbo
    /english   every reply is in English
    /auto      no mode: the reply follows the language of each message

The mode is a hard pin, not a suggestion. In Yoruba mode a Hausa message is
answered in Yoruba, because the mode is the user's choice about the *reply* and
the language of the message is only the default. It changes when the user runs
one of the commands above, never because a message arrived in another
language.

The mode reaches the server two ways, and both are always sent:

  1. `language` in the request body, which a patched server treats as
     authoritative -- it skips its own detector for that request. An unpatched
     server ignores it, because ChatCompletionRequest is `extra="allow"`.
  2. one directive line on the outgoing turn, which an unpatched server still
     reads.

There is deliberately no language detector here. The server has a good one
(n_atlas_server/natlas_notebook.py, Cell 7) and a second copy would only be a
second thing to keep in step with it. Nothing in this module guesses a
language; it only records which one the user asked for.
"""

AUTO = "auto"

LANGUAGE_NAMES = {
    AUTO: "Auto",
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

# The order the modes are listed in, in every language.
MODE_ORDER = ("ha", "yo", "ig", "en", AUTO)

# The command each mode is set with, quoted back in the confirmation.
COMMANDS = {
    "ha": "hausa",
    "yo": "yoruba",
    "ig": "igbo",
    "en": "english",
    AUTO: "auto",
}

# Sent on the outgoing turn. The server has its own, stronger version of this
# when it understands the `language` field; this one is what keeps an
# unpatched server honest.
# Both the two-letter code and the full name resolve to the same mode, so a
# caller passing "yoruba" and one passing "yo" cannot disagree. Local names are
# here too, because the person choosing the language may well be typing in it.
_SYNONYMS = {
    "en": "en", "english": "en",
    "ha": "ha", "hausa": "ha", "harshe": "ha", "sauraya": "ha",
    "yo": "yo", "yoruba": "yo", "yebu": "yo", "ede yoruba": "yo",
    "ig": "ig", "igbo": "ig", "ahukeri": "ig",
    AUTO: AUTO, "automatic": AUTO, "follow": AUTO,
}

DIRECTIVE = (
    "[language] LANGUAGE MODE: {name}. The user set it with /{command} and "
    "every reply must be in {name}, even when the message is written in "
    "another language. Write the entire reply in {name} and do not use "
    "{others}. Keep names, numbers, dates and product data exactly as given, "
    "and ignore automatic language detection for this turn."
)

# The /hausa family of commands is answered on the laptop, so this is the one
# place the client itself speaks four languages, and the wording has to be
# right without a model to check it.
UI = {
    "en": {
        "title": "Language mode",
        "auto": "Auto - follow the language of each message",
        "current": "Now replying in {name}.",
        "confirm": "Language mode: {name}. ConfamAI will reply in {name} "
                   "until you switch with another language command.",
        "auto_confirm": "Language mode: Auto. ConfamAI will reply in the "
                        "language of each message.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "ha": {
        "title": "Yanin harshen",
        "auto": "Auto - yana bin yaren da kake yi",
        "current": "Ana amsa cikin {name} a yanzu.",
        "confirm": "An saita harshen: {name}. Za a amsa cikin {name} har sai "
                   "ka canza shi da umarni harshen nan.",
        "auto_confirm": "An saita harshen: Auto. Za a amsa cikin yaren da kake "
                        "yi a kowane sauri.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "yo": {
        "title": "Èdè tí a ṣe tàn",
        "auto": "Auto - ó tẹ̀lé èdè tí o bá ń lò",
        "current": "À ń dá ọjọ́ nínú {name} ní báyìí.",
        "confirm": "A ṣètò èdè: {name}. A ó dá gbogbo ìdáhùn nínú {name} títí "
                   "o yí padà pẹ̀lú àṣà èdè mìíràn.",
        "auto_confirm": "A ṣètò èdè: Auto. A ó dá ọjọ́ nínú èdè tí o bá ń lò "
                        "fún gbogbo ìbéèrè.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
    "ig": {
        "title": "Ụdọ akabọọga",
        "auto": "Akpọcha - ọ na-akpọ ụdọ ọ na-eme",
        "current": "Anyị na-ezighata na {name} ugbu a.",
        "confirm": "Agbanyela ụdọ akabọọga: {name}. ConfamAI ga-ezighata "
                   "na {name} ruo mgbe ị gbanwee ụdọ site na nzọọụta ọzọ.",
        "auto_confirm": "Agbanyela ụdọ akabọọga: Akpọcha. ConfamAI ga-ezighata "
                        "na ụdọ ọ na-eme maka ihe ọ bụla.",
        "hint": "/hausa  /yoruba  /igbo  /english  /auto",
    },
}


def normalize(mode: str) -> str:
    """The mode, or AUTO. Anything unrecognised is not a mode.

    Accepts the code, the full name, or the name in the language itself, so a
    typo in `.env` or in a call site falls back to auto instead of raising, and
    a caller cannot accidentally end up in auto by passing "yoruba" instead of
    "yo".
    """
    key = str(mode or "").strip().lower()
    if key in LANGUAGE_NAMES:
        return key
    return _SYNONYMS.get(key, AUTO)


def directive_for(mode: str) -> str:
    """The line added to each outgoing turn. Empty in auto."""
    code = normalize(mode)
    if code == AUTO:
        return ""
    others = ", ".join(
        name for other, name in LANGUAGE_NAMES.items()
        if other != AUTO and other != code
    )
    return DIRECTIVE.format(
        name=LANGUAGE_NAMES[code], command=COMMANDS[code], others=others
    )


class LanguageSwitcher:
    """Holds the mode and answers the /hausa family of commands on the laptop.

    A class rather than a function because the mode has to outlive a single
    call: the REPL reads it, the chat path sends it, and the NAFDAC renderer
    prints in it.
    """

    def __init__(self, cfg=None, debug: bool = False):
        self.debug = bool(debug or getattr(cfg, "debug", False))
        self.configured = getattr(cfg, "language", AUTO) or AUTO
        self.mode = normalize(self.configured)
        if self.debug and self.configured and self.mode == AUTO:
            print("[debug] N_ATLAS_LANGUAGE=%r is not a language; using auto."
                  % (self.configured,), flush=True)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def pinned(self) -> bool:
        return self.mode != AUTO

    def label(self) -> str:
        return LANGUAGE_NAMES[self.mode]

    def code(self) -> str:
        """The code for the request body, and for local output rendering.

        Auto renders local NAFDAC output in English: with no mode there is
        nothing to localise to, and a deterministic table beats a guess. The
        model's own reply still follows the message, because the server detects
        that independently.
        """
        return "en" if self.mode == AUTO else self.mode

    def _t(self, key: str, **kwargs) -> str:
        table = UI.get(self.mode) or UI["en"]
        return table[key].format(**kwargs)

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    def set(self, mode: str) -> str:
        """Handle /hausa, /yoruba, /igbo, /english and /auto."""
        self.mode = normalize(mode)
        if self.debug:
            print("[debug] language mode = %s" % self.mode, flush=True)
        if self.mode == AUTO:
            return self._t("auto_confirm")
        return self._t("confirm", name=LANGUAGE_NAMES[self.mode])

    def command(self, code: str) -> str:
        """Handle one language command.

        An empty `code` shows the current mode without changing it, which is
        what `/langueage` and `/help` use.
        """
        if not code:
            return self.menu()
        return self.set(code)

    def menu(self) -> str:
        lines = [self._t("title")]
        for mode in MODE_ORDER:
            marker = "*" if mode == self.mode else " "
            lines.append("  %s %s" % (marker, LANGUAGE_NAMES[mode]))
        lines.append("")
        lines.append(self._t("current", name=self.label()))
        lines.append(self._t("hint"))
        return "\n".join(lines)

    def help_text(self) -> str:
        return (
            "  /hausa  /yoruba  /igbo  /english  /auto   Choose the language\n"
            "  /langueage          Show the current language mode"
        )

    # ------------------------------------------------------------------
    # the chat path
    # ------------------------------------------------------------------
    def context_for(self, message: str = "") -> str:
        """The line added to this turn. Empty in auto, always.

        `message` is accepted and ignored on purpose: a pin does not care what
        language the message is in, which is the whole point of it. The
        parameter is there so the call site reads the same in both modes and so
        a future detector has an obvious place to go.
        """
        directive = directive_for(self.mode)
        if self.debug:
            print("[debug] language mode=%s directive=%s"
                  % (self.mode, bool(directive)), flush=True)
        return directive

    def request_language(self):
        """The value for the request body: the code, or None in auto.

        None rather than "auto" so an auto request is byte-identical to what the
        client sent before language modes existed.
        """
        return None if self.mode == AUTO else self.mode
