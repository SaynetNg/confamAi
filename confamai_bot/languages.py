"""Language modes and the opt-out keyword, handled without the model.

A WhatsApp user has no slash-command help text in front of them, so these have
to work by guessable spelling in four languages. That is why the table below is
a table and not four separate checks: someone who types `/harshe` and gets a
reply in Hausa immediately has no way to know `/hausa` was the canonical word,
and no reason to care.

The mode reaches the server two ways, both from `natlas.py`:

    1. the `language` field of the request, which the server treats as
       authoritative and skips its own detector for this turn
    2. the server's own directive, which the model reads

`/auto` sends neither and hands the choice back to the server's detector.

These commands are answered locally, without a model call. A `/hausa` that cost
13 seconds of GPU time to be answered with a two-word confirmation would be a
worse version of something already implemented.

The opt-out keyword is here too, and it is not optional: NDPA 2023 (Nigeria's
Data Protection Act) requires an easy way to stop. `STOP` and its variants clear
the stored conversation and record the opt-out, so no further message is
accepted or answered from that number.
"""

from __future__ import annotations

AUTO = ""

LANGUAGE_NAMES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

MODE_ORDER = ("ha", "yo", "ig", "en", AUTO)

# Command word -> mode code. `None` means "show the current mode and change
# nothing"; `AUTO` (the empty string) means "drop the pin". Those two must stay
# distinct: when both were `""`, `/auto` matched the read-only branch, so the
# command appeared to work while silently leaving the pin in place — a user
# who typed /auto to leave Hausa was still answered in Hausa.
LANGUAGE_COMMANDS = {
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
    "english": "en",
    "auto": AUTO,
    # Read-only, three spellings because nobody remembers which.
    "langueage": None,
    "language": None,
    "lang": None,
    "yare": None,   # "language" in Hausa
    "ede": None,    # "language" in Yoruba
}

# Every spelling accepted for each mode. The canonical command name is included
# automatically, so /hausa works without being repeated here. Local-language
# names are present because the person choosing the language may well be typing
# in it — `harshe` is "language" in Hausa, and it is the word someone typing in
# Hausa is most likely to try.
_SYNONYMS = {
    "en": "en", "english": "en",
    "ha": "ha", "hausa": "ha", "harshe": "ha", "sauraya": "ha",
    "yo": "yo", "yoruba": "yo", "yebu": "yo", "ede yoruba": "yo", "yorùbá": "yo",
    "ig": "ig", "igbo": "ig", "ahukeri": "ig",
    AUTO: AUTO, "automatic": AUTO, "follow": AUTO,
}

# Opt-out keywords, matched case-insensitively on the whole message.
# "STOP" and "QUIT" are the two Meta's own templates conventionally use, so
# they are the two a user has most likely already been trained to type.
OPT_OUT_WORDS = frozenset(
    {"stop", "quit", "unsubscribe", "block", "cancel", "end", "bye"}
)

# Accent-carrying variants, because "unsubscribé" and "block" arrive from
# Nigerian phones often enough to matter, and the cost of missing an opt-out is
# a legal problem rather than a cosmetic one.
_OPT_OUT_EXTRA = frozenset(
    {"unsubscribé", "souscri", "blocke", "katsi", "daina", "sara", "awo"}
)

_UI = {
    "en": {
        "current": "Language mode: {name}. I will reply in {name} from now on.",
        "auto": "Language mode: auto. I will reply in the language you write in.",
        "shown": "Language mode: {name}.",
        "cleared": "Conversation cleared. Send your question again.",
        "image": (
            "I can only read text messages for now. Please type the NAFDAC "
            "number from the pack — it looks like A4-100160."
        ),
        "voice_listening": "🎧 Listening to your voice note...",
        "voice_failed": (
            "Sorry, I could not hear that clearly. Please type the NAFDAC "
            "number instead — it looks like A4-100160."
        ),
        "optout": (
            "You have been unsubscribed and your conversation was deleted. "
            "No messages will be sent. Send START to begin again."
        ),
        "opted_out": (
            "This number is unsubscribed. Send START to begin again."
        ),
        "not_allowed": "This service is not available on this number.",
        "busy": "I am checking another message for you. Please send again in a moment.",
        "unavailable": (
            "I cannot reach the NAFDAC records right now. Please try again in a "
            "few minutes — I would rather say this than guess."
        ),
        "blank": "I did not understand that. Type a NAFDAC number like A4-100160.",
    },
    "ha": {
        "current": "Yanin harshen: {name}. Zan amsa cikin {name} daga yanzu.",
        "auto": "Yanin harshen: auto. Zan amsa cikin yaren da kake yi.",
        "shown": "Yanin harshen: {name}.",
        "cleared": "An sume sabisawa. Sake aika tambayarka.",
        "image": (
            "Na iya karanta saƙo mai rubanni kawai. Ka buga lamarin NAFDAC daga "
            "karkashin mai'amfani, kamar A4-100160."
        ),
        "voice_listening": "🎧 Ina sauraron saƙon muryarka...",
        "voice_failed": (
            "Ku yi haƙuri, ba na iya ji da kyau ba. Ka buga lambar NAFDAC "
            "maimakon haka — kamar A4-100160."
        ),
        "optout": (
            "An soke rajinka kuma an delete sabisawa. Za a iya tura saƙo kai. "
            "Aika START don sake farawa."
        ),
        "opted_out": "Wannan lamba an soke rajanta. Aika START don sake farawa.",
        "not_allowed": "Wannan sabis ba ga wannan lambar ba.",
        "busy": "In duba wata saƙo mai zuwa. Ka sake aika bayan ɗan lokaci.",
        "unavailable": (
            "Ban iya isa ga bayanan NAFDAC a yanzu. Ka gwada sake bayan mintuna "
            "kadan. Na fi ya san hakan fiye da hasashe."
        ),
        "blank": "Ban fahimta ba. Ka buga lamarin NAFDAC kamar A4-100160.",
    },
    "yo": {
        "current": "Èdè ìjẹ́rìí: {name}. Ó máa dá sí ni nínú {name} láti ní báyìí.",
        "auto": "Èdè ìjẹ́rìí: àìtò. Ó máa dá sí ní èdè tí o bá kọ.",
        "shown": "Èdè ìjẹ́rìí: {name}.",
        "cleared": "A fagilé àwọn ìjẹ́rìí. Tún ìbéèrè rẹ̀ lẹ́ẹ̀kansí.",
        "image": (
            "Mo lè kà nàkoko tí ó jẹ́ ọ̀rọ̀ nìkan. Jọ̀wọ́ kọ ìlà NRN láti inú "
            "ìjàngbọ̀n, bí A4-100160."
        ),
        "voice_listening": "🎧 Ń gbọ́ ìkọ̀kọ̀ rẹ...",
        "voice_failed": (
            "Ẹ jọ̀wọ́, èmi kò lè gbọ́ ọ̀rọ̀ yẹn dáadáa. Jọ̀wọ́ kọ ìlà NAFDAC "
            "lẹ́ẹ̀kansí — bí A4-100160."
        ),
        "optout": (
            "A yí ọ kúnáàti. A ti fagilé ìjẹ́rìí rẹ̀. A kò ní fi ìránṣẹ́ sí i. "
            "Fi ìránṣẹ́ START láti bẹ̀rẹ̀ sí i."
        ),
        "opted_out": "Àmì àyè tí a ti yí àkọkọ́ ọ kúnáàti. Fi START láti bẹ̀rẹ̀ sí i.",
        "not_allowed": "Àmáìṣè yìí kò wà lórí nọ́mbà náà.",
        "busy": "Ń ṣàyẹ̀wò ọ̀rọ̀ mìíràn. Rán ọ̀rọ́ rẹ̀ lẹ́ẹ̀kansí.",
        "unavailable": (
            "Kò bá lè dé ibi ìwé NAFDAC báyìí. gbìyànjú lẹ́ẹ̀kansí láàárín ìṣẹ́jú "
            "mẹ́wàá. Mo fẹ́ sọ ọ̀rọ̀ yìí ju síkí ìṣàlàyé."
        ),
        "blank": "Èmi kò gbọ́. Jọ̀wọ́ kọ ìlà NRN bí A4-100160.",
    },
"ig": {
        "current": "Ụdọka bekee: {name}. Ga-eme site na ọ ga-eme n'ime {name}.",
        "auto": "Ụdọka bekee: akụkwọ. Ga-eme n'ụdọka ị bụrị na ị na-akpụtụrị.",
        "shown": "Ụdọka bekee: {name}.",
        "cleared": "Akagbuola akụkọ. Zigharịa ọzọ.",
        "image": (
            "Enweghị ike ịgụ naanị ederede. Biko pụta nọ́mba NAFDAC dị "
            "na ngọ́ ọrụ, dịka A4-100160."
        ),
        "voice_listening": "🎧 Na-anụ olu gị...",
        "voice_failed": (
            "Nnọọ, enweghị m ike ịnụ ya nke ọma. Biko pụta nọ́mba NAFDAC "
            "kama — dịka A4-100160."
        ),
        "optout": (
            "Aka-akịkwela gị site-na-naanị. Akagbuola akụkọ gị. Enweghị "
            "ozi-e ga-abị gị. Zigharịa START ka ị bụrụ ọzọ."
        ),
        "opted_out": "Aka-akịkwela nọ́mbà ndị a. Zigharịa START ka ị bụrụ ọzọ.",
        "not_allowed": "Ọrụ a dịghị na nọ́mbà ndị a.",
        "busy": "Ana m lelee ozi ọzọ. Zigharịa ọzọ n'oge di.",
        "unavailable": (
            "Enweghị ike ịrụ nke NAFDAC ugbu a. Biko gbalịa ọzọ n'ime nkeji "
            "ole na ole. Na-aga m ịka akacha ọrụ aka."
        ),
        "blank": "Ụkwọghị m. Biko pụta nọ́mba NAFDAC dịka A4-100160.",
    },
}

# START is the counterpart to STOP: an opt-out is supposed to be reversible,
# and a user who wants to come back must not have to find a phone number.
START_WORDS = frozenset({"start", "begin", "resume", "restart", "sub"})


def is_language_command(text: str) -> tuple[bool, str | None, str]:
    """(is_command, mode, argument) for `/hausa`-style input.

    `mode` is `None` for a read-only `/lang`, `""` for `/auto`, and a language
    code otherwise. The argument is anything after the command word, so
    `/hausa please` still resolves.
    """
    if not text:
        return False, None, ""
    stripped = str(text).strip()
    if not stripped.startswith("/"):
        return False, None, ""

    body = stripped[1:].strip()
    if not body:
        return False, None, ""

    head, _, argument = body.partition(" ")
    head = head.strip().lower()

    if head not in LANGUAGE_COMMANDS:
        return False, None, ""

    # `in` rather than `!= ""`, so the read-only `None` is matched correctly and
    # AUTO (""), which is a real mode, is not mistaken for read-only.
    if head in LANGUAGE_COMMANDS and LANGUAGE_COMMANDS[head] is None:
        return True, None, argument.strip()
    return True, LANGUAGE_COMMANDS[head], argument.strip()


def resolve_alias(text: str) -> str:
    """The mode code for a language word, or "" when it is not one."""
    key = str(text or "").strip().lower()
    if not key:
        return ""
    return _SYNONYMS.get(key, "")


def is_opt_out(text: str) -> bool:
    """True when this message is an opt-out.

    Only a short message counts. `stop` inside a sentence is somebody asking
    their blood pressure medicine to stop being checked, and honouring that as
    an opt-out would delete a real user's conversation mid-question.
    """
    words = str(text or "").strip().lower()
    if not words or len(words) > 40:
        return False
    tokens = set(words.replace("!", "").replace(".", "").replace(",", "").split())
    return bool(tokens & (OPT_OUT_WORDS | _OPT_OUT_EXTRA))


def is_start(text: str) -> bool:
    words = str(text or "").strip().lower()
    if not words or len(words) > 40:
        return False
    tokens = set(words.replace("!", "").replace(".", "").replace(",", "").split())
    return bool(tokens & START_WORDS)


def label(mode: str) -> str:
    return LANGUAGE_NAMES.get(mode, "automatic")


def confirm(mode: str) -> str:
    """The confirmation for a pinned mode, in that language."""
    if mode == AUTO:
        return _UI["en"]["auto"]
    return _UI.get(mode, _UI["en"])["current"].format(name=LANGUAGE_NAMES[mode])


def shown(mode: str) -> str:
    if mode == AUTO:
        return _UI["en"]["auto"]
    return _UI.get(mode, _UI["en"])["shown"].format(name=LANGUAGE_NAMES[mode])


def message(key: str, mode: str = "en") -> str:
    """A local UI string by key, in the user's language.

    Falls back to English. A missing translation must never be the reason a
    user is told nothing, so every caller passes through here rather than
    indexing the dict directly.
    """
    return _UI.get(mode, _UI["en"]).get(key) or _UI["en"][key]