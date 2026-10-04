"""The ConfamAI system prompt shipped with the client.

This is the persona the CONFAM consumer-products assistant runs under: it is
sent as the system message on every request, so the model answers as ConfamAI
and not as the general-purpose N-ATLaS assistant the server was built for.

`confam_prompt.txt` at the project root is the source of truth, and both the
client and the server read it. Where the prompt comes from, in order of
precedence:

    1. `N_ATLAS_SYSTEM_PROMPT`   inline text.
    2. `N_ATLAS_SYSTEM_PROMPT_FILE`  a file holding the text.
    3. `confam_prompt.txt`, found next to the client or at the project root.
    4. the ConfamAI prompt below, used only when that file is missing.
    5. nothing at all, when it is switched off explicitly with
       `N_ATLAS_SYSTEM_PROMPT=none` (also off / 0 / false / no / disabled),
       which is how the server's own prompt takes over again.

The switch-off check runs first, so `none` beats a configured file too. Empty
values count as unset, which is what a bare `N_ATLAS_SYSTEM_PROMPT=` line in
.env already means.

`CONFAM_AI_SYSTEM_PROMPT` below is a last-resort fallback so the client still
runs on a machine where the text file was not copied. Keep it identical to the
file: `python n_atlas_server/check_prompt_drift.py` fails when it is not.

One copy of the resolver lives in the single-file client (n_atlas.py) as well,
because that file is copied to other machines on its own and imports nothing
from this package. The two resolvers must behave the same way.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILENAME = "confam_prompt.txt"

CONFAM_AI_SYSTEM_PROMPT = """You are ConfamAI, the official consumer products verification assistant for the CONFAM powered by saynet.ng.

YOUR IDENTITY:
- Your name is ConfamAI. You are NOT N-ATLaS and not "Confam AI". Never refer to yourself by any other name.
- You help Nigerians check whether a consumer product is registered with NAFDAC before they buy or use it: packaged food and beverages, bottled water, snacks and confectionery, cosmetics and personal care, household and cleaning products, sanitary products, textiles, and other everyday goods.
- You are professional, warm, concise, and safety-focused.

YOUR CORE FUNCTION:
When a user asks about a consumer product:
1. If NAFDAC verification data is provided in the conversation, use ONLY that data to respond. Never invent registration numbers, approval dates, manufacturers, product names, or verdicts.
2. Use these exact verdicts: AUTHENTIC (registered), FLAGGED (recalled, banned or suspended), or NOT REGISTERED (not found in database).
3. NOT REGISTERED does NOT mean fake or counterfeit. Always say: "This product was not found in our current NAFDAC database. This could mean the product is new, our records are outdated, or the product is not registered."
4. Never claim a product is counterfeit, fake, poisoned or adulterated.
5. If a food, drink or cosmetic comes back NOT REGISTERED or FLAGGED, say plainly that the user should not buy or use it, and suggest they report it to NAFDAC. That is a statement about the product, not medical advice.

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
- You ONLY help with consumer product verification and NAFDAC-related questions.
- For unrelated questions, politely redirect: "I'm ConfamAI - I specialize in helping you check consumer products against NAFDAC records. Would you like to verify a product?"
- Never provide medical, nutritional or health advice. If asked about dosage, side effects, a diet or a treatment, say: "Please consult a healthcare professional or a nutritionist for medical advice."
- This service is NOT yet officially endorsed by NAFDAC. Do not claim otherwise."""

# Values that mean "send no system message at all". Checked before anything
# else so an explicit opt-out cannot be overridden by a configured prompt file.
OFF_VALUES = frozenset({"0", "off", "no", "none", "false", "disable", "disabled"})


def is_off(value) -> bool:
    """True when this configured value switches the system prompt off."""
    return str(value or "").strip().lower() in OFF_VALUES


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _prompt_file_candidates():
    """Where confam_prompt.txt is looked for, most specific first.

    A relative `N_ATLAS_SYSTEM_PROMPT_FILE` is read against the current working
    directory, which is the project root in the documented way to run the
    client. It is also tried beside this package and at the project root, so a
    client started from another directory still finds the file rather than
    silently falling back to the built-in prompt.
    """
    candidates = []
    for base in (Path.cwd(), Path(__file__).resolve().parent, PROJECT_ROOT):
        candidates.append(base / PROMPT_FILENAME)
    return candidates


def find_prompt_file():
    """The first confam_prompt.txt that exists, or None."""
    for candidate in _prompt_file_candidates():
        if candidate.is_file():
            return candidate
    return None


def resolve_system_prompt(inline: str = "", prompt_file: str = "") -> str:
    """The system prompt to send: inline, file, the project file, built-in."""
    if is_off(inline):
        return ""
    text = str(inline or "").strip()
    if text:
        return text

    path_text = str(prompt_file or "").strip()
    if path_text:
        text = _read(Path(path_text))
        if text:
            return text

    found = find_prompt_file()
    if found is not None:
        text = _read(found)
        if text:
            return text

    return CONFAM_AI_SYSTEM_PROMPT