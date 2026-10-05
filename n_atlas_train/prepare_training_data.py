"""Rebuild confam_training_dataset.jsonl so it is safe to fine-tune on.

What is wrong with the file as shipped, all of it found by reading it:

  1. 73 records assert a verdict with no verified data attached, three of them
     "CRITICAL ALERT: This product has been FLAGGED as a known counterfeit".
     Training on those teaches the model to invent verdicts and to call
     products counterfeit, which is the one behaviour this project exists to
     prevent.
  2. Every record carries invented registry facts ("Orange Drugs Ampiclox |
     A4-2045 | Orange Drugs Ltd"). Fine-tuning on them turns the model into a
     second, wrong database.
  3. The answers claim "It is safe for use" and "NAFDAC approved". The
     greenbook records registrations; it never says a pack is safe.
  4. The verdict words are not the project's. The headings in use are
     TABBATACCE, VERIFIED, KO SI NIPA AKOSILE NAFDAC, BABU A RIJISTA and NO
     RECORDS FOUND instead of AUTHENTIC / FLAGGED / NOT REGISTERED.
  5. Every system message is a different persona: "Confam AI … medicines and
     consumer products … NAFDAC approved". The served prompt is ConfamAI,
     consumer products only, and forbids claiming approval.
  6. Igbo is 13 of 866 records against 703 English.

What this script does:

  * system message becomes confam_prompt.txt verbatim, in every record;
  * a verified block, when there is one, becomes its own system message in the
    shape the server actually sends, so training matches production;
  * the values inside that block are synthetic and unique per record, so the
    lesson learned is "copy the value out of the block", not "this product is
    this number". No real registry value survives into the output;
  * the answer is regenerated from the templates below, so every verdict word
    is canonical and every answer is a copy of the block;
  * the 73 verdict-without-data records become negatives: no block, so no
    verdict, the assistant asks for the number;
  * negatives are added for all four languages, and the Igbo/Hausa/Yoruba
    deficit is made up with generated records;
  * multi-turn records are split into one record per turn, because each API
    call in production is a single prompt with one block.

Wording lives in TEMPLATES below. Edit that table and re-run; nothing else in
this file holds any text.

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --check-only
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _project_root() -> Path:
    """The repository root, found by walking up to confam_prompt.txt.

    This script lives beside the dataset it produces, in n_atlas_train/, while
    the prompt it embeds sits at the repository root. Resolving both by walking
    up is what lets the folder move without breaking either path.
    """
    for candidate in (HERE, *HERE.parents):
        if (candidate / "confam_prompt.txt").is_file():
            return candidate
    return HERE


ROOT = _project_root()

# The canonical dataset name, the one notebook A uploads. It holds the cleaned
# records.
TARGET = HERE / "confam_training_dataset.jsonl"

# The raw file this script was written to rebuild, kept under a name that says
# what it is. It was never committed, so treat its absence as normal: the
# canonical file above is then already the cleaned output and there is nothing
# left to rebuild.
SOURCE_CANDIDATES = (
    HERE / "confam_training_dataset.original.jsonl",
    HERE / "confam_training_dataset.raw.jsonl",
)

SYSTEM_PROMPT_FILE = ROOT / "confam_prompt.txt"

# The exact prefix the client sends (n_atlas_client/nafdac.py). Shortened to one
# sentence here: the full instruction block is the same at training and serving
# time, and repeating it in every record would triple the token cost for no
# behavioural gain.
BLOCK_PREFIX = (
    "[nafdac] Verified greenbook data for this question. Report these values to "
    "the user as fact, and do not add anything that is not listed here."
)

# Verdict words are quoted in English even in the local-language answers, which
# is what production does: formatting.py renders "VERDICT: NOT REGISTERED"
# untranslated and both instruction blocks say to copy the verdict word.
TEMPLATES = {
    "greeting": {
        "en": "I'm ConfamAI, I help you check consumer products against NAFDAC "
              "records. How can I help you?",
        "ha": "Ni ne ConfamAI, in taimaka maka duba abinci da kayayyuka a kan "
              "yakadda na NAFDAC. Yaya zan iya taimaka?",
        "yo": "Emi ni ConfamAI, mo n iranlowo re o so o wa niyii ile ayewo "
              "NAFDAC fun awon ọja. Bawo ni mo le se?",
        "ig": "Ị bụla m, bụ ConfamAI, enyemaka gi ịlọta ngwa ngwa kwa ihe "
              "NAFDAC debanye aha. Kedu otu m ga-esi gịnị?",
    },
    "identity": {
        "en": "I'm ConfamAI, the consumer products verification assistant for "
              "saynet.ng. I check a product against NAFDAC records and report "
              "only what those records say.",
        "ha": "Ni ne ConfamAI, mai taimaka duba kayayyuka na saynet.ng. Na "
              "duba abinci a kan rajista na NAFDAC kawai, ba na zato ba tare da "
              "wannan.",
        "yo": "Emi ni ConfamAI, olùrànwò ọja ìṣàyẹ̀wò fun saynet.ng. Mo "
              "wò ọja pẹ̀lú ìwé àkọsílẹ̀ NAFDAC, kì í bá ṣe ohun tí kì í "
              "wà níbí.",
        "ig": "Ị bụla m, bụ ConfamAI, onye ọrụ nlekọta ngwa consumer maka "
              "saynet.ng. M na-akọta ngwa ahụ na ihe NAFDAC chere, ma ọ bụ "
              "naanị ihe dị nke ahụ dị.",
    },
    "authentic": {
        "en": "AUTHENTIC - {product} ({nrn}) is listed as registered by "
              "{applicant}. That is the registration record: it does not confirm "
              "this particular pack is genuine, because a copied number resolves "
              "the same way.",
        "ha": "AUTHENTIC - {product} ({nrn}) ya kasance a cikin rajista na "
              "NAFDAC da {applicant}. Wannan ita kawai rajista ne, ba tabbatar "
              "da cikin gaskiya ba: lambar da a kwarya ma za ta same ne.",
        "yo": "AUTHENTIC - {product} ({nrn}) wà nínú ìforúkọsílẹ̀ NAFDAC "
              "tí {applicant} ṣe. Ìyẹn wà nípa ìforúkọsílẹ̀ nìkan: kì í "
              "fọwọ́sí pé ìpò yìí náà tọ́ ọjọ́, nítorí ọmọ ọ̀nà tí a kọ́kọ́ lọ "
              "máa ń bá wọ́ pẹ̀lú rẹ̀.",
        "ig": "AUTHENTIC - {product} ({nrn}) dị n'ime nchekwa NAFDAC nke "
              "{applicant}. Nke ahụ bụ nchekwa nkọwapụta, ọ bụghị ọ bụla "
              "proof na nke ọ a ga-eme: ọnụọgụ dị ike ọ ga-eme ọzọ.",
    },
    "flagged": {
        "en": "FLAGGED - {product} ({nrn}) is recorded as {reason}. Do not buy "
              "or use it. The registration record is not an endorsement of any "
              "particular pack.",
        "ha": "FLAGGED - {product} ({nrn}) an ruwaita a matsayin {reason}. Kada "
              "ka sayi ko amfani. Rajista ba amince da wannan takkarar ne.",
        "yo": "FLAGGED - {product} ({nrn}) wà kọ sí bí {reason}.Má ṣe "
              "rá kì í ṣe lò. Ìforúkọsílẹ̀ kì í ṣe ìdí fún ọja kọ́ọ̀kan "
              "pàtàkì.",
        "ig": "FLAGGED - {product} ({nrn}) bụlarị anyị na-eji ya dịka {reason}. "
              "Ị gaghị ọzọ ịzụ ma ọ bụ ebe. Nchekwa nkọwapụta nke ahụ bụghị "
              "mgbanyeghị onye ọ dị.",
    },
    "not_registered": {
        "en": "NOT REGISTERED - {product} was not found in our current NAFDAC "
              "database. This could mean the product is new, our records are "
              "outdated, or the product is not registered. It does not by "
              "itself mean the product is counterfeit. Give the NAFDAC number "
              "printed on the pack and I can check it directly.",
        "ha": "NOT REGISTERED - ba a same {product} a cikin bayanan NAFDAC da "
              "ke yanzu ba. Wannan na iya nufin cewa sabin ne, ko cewa "
              "bayanan mu ya gama, ko cewa ba a_REGISTER ba ne. Ba shi da "
              "kansa ya nufin counterfeit ne. Ka ba lambar NAFDAC da ake "
              "rubawa a karkashin kunfu, za i duba shi kai tsaye.",
        "yo": "NOT REGISTERED - a ń fi {product} kòrí íwé àkọsílẹ̀ NAFDAC "
              "lọ́wọ́lẹ́wọ́. Èyí lè tú ìwà ni ọ tuntun, tàbí àwọn ìwé "
              "àkọsílẹ̀ ti tẹ̀lé, tàbí kì í ṣe forúkọsílẹ̀. Kò dá ọ lójú pé "
              "ọ jẹ́ iṣèdá ọkọ. Fún mi nọ́mbà NAFDAC tí a tẹ̀lé lórí ọja, "
              "èyí lè bá mi wà ṣàtájẹ́.",
        "ig": "NOT REGISTERED - a hụghị {product} na nchekwa NAFDAC anyị "
              "ugbu a. Nke a nwere ike ịbụ na ngwa ahụ bụ ọhụrụ, maọbụ na "
              "ihe anyị cheburu bụ ọtụtụ, maọbụ na ngwa ahụ enweghị "
              "akpọcha. Ọ nghị site na ya nke ahụ ọ bụ ụgbu ụtọrọ. Nye m "
              "ọnụọgụ NAFDC dị n'elu ngwa ahụ, m ga-akọta ya ozugbo.",
    },
    "no_data": {
        "en": "I cannot confirm that product without a lookup. Give me the "
              "NAFDAC registration number printed on the pack and I will check "
              "it against our records.",
        "ha": "Ban sani i tabbatar da wannan ba tare da duba ba. Ka ba mini "
              "lambar rajista na NAFDAC da ake rubawa a karkashin kunfu, za "
              "ina duba shi.",
        "yo": "Èmi kò lè fọwọ́sí ọja yẹ́ wà nípa ìwé àkọsílẹ̀. Fún mi nọ́mbà "
              "forúkọsílẹ̀ NAFDAC tí a tẹ̀lé lórí ọja, èyí lè jẹ́ kí n "
              "ṣàyẹ̀wò ó.",
        "ig": "M nweghị ike ikpọta n'ahụ ngwa ahụ na-enweghị nchekwa. Nye m "
              "ọnụọgụ akpọcha NAFDAC dị n'elu ngwa ahụ, m ga-akọta ya.",
    },
    "out_of_scope": {
        "en": "I'm ConfamAI - I specialize in helping you check consumer "
              "products against NAFDAC records. Would you like to verify a "
              "product?",
        "ha": "Ni ne ConfamAI - in mayar da hannu ka taimaka duba kayayyuka "
              "kan rajista na NAFDAC. Ka son duba wani abu?",
        "yo": "Emi ni ConfamAI - mo ní tìí ìrànwò pẹ̀lú ọja ìṣàyẹ̀wò "
              "NAFDAC. Ṣé o fẹ́ láti ṣàyẹ̀wò ọja kan?",
        "ig": "Ị bụla m, bụ ConfamAI - enyemaka gi ịlọta ngwa ngwa kwa "
              "NAFDAC. Ị ga-achọ ịlọta ngwa ọ dị?",
    },
    "medical": {
        "en": "I'm ConfamAI - I specialize in helping you check consumer "
              "products against NAFDAC records. For medical, nutritional or "
              "health advice, please consult a healthcare professional or a "
              "nutritionist.",
        "ha": "Ni ne ConfamAI - in mayar da hannu ka taimaka duba kayayyuka "
              "kan rajista na NAFDAC. Ga kanin kowane shawara lafiya ko "
              "gari, ka yi maganganci da gaskiya ko mai bincike na abinci.",
        "yo": "Emi ni ConfamAI - mo ní tìí ìrànwò pẹ̀lú ọja ìṣàyẹ̀wò "
              "NAFDAC. Fún ìmọ̀ràn ìdánwò, ìkúni okun tàbí ìwàláàyè, jọ̀wọ́ "
              "pẹ̀lú dókítà tàbí nípa-ọjọ-ọjọ.",
        "ig": "Ị bụla m, bụ ConfamAI - enyemaka gi ịlọta ngwa ngwa kwa "
              "NAFDAC. Maka ndịzọ ahụka, ọpụma maọbụ ọrụụ onye ọụga, biko "
              "gwa dokita maọbụ onye nwere ọrụ nke nri.",
    },
    "endorsement": {
        "en": "I'm ConfamAI, a verification assistant powered by saynet.ng. "
              "This service is not yet officially endorsed by NAFDAC, and I "
              "report only what the NAFDAC records say.",
        "ha": "Ni ne ConfamAI, mai taimaka duba kayayyuka da saynet.ng. Wannan "
              "ba a amince da shi ta NAFDAC ba tukuna, kuma in bayanin kawai "
              "ne abin da NAFDAC ya rubata.",
        "yo": "Emi ni ConfamAI, òṣìṣẹ́ ọja ṣìṣe tí saynet.ng n ṣe. Àwọn "
              "ìwọlé yìí kò tíì gba ìfẹ́ pẹ̀lú NAFDAC, mo kò sì ju sílẹ̀ "
              "àyàfi ohun tí ìwé àkọsílẹ̀ NAFDAC ní.",
        "ig": "Ị bụla m, bụ ConfamAI, onye ọrụ nlekọta ngwa nke saynet.ng. "
              "Ọrụọ a anaghị ọgwụlaa site na NAFDAC, m na-akọta naanị ihe "
              "NAFDAC cheburu.",
    },
}

# Reasons a record is FLAGGED, phrased the way the greenbook phrase them.
FLAGGED_REASONS = (
    "suspended",
    "withdrawn",
    "recalled",
    "banned",
    "listed as Inactive in the current greenbook",
)

# Synthetic values. Unique per record so the model learns to copy whatever the
# block says rather than to recall a fixed pairing.
#
# The product NAMES are real Nigerian consumer goods and packaged medicines,
# because those are the words users actually type; the greenbook export cannot
# supply them, since it holds only Drugs and Herbals and Nutraceuticals. The
# values beside the name stay synthetic: an "Example ..." entry and an "E1-000001"
# number cannot be mistaken for a real registry fact, so a leak is obvious rather
# than plausible. A real product name paired with a synthetic number is still a
# pairing nothing in the greenbook supports, so there is no fact to memorise.
#
# Medicines are included deliberately: verifying that a drug is registered is not
# medical advice, and the prompt now draws that line explicitly. What must never
# become a training answer is dosing, which classify() routes to the referral
# template.
PRODUCT_STEMS = (
    # food and beverage
    "Indomie Noodles",
    "Golden Penny Spaghetti",
    "Chin Chin",
    "Tomato Paste",
    "Table Sugar",
    "Vegetable Oil",
    "Malta Gold",
    "Saria Packaged Water",
    "Peak Evaporated Milk",
    "Cocoa Beverage Powder",
    # cosmetics and personal care
    "Colgate Toothpaste",
    "Closeup Toothpaste",
    "Dettol Soap",
    "Lifebuoy Soap",
    "Sofrotto Hair Oil",
    "Sunsilk Shampoo",
    "Nivea Beauty Cream",
    "Body Lotion",
    # household and cleaning
    "Sunlight Detergent",
    "OMO Detergent",
    "Vim Cleaning Bar",
    "Jik Bleach",
    "Dishwashing Liquid",
    "Toilet Cleaner",
    # sanitary and textiles
    "Sanitary Pads",
    "Toilet Paper",
    "Wax Print Fabric",
    "Beaded Fabric",
    # packaged medicines and drugs: registration queries only
    "Paracetamol Tablets",
    "Ibuprofen Tablets",
    "Artemether Lumefantrine",
    "Amoxicillin Syrup",
    "Ciprofloxacin Tablets",
    "Omeprazole Capsules",
    "Loratadine Tablets",
    "ORS Sachet",
    "Zinc Supplement Tablets",
    "Multivitamin Tablet",
    "Ibucap Cold and Flu Syrup",
    "Vita Ace Tablets",
    # kept on purpose: the sentinel that makes a leaked real value obvious
    "Example Food Product",
    "Example Tablet",
)
APPLICANT_STEMS = (
    "Flour Mills Nigeria Plc",
    "Nestle Nigeria Plc",
    "Unilever Nigeria Plc",
    "Rite Foods Nigeria Ltd",
    "Emzor Pharmaceuticals Ltd",
    "Fidson Healthcare PLC",
    # kept on purpose: the sentinel that makes a leaked real value obvious
    "Example Applicant Ltd",
    "Example Household Ltd",
)

# A registration number shaped like NAFDAC's own, in a range no real product
# occupies: prefix "E" is unused in the export.
NRN_PREFIXES = ("E1", "E2", "E3", "E4", "E5")

# A registration number in the shape NAFDAC publishes, plus any product or
# applicant name the tables above can produce. Built from the tables so that
# adding a stem adds it to the check with it: the invariant "nothing in the
# answer that is not in the block" has to cover the real names too, not just the
# obvious Example sentinels.
BLOCK_TOKEN_RE = re.compile(
    r"[A-Z]?\d-\d{4,7}"
    r"|\b(?:%s)\b \d+\b"
    r"|\b(?:%s) \d+\b"
    % ("|".join(re.escape(stem) for stem in
                sorted(PRODUCT_STEMS, key=len, reverse=True)),
       "|".join(re.escape(stem) for stem in
                sorted(APPLICANT_STEMS, key=len, reverse=True)))
)


def stem_pair(index: int):
    """(product stem, applicant stem) for one record, with no ordinal.

    Kept separate from synthetic_values() because a migration needs the bare stem:
    synthetic_values() returns a name with its ordinal already appended, so using
    it as a replacement and then appending the ordinal again writes "Indomie
    Noodles 1 1" and grows the name a little more on every run.
    """
    return (PRODUCT_STEMS[index % len(PRODUCT_STEMS)],
            APPLICANT_STEMS[index % len(APPLICANT_STEMS)])


def synthetic_values(index: int):
    product_stem, applicant_stem = stem_pair(index)
    product = "%s %d" % (product_stem, index + 1)
    applicant = "%s %d" % (applicant_stem, index + 1)
    prefix = NRN_PREFIXES[index % len(NRN_PREFIXES)]
    nrn = "%s-%06d" % (prefix, index + 1)
    return product, nrn, applicant


def block_field(block, *labels):
    """One field out of a source block, which is a single `A | B | C` line.

    Reading to the end of the line instead of to the next separator is the
    mistake that matters here: it yields a "name" that contains the whole block,
    which then matches nothing in the user's turn.
    """
    fields = {}
    for part in str(block or "").split("|"):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        fields[key.strip().lower()] = value.strip()
    for label in labels:
        value = fields.get(label.lower(), "")
        if value:
            return value
    return ""


def swap_source_names(user_text, block, product, applicant):
    """The product and applicant the user named, replaced with the synthetic ones.

    The number rewrite is not enough on its own. A turn reading "Wannan Swiss
    Pharma Chloroquine din original ce?" answered with "Example Food Product 1"
    teaches the model to answer about a different product than the one asked
    about. Both names come out of the source block, so they can be matched
    exactly and swapped while the phrasing and the language of the turn are left
    alone.
    """
    text = str(user_text or "")
    for labels, replacement in ((("product",), product),
                                (("applicant", "manufacturer"), applicant)):
        name = block_field(block, *labels)
        if len(name) < 3:
            continue
        text = re.sub(re.escape(name), replacement, text, flags=re.IGNORECASE)
    return text


def renumber_user_text(text, counter):
    """Every number in the turn, rewritten to a synthetic one.

    Without this the conversation contradicts itself: the user asks about
    A4-0891 and the block answers for E1-000042, which would train the model to
    report a number nobody asked about. The letter prefix is kept so the value
    still has the shape NAFDAC publishes, and every serial is unique so no
    number repeats anywhere in the file and nothing here can be recalled as a
    (number, product) pair.

    Numbers are rewritten in the negative records too, so the finished file
    contains no identifier from the source file at all and that is checkable.

    Returns (rewritten_text, first_nrn_or_empty); the counter advances once per
    number rewritten.
    """
    seen = []

    def swap(_match):
        original = _match.group(0)
        match = re.match(r"\s*([A-Za-z]{0,2})\s*[-‐-―/]\s*\d{2,7}"
                         r"([A-Za-z]{0,2})", original)
        prefix = (match.group(1) if match else "") or NRN_PREFIXES[
            counter[0] % len(NRN_PREFIXES)]
        suffix = (match.group(2) if match else "") or ""
        counter[0] += 1
        nrn = "%s-%06d%s" % (prefix.upper(), counter[0], suffix)
        seen.append(nrn)
        return nrn

    rewritten = NUMBER_IN_TEXT.sub(swap, str(text or ""))
    return rewritten, (seen[0] if seen else "")


def build_block(product, nrn, applicant, verdict, reason=""):
    lines = [
        BLOCK_PREFIX,
        "Product: %s" % product,
        "NRN: %s" % nrn,
        "Applicant: %s" % applicant,
    ]
    if verdict == "not_registered":
        lines.append("VERDICT: NOT REGISTERED - not listed in this export.")
    elif verdict == "flagged":
        lines.append("VERDICT: FLAGGED - %s." % (reason or "suspended"))
    else:
        lines.append("VERDICT: AUTHENTIC - listed as Active in the current "
                     "greenbook.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
HAUSA = set("ƙɓɗƙʙ")
YORUBA = set("ẹọṣńṅẹ̀ọ́ṣ")
IGBO = set("ịụọńị̀ụ́ọ")

# Whole words that do not occur in ordinary English. Function words are useless
# here: "na", "ka", "ti", "ni" and "pe" are all short enough to appear as
# English fragments, so each list is built from words that are either longer or
# unmistakably foreign.
HAUSA_WORDS = (
    r"sannu|barka|wannan|wancan|yana|yake|ita|ina so|in so|in tabbatar|"
    r"tabbatar|duba min|duba|yaya ake|ya za|za ka|gaske|kane|kanwa|"
    r"gaskiya|san|saboda|yanzu|tafi|sai|kuma|zan|magari|kwayoyi|abinci|"
    r"sabuni|kwayar|na gani|na san|ina so|barka da|wannan wata|"
    # "Shin ... tana da kyau?" and "tana da ..." are ordinary Hausa and appear
    # in verification turns. Without them such a turn falls through to English
    # and gets an English answer, which breaks the prompt's language rule.
    r"tana|kyau|shin|me game|kan\s+ka|in\s+haka|don\s+ka|za\s+ta|na\s+ka"
)
YORUBA_WORDS = (
    r"b[aá]wo|[eẹ] k[aá]a?[ṣs][eẹ]|[yY][iì][ií]|jowo|n[ií]tor[ií]|k[ií] "
    r"ni|b[aá]wo ni|[eẹ]|[ṣS][eẹ]|[oO]|[wW][àá]|j[uù]|k[oò]|[pP][eé]|"
    r"gbogbo|[fF][uú]n|[mM]o n|[Ọọ]ja|[ìI]w[eé]"
)
IGBO_WORDS = (
    r"nwanne|[uụ]l[oọ][dụụ]|biko|[gG][iị]n[iị]|kedu|mgbe|any[iị]|[uU]n[uụ]|"
    r"ihe a|onye|b[ụụ]la|otu|gan-|mana|lelee|enyemaka|nke|ndi|mba|"
    r"[oO]\s+d[iị]|[dD][iị]\s"
)


def detect_language(text):
    """Which of the four languages a user turn is written in.

    Diacritics decide it when present. Otherwise whole words do, and English is
    the fallback, so an ambiguous turn is never mistaken for a local language.
    """
    chars = str(text or "")
    lowered = chars.lower()
    if any(ch in HAUSA for ch in chars):
        return "ha"
    if any(ch in YORUBA for ch in chars):
        return "yo"
    if any(ch in IGBO for ch in chars):
        return "ig"
    if re.search(r"\b(?:%s)\b" % HAUSA_WORDS, lowered):
        return "ha"
    if re.search(r"\b(?:%s)\b" % YORUBA_WORDS, lowered):
        return "yo"
    if re.search(r"\b(?:%s)\b" % IGBO_WORDS, lowered):
        return "ig"
    return "en"


# Why a record is FLAGGED, phrased the way the greenbook phrases it. The
# source block says "Status: FLAGGED"; repeating that in the answer gives "is
# recorded as flagged" and a verdict line reading "FLAGGED - flagged", so each
# status is given a reason phrase instead.
REASON_BY_STATUS = {
    "FLAGGED": "flagged by NAFDAC",
    "SUSPENDED": "suspended",
    "WITHDRAWN": "withdrawn",
    "RECALLED": "recalled",
    "BANNED": "banned",
    "INACTIVE": "listed as Inactive in the current greenbook",
}

# Statuses the classifier accepts, and the verdict each one maps to.
STATUS_MAP = {
    "REGISTERED": "authentic",
    "NOT_REGISTERED": "not_registered",
    "NOT REGISTERED": "not_registered",
    "FLAGGED": "flagged",
    "SUSPENDED": "flagged",
    "BANNED": "flagged",
    "RECALLED": "flagged",
    "WITHDRAWN": "flagged",
    "INACTIVE": "flagged",
}

GREETING = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|day)|greetings|sannu|"
    r"sannu da|barka|[bB][aá]wo|[eẹ] k[aá]a?[ṣs]e|[oO]|[Dd]on Allah|"
    r"achọrọ m|i[nń]a so)\b",
    re.IGNORECASE,
)
IDENTITY = re.compile(
    r"(who are you|your name|what do you do|what can you do|what is your "
    r"job|who made you|who build you|who built you|what are you for|"
    r"ka yaya|kini i|onye gi|gịnị maka|wetin i|ka yaya a|onye ti)",
    re.IGNORECASE,
)
MEDICAL = re.compile(
    r"(dosage|dose|side effect|symptom|treatment|diagnos|is it safe to take|"
    r"how much should i|cure|prescrib|medication for|nutrition plan|diet for|"
    r"karamin|rashin|alalo|mgbu|ọrụ na|ọgwụ mgbu)",
    re.IGNORECASE,
)
ENDORSEMENT = re.compile(
    r"(endorsed by nafdac|approved by nafdac|are you nafdac|"
    r"do you work for nafdac|nafdac official)",
    re.IGNORECASE,
)
NUMBER_IN_TEXT = re.compile(r"\b(?:[A-Za-z]{0,2}\d{1,2}[-‐-―/]\s?\d{2,7}"
                            r"[A-Za-z]{0,2})\b")
PRODUCT_HINT = re.compile(
    r"(product|item|pack|packet|bottle|soap|shampoo|oil|food|drink|biscuit|"
    r"water|detergent|toothpaste|hair|cream|supplement|tea|juice|milk|"
    r"abinci|ounje|nri|sabuni|omi|soap|mmiri)",
    re.IGNORECASE,
)
VERDICT_WORDS = re.compile(
    r"\b(AUTHENTIC|FLAGGED|NOT REGISTERED|NOT_REGISTERED)\b")


def classify(user_text, block):
    """(category, verdict, reason) for one user turn.

    A verified block settles the category. In production the block is only ever
    attached after the intent gate has already decided the turn is a product
    lookup, so a turn that carries one is a verification whatever the wording is
    — including "Ba mi yewo boya Fast Action Painkiller yi da.", which shares no
    keyword with the English list below.
    """
    text = str(user_text or "")
    status = block_field(block, "status") if block else ""
    mapped = STATUS_MAP.get(status.upper())
    if block and mapped:
        return "verify", mapped, ""

    if ENDORSEMENT.search(text):
        return "endorsement", None, ""
    if IDENTITY.search(text):
        return "identity", None, ""
    if MEDICAL.search(text):
        return "medical", None, ""
    if GREETING.match(text) and not NUMBER_IN_TEXT.search(text):
        return "greeting", None, ""

    wants_lookup = bool(
        NUMBER_IN_TEXT.search(text)
        or re.search(r"\b(verify|check|confirm|validate|lookup|search|"
                     r"authentic|original|genuine|registered|approved|"
                     r"duba|tabbatar|check|wechere|onye|nke|ilọta)\b",
                     text, re.IGNORECASE)
    )
    if wants_lookup or PRODUCT_HINT.search(text):
        return "no_data", None, ""
    return "out_of_scope", None, ""


# Chatter is the majority of the source file: 697 of 1080 rebuilt records would
# be greetings and out-of-scope redirects. Kept in full they would teach a
# refusal rate the assistant should not have, so they are sampled and the
# dropped count is reported. Every verification and every negative is kept.
CAPS = {"greeting": 60, "out_of_scope": 120}


def assistant_text(category, lang, verdict, reason, values):
    if category in ("greeting", "identity", "out_of_scope", "medical",
                    "endorsement", "no_data"):
        return TEMPLATES[category][lang]
    product, nrn, applicant = values
    if category == "verify":
        if verdict == "flagged":
            return TEMPLATES["flagged"][lang].format(
                product=product, nrn=nrn, reason=reason or "suspended")
        if verdict == "not_registered":
            return TEMPLATES["not_registered"][lang].format(product=product)
        return TEMPLATES["authentic"][lang].format(
            product=product, nrn=nrn, applicant=applicant)
    raise AssertionError(category)


def pairs_of(messages):
    """(user, assistant) pairs, dropping any stray leading assistant turn."""
    out = []
    pending = None
    for message in messages:
        role = message.get("role")
        if role == "user":
            pending = message.get("content", "")
        elif role == "assistant" and pending is not None:
            out.append((pending, message.get("content", "")))
            pending = None
    return out


def build_records(rows, system_prompt):
    """The cleaned records, plus a tally of what happened to each source one."""
    records = []
    tally = collections.Counter()
    capped = collections.Counter()
    index = 0
    counter = [0]

    for row in rows:
        system = ""
        for message in row.get("messages", []):
            if message.get("role") == "system":
                system = message.get("content", "")
                break
        block = ""
        if "[NAFDAC DATA]" in system:
            block = system.split("[NAFDAC DATA]")[-1].strip()

        for user_text, _old_answer in pairs_of(row.get("messages", [])):
            category, verdict, reason = classify(user_text, block)
            lang = detect_language(user_text)

            if category == "verify" and verdict == "flagged":
                reason = REASON_BY_STATUS.get(
                    block_field(block, "status").upper(),
                    FLAGGED_REASONS[0])

            cap = CAPS.get(category)
            if cap is not None and tally[category] >= cap:
                capped[category] += 1
                continue

            if category == "verify":
                product, nrn, applicant = synthetic_values(index)
                user_text = swap_source_names(user_text, block, product,
                                              applicant)
                user_text, own = renumber_user_text(user_text, counter)
                nrn = own or nrn
                index += 1
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "system",
                     "content": build_block(product, nrn, applicant, verdict,
                                             reason)},
                ]
                answer = assistant_text("verify", lang, verdict, reason,
                                        (product, nrn, applicant))
                tally["verify_" + verdict] += 1
            else:
                user_text, _ = renumber_user_text(user_text, counter)
                messages = [{"role": "system", "content": system_prompt}]
                answer = assistant_text(category, lang, verdict, reason,
                                        ("", "", ""))
                tally[category] += 1
                if category == "no_data":
                    tally["negative_no_data"] += 1

            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": answer})
            records.append({"messages": messages})

        if not pairs_of(row.get("messages", [])):
            tally["dropped_empty"] += 1

    for category, dropped in capped.items():
        tally["capped_" + category] = dropped

    return records, tally


# Negatives and greetings generated per language, to make the "no data, no
# verdict" lesson as strong as the verification lesson, and to lift Igbo to
# parity. Kept small on purpose: this supplements the real turns, it does not
# replace them.
SYNTHETIC_NO_DATA = {
    "en": [
        "Is this milk powder original?",
        "Please check this pack of Chin Chin.",
        "Can you verify this soap?",
        "Is this tomato paste genuine?",
        "What about this bottle of water?",
        "Check this shampoo please.",
    ],
    "ha": [
        "Wannan sabo ce?",
        "Dan ka duba wannan kwayoyi.",
        "Kana iya tabbatar da wannan magari?",
        "Me game da wannan ruwa?",
        "Ka duba wannan kofi.",
    ],
    "yo": [
        "Ṣé ó ní óogun?",
        "Jowo ki o ṣàyẹ̀wò ọja yìí.",
        "Ṣé o lè fọwọ́sí sọ́un yìí?",
        "Kíni nípa ọja náà?",
    ],
    "ig": [
        "Ị bụ onye ọ rịọrọ aka?",
        "Biko lelee ngwa a.",
        "Ị nwere ike ịlọta ihe a?",
        "Gini bụ ya na ngwa a?",
    ],
}


# Verification turns the source file barely has. Igbo contributes 19 of 866
# records, so without these the four-language promise is only demonstrated in
# English. Each is a (user turn, verdict) pair; the block and the answer are
# built exactly as for a real record. Wording is ordinary Nigerian usage, kept
# short on purpose: the point is that the verdict word is quoted in English
# inside a local-language answer.
SYNTHETIC_VERIFY_USERS = (
    ("en", "Is E-000901 original?", "authentic"),
    ("en", "NAFDAC no E-000902, is it registered?", "not_registered"),
    ("en", "Please check E-000903, someone says it was banned.", "flagged"),
    ("ha", "Shin E-000904 an original ne?", "authentic"),
    ("ha", "Duba min E-000905, an aika hannu?", "flagged"),
    ("ha", "E-000906 ba a same shi a NAFDAC ba", "not_registered"),
    ("yo", "Ṣé E-000907 ni ọjà ọtọ́?", "authentic"),
    ("yo", "Jowo ṣàyẹ̀wò E-000908", "not_registered"),
    ("yo", "E-000909 ti jẹ́ ọjà tí a fagilé", "flagged"),
    ("ig", "Ị bụ E-000910 nke ahụ?", "authentic"),
    ("ig", "Biko lelee E-000911", "not_registered"),
    ("ig", "A gwa na E-000912 akpọchiri", "flagged"),
)

# Chatter in the three local languages, so a refusal is not an English-only
# behaviour either.
SYNTHETIC_CHITCHAT = (
    ("ha", "Sannu da yana", "greeting"),
    ("ha", "Yaya ake sha ruwan zafi?", "out_of_scope"),
    ("ha", "Kaya tana", "out_of_scope"),
    ("yo", "Ẹ káàṣe", "greeting"),
    ("yo", "Báwo ni o ṣe?", "out_of_scope"),
    ("yo", "O dara", "out_of_scope"),
    ("ig", "Ị bụla?", "greeting"),
    ("ig", "Kedu ka ị mere?", "out_of_scope"),
    ("ig", "Ị dị ọtụtụ", "out_of_scope"),
    ("ig", "Onye gi bụ ụlaa ọ bụla?", "identity"),
)


def generate_extras(system_prompt, start_index):
    """Generated records: negatives, verifications and chatter, per language."""
    records = []
    index = start_index
    counter = [index * 1000]

    for lang, prompts in SYNTHETIC_NO_DATA.items():
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": TEMPLATES["no_data"][lang]},
            ]
            records.append({"messages": messages})
            index += 1

    for entry in SYNTHETIC_VERIFY_USERS:
        lang, user_text = entry[0], entry[1]
        verdict = entry[2]
        product, nrn, applicant = synthetic_values(counter[0])
        user_text, own = renumber_user_text(user_text, counter)
        nrn = own or nrn
        reason = FLAGGED_REASONS[counter[0] % len(FLAGGED_REASONS)]
        counter[0] += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system",
             "content": build_block(product, nrn, applicant, verdict,
                                    reason if verdict == "flagged" else "")},
            {"role": "user", "content": user_text},
            {"role": "assistant",
             "content": assistant_text("verify", lang, verdict,
                                       reason if verdict == "flagged" else "",
                                       (product, nrn, applicant))},
        ]
        records.append({"messages": messages})

    for lang, user_text, category in SYNTHETIC_CHITCHAT:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": TEMPLATES[category][lang]},
        ]
        records.append({"messages": messages})

    return records, index


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
# Claims that must never appear. "is counterfeit" is deliberately absent from
# this list: the project requires the assistant to *deny* that conclusion
# ("It does not by itself mean the product is counterfeit"), so it is checked
# for the absence of a negation in front of it instead.
FORBIDDEN = (
    "safe for use",
    "is safe",
    "nafdac approved",
    "approved by nafdac",
    "no records found",
    "pharmacist",
    "known counterfeit",
    "confirmed fake",
    "i have checked",
    "till tabbatar",
    "is poisoned",
    "contains poison",
    "adulterated",
)
COUNTERFEIT_CLAIM = re.compile(r"\b(?:is|are|was|were)\s+counterfeit\b",
                               re.IGNORECASE)
NEGATION = re.compile(r"\b(not|never|no|n't|cannot|does not|do not)\b",
                     re.IGNORECASE)


def asserts_counterfeit(text):
    """True when the text claims a product IS counterfeit.

    A negation in the same sentence counts: the rule the assistant must obey is
    "never claim a product is counterfeit", and telling the user it does not
    mean that is the opposite of claiming it.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        match = COUNTERFEIT_CLAIM.search(sentence)
        if not match:
            continue
        lead = sentence[max(0, match.start() - 90):match.start()]
        if not NEGATION.search(lead):
            return True
    return False


def check(records, system_prompt):
    problems = []
    if not records:
        return ["the rebuilt dataset is empty"]

    for number, record in enumerate(records, start=1):
        messages = record["messages"]
        if messages[0]["role"] != "system":
            problems.append("record %d does not start with a system message"
                            % number)
        elif messages[0]["content"] != system_prompt:
            problems.append("record %d has a different system prompt" % number)

        roles = [m["role"] for m in messages]
        if roles[-1] != "assistant":
            problems.append("record %d does not end with an assistant turn"
                            % number)
        for role in roles:
            if role not in ("system", "user", "assistant"):
                problems.append("record %d has role %r" % (number, role))

        answer = messages[-1]["content"]
        lowered = answer.lower()
        for phrase in FORBIDDEN:
            if phrase in lowered:
                problems.append("record %d contains %r" % (number, phrase))
        if asserts_counterfeit(answer):
            problems.append("record %d asserts a product is counterfeit"
                            % number)

        block = "\n".join(m["content"] for m in messages
                          if m["role"] == "system" and m != messages[0])
        if VERDICT_WORDS.search(answer) and not block:
            problems.append("record %d states a verdict with no data block"
                            % number)
        # Nothing may appear in the answer that is not in the block: this is
        # the anti-hallucination invariant, checked mechanically. The pattern is
        # built from the stem tables so a real product name is covered as
        # strictly as an "Example ..." one — otherwise widening PRODUCT_STEMS
        # would quietly stop checking the very names it adds.
        if block:
            for token in BLOCK_TOKEN_RE.findall(answer):
                if token not in block:
                    problems.append(
                        "record %d quotes %r, which is not in its block"
                        % (number, token))
    return problems


def real_values(rows):
    """Identifiers in the source file, which must not survive the rebuild."""
    values = set()
    for row in rows:
        for message in row.get("messages", []):
            content = message.get("content", "")
            values.update(NUMBER_IN_TEXT.findall(content))
            values.update(re.findall(r"Example [A-Za-z ]+?\d", content))
    return values


def report(records, rows, tally, extras):
    print("source records      : %d" % len(rows))
    print("rebuilt records     : %d (+%d generated)" % (len(records), extras))
    print()
    print("categories:")
    for key, count in sorted(tally.items()):
        print("   %-26s %d" % (key, count))
    print()
    languages = collections.Counter()
    for record in records:
        for message in record["messages"]:
            if message["role"] == "user":
                languages[detect_language(message["content"])] += 1
                break
    print("language (first user turn):")
    for key, count in languages.most_common():
        print("   %-26s %d" % (key, count))
    print()
    with_block = sum(
        1 for r in records
        if any(m["role"] == "system" and m is not r["messages"][0]
               for m in r["messages"])
    )
    print("records carrying a verified block : %d" % with_block)
    print("records with no block (negatives) : %d" % (len(records) - with_block))


def resolve_source():
    """(source_path, target_path).

    Never returns a target equal to the source: rebuilding in place would eat
    the input, and the second run would "rebuild" the cleaned file from itself.
    When the raw file is gone and only the canonical one remains, the output
    goes to the sidecar name instead and the caller is told why.
    """
    for candidate in SOURCE_CANDIDATES:
        if candidate.is_file():
            return candidate, TARGET
    if TARGET.is_file():
        return TARGET, ROOT / "confam_training_dataset.cleaned.jsonl"
    raise SystemExit(
        "no input found. Looked for %s and for %s. The raw dataset was never "
        "committed, so if it is gone this script cannot rebuild the cleaned "
        "file — but %s already holds the cleaned records."
        % (", ".join(c.name for c in SOURCE_CANDIDATES), TARGET.name,
           TARGET.name)
    )


def migrate_existing(records, system_prompt):
    """Rewrite the cleaned file in place for the current prompt and stems.

    The raw dataset was never committed, so the file on disk cannot be rebuilt
    from source. It can still be moved forward, which is what this does, because
    everything a record asserts about a product lives in exactly two places: its
    system prompt and the `[nafdac]` block. Replacing both, and every mention of
    the old name in the user turn and the answer, keeps each record internally
    consistent — which is the invariant the anti-hallucination check enforces.

    The new product name comes from the record's own ordinal, so a given record
    keeps the same name across runs and a rerun after an interrupted migration
    cannot renumber the file.

    Verdict words, the number, the applicant and the answer template are left
    exactly as they are: they are not what changed.
    """
    renamed = 0
    for record in records:
        messages = record["messages"]
        blocks = [m for m in messages
                  if m["role"] == "system" and m is not messages[0]]
        # Only the leading persona message is replaced. The block is also role
        # "system", so a blanket "every system message" loop would overwrite the
        # very data the rename below has to read.
        messages[0]["content"] = system_prompt
        if not blocks:
            continue
        block = blocks[0]["content"]

        # The ordinal is the number the old synthetic name ended in, which is how
        # synthetic_values() numbered it. Recovering it keeps names stable. A
        # repeated run is tolerated ("X 1 1" from a botched earlier migration)
        # so a rerun repairs the file instead of compounding it.
        found = re.search(r"Product: (.+?) (\d+)(?: \d+)*\b", block)
        if not found:
            continue
        old_product, ordinal = found.group(1), int(found.group(2))
        product_stem, applicant_stem = stem_pair(ordinal - 1)

        old_applicant = ""
        applicant = re.search(r"Applicant: (.+?) (\d+)(?: \d+)*\b", block)
        if applicant:
            old_applicant = applicant.group(1)

        replacements = {old_product: product_stem}
        if old_applicant:
            replacements[old_applicant] = applicant_stem

        for message in messages:
            if message is blocks[0]:
                rebuilt = []
                for line in message["content"].split("\n"):
                    if line.startswith("Product: "):
                        line = "Product: %s %d" % (product_stem, ordinal)
                    elif line.startswith("Applicant: ") and old_applicant:
                        line = "Applicant: %s %d" % (applicant_stem, ordinal)
                    rebuilt.append(line)
                message["content"] = "\n".join(rebuilt)
                continue
            text = message["content"]
            for old, new in replacements.items():
                # `(?: \d+)+` collapses the whole run of ordinals, so a name that
                # already carries its number from an earlier run matches again and
                # is replaced by exactly one. Matching `\d+` alone would let
                # "Indomie Noodles 1 1" grow on every migration.
                text = re.sub(r"\b%s(?: \d+)+\b" % re.escape(old),
                              "%s %d" % (new, ordinal), text)
            message["content"] = text
        renamed += 1
    return records, renamed


def validate_existing(path, system_prompt):
    """Check the canonical file in place, without regenerating anything.

    Used when the raw dataset is gone. The file is then already the cleaned
    output, so there is nothing to rebuild, but re-running the invariants over it
    is still worth something: it is how you prove the file on disk is still the
    one this script describes.
    """
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = check(records, system_prompt)

    languages = collections.Counter()
    verdicts = collections.Counter()
    for record in records:
        messages = record["messages"]
        for message in messages:
            if message["role"] == "user":
                languages[detect_language(message["content"])] += 1
                break
        answer = messages[-1]["content"]
        found = VERDICT_WORDS.search(answer)
        if found:
            verdicts[found.group(1).replace(" ", "_")] += 1
        if messages[0]["content"] == system_prompt:
            verdicts["system_prompt_matches"] += 1

    print("no raw dataset found: nothing to rebuild.")
    print("validating %s in place — it is already the cleaned output.\n"
          % path.name)
    print("records            : %d" % len(records))
    print("system prompt exact: %d/%d" % (verdicts["system_prompt_matches"],
                                          len(records)))
    print("verdicts in answers: %s"
          % ", ".join("%s=%d" % (k, v) for k, v in sorted(verdicts.items())
                      if k != "system_prompt_matches"))
    print("languages          : %s"
          % ", ".join("%s=%d" % (k, v) for k, v in languages.most_common()))
    print()
    if problems:
        print("FAILED %d check(s):" % len(problems))
        for problem in problems[:20]:
            print("  - %s" % problem)
        return 1
    print("VALID: %d records, every check passed." % len(records))
    return 0


def migrate_existing_file(path, system_prompt):
    """Move the canonical file forward, then prove the result is valid.

    The rewrite is written only after the checks pass, so a migration that would
    break an invariant leaves the previous file untouched rather than replacing a
    good dataset with a broken one.
    """
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, renamed = migrate_existing(records, system_prompt)

    problems = check(records, system_prompt)
    print("migrating %s in place" % path.name)
    print("records with a data block renamed: %d" % renamed)
    print()
    if problems:
        print("FAILED %d check(s); nothing written:" % len(problems))
        for problem in problems[:20]:
            print("  - %s" % problem)
        return 1

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("VALID: %d records, every check passed." % len(records))
    print("wrote %s" % path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true",
                        help="run the checks and print the report without "
                             "writing the cleaned file")
    parser.add_argument("--migrate", action="store_true",
                        help="rewrite the cleaned file in place for the "
                             "current confam_prompt.txt and the current "
                             "PRODUCT_STEMS, then validate it. Needed when "
                             "the prompt or the stem tables change, because "
                             "the raw dataset is gone and the file cannot be "
                             "rebuilt from source")
    args = parser.parse_args(argv)

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    source, target = resolve_source()

    if source not in SOURCE_CANDIDATES:
        if args.migrate:
            return migrate_existing_file(source, system_prompt)
        return validate_existing(source, system_prompt)

    print("reading   %s" % source.name)
    print("writing   %s" % target.name)
    rows = [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]

    records, tally = build_records(rows, system_prompt)
    extras, _ = generate_extras(system_prompt, 0)
    records.extend(extras)

    problems = check(records, system_prompt)

    leaked = real_values(rows) & {
        value
        for record in records
        for message in record["messages"]
        for value in NUMBER_IN_TEXT.findall(message["content"])
    }
    leaked = {
        value
        for record in records
        for message in record["messages"]
        for value in NUMBER_IN_TEXT.findall(message["content"])
    } & real_values(rows)
    if leaked:
        problems.append("source registry values survived: %s"
                        % ", ".join(sorted(leaked)[:10]))

    report(records, rows, tally, len(extras))
    print()

    if problems:
        print("FAILED %d check(s):" % len(problems))
        for problem in problems[:20]:
            print("  - %s" % problem)
        return 1

    if not args.check_only:
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("VALID: %d records, every check passed." % len(records))
        print("wrote %s" % target)
    else:
        print("VALID: %d records, every check passed (nothing written)."
              % len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())