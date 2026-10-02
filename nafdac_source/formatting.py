"""Terminal and prompt formatting for NAFDAC results.

Kept apart from `verify.py` so the presentation of a result can change without
touching the lookup logic, and so the same result can be rendered three ways:

    format_verification()    a table for a terminal
    VerificationResult.render()   a plain block for the model's context
    format_search_results()  a ranked list for /search

All three are plain text with no markdown syntax, because the model's block is
fed straight into a prompt and a stray `**` would reach the user verbatim.

Localisation
------------
`format_*` is terminal output, so it takes a `lang` and renders its labels and
prose in that language. Two rules keep that honest:

    1. Values are never translated. A product name, an applicant, a strength and
       an approval date are printed exactly as the greenbook holds them. Only
       labels, verdicts and sentences change.
    2. `VerificationResult.render()` and `render_record()` are untouched, and
       always English. That text is fed to the model as verified data, so
       putting a translation in front of a verifier is how a wrong value gets
       reported as fact.

`_t` falls back to English one key at a time, so a translation that is missing
a key degrades to a readable English label rather than printing a key name at
someone who is checking whether a medicine is genuine.
"""

from typing import Iterable, List, Optional

from .records import (
    VERDICT_AUTHENTIC,
    VERDICT_FLAGGED,
    ProductRecord,
)

# Terminal width for the boxes below. 78 keeps the table inside an 80-column
# window, which is still the common case in cmd.exe.
WIDTH = 78

SUPPORTED_LANGUAGES = ("en", "ha", "yo", "ig")

# Keys are shared by all four tables on purpose: adding a string means adding it
# once here and four times in the tables, and the test suite checks that every
# language answers every key.
STRINGS = {
    "en": {
        "verdict_authorised": "AUTHORISED",
        "verdict_not_authorised": "NOT CURRENTLY AUTHORISED",
        "verdict_unknown": "STATUS NOT RECORDED",
        "status_authorised": "AUTHORISED",
        "status_not_authorised": "NOT AUTHORISED",
        "status_unknown": "STATUS NOT RECORDED",
        "label_number": "Number",
        "label_ingredients": "Ingredients",
        "label_strength": "Strength",
        "label_form": "Form",
        "label_category": "Category",
        "label_marketing": "Marketing",
        "label_applicant": "Applicant",
        "label_manufacturer": "Manufacturer",
        "label_pack": "Pack size",
        "label_approved": "Approved",
        "label_expires": "Expires",
        "label_atc": "ATC code",
        "label_status": "Status",
        # "Greenbook" is NAFDAC's own name for the register, so it stays as one
        # word in every language.
        "label_greenbook": "Greenbook",
        "label_confidence": "Confidence",
        "label_unavailable": "Unavailable",
        "label_note": "Note",
        "label_source": "Source",
        "label_greenbook_page": "Greenbook page",
        "not_recorded": "not recorded",
        "name_not_recorded": "(name not recorded)",
        "duplicate_entry": "duplicate entry",
        "no_number": "no number",
        "headline": "NAFDAC {nrn} - {verdict}",
        "headline_ambiguous": "NAFDAC {nrn} - {count} different products share this number:",
        "not_found": "NAFDAC {nrn} - not found in the greenbook.",
        "not_found_absent": "This number is not in any data source N-ATLaS can reach.",
        "not_found_checked": "Checked: {sources}",
        "not_found_1": "That is not proof the product is fake. The export may be",
        "not_found_2": "out of date, or the number may be mistyped. Check the",
        "not_found_3": "number against the pack, and try the search command if the product has a different number.",
        "no_number_in_message": "No NAFDAC number found in that. It looks like A4-100160 or 04-3877 - the number NAFDAC printed on the pack.",
        "search_none": "No product matched {query!r}. Try a shorter word, or an active ingredient such as 'paracetamol' or 'artemether'.",
        "search_matched": "{count} product(s) matched {query!r}{suffix}:",
        "search_suffix": " (showing {shown})",
        "search_verify_hint": "Verify one with: /verify <number>",
        "status_title": "NAFDAC data sources",
        "status_note_1": "The PostgreSQL source carries superseded and withdrawn",
        "status_note_2": "products. Without it, a number that has since been",
        "status_note_3": "withdrawn will not be distinguishable from one that never existed.",
        "usage_verify": "Usage: /verify <nafdac number>   e.g. /verify A4-100160",
        "usage_search": "Usage: /search <text>   e.g. /search malaria",
        "help_title": "NAFDAC verification",
        "help_verify": "  /verify <number>   Check one NAFDAC number, e.g. /verify A4-100160",
        "help_search": "  /search <text>     Search products by name or ingredient, e.g. /search malaria",
        "help_nafdac": "  /nafdac            Show which data sources are available",
        "help_1": "You can also just ask in your own words. \"Is A4-100160 genuine?\" is",
        "help_2": "verified against the greenbook before N-ATLaS answers, and the model is",
        "help_3": "given the verified values so it cannot invent them.",
        # A product-name search, for `/verify <product name>`.
        "verdict_line": "VERDICT: {verdict} - {reason}",
        "verdict_not_listed": "VERDICT: NOT REGISTERED - not listed in this greenbook export.",
        "verdict_unknown_multi": "VERDICT: UNKNOWN - these are different products, so this is not an answer about one pack.",
        "search_one": "Greenbook search for {query}: one product matched.",
        "search_many": "{count} products matched {query} (showing {shown}):",
        "ask_which": "  These are different products. Which one did you mean?",
        "flagged_count": "{count} of them are flagged by NAFDAC.",
        "registered_caveat": "The greenbook lists this as currently authorised. It is registered, but the greenbook cannot confirm this particular pack is genuine: a copied number resolves here too.",
        "flagged_warn": "This product is flagged and should not be used. The reason is above.",
        "search_not_listed_1": "This is not proof the product is counterfeit. The export may be out of date, or the name may be spelled differently on the pack.",
        "search_not_listed_2": "Give the NAFDAC number printed on the pack, which looks like A4-100160 or 04-3877, and it can be checked directly.",
        "more_not_shown": "...and {count} more not shown.",
        "searched_none": "Nothing to look up. Give a NAFDAC number such as A4-100160, or a product name such as paracetamol.",
        "usage_verify_words": "Usage: /verify <number or product name>   e.g. /verify A4-100160",
        "usage_verify_words_short": "Usage: /verify <number or product name>",
    },
    "ha": {
        "verdict_authorised": "AN AMMA",
        "verdict_not_authorised": "BA A AMMA A YANZU",
        "verdict_unknown": "BAI KUNYA AN KAYYALE HALI",
        "status_authorised": "AN AMMA",
        "status_not_authorised": "BA A AMMA",
        "status_unknown": "BAI KUNYA AN KAYYALE HALI",
        "label_number": "Lamba",
        "label_ingredients": "Maimaitawa",
        "label_strength": "Karfi",
        "label_form": "Tsari",
        "label_category": "Sifa",
        "label_marketing": "Tsiyar",
        "label_applicant": "Mai neman",
        "label_manufacturer": "Mai yin",
        "label_pack": "Girma",
        "label_approved": "An amma",
        "label_expires": "Zai kare",
        "label_atc": "Lamba ATC",
        "label_status": "Hali",
        "label_greenbook": "Greenbook",
        "label_confidence": "Tabbaci",
        "label_unavailable": "Babu",
        "label_note": "Bayanai",
        "label_source": "Tushen",
        "label_greenbook_page": "Shafi na Greenbook",
        "not_recorded": "ba a kayyale",
        "name_not_recorded": "(sunan ba a kayyale)",
        "duplicate_entry": "shafi mai yawa",
        "no_number": "babu lamba",
        "headline": "NAFDAC {nrn} - {verdict}",
        "headline_ambiguous": "NAFDAC {nrn} - {count} kayawa ne masu bambancin suna amfani da wannan lamba:",
        "not_found": "NAFDAC {nrn} - ba a same shi a cikin Greenbook.",
        "not_found_absent": "Wannan lamba ba ta cikin ko wata bayanai da N-ATLaS zai iya samu.",
        "not_found_checked": "An duba: {sources}",
        "not_found_1": "Wannan ba tabbacin cewa gurbin hakan ne ba.",
        "not_found_2": "Fasashar da aka fita na iya girmama kwanaki, ko an yi kuskure a lamba.",
        "not_found_3": "Duba lambar da ke a kafin sa, ka kuma gwada umarar bincike idan wannan samfur ya samu lamba dabam.",
        "no_number_in_message": "Ba a samu lamba NAFDAC a cikin wannan. Kamar yadda A4-100160 ko 04-3877 yake - lambar da NAFDAC ya buga a karkashin.",
        "search_none": "Babin samfur da ya yi dawa da {query!r}. Ka gwada kalma mai girma, ko wata maimaitawa mai aiki kamar 'paracetamol' ko 'artemether'.",
        "search_matched": "{count} samfur sun yi dawa da {query!r}{suffix}:",
        "search_suffix": " (ana nuna {shown})",
        "search_verify_hint": "Ka duba wannan da: /verify <lamba>",
        "status_title": "Mabulbin NAFDAC",
        "status_note_1": "Tushen PostgreSQL yana ɗauki samfura da aka maye da suka ja",
        "status_note_2": "daga riji. tare da shi, lambar da aka janye ba za ta iya",
        "status_note_3": "bambanta da wacda ta taba nan ba.",
        "usage_verify": "Yadda ake amfani: /verify <lamba NAFDAC>  misali /verify A4-100160",
        "usage_search": "Yadda ake amfani: /search <rubutacci>  misali /search malaria",
        "help_title": "Tabbatar da NAFDAC",
        "help_verify": "  /verify <lamba>   Duba lamba NAFDAC guda, misali /verify A4-100160",
        "help_search": "  /search <rubutacci>     Nemo samfura da suna ko maimaitawa, misali /search malaria",
        "help_nafdac": "  /nafdac            Nuna mabulbin bayanai da ke",
        "help_1": "Kuma kana iya tambaya kawai a cikin kalmominka.",
        "help_2": "An tabbatar da shi kafin N-ATLaS ya amsa, anu kuma ba wa",
        "help_3": "kanonai gaskiya ko don hakan ya iya ɗauka.",
        "verdict_line": "HUKUCI: {verdict} - {reason}",
        "verdict_not_listed": "HUKUCI: BA A LISSACE - ba a lissace shi a cikin wannan bayanani.",
        "verdict_unknown_multi": "HUKUCI: BABU AMMA - wa da bambancin ne, wannan ba amsa game da wata kofiya.",
        "search_one": "Bincike na Greenbook don {query}: samfur daya ya yi dawa.",
        "search_many": "{count} kayawa sun yi dawa da {query} (ana nuna {shown}):",
        "ask_which": "  Suna bambancin ne. Wa kake nema?",
        "flagged_count": "{count} daga cikin su an zage su.",
        "registered_caveat": "Greenbook ya lissace shi a yanzu. An ruce shi, amma Greenbook ba za iya tabbatar da gurbin wannan kofiya ba: lamba da a kwanta za ta same shi a nan ma.",
        "flagged_warn": "An zage wannan samfur, kada a yi amfani da shi. Dalilin yana sama.",
        "search_not_listed_1": "Wannan ba tabbacin cewa gurbin hakan ne ba. Bayanan na iya tsafe, ko kuma sunan ya bambanta a karkashin kofiya.",
        "search_not_listed_2": "Ka ba da lambar NAFDAC da ke a karkashin kofiya, kamar A4-100160 ko 04-3877, za a iya duba shi kai tsaye.",
        "more_not_shown": "...da {count} karfe da ba a nuna ba.",
        "searched_none": "Babin abin da za a duba. Ka ba da lamba NAFDAC kamar A4-100160, ko samarfi kamar paracetamol.",
        "usage_verify_words": "Yadda ake amfani: /verify <lamba ko samarfi>  misali /verify A4-100160",
        "usage_verify_words_short": "Yadda ake amfani: /verify <lamba ko samarfi>",
    },
    "yo": {
        "verdict_authorised": "A TI FI LỌ́WỌ́",
        "verdict_not_authorised": "A TI JẸ́ KÌ Í FI LỌ́WỌ́",
        "verdict_unknown": "IPO KÌ NÍ",
        "status_authorised": "A FI LỌ́WỌ́",
        "status_not_authorised": "A TI JẸ́ KÌ Í FI LỌ́WỌ́",
        "status_unknown": "IPO KÌ NÍ",
        "label_number": "Nọ́mbà",
        "label_ingredients": "Àwọn àrọ́",
        "label_strength": "Agbára",
        "label_form": "Ìwọ̀n",
        "label_category": "Ẹ̀ka",
        "label_marketing": "Ìtajà",
        "label_applicant": "Olùbẹ̀rè",
        "label_manufacturer": "Olùṣe",
        "label_pack": "Ìwọ̀n àpò",
        "label_approved": "A fi lọ́wọ́",
        "label_expires": "Yóò parí",
        "label_atc": "Kóòdù ATC",
        "label_status": "Ipò",
        "label_greenbook": "Greenbook",
        "label_confidence": "Ìdánilójú",
        "label_unavailable": "Kò sí",
        "label_note": "Àkíyèsí",
        "label_source": "Ìtumọ̀",
        "label_greenbook_page": "Ojú ìwé Greenbook",
        "not_recorded": "a kò kíyèsí",
        "name_not_recorded": "(orúkọ a kò kíyèsí)",
        "duplicate_entry": "ẹ̀yì ìdupẹ́",
        "no_number": "kò sí nọ́mbà",
        "headline": "NAFDAC {nrn} - {verdict}",
        "headline_ambiguous": "NAFDAC {nrn} - {count} ọjarí ọ̀tọ̀ọ̀kan ń pín nọ́mbà náà:",
        "not_found": "NAFDAC {nrn} - a kò rí i nínú Greenbook.",
        "not_found_absent": "Nọ́mbà náà kò sí nínú èyíkéyíí orísun afojúmọ̀ tí N-ATLaS lè dédé.",
        "not_found_checked": "A ṣàyẹ̀wò: {sources}",
        "not_found_1": "Èyí kì í ṣe ìlọ́wọ́ pé ó ní ìdánwò jẹ́.",
        "not_found_2": "Àtìjẹ́dá tí a fì jáde kó̀ lè ti tẹ̀ ọjọ́, tàbí kí a kò kọ́ nọ́mbà náà tàbí.",
        "not_found_3": "Wo nọ́mbà náà sí eré náà, kí o sì gbìyànjú àṣà wáàjí tí ọja náà ní nọ́mbà mìíràn.",
        "no_number_in_message": "A kò rí nọ́mbà NAFDAC nínú èyí. Ó dà bí A4-100160 tàbí 04-3877 - nọ́mbà tí NAFDAC tẹ̀ sílẹ̀ lórí.",
        "search_none": "Kò sí ọja tí ó bá ń ṣe àdàrò {query!r}. Gbìyànjú ọ̀rọ̀ kúkúrú, tàbí àrọ́ ọ̀rọ̀ tí ń ṣe àdàrò bí 'paracetamol' tàbí 'artemether'.",
        "search_matched": "{count} ọja ṣe àdàrò {query!r}{suffix}:",
        "search_suffix": " (à ń fi {shown} nìkan hàn)",
        "search_verify_hint": "Ṣàyẹ̀wò ọ̀kan nínú: /verify <nọ́mbà>",
        "status_title": "Orísun afojúmọ̀ NAFDAC",
        "status_note_1": "Orísun PostgreSQL gbé àwọn ọja tí a ti fagilé tàbí",
        "status_note_2": "tí a ti mú kúrú lọ. Lẹ́yìn náà, nọ́mbà tí",
        "status_note_3": "a mú kúrú kò lè yà sọ́ ọ̀kan tí kò sí tẹ́lẹ̀.",
        "usage_verify": "Ilo lórí: /verify <nọ́mbà NAFDAC>  fún àpẹẹẹrẹ /verify A4-100160",
        "usage_search": "Ilo lórí: /search <ọ̀rọ̀>  fún àpẹẹẹrẹ /search malaria",
        "help_title": "Ìfẹ̀rí NAFDAC",
        "help_verify": "  /verify <nọ́mbà>   Wo nọ́mbà NAFDAC kan, fún àpẹẹẹrẹ /verify A4-100160",
        "help_search": "  /search <ọ̀rọ̀>     Wá ọja pẹ̀lú orúkọ tàbí àrọ́, fún àpẹẹẹrẹ /search malaria",
        "help_nafdac": "  /nafdac            Fi ìtumọ̀ orísun afojúmọ̀ tí wà",
        "help_1": "O tún bẹ̀rè pẹ̀lú ọ̀rọ̀ rẹ ara. \"A4-100160 bá tọ́ jẹ́?\"",
        "help_2": "a fi ṣàyẹ̀wò nígbà tí N-ATLaS ṣe àdá, a sì fi",
        "help_3": "ìyẹn ọ̀nà tó dára lọ́wọ́ sókè ló bá lè ṣe àrékérékè.",
        "verdict_line": "ÌDÁ: {verdict} - {reason}",
        "verdict_not_listed": "ÌDÁ: KÌ FI FORÚKÌSÌLẸ̀ - kò sí nínú àtúnjade Greenbook yìí.",
        "verdict_unknown_multi": "ÌDÁ: KÌ MÒ - àwọn ọjọ́ ọ̀tọ̀ọ̀kan ni, nítorí náà kì í ṣe àdápa sí ọja kan.",
        "search_one": "Ìwájì Greenbook fún {query}: ọ̀kan ọja bá pèdá.",
        "search_many": "{count} ọja bá pèdá {query} (à ń fi {shown} nìkan hàn):",
        "ask_which": "  Wọ́n jẹ́ ọjọ́ ọ̀tọ̀ọ̀kan. Ọ̀nà ni o fẹ́rẹ?",
        "flagged_count": "{count} nínú wọ́n ni a ti fi ọkà àmì sí.",
        "registered_caveat": "Greenbook ti fi ọja náà lọ́wọ́ nísinsìnyí. A forúkìsílẹ̀, ṣùgbọ́n Greenbook kò lè fẹ̀rí àdàrà pé inú ìwé yìí ni ọ̀tọ̀: nọ́mbà tí a bá ṣe àkọ́pọ̀ níbí náà ni a máa rí.",
        "flagged_warn": "A ti fi ọmà ọkà àmì sí ọja yìí, kí o má ṣe lò. Ìdí wà lókè.",
        "search_not_listed_1": "Èyí kì í ṣe ìlọ́wọ́ pé ó ní ìdánwò jẹ́. Àtìjẹ́dá kó là lẹ́tà, tàbí kí orúkọ náà kò bá kọ́ lórí ẹ̀pò.",
        "search_not_listed_2": "Fi nọ́mbà NAFDAC tí ó wà lórí ẹ̀pò náà dọ́ o, bí A4-100160 tàbí 04-3877, a lè ṣàyẹ̀wò rẹ̀ tààrà.",
        "more_not_shown": "...àti {count} mìíràn tí a kò fi hàn.",
        "searched_none": "Kò sí ohun tí a fẹ́ ṣàyẹ̀wò. Fi nọ́mbà NAFDAC bí A4-100160, tàbí orúkọ ọja bí paracetamol.",
        "usage_verify_words": "Ilo lórí: /verify <nọ́mbà tàbí orúkọ ọja>  fún àpẹẹẹrẹ /verify A4-100160",
        "usage_verify_words_short": "Ilo lórí: /verify <nọ́mbà tàbí orúkọ ọja>",
    },
    "ig": {
        "verdict_authorised": "E KWADORO LA",
        "verdict_not_authorised": "E NỌGHỊ EKWEORO NTA",
        "verdict_unknown": "ỌNWEKWERE ỌBỤNỤỌ̀ NÀ À",
        "status_authorised": "E KWADORO LA",
        "status_not_authorised": "E NỌGHỊ EKWEORO NTA",
        "status_unknown": "ỌNWEKWERE ỌBỤNỤỌ̀ NÀ À",
        "label_number": "Nọ́mba",
        "label_ingredients": "Ihe ndị akịrị",
        "label_strength": "Ebe ọrụ",
        "label_form": "Usoro",
        "label_category": "Klass",
        "label_marketing": "Mbụ",
        "label_applicant": "Onye na-achọọ ọrụ",
        "label_manufacturer": "Onye na-eme",
        "label_pack": "Nkụwu",
        "label_approved": "A kwadoro ya",
        "label_expires": "Ọ ga-adịghị",
        "label_atc": "Koodu ATC",
        "label_status": "Ọnwekwere",
        "label_greenbook": "Greenbook",
        "label_confidence": "Ụdọ aka",
        "label_unavailable": "Ọ nweghị",
        "label_note": "Nkọwa",
        "label_source": "Ebe si",
        "label_greenbook_page": "Peeji Greenbook",
        "not_recorded": "akwala",
        "name_not_recorded": "(aha a na-akwala)",
        "duplicate_entry": "inye dị ọzọ",
        "no_number": "enweghị nọ́mba",
        "headline": "NAFDAC {nrn} - {verdict}",
        "headline_ambiguous": "NAFDAC {nrn} - ebe ahụ dị naanị ụdọ {count} na-eji nọ́mba a:",
        "not_found": "NAFDAC {nrn} - a chọtaghị ya na Greenbook.",
        "not_found_absent": "Nọ́mba a dịghị na nke ọ bụla nke N-ATLaS nwere ike ịrụ.",
        "not_found_checked": "Anyị chèta: {sources}",
        "not_found_1": "Nke a adịghị ọ bụla ihe na-eme na ọ bụla ihe.",
        "not_found_2": "Mbụkọ na abụọ nwere ike ọ bụla ochie, maọbụ na ị ga-ehie nọ́mba.",
        "not_found_3": "Lelee nọ́mba ahụ na nkata nke ya, gbalịa ka anyị gbalịa iji okwu chọọ rọrọ ọzọ ma ọ bụrụ na ihe ahụ nwere nọ́mba ọzọ.",
        "no_number_in_message": "A chọtaghị nọ́mba NAFDAC na nke a. Ọ dị ka A4-100160 maọbụ 04-3877 - nọ́mba NAFDAC dega ya na nkata.",
        "search_none": "Ọ dịghị ngwa ihe dakọtara {query!r}. Gbalịa okwu dị mkrumike, maọbụ ihe ọrụ ọrụ dị ọrụka dị ka 'paracetamol' maọbụ 'artemether'.",
        "search_matched": "{count} ngwa ihe dakọtara {query!r}{suffix}:",
        "search_suffix": " (anyị na-egbu {shown} kama)",
        "search_verify_hint": "Gbado otu site na: /verify <nọ́mba>",
        "status_title": "Ebe ọrụ NAFDAC",
        "status_note_1": "Ebe ọrụ PostgreSQL nwere ngwa ihe enweghị ike ma ọ bụ",
        "status_note_2": "ewetara. Ya mere, nọ́mba a bụla ewetara n’oge",
        "status_note_3": "gị ga-abụ otu ị na-apụta ọzọ.",
        "usage_verify": "Ịji: /verify <nọ́mba NAFDAC>   dụ́ mba /verify A4-100160",
        "usage_search": "Ịji: /search <okwu>   dụ́ mba /search malaria",
        "help_title": "Nkwenye NAFDAC",
        "help_verify": "  /verify <nọ́mba>   Gbado nọ́mba NAFDAC otu, dụ́ mba /verify A4-100160",
        "help_search": "  /search <okwu>     Chọta ngwa ihe site na aha maọbụ ihe ndị akịrị, dụ́ mba /search malaria",
        "help_nafdac": "  /nafdac            Gbu ebe ọrụ dị",
        "help_1": "Ị nwere ike ịjụọ site na okwu gị. \"A4-100160 bụ eziọlụ?\" a",
        "help_2": "gbado ya tupu N-ATLaS anya, a site na",
        "help_3": "mgbe eziọlụ dị ezo dị ọ",
        "verdict_line": "EMECHI: {verdict} - {reason}",
        "verdict_not_listed": "EMECHI: AKWADOGHIHIE - ọ dịghị na nkeji export a.",
        "verdict_unknown_multi": "EMECHI: EBE EZI - ndị dị ọzọ nke ukwuu, nke a ọ bụghị ọdịnaya maka otu nka.",
        "search_one": "Nchọka Greenbook maka {query}: otu ngwa ihe dakọtara.",
        "search_many": "{count} ngwa ihe dakọtara {query} (anyị na-egbu {shown} kama):",
        "ask_which": "  Ndi a dị ọzọ nke ukwuu. Ị chọrị nke?",
        "flagged_count": "{count} n'ime ha akpọcha.",
        "registered_caveat": "A kwadoro ya na Greenbook ugbu a. A debere ya, mana Greenbook nwere ike ịkwado na nka a nke ukwuu bụ eziọlụ: nọ́mba nke a kpọtụrụ a na-akpọta ebe a.",
        "flagged_warn": "A akpọcha ngwa ihe a, ka ị ghara ịji ya. Ihe mere dị n'elu.",
        "search_not_listed_1": "Nke a adịghị ọ bụla ihe na-eme na ọ bụla ihe. Mbụkọ na abụọ nwere ike ochie, maọbụ na aha dị ọzọ na akpụ ya.",
        "search_not_listed_2": "Nyere nọ́mba NAFDAC dị na akpụ ya, dịka A4-100160 maọbụ 04-3877, a ga-esi gbado ya ozugbo.",
        "more_not_shown": "...na {count} ọzọ anyị họghị gbu.",
        "searched_none": "Ọ dịghị ihe anyị ga-eme. Nyere nọ́mba NAFDAC dịka A4-100160, maọbụ aha ngwa ihe dịka paracetamol.",
        "usage_verify_words": "Ịji: /verify <nọ́mba maọbụ aha ngwa ihe>   dụ́ mba /verify A4-100160",
        "usage_verify_words_short": "Ịji: /verify <nọ́mba maọbụ aha ngwa ihe>",
    },
}

# The English help block, kept as a module constant because the Kaggle server
# and older callers read it as one. `nafdac_help()` is the localised entry.
NAFDAC_HELP = "\n".join(
    [
        STRINGS["en"]["help_title"],
        STRINGS["en"]["help_verify"],
        STRINGS["en"]["help_search"],
        STRINGS["en"]["help_nafdac"],
        "",
        STRINGS["en"]["help_1"],
        STRINGS["en"]["help_2"],
        STRINGS["en"]["help_3"],
    ]
)


def _t(lang: str, key: str, **kwargs) -> str:
    """One string in one language, falling back to English key by key.

    The fallback is per key rather than per table on purpose: a translation
    that is missing one sentence still shows that sentence in English instead
    of showing the reader a variable name.
    """
    table = STRINGS.get(str(lang or "").strip().lower()) or {}
    text = table.get(key) or STRINGS["en"].get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def nafdac_help(lang: str = "en") -> str:
    """The /nafdac help block in `lang`."""
    return "\n".join(
        [
            _t(lang, "help_title"),
            _t(lang, "help_verify"),
            _t(lang, "help_search"),
            _t(lang, "help_nafdac"),
            "",
            _t(lang, "help_1"),
            _t(lang, "help_2"),
            _t(lang, "help_3"),
        ]
    )


def format_usage_verification(lang: str = "en") -> str:
    """The `/verify` with no argument message. Lives here so all NAFDAC prose
    is one table, rather than split between this package and the client."""
    return _t(lang, "usage_verify")


def format_usage_search(lang: str = "en") -> str:
    """The `/search` with no argument message."""
    return _t(lang, "usage_search")


def format_usage_verification_words(lang: str = "en") -> str:
    """The `/verify <product name>` usage message."""
    return _t(lang, "usage_verify_words")


def format_usage_verification_words_short(lang: str = "en") -> str:
    """The same usage message without the example, for the word-search path."""
    return _t(lang, "usage_verify_words_short")


def format_searched_nothing(lang: str = "en") -> str:
    """What to say when a message held neither a number nor a product name."""
    return _t(lang, "searched_none")


def format_no_number_found(lang: str = "en") -> str:
    """What to say when a /verify argument carried no NAFDAC number at all."""
    return _t(lang, "no_number_in_message")


def format_search_verdict(result, limit: int = 10, show_data: bool = True,
                          lang: str = "en") -> str:
    """Render a product-name search as candidates plus a verdict.

    The terminal counterpart of `SearchResult.render()`: same candidate list,
    same rules, but for a person rather than for the model. Two behaviours are
    deliberate and come straight from the data model:

      - with one candidate the verdict is meaningful, so it is stated
      - with several it is not, so the verdict is UNKNOWN and the user is asked
        which product they meant rather than handed the top-ranked one

    A miss says the export does not list the product. It never says counterfeit.

    `verdict_reason` is machine-generated prose from the record layer and stays
    English inside a translated verdict line; the verdict word itself and every
    sentence around it are localised.
    """
    if not result or not result.searched:
        return format_searched_nothing(lang)

    if not result.records:
        lines = [
            _t(lang, "not_found", nrn=result.query or ""),
            _rule("="),
            "",
            "  " + _t(lang, "search_not_listed_1"),
            "  " + _t(lang, "search_not_listed_2"),
        ]
        if result.sources_checked:
            lines.insert(3, "  " + _t(lang, "not_found_checked",
                                      sources=", ".join(result.sources_checked)))
        failed = _format_failures(result, lang)
        if failed:
            lines.insert(3, failed)
        return "\n".join(lines)

    shown = result.records[: max(1, limit)]
    lines: List[str] = []
    if len(shown) == 1:
        record = shown[0]
        lines.append(_t(lang, "search_one", query=result.query))
        lines.append(
            "  " + _t(lang, "verdict_line", verdict=record.verdict.upper(),
                      reason=record.verdict_reason)
        )
    else:
        lines.append(_t(lang, "search_many", count=result.total or len(result.records),
                        query=result.query, shown=len(shown)))
        lines.append("  " + _t(lang, "verdict_unknown_multi"))
        lines.append(_t(lang, "ask_which"))
        if result.flagged_count:
            lines.append(_t(lang, "flagged_count", count=result.flagged_count))
    lines.append(_rule("="))

    for record in shown:
        lines.extend(_format_record(record, show_data, lang))
        lines.append("")
    if len(result.records) > len(shown):
        lines.append("  " + _t(lang, "more_not_shown",
                               count=len(result.records) - len(shown)))

    if len(shown) == 1:
        record = shown[0]
        if record.verdict == VERDICT_AUTHENTIC:
            lines.append("  " + _t(lang, "registered_caveat"))
        elif record.verdict == VERDICT_FLAGGED:
            lines.append("  " + _t(lang, "flagged_warn"))

    failed = _format_failures(result, lang)
    if failed:
        lines.append(failed)
    for note in result.notes:
        lines.append("  %s: %s" % (_t(lang, "label_note"), note))

    return "\n".join(lines).rstrip()


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _label_value(label: str, value: str, lang: str = "en", width: int = 16) -> str:
    return "  %-*s %s" % (width, _t(lang, label) + ":", value)


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _status_line(record: ProductRecord, lang: str = "en") -> str:
    """One line describing whether a product is currently authorised."""
    authorised = record.authorised
    if authorised is True:
        state = _t(lang, "status_authorised")
    elif authorised is False:
        state = _t(lang, "status_not_authorised")
    else:
        state = _t(lang, "status_unknown")
    return f"{record.status or 'Unknown'} - {state}"


def format_verification(result, show_data: bool = True, lang: str = "en") -> str:
    """Render a VerificationResult as a terminal table.

    `show_data=False` gives just the verdict, which is what `/nafdac` needs when
    it is summarising several lookups. `lang` localises the labels and the
    verdict words only; every value comes from the record untouched.
    """
    if not result.searched:
        return _t(lang, "no_number_in_message")

    if not result.found:
        return format_not_found(result, lang=lang)

    lines: List[str] = []

    if result.ambiguous:
        lines.append(
            _t(lang, "headline_ambiguous",
               nrn=result.nrn, count=len(result.matches))
        )
    else:
        if result.authorised is True:
            verdict = _t(lang, "verdict_authorised")
        elif result.authorised is False:
            verdict = _t(lang, "verdict_not_authorised")
        else:
            verdict = _t(lang, "verdict_unknown")
        lines.append(_t(lang, "headline", nrn=result.nrn, verdict=verdict))
    lines.append(_rule("="))

    for record in result.matches:
        lines.extend(_format_record(record, show_data, lang))
        lines.append("")

    if result.greenbook_url:
        lines.append("  %s: %s" % (_t(lang, "label_greenbook"), result.greenbook_url))
    lines.append("  %s: %s" % (_t(lang, "label_confidence"), result.confidence))

    failed = _format_failures(result, lang)
    if failed:
        lines.append(failed)

    for note in result.notes:
        lines.append("  %s: %s" % (_t(lang, "label_note"), note))

    return "\n".join(lines).rstrip()


def _format_record(record: ProductRecord, show_data: bool, lang: str = "en") -> List[str]:
    """The per-product block inside a verification result."""
    lines: List[str] = []
    name = record.product_name or _t(lang, "name_not_recorded")
    if result_is_duplicate(record):
        name += "  [%s]" % _t(lang, "duplicate_entry")
    lines.append(f"  {name}")
    lines.append(_label_value("label_number", record.nrn or _t(lang, "not_recorded"), lang))

    if not show_data:
        lines.append(_label_value("label_status", _status_line(record, lang), lang))
        return lines

    if record.active_ingredients:
        lines.append(_label_value("label_ingredients", _truncate(record.active_ingredients, 60), lang))
    if record.strengths:
        lines.append(_label_value("label_strength", _truncate(record.strengths, 60), lang))
    if record.form or record.route:
        shape = " / ".join(part for part in (record.form, record.route) if part)
        lines.append(_label_value("label_form", shape, lang))
    if record.product_category:
        lines.append(_label_value("label_category", record.product_category, lang))
    if record.marketing_category:
        lines.append(_label_value("label_marketing", record.marketing_category, lang))
    if record.applicant_name:
        lines.append(_label_value("label_applicant", _truncate(record.applicant_name, 60), lang))
    if record.manufacturer_name or record.manufacturer_country:
        who = " / ".join(
            part for part in (record.manufacturer_name, record.manufacturer_country) if part
        )
        lines.append(_label_value("label_manufacturer", _truncate(who, 60), lang))
    if record.pack_size:
        lines.append(_label_value("label_pack", _truncate(record.pack_size, 60), lang))
    if record.approval_date:
        lines.append(_label_value("label_approved", record.approval_date, lang))
    if record.expiry_date:
        lines.append(_label_value("label_expires", record.expiry_date, lang))
    if record.atc_code:
        lines.append(_label_value("label_atc", record.atc_code, lang))
    lines.append(_label_value("label_status", _status_line(record, lang), lang))
    return lines


def result_is_duplicate(record: ProductRecord) -> bool:
    return bool(getattr(record, "duplicate_marker", False))


def format_not_found(result, lang: str = "en") -> str:
    """Render a miss without over-claiming.

    The wording is deliberate: not being in the greenbook is not the same as
    being fake, and a user who is told a real product is counterfeit because of
    a stale export will not trust the next answer either.
    """
    lines = [_t(lang, "not_found", nrn=result.nrn), _rule("=")]
    lines.append("")
    lines.append("  " + _t(lang, "not_found_absent"))
    if result.sources_checked:
        lines.append("  " + _t(lang, "not_found_checked",
                               sources=", ".join(result.sources_checked)))
    failed = _format_failures(result, lang)
    if failed:
        lines.append(failed)
    lines.append("")
    lines.append("  " + _t(lang, "not_found_1"))
    lines.append("  " + _t(lang, "not_found_2"))
    lines.append("  " + _t(lang, "not_found_3"))
    return "\n".join(lines)


def _format_failures(result, lang: str = "en") -> str:
    if not result.sources_failed:
        return ""
    parts = [f"{name} ({why})" for name, why in result.sources_failed.items()]
    return f"  {_t(lang, 'label_unavailable')}: {'; '.join(parts)}"


def format_search_results(
    records: Iterable[ProductRecord], query: str, limit: int = 20, lang: str = "en"
) -> str:
    """Render a free-text search as a ranked list."""
    records = list(records)
    if not records:
        return _t(lang, "search_none", query=query)

    shown = records[: max(1, limit)]
    suffix = ""
    if len(records) > len(shown):
        suffix = _t(lang, "search_suffix", shown=len(shown))
    lines = [
        _t(lang, "search_matched", count=len(records), query=query, suffix=suffix),
        _rule("="),
    ]
    for record in shown:
        name = _truncate(record.product_name or _t(lang, "name_not_recorded"), 46)
        number = record.nrn or _t(lang, "no_number")
        lines.append(f"  {name}  [{number}]")
        detail = " / ".join(
            part
            for part in (
                _truncate(record.active_ingredients, 40),
                _truncate(record.applicant_name, 34),
            )
            if part
        )
        if detail:
            lines.append(f"    {detail}")
    lines.append("")
    lines.append("  " + _t(lang, "search_verify_hint"))
    return "\n".join(lines)


def format_status(sources: Iterable, lang: str = "en") -> str:
    """Render which sources are available, for /nafdac.

    Each source's own `describe()` is left as it is: those strings name files,
    hosts and drivers, and translating them would make a connection error
    harder to read rather than easier.
    """
    sources = list(sources)
    lines = [_t(lang, "status_title"), _rule("=")]
    for source in sources:
        try:
            description = source.describe()
        except Exception as exc:
            description = f"error: {exc}"
        lines.append(f"  {source.name:<9} {description}")
    lines.append("")
    lines.append("  " + _t(lang, "status_note_1"))
    lines.append("  " + _t(lang, "status_note_2"))
    lines.append("  " + _t(lang, "status_note_3"))
    return "\n".join(lines)


def source_status(sources: Optional[Iterable], lang: str = "en") -> str:
    return format_status(sources or [], lang=lang)
