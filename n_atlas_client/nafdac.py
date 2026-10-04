"""NAFDAC verification for the laptop client.

The client intercepts a message before it reaches N-ATLaS. When the message
carries a NAFDAC number, the number is verified against the greenbook here, on
the laptop, and the verified facts are handed to the model as context.

Why the client does this rather than the server:

    the greenbook data and the PostgreSQL connection live on this machine, so
    the lookup happens where the data is. N-ATLaS still does the talking: it
    receives the verified block as context and writes the reply in the user's
    language, which is what it is good at.

The important consequence is that the model never guesses. It is told the
product name, the status, the applicant and the dates that the database holds,
and it is told to say a field is not recorded rather than invent one.

Two paths through here:

    /verify <nrn>     a local command, answered without touching the server
    ordinary chat     an intercepted message, answered with the server
"""

from .nafdac_source import (
    NAFDAC_HELP,
    SearchResult,
    VerificationResult,
    extract_nrn,
    extract_search_terms,
    find_products,
    format_not_found,
    format_no_number_found,
    format_search_results,
    format_search_verdict,
    format_searched_nothing,
    format_status,
    format_usage_search,
    format_usage_verification,
    format_usage_verification_words,
    format_verification,
mentions_nafdac,
    search_products,
    verify_nafdac_product,
)

# Prepended to every context block, whether it came from a number lookup or a
# search. Kept as one constant so the two paths cannot drift apart on what the
# model is allowed to say.
_CONTEXT_PREFIX = (
    "[nafdac] Verified greenbook data for this question. Report these values "
    "to the user as fact. Do not add a product name, applicant, date or status "
    "that is not listed here, and do not say a field is missing when it is "
    "present. If the user asks something the data does not cover, say so "
    "plainly.\n"
    "- Use the VERDICT line to answer 'is it genuine?': AUTHENTIC, FLAGGED or "
    "NOT_REGISTERED.\n"
    "- NEVER tell the user a product is counterfeit, fake or poisoned. The "
    "greenbook records registrations, not counterfeiting. Not being listed is "
    "not the same as being fake.\n"
    "- AUTHENTIC means registered, not that this particular pack is genuine: a "
    "copied number resolves just as well.\n"
    "- If several candidates are listed with no verdict, show them and ask which "
    "product the user means. Never pick one for them.\n"
    "- Answer in the user's language, including Hausa, Yoruba or Igbo. Keep "
    "product names, NAFDAC numbers, dates and statuses exactly as written.\n\n"
)


class NafdacInterceptor:
    """Decides whether a message is a NAFDAC lookup and prepares the context.

    Kept as a class rather than a function so the sources are built once and
    reused: building them re-reads the greenbook, which takes seconds.
    """

    def __init__(self, cfg, debug: bool = False, language=None):
        self.cfg = cfg
        self.debug = bool(getattr(cfg, "debug", False) or debug)
        self.enabled = bool(getattr(cfg, "nafdac_enabled", True))
        self.show_data = bool(getattr(cfg, "nafdac_show_data", True))
        self.max_results = int(getattr(cfg, "nafdac_max_search_results", 10) or 10)
        self._sources = None
        self.last_result: VerificationResult | None = None
        self.last_search: SearchResult | None = None
        # A zero-argument callable returning the current language code, so
        # `/hausa` changes the language these tables print in without
        # rebuilding the interceptor and re-reading the greenbook. A callable
        # rather than a value also keeps this module free of any dependency on
        # language.py, so it stays testable on its own.
        self._language_code = language or (lambda: "en")

    def _lang(self) -> str:
        """The current language code, or English if the holder misbehaves.

        A broken language holder must never be the reason a greenbook lookup
        fails, so an exception here degrades to the default rendering.
        """
        try:
            return str(self._language_code() or "en")
        except Exception:
            return "en"

    # ------------------------------------------------------------------
    def _get_sources(self):
        """Build the source list on first use, then reuse it."""
        if self._sources is None:
            from .config import apply_nafdac_env

            apply_nafdac_env(self.cfg)
            from .nafdac_source import build_sources

            self._sources = build_sources()
            if self.debug:
                for source in self._sources:
                    print("[debug] nafdac source %s: %s"
                          % (source.name, source.describe()), flush=True)
        return self._sources

    def status(self) -> str:
        """A short report on which sources are usable, for /nafdac."""
        return format_status(self._get_sources(), lang=self._lang())

    def looks_like_lookup(self, message: str) -> bool:
        """True when this message should bring greenbook data with it.

        There is no longer a test for "is this about a product". Every message
        is searched, and a hit is passed on whether or not the user asked about a
        product, because greenbook names are ordinary English words: "good
        morning" matches products called "Morning", and no word list can tell
        that apart from a genuine product question. The model reads the block and
        decides whether it is relevant.

        What this returns is narrower: whether there is anything at all to pass
        on. An unknown number or a search with no hits returns False, and the
        model answers the message conversationally.
        """
        if not self.enabled or not message:
            return False
        if mentions_nafdac(message):
            return True

        if extract_nrn(message):
            result = self.verify(message, quiet=True)
            return bool(result and result.found)

        search = self.search(message, quiet=True)
        return bool(search and search.found)

    # ------------------------------------------------------------------
    def search(self, message: str, quiet: bool = False) -> SearchResult:
        """Free-text search whatever product words are in `message`.

        `quiet` suppresses the data block and is used for the
        looks-like-a-lookup probe, so a decision is not printed twice.
        """
        if not self.enabled:
            return SearchResult()
        terms = extract_search_terms(message)
        if not terms:
            return SearchResult(query=(message or "").strip())

        result = find_products(
            terms, sources=self._get_sources(), limit=self.max_results
        )
        self.last_search = result

        if self.debug:
            print(
                "[debug] nafdac search %s query=%r found=%s total=%s "
                "verdict=%s"
                % (
                    "probe" if quiet else "search",
                    result.query,
                    result.found,
                    result.total,
                    result.verdict,
                ),
                flush=True,
            )
        return result

    # ------------------------------------------------------------------
    def verify(self, message: str, quiet: bool = False) -> VerificationResult:
        """Verify whatever NAFDAC number is in `message`.

        `quiet` suppresses the data block and is used for the
        looks-like-a-lookup probe, so a decision is not printed twice.
        """
        if not self.enabled:
            return VerificationResult()
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

    def verify_command(self, argument: str) -> str:
        """Handle `/verify <nrn or product name>`, answered on the laptop.

        A number is verified by number. Anything else is searched instead of
        rejected, because `/verify paracetamol` and `/verify is this genuine?`
        are the same question a user types in words, and answering "no NAFDAC
        number found" to those is a non-answer.
        """
        argument = (argument or "").strip()
        if not argument:
            return format_usage_verification_words(self._lang())

        if not extract_nrn(argument):
            return self.verify_word_command(argument)

        result = self.verify(argument)
        if not result.searched:
            return format_no_number_found(self._lang())
        return format_verification(
            result, show_data=self.show_data, lang=self._lang()
        )

    def verify_word_command(self, argument: str) -> str:
        """Search for the product words in `argument` and report the verdict.

        The search half of `verify_command`. A miss is worded as "not in this
        export", never as counterfeit, for the reason the whole package exists.
        """
        argument = (argument or "").strip()
        if not argument:
            return format_usage_verification_words_short(self._lang())

        result = self.search(argument)
        if not result.searched:
            return format_searched_nothing(self._lang())
        return format_search_verdict(
            result, limit=self.max_results, show_data=self.show_data,
            lang=self._lang(),
        )

    def search_command(self, argument: str) -> str:
        """Handle `/search <text>`, a free-text greenbook search."""
        argument = (argument or "").strip()
        if not argument:
            return format_usage_search(self._lang())
        # Normalised to terms rather than passed as one string. `search_products`
        # matches whole words, so a multi-word argument became a single needle
        # that could never match anything: `/search paracetamol 500mg` reported
        # "no product matched" while `/verify paracetamol 500mg` found 257.
        records = search_products(
            extract_search_terms(argument),
            sources=self._get_sources(),
            limit=self.max_results,
        )
        return format_search_results(
            records, argument, limit=self.max_results, lang=self._lang()
        )

    def status_command(self) -> str:
        """Handle `/nafdac`."""
        return self.status()

    def help_text(self) -> str:
        return nafdac_help(self._lang())

    # ------------------------------------------------------------------
    def context_for(self, message: str, result: VerificationResult) -> str:
        """The block handed to N-ATLaS as context for this turn.

        Prefixed so the model can tell verified database output from the
        conversation around it, and so it is obvious the values are not
        something the model is expected to recall.
        """
        return _CONTEXT_PREFIX + result.render()

    def context_for_search(self, message: str, result: SearchResult) -> str:
        """The context block for a product-name search.

        Same prefix as the number path, so the model is held to the same rule in
        both cases: only report what the block lists, and never call a product
        counterfeit because the export does not list it.
        """
        if not result or not result.searched:
            return ""
        return _CONTEXT_PREFIX + result.render(limit=self.max_results)

    def answer_locally(self, result: VerificationResult) -> str:
        """A direct answer with no model call at all.

        Used when the server is unreachable but the greenbook is not. A verified
        table beats an error message, and the user still gets their answer.
        """
        if not result.searched:
            return ""
        if not result.found:
            return format_not_found(result, lang=self._lang())
        return format_verification(result, show_data=self.show_data, lang=self._lang())
