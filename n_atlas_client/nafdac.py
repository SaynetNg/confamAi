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
    VerificationResult,
    extract_nrn,
    format_not_found,
    format_search_results,
    format_status,
    format_verification,
    mentions_nafdac,
    search_products,
    verify_nafdac_product,
)


class NafdacInterceptor:
    """Decides whether a message is a NAFDAC lookup and prepares the context.

    Kept as a class rather than a function so the sources are built once and
    reused: building them re-reads the greenbook, which takes seconds.
    """

    def __init__(self, cfg, debug: bool = False):
        self.cfg = cfg
        self.debug = bool(getattr(cfg, "debug", False) or debug)
        self.enabled = bool(getattr(cfg, "nafdac_enabled", True))
        self.show_data = bool(getattr(cfg, "nafdac_show_data", True))
        self.max_results = int(getattr(cfg, "nafdac_max_search_results", 10) or 10)
        self._sources = None
        self.last_result: VerificationResult | None = None

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
        return format_status(self._get_sources())

    def looks_like_lookup(self, message: str) -> bool:
        """True when this message should be verified rather than just sent on.

        The cheap test comes first so ordinary conversation never pays for a
        data lookup. A message that names a number but not NAFDAC is still
        checked, because a bare `A4-100160` is usually a lookup; what stops a
        date or a quantity from being treated as one is that the greenbook has
        to actually know the number.
        """
        if not self.enabled or not message:
            return False
        if mentions_nafdac(message):
            return True
        if not extract_nrn(message):
            return False
        result = self.verify(message, quiet=True)
        return bool(result and result.found)

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
        """Handle `/verify <nrn>`, answered entirely on the laptop."""
        argument = (argument or "").strip()
        if not argument:
            return "Usage: /verify <nafdac number>   e.g. /verify A4-100160"

        result = self.verify(argument)
        if not result.searched:
            return (
                "No NAFDAC number found in that. It looks like A4-100160 or "
                "04-3877 — the number NAFDAC printed on the pack."
            )
        return format_verification(result, show_data=self.show_data)

    def search_command(self, argument: str) -> str:
        """Handle `/search <text>`, a free-text greenbook search."""
        argument = (argument or "").strip()
        if not argument:
            return "Usage: /search <text>   e.g. /search malaria"
        records = search_products(
            argument, sources=self._get_sources(), limit=self.max_results
        )
        return format_search_results(records, argument, limit=self.max_results)

    def status_command(self) -> str:
        """Handle `/nafdac`."""
        return self.status()

    def help_text(self) -> str:
        return NAFDAC_HELP

    # ------------------------------------------------------------------
    def context_for(self, message: str, result: VerificationResult) -> str:
        """The block handed to N-ATLaS as context for this turn.

        Prefixed so the model can tell verified database output from the
        conversation around it, and so it is obvious the values are not
        something the model is expected to recall.
        """
        return (
            "[nafdac] Verified greenbook data for this question. Report these "
            "values to the user as fact. Do not add a product name, applicant, "
            "date or status that is not listed here, and do not say a field is "
            "missing when it is present. If the user asks something the data "
            "does not cover, say so plainly.\n\n"
            + result.render()
        )

    def answer_locally(self, result: VerificationResult) -> str:
        """A direct answer with no model call at all.

        Used when the server is unreachable but the greenbook is not. A verified
        table beats an error message, and the user still gets their answer.
        """
        if not result.searched:
            return ""
        if not result.found:
            return format_not_found(result)
        return format_verification(result, show_data=self.show_data)
