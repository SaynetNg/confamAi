"""Locate the `nafdac_source` package from inside the client.

`nafdac_source` lives at the project root, next to `n_atlas_client`, not inside
it. That is deliberate: the Kaggle server imports the same package, and the
scraper can use it too. This module is the single place that knows how to find
it, so the rest of the client can just `from .nafdac_source import ...`.

Two situations are handled:

  * running from a checkout, where the project root is one level up;
  * a copied-out single-file client, where the sibling directory may not exist
    at all, in which case importing raises a clear error rather than an
    obscure one from deep inside an import chain.
"""

import sys
from pathlib import Path

# The directory containing n_atlas_client, i.e. the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import nafdac_source as _nafdac_source
except ImportError as exc:  # pragma: no cover - only when the tree is broken
    raise ImportError(
        "The nafdac_source package could not be imported from %s. N-ATLaS "
        "product verification needs it. Keep nafdac_source next to "
        "n_atlas_client, or add the project root to PYTHONPATH. (original "
        "error: %s)" % (_PROJECT_ROOT, exc)
    ) from exc

# Re-export the public surface so callers can do
# `from .nafdac_source import verify_nafdac_product` without a try/except.
from nafdac_source import (  # noqa: E402,F401
    NAFDAC_URL_TEMPLATE,
    SearchResult,
    VerificationResult,
    build_sources,
    clear_source_cache,
    extract_nrn,
    extract_nrn_candidates,
    extract_search_terms,
    find_products,
    looks_like_nrn,
    mentions_nafdac,
    normalize_nrn,
    normalize_nrn_key,
    search_products,
    verify_nafdac_product,
)
from nafdac_source.formatting import (  # noqa: E402,F401
    NAFDAC_HELP,
    STRINGS,
    SUPPORTED_LANGUAGES,
    format_no_number_found,
    format_not_found,
    format_searched_nothing,
    format_search_results,
    format_search_verdict,
    format_status,
    format_usage_search,
    format_usage_verification,
    format_usage_verification_words,
    format_usage_verification_words_short,
    format_verification,
    nafdac_help,
    source_status,
)

__all__ = [
    "NAFDAC_HELP",
    "NAFDAC_URL_TEMPLATE",
    "STRINGS",
    "SUPPORTED_LANGUAGES",
    "SearchResult",
    "VerificationResult",
    "build_sources",
    "clear_source_cache",
    "extract_nrn",
    "extract_nrn_candidates",
    "extract_search_terms",
    "find_products",
    "format_no_number_found",
    "format_not_found",
    "format_searched_nothing",
    "format_search_results",
    "format_search_verdict",
    "format_status",
    "format_usage_search",
    "format_usage_verification",
    "format_usage_verification_words",
    "format_usage_verification_words_short",
    "format_verification",
    "looks_like_nrn",
"mentions_nafdac",
    "normalize_nrn",
    "normalize_nrn_key",
    "search_products",
    "source_status",
    "verify_nafdac_product",
]
