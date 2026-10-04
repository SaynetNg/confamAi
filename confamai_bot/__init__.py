"""Confam AI — WhatsApp bot for the CONFAM powered by saynet.ng.

A third client for the N-ATLaS API, a peer of `n_atlas_client`. It is not a mode
of the terminal client and not a copy of the single-file `n_atlas.py`: that file
exists to be copied onto one laptop alone and shares no code with the package,
so forking it guarantees drift.

What this package is:

    WhatsApp  ->  POST /webhook  ->  POST /v1/chat/completions  ->  reply

The bot owns three things and nothing else:

    * transport  — the Cloud API webhook and the Graph API sender
    * sessions   — per-phone conversation history, because the server is
                   stateless per request
    * formatting — turning a reply into text WhatsApp will render cleanly

What it deliberately does NOT own, because the server already does and owns it
better:

    * the NAFDAC greenbook lookup
    * the intent gate that decides whether to look anything up
    * the verdict
    * the system prompt

There is no second retrieval authority here. That is not a shortcut, it is the
invariant the whole project is built on: one implementation of "is this product
registered", so the two front ends can never disagree. `n_atlas_client/api.py`
already reached the same conclusion about itself and returns an empty context
block whenever a server is reachable (`nafdac_server_authoritative`).

Consequently this package imports NOTHING from `nafdac_source`,
`n_atlas_client` or `n_atlas_server`. Its only dependencies are httpx and
FastAPI. That is also why it never verifies anything when the N-ATLaS server is
down: there is no second source of truth to fall back to, by design.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]