"""Speech-to-text for voice notes.

WhatsApp delivers a voice note as .ogg/opus audio, and the verification
question inside has to become text before the N-ATLaS server can see
it. There are two ways to get that text, and this module owns both:

* **Server transcription (default).** The note is POSTed to the
  server's `/v1/audio/transcriptions`, the same OpenAI-compatible
  endpoint the website's microphone uses. The server already holds the
  four NCAIR1 models, preloaded in Cell 4 and running on the T4, so the
  bot needs no torch, no GPU and no model download. This is why the
  server endpoint was built, and it is what `mode="server"` /
  `mode="auto"` use.
* **Local transcription (fallback).** Loading the NCAIR1 models in this
  process, for a deployment that points no server at voice or wants to
  keep working when the server is down. Lazy on the first note, and
  `torch`/`transformers` are deliberately not in requirements.txt, so a
  deployment without them still boots and only voice is affected.

Alignment with the server matters here, because the two paths have the
same three traps and both must avoid them:

* The pipeline takes **raw `bytes`**, not a file object. Passing
  `io.BytesIO` raises "We expect a numpy ndarray or torch tensor".
* A clip longer than **30 seconds** makes Whisper refuse a single pass
  and demand long-form generation; a real WhatsApp voice note is
  routinely longer, so the decode is retried with `chunk_length_s=30`.
* A model that cannot be reached is a **retryable** failure, and the
  user is told so rather than the error being swallowed.

`WHATSAPP_VOICE_TRANSCRIPTION` selects the path: `auto` (try the
server, fall back to local), `server` (server only), or `local` (skip
the server entirely).
"""

from __future__ import annotations

import io
import os

# Hugging Face model ids, one per supported language. Mirrors
# n_atlas_server/natlas_notebook.py VOICE_MODELS so both paths speak the
# same four codes.
_MODELS = {
    "en": "NCAIR1/NigerianAccentedEnglish",
    "ha": "NCAIR1/Hausa-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "ig": "NCAIR1/Igbo-ASR",
}

_MODE_ENV = "WHATSAPP_VOICE_TRANSCRIPTION"


def _mode() -> str:
    value = (os.environ.get(_MODE_ENV) or "auto").strip().lower()
    return value if value in {"auto", "server", "local"} else "auto"


class VoiceTranscriptionError(Exception):
    """Transcription could not run. Carries whether it might work later."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class VoiceTranscriber:
    """Speech-to-text, server-first with a local NCAIR1 fallback.

    Cheap to construct: `bind()` attaches the N-ATLaS client the router
    already has, and each local pipeline is built on the first call for
    its language and then kept.
    """

    def __init__(self):
        self.models: dict[str, object] = {}
        self._natlas = None

    def bind(self, natlas) -> None:
        """Attach the N-ATLaS client used for server transcription.

        Called once by the router. Kept as a separate setter rather than a
        constructor argument so the module-level singleton can exist before
        the client does, and so tests can replace the singleton wholesale.
        """
        self._natlas = natlas
        return self

    def transcribe(self, audio_bytes: bytes, language_code: str = "en") -> str:
        """Convert one voice note into text.

        Synchronous on purpose: the local path is CPU/GPU-bound and the
        server path is a blocking HTTP call, so the router runs this in a
        worker thread rather than on the event loop, where it would stall
        every other user's webhook.
        """
        if not audio_bytes:
            raise VoiceTranscriptionError("The voice note arrived empty.")

        code = language_code if language_code in _MODELS else "en"
        mode = _mode()

        if mode in ("auto", "server") and self._natlas is not None:
            try:
                return self._server_transcribe(audio_bytes, code)
            except _ServerTranscriptionError as exc:
                if mode == "server":
                    raise VoiceTranscriptionError(str(exc), retryable=True) from exc
                print(
                    "[audio] server transcription failed (%r); falling back to "
                    "the local model" % (exc,),
                    flush=True,
                )

        if mode == "server":
            raise VoiceTranscriptionError(
                "Server transcription is selected but no N-ATLaS client is "
                "bound." if self._natlas is None else "Server transcription failed.",
                retryable=True,
            )

        return self._local_transcribe(audio_bytes, code)

    # ------------------------------------------------------------------
    def _server_transcribe(self, audio_bytes: bytes, code: str) -> str:
        """Transcribe via /v1/audio/transcriptions on the N-ATLaS server.

        Runs the async client method on a private loop: `transcribe()` is
        called from a worker thread (asyncio.to_thread), where there is no
        running loop to await on.
        """
        import asyncio

        from .natlas import NAtlasError

        print("[audio] transcribing with the server (%s)..." % code, flush=True)
        try:
            text = asyncio.run(self._natlas.transcribe(audio_bytes, language=code))
        except NAtlasError as exc:
            raise _ServerTranscriptionError(str(exc)) from exc
        except RuntimeError as exc:
            # asyncio.run() refuses to run inside an already-running loop.
            raise _ServerTranscriptionError(str(exc)) from exc
        print("[audio] server transcribed %d chars" % len(text or ""), flush=True)
        return text

    # ------------------------------------------------------------------
    def _local_transcribe(self, audio_bytes: bytes, code: str) -> str:
        asr = self._pipeline(code)
        print("[audio] transcribing with the local %s model..." % code, flush=True)
        try:
            # The pipeline takes raw bytes, not a file object. It also refuses a
            # single pass over >30s audio, which is why the retry below exists;
            # a WhatsApp voice note is routinely longer than that.
            try:
                result = asr(audio_bytes)
            except Exception:
                result = asr(audio_bytes, chunk_length_s=30, return_timestamps=True)
        except Exception as exc:
            # A corrupt or truncated note, an undecodable codec, an OOM.
            # All of them are worth telling the user about, and some
            # (memory pressure) may succeed on a retry.
            raise VoiceTranscriptionError(
                "The voice note could not be transcribed: %s" % exc,
                retryable=True,
            ) from exc

        if isinstance(result, dict):
            text = str(result.get("text") or "").strip()
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            text = str(result[0].get("text") or "").strip()
        else:
            text = str(result or "").strip()
        print("[audio] local model transcribed %d chars" % len(text), flush=True)
        return text

    def _pipeline(self, code: str):
        """The local ASR pipeline for one language, loading it on first use."""
        if code not in self.models:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise VoiceTranscriptionError(
                    "Speech-to-text is not available: the server path is not "
                    "configured and torch/transformers are not installed. Set "
                    "WHATSAPP_VOICE_TRANSCRIPTION=server, or install them with: "
                    "pip install torch transformers"
                ) from exc
            print("[audio] loading the NCAIR1 %s model..." % code, flush=True)
            self.models[code] = pipeline(
                "automatic-speech-recognition", model=_MODELS[code]
            )
            print("[audio] the %s model is ready" % code, flush=True)
        return self.models[code]


class _ServerTranscriptionError(Exception):
    """Internal: the server path failed, so `auto` may fall back."""


# The single instance the router uses. Safe to create at import time
# because nothing is loaded until the first voice note arrives, and the
# server client is attached later through bind().
transcriber = VoiceTranscriber()