"""Speech-to-text for voice notes.

The NCAIR1 models are what make a voice note answerable at all:
WhatsApp delivers them as .ogg/opus audio, and the verification
question inside has to become text before the N-ATLaS server can
see it.

Loading is lazy — on the first voice note, not at import. This
process is a webhook receiver first: `python -m confamai_bot
--status` and `--test` must not download four models to print one
line, and `torch`/`transformers` are deliberately not in
requirements.txt, so a deployment without them still boots and only
voice notes are affected.

The models are keyed by the same four codes the session store pins
(`en`, `ha`, `yo`, `ig`). Anything else falls back to the
Nigerian-accented English model, which is also the default for a
brand-new session.
"""

from __future__ import annotations

import io

# Hugging Face model ids, one per supported language.
_MODELS = {
    "en": "NCAIR1/NigerianAccentedEnglish",
    "ha": "NCAIR1/Hausa-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "ig": "NCAIR1/Igbo-ASR",
}


class VoiceTranscriptionError(Exception):
    """Transcription could not run. Carries whether it might work later."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class VoiceTranscriber:
    """NCAIR1 speech-to-text, one pipeline per supported language.

    Cheap to construct: each pipeline is built on the first call for
    its language and then kept, so a voice note pays the load cost
    once per process instead of on every message.
    """

    def __init__(self):
        self.models: dict[str, object] = {}

    def transcribe(self, audio_bytes: bytes, language_code: str = "en") -> str:
        """Convert one voice note into text.

        Synchronous on purpose: the call is CPU/GPU-bound and can take
        tens of seconds, so the router runs it in a worker thread
        rather than on the event loop, where it would stall every
        other user's webhook.
        """
        if not audio_bytes:
            raise VoiceTranscriptionError("The voice note arrived empty.")

        code = language_code if language_code in _MODELS else "en"
        asr = self._pipeline(code)

        print("[audio] transcribing with the %s model..." % code, flush=True)
        try:
            result = asr(io.BytesIO(audio_bytes))
        except Exception as exc:
            # A corrupt or truncated note, an undecodable codec, an OOM.
            # All of them are worth telling the user about, and some
            # (memory pressure) may succeed on a retry.
            raise VoiceTranscriptionError(
                "The voice note could not be transcribed: %s" % exc,
                retryable=True,
            ) from exc

        text = str((result or {}).get("text") or "").strip()
        print("[audio] transcribed %d chars" % len(text), flush=True)
        return text

    def _pipeline(self, code: str):
        """The ASR pipeline for one language, loading it on first use."""
        if code not in self.models:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise VoiceTranscriptionError(
                    "Speech-to-text is not installed on this deployment. "
                    "Install it with: pip install torch transformers"
                ) from exc
            print("[audio] loading the NCAIR1 %s model..." % code, flush=True)
            self.models[code] = pipeline(
                "automatic-speech-recognition", model=_MODELS[code]
            )
            print("[audio] the %s model is ready" % code, flush=True)
        return self.models[code]


# The single instance the router uses. Safe to create at import time
# because nothing is loaded until the first voice note arrives.
transcriber = VoiceTranscriber()
