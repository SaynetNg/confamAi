"""Offline test doubles for the N-ATLaS Kaggle notebook.

These stubs let the full notebook (Cells 2-17) run on a CPU-only laptop so the
FastAPI layer, auth, schema validation, context budgeting, response cleaning,
error mapping and the generation lock can be verified without a T4.

They intentionally emit the exact failure modes the real model produced
(template artifacts and a runaway "Question:" continuation) so the response
cleaner is genuinely exercised.
"""

import re
import types
from typing import Any, Dict, List, Optional

import torch

MOCK_STATE: Dict[str, Any] = {
    "force_oom": False,
    "force_error": False,
    "calls": 0,
    "last_prompt_text": "",
    # Voice: set by verify_offline.py to simulate a first-load failure, and
    # read by the fake ASR pipeline below.
    "force_transcribe_error": False,
    "force_long_form": False,
    "transcribe_calls": 0,
    "last_audio_bytes": 0,
    "last_transcribe_language": "",
    "last_chunk_length_s": 0,
}

PAD_ID = 0
EOS_ID = 2


class MockTokenizer:
    """Character-exact tokenizer with a ChatML-shaped template."""

    def __init__(self) -> None:
        self.eos_token = "</s>"
        self.eos_token_id = EOS_ID
        self.pad_token = "</s>"
        self.pad_token_id = PAD_ID
        self.unk_token = "<unk>"
        self.vocab_size = 512
        self.model_max_length = 8192
        self.chat_template = "{{ '<|im_start|>' }}mock chatml template{{ '<|im_end|>' }}"
        self._chars: List[str] = ["\x00", "\x01", "\x02"]
        self._index: Dict[str, int] = {c: i for i, c in enumerate(self._chars)}

    def __len__(self) -> int:
        return max(len(self._chars), self.vocab_size)

    def get_vocab(self) -> Dict[str, int]:
        # The real tokenizer's interface: Cell 4 resolves <|eot_id|>
        # from it. The mock has no such token, so the lookup returns
        # None and the stop list falls back to eos alone.
        return dict(self._index)

    def _encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for char in text:
            if char not in self._index:
                self._index[char] = len(self._chars)
                self._chars.append(char)
                self.vocab_size = max(self.vocab_size, len(self._chars))
            ids.append(self._index[char])
        return ids

    def _decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        out = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (PAD_ID,):
                continue
            if 0 <= token_id < len(self._chars):
                out.append(self._chars[token_id])
        return "".join(out)

    def encode(self, text: str, **_: Any) -> List[int]:
        return self._encode(text)

    def decode(self, ids, skip_special_tokens: bool = True, **_: Any) -> str:
        return self._decode(ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **_: Any,
    ):
        parts = ["<|begin_of_text|>"]
        for message in messages:
            parts.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if tokenize:
            return self._encode(text)
        return text

    def __call__(self, text: str, return_tensors: Optional[str] = None, **_: Any):
        from transformers import BatchEncoding

        ids = self._encode(text)
        if return_tensors == "pt":
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = ids
            attention_mask = [1] * len(ids)
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


class _Param:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


CANNED: List[tuple] = [
    (
        ("sannu", "kana lafiya", "ina lafiya", "yaya kake"),
        "Ina lafiya, sai an jima. Yaya ranar ka? Mun kasance da kyau a cikin harshhe.",
    ),
    (
        ("báwo ni", "bawo ni", "báwọ ni", "bawo"),
        "Báwo ni? Ó dara pupọ, Ṣé ibo ni? Mo ti gbogbo ibi lọ.",
    ),
    (
        ("kedu ka", "kedu", "kedu ká"),
        "Kedu ka ị mere? Nna m anyị dị ọma, ihe niile dị nma.",
    ),
    (
        ("how are you", "yaya kake"),
        "I'm doing great, thanks for asking! Ina lafiya, yaya kake? How can I help?",
    ),
    (
        ("what is my name", "what's my name", "my name"),
        "Your name is Ahmed. I remembered it from earlier in our conversation.",
    ),
    (
        ("answer in hausa",),
        "To za a yi magani a cikin harshhe na Hausa. Sannu da zuwa!",
    ),
    (
        ("answer in english", "now answer in english"),
        "Sure, from now on I will answer in English. How can I help you?",
    ),
    (
        ("ask you to do first",),
        "You first asked me to answer in Hausa.",
    ),
    (
        ("name is ahmed",),
        "Nice to meet you, Ahmed. How can I help you today?",
    ),
    (
        ("explain what n-atlas", "explain what natlas"),
        "N-ATLaS is a multilingual assistant that understands English, Hausa, Igbo "
        "and Yoruba, and can switch language whenever you ask.",
    ),
]

GENERIC = "Hello! I'm here and ready to help. Tell me what you need."


def _canned_answer(prompt: str) -> str:
    lowered = prompt.lower()
    for needles, answer in CANNED:
        if any(needle in lowered for needle in needles):
            return answer
    return GENERIC


class MockModel:
    """Emits a ChatML-leaking, self-continuing completion like the real model did."""

    def __init__(self, tokenizer: MockTokenizer) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = types.SimpleNamespace(use_cache=True)
        self._parameters = [_Param(8_000_000_000), _Param(2_000_000)]

    def parameters(self):
        return iter(self._parameters)

    def named_parameters(self, *a, **k):
        for index, param in enumerate(self._parameters):
            yield f"mock.weight_{index}", param

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def generate(self, **kwargs):
        if MOCK_STATE["force_oom"]:
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 20.00 GiB"
            )
        if MOCK_STATE["force_error"]:
            raise RuntimeError("mock generation explosion")

        input_ids = kwargs["input_ids"]
        prompt_text = self.tokenizer.decode(input_ids[0])
        MOCK_STATE["last_prompt_text"] = prompt_text
        MOCK_STATE["calls"] += 1

        last_user = ""
        matches = re.findall(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt_text, re.S
        )
        if matches:
            last_user = matches[-1].strip()

        answer = _canned_answer(last_user)
        leaked = (
            f"{answer}<|im_end|>\n<|im_start|>user\nAnything else?"
            "\nQuestion: Ah, ina farin ciki... Who am I talking to?"
        )

        max_new_tokens = int(kwargs.get("max_new_tokens") or 64)
        new_ids = self.tokenizer._encode(leaked)[:max_new_tokens]
        if not new_ids:
            new_ids = self.tokenizer._encode(answer)[:1]

        return torch.cat(
            [input_ids, torch.tensor([new_ids], dtype=input_ids.dtype)], dim=1
        )


def install() -> MockTokenizer:
    """Patch transformers so the notebook loads the mocks. Returns the tokenizer."""
    import transformers

    tokenizer = MockTokenizer()
    model = MockModel(tokenizer)

    def fake_tokenizer_from_pretrained(*_args, **_kwargs):
        return tokenizer

    def fake_model_from_pretrained(*_args, **_kwargs):
        return model

    transformers.AutoTokenizer.from_pretrained = staticmethod(fake_tokenizer_from_pretrained)
    transformers.AutoModelForCausalLM.from_pretrained = staticmethod(
        fake_model_from_pretrained
    )

    # Cell 1 calls huggingface_hub.login(token=HF_TOKEN), which validates the
    # token against huggingface.co/whoami. The offline suite runs with a dummy
    # token and no network, so the call is stubbed to a no-op. This only affects
    # the verifier: mock_backend is never imported by the notebook on Kaggle.
    try:
        import huggingface_hub

        huggingface_hub.login = lambda *a, **k: None
    except Exception:  # noqa: BLE001 - absence just means Cell 1 will fail loudly
        pass

    return tokenizer


class MockASR:
    """A deterministic stand-in for one NCAIR1 ASR pipeline.

    Cell 6 builds one pipeline per language on first use. The offline suite
    injects a factory returning this in place of the notebook's `_voice_pipeline`
    (see verify_offline.py), which is the same seam verify_nafdac.py uses for the
    intent extractor — it avoids the lazy `from transformers import pipeline`
    import entirely. Records enough to prove the route passes the clip and the
    language through, and returns a fixed transcript so no real model or ffmpeg
    is needed. `force_transcribe_error` simulates a corrupt clip.
    """

    def __init__(self, language: str = "en") -> None:
        self.language = language

    def __call__(self, audio, **kwargs):
        if MOCK_STATE["force_transcribe_error"]:
            raise ValueError("mock: could not decode the audio")
        # Mirror the real pipeline's contract: it accepts raw bytes, a path or a
        # numpy array, but NOT a file object. Rejecting BytesIO here is what
        # stops the mock from hiding a `asr(BytesIO(...))` bug in Cell 6.
        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        elif hasattr(audio, "read") and not isinstance(audio, (str, bytes, bytearray)):
            raise TypeError(
                "We expect a numpy ndarray or torch tensor as input, got "
                f"`{type(audio)}`"
            )
        else:
            data = audio
        # Mirror Whisper's long-form behaviour: a single pass over >30 s audio
        # raises and demands chunked generation, which the engine then retries
        # with chunk_length_s. The mock raises on the first (unchunked) call when
        # force_long_form is set, and succeeds once chunk_length_s arrives.
        if MOCK_STATE["force_long_form"] and not kwargs.get("chunk_length_s"):
            raise ValueError(
                "You have passed more than 3000 mel input features (> 30 seconds) "
                "which automatically enables long-form generation which requires "
                "the model to predict timestamp tokens."
            )
        MOCK_STATE["transcribe_calls"] += 1
        MOCK_STATE["last_audio_bytes"] = len(data or b"")
        MOCK_STATE["last_transcribe_language"] = self.language
        MOCK_STATE["last_chunk_length_s"] = kwargs.get("chunk_length_s") or 0
        return {"text": f"mock transcript ({self.language})"}


# The four gated NCAIR1 ASR repos, mapped to the language code Cell 4 loads them
# under. install_voice() uses this to hand back the right MockASR.
_VOICE_CODE_BY_MODEL_ID = {
    "NCAIR1/NigerianAccentedEnglish": "en",
    "NCAIR1/Hausa-ASR": "ha",
    "NCAIR1/Yoruba-ASR": "yo",
    "NCAIR1/Igbo-ASR": "ig",
}


def voice_asr_factory():
    """Return a factory matching the notebook's VOICE_ASR_FACTORY signature.

    verify_offline.py assigns this to the notebook global VOICE_ASR_FACTORY
    before Cells 1-18 run, so Cell 4's eager preload builds MockASR pipelines
    instead of downloading the four gated NCAIR1 models. Injecting the notebook
    global is the seam the repo already uses for the NAFDAC hooks; patching
    transformers.pipeline does not work because transformers is a lazy module
    that restores tampered top-level attributes.
    """

    def factory(model_id: str, device_index: int, token) -> MockASR:
        return MockASR(_VOICE_CODE_BY_MODEL_ID.get(model_id, "en"))

    return factory
