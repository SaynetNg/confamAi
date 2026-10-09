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

    return tokenizer
