"""
Confam AI diagnostic. Run in the same environment that serves the model (Colab/terminal).
Prints facts only; makes no changes. Paste the full output back.

    pip install -U transformers peft accelerate bitsandbytes
    python confam_diagnose.py path/to/training_file.jsonl
"""
import sys, json, time, statistics as st, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

BASE = "NCAIR1/N-ATLaS"
ADAPTER = "Saynetng/confam-ai-adapter-v1.1"
DATA = sys.argv[1] if len(sys.argv) > 1 else "confam_training_dataset_corrected.jsonl"

def hr(t): print("\n" + "=" * 8, t, "=" * 8)

# ---------- 1. Tokenizer / stop-token facts ----------
hr("1. STOP TOKENS")
tok = AutoTokenizer.from_pretrained(BASE)
eot = tok.convert_tokens_to_ids("<|eot_id|>")
print("tokenizer.eos_token:", tok.eos_token, tok.eos_token_id)
print("<|eot_id|> id:", eot)
gc = GenerationConfig.from_pretrained(BASE)
print("base generation_config.eos_token_id:", gc.eos_token_id)
ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
print("-> generate() stops on <|eot_id|> by default:", eot in ids)

hr("2. DOUBLE-BOS CHECK")
msgs = [{"role": "user", "content": "hi"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
a = tok(text, add_special_tokens=True)["input_ids"]
b = tok(text, add_special_tokens=False)["input_ids"]
print("BOS count with default tokenization:", a.count(tok.bos_token_id),
      "| with add_special_tokens=False:", b.count(tok.bos_token_id))

# ---------- 3. Training-data facts ----------
hr("3. TRAINING DATA vs TEMPLATE")
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
lens, reply_lens, bad_end = [], [], 0
for r in rows:
    full = tok.apply_chat_template(r["messages"], tokenize=False)
    lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
    reply_lens.append(len(tok(r["messages"][-1]["content"], add_special_tokens=False)["input_ids"]))
    if not full.rstrip().endswith("<|eot_id|>"):
        bad_end += 1
print(f"rows={len(rows)}  tokens/example mean={st.mean(lens):.0f} max={max(lens)}")
print(f"assistant reply tokens mean={st.mean(reply_lens):.0f} p90={sorted(reply_lens)[int(.9*len(reply_lens))]} max={max(reply_lens)}")
print("rows whose rendered text does NOT end with <|eot_id|>:", bad_end)
print("-> your training code must put <|eot_id|> inside the supervised labels (not masked out).")

# ---------- 4. Generation behaviour ----------
hr("4. GENERATION TEST (card settings vs fixed settings)")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

system = rows[0]["messages"][0]["content"]
data_row = next(r for r in rows if sum(m["role"] == "system" for m in r["messages"]) > 1)
tests = {
    "greeting": [{"role": "system", "content": system}, {"role": "user", "content": "Hi"}],
    "nrn_train_format": [{"role": "system", "content": system}, {"role": "user", "content": "E1-000011"}],
    "nrn_real_format": [{"role": "system", "content": system}, {"role": "user", "content": "A2-102382"}],
    "with_data": data_row["messages"][:-1],
}

def has_loop(ids, n=12, k=3):
    c = collections.Counter(tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n)))
    return any(v >= k for v in c.values())

def run(name, messages, **kw):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = enc["input_ids"].shape[1]
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, pad_token_id=tok.eos_token_id, **kw)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = out[0][n_prompt:].tolist()
    stopped = bool(new) and new[-1] in (tok.eos_token_id, eot)
    print(f"[{name}] prompt_tok={n_prompt} new_tok={len(new)} stopped_on_stop_token={stopped} "
          f"loop={has_loop(new)} time={dt:.1f}s tok/s={len(new)/dt:.1f}")
    print("   ", tok.decode(new, skip_special_tokens=True)[:200].replace("\n", " "))

card = dict(max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
fixed = dict(max_new_tokens=150, do_sample=False, repetition_penalty=1.12,
             eos_token_id=[tok.eos_token_id, eot])
for name, m in tests.items():
    print(f"\n--- {name}")
    run("card ", m, **card)
    run("fixed", m, **fixed)

# prefill cost of the long system prompt
enc = tok(tok.apply_chat_template(tests["greeting"], tokenize=False, add_generation_prompt=True),
          return_tensors="pt", add_special_tokens=False).to(model.device)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    model.generate(**enc, max_new_tokens=1, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
print(f"\nPREFILL (system prompt + 1 token): {time.time()-t0:.2f}s for {enc['input_ids'].shape[1]} prompt tokens")
