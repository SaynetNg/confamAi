"""
Confam AI decisive diagnostic (repo-aware). Run where the model runs (Kaggle/Colab T4).
Prints facts only; changes nothing.

    pip install -U transformers peft accelerate bitsandbytes "trl==0.12.2"
    python confam_diagnose.py --repo /path/to/confamAi-main

Answers four questions:
  Q1  What are eos_token / pad_token / <|eot_id|>?  Does the trainer's default collator
      erase <|eot_id|> from the loss (pad == eos)?
  Q2  How long is the prompt the model is TRAINED on vs the prompt it is SERVED?
  Q3  On the served prompt, does the model emit <|eot_id|>, and does the server's current stop list stop on it?
  Q4  Does stopping on <|eot_id|> end the run-on?
"""
import argparse, ast, json, time, collections
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling)
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--base", default="NCAIR1/N-ATLaS")
ap.add_argument("--adapter", default="Saynetng/confam-ai-adapter-v1.1")
ap.add_argument("--max_new", type=int, default=512)   # the server's own ceiling
a = ap.parse_args()
R = a.repo.rstrip("/")

def hr(t): print("\n" + "=" * 8, t, "=" * 8)

tok = AutoTokenizer.from_pretrained(a.base)
eot = tok.convert_tokens_to_ids("<|eot_id|>")
eos = tok.eos_token_id

hr("Q1  TOKENS AND TRAINING MASK")
print("eos_token:", repr(tok.eos_token), eos, "| pad_token:", repr(tok.pad_token), tok.pad_token_id,
      "| <|eot_id|>:", eot)
print("eos == <|eot_id|>:", eos == eot)
pad_after_train_setup = tok.pad_token_id if tok.pad_token_id is not None else eos   # what train_notebook.py does
print("train_notebook sets pad -> id", pad_after_train_setup, "| pad == <|eot_id|>:", pad_after_train_setup == eot)
rec = json.loads(open(f"{R}/n_atlas_train/confam_training_dataset.jsonl", encoding="utf-8").readline())
text = tok.apply_chat_template(rec["messages"], tokenize=False)
ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
tok.pad_token_id = pad_after_train_setup
col = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
batch = col([{"input_ids": ids[0].tolist()}])
lab = batch["labels"][0]
n_eot = int((ids[0] == eot).sum())
n_eot_masked = int(((ids[0] == eot) & (lab == -100)).sum())
print(f"<|eot_id|> tokens in one training record: {n_eot}; masked out of the loss (-100): {n_eot_masked}")
print("=> if masked == total, the model was NEVER trained to emit <|eot_id|>.")

hr("Q2  TRAINED vs SERVED PROMPT")
src = open(f"{R}/n_atlas_server/natlas_notebook.py", encoding="utf-8").read()
instr = next(ast.literal_eval(n.value) for n in ast.parse(src).body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NAFDAC_INSTRUCTION")
sysp = open(f"{R}/confam_prompt.txt", encoding="utf-8").read()
train_hdr = ("[nafdac] Verified greenbook data for this question. Report these values to the user as "
             "fact, and do not add anything that is not listed here.\n")
block = "NAFDAC A2-102382 - not found in the greenbook.\nVERDICT: NOT_REGISTERED"   # approximation of the server's block
train_msgs = [{"role": "system", "content": sysp},
              {"role": "system", "content": train_hdr + "NRN: A2-102382\nVERDICT: NOT REGISTERED"},
              {"role": "user", "content": "A2-102382"}]
serve_msgs = [{"role": "system", "content": sysp},
              {"role": "system", "content": instr + "\n\n" + block},
              {"role": "user", "content": "A2-102382"}]
def ntok(m): return len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                            add_special_tokens=False)["input_ids"])
print("trained-shape prompt tokens:", ntok(train_msgs), "| served-shape prompt tokens:", ntok(serve_msgs),
      "| training max_seq_length: 2048 (before history)")

hr("Q3/Q4  GENERATION ON THE SERVED PROMPT")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, a.adapter).eval()

def loops(ids, n=12, k=3):
    c = collections.Counter(tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n)))
    return any(v >= k for v in c.values())

def run(label, msgs, stop_ids, temperature):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
    n0 = enc["input_ids"].shape[1]
    kw = dict(max_new_tokens=a.max_new, repetition_penalty=1.10, eos_token_id=stop_ids,
              pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos)
    kw.update(dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature > 0 else dict(do_sample=False))
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): out = model.generate(**enc, **kw)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = out[0][n0:].tolist()
    first_eot = new.index(eot) if eot in new else None
    print(f"\n[{label}] prompt={n0} new={len(new)} hit_cap={len(new) >= a.max_new} loop={loops(new)} "
          f"<|eot_id|>_emitted_at={first_eot} time={dt:.1f}s ({len(new)/dt:.1f} tok/s)")
    print("   raw:", tok.decode(new[:120], skip_special_tokens=False)[:400].replace("\n", "\\n"))

server_stop = [eos]            # what _resolve_eos_ids() returns today
fixed_stop = sorted({eos, eot})
for temp in (0.7, 0.1):        # clients send 0.7; the server default is 0.1
    run(f"SERVED prompt | server stop={server_stop} | T={temp}", serve_msgs, server_stop, temp)
    run(f"SERVED prompt | fixed  stop={fixed_stop} | T={temp}", serve_msgs, fixed_stop, temp)
run(f"TRAINED-shape prompt | fixed stop={fixed_stop} | T=0.1", train_msgs, fixed_stop, 0.1)
print("\nREAD: eot emitted + server-stop run hit the cap  -> serving stop-list bug (apply confam_fix.patch, server part).")
print("      eot NEVER emitted in any run                -> training never taught it (apply the training part and retrain).")
print("      loops only on the SERVED prompt             -> train/serve prompt mismatch (use the same header in both).")
