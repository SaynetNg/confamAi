"""Held-out coverage check: do real user turns land on categories we trained?

Reads the saved conversations from the client (gitignored, so genuinely
held-out), classifies every user turn with the same classifier the dataset
builder uses, and reports the shape of real traffic against what the cleaned
dataset covers. Nothing here writes anything.
"""

import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_training_data as P  # noqa: E402

DATASET_PATH = P.TARGET
TRANSCRIPT_GLOBS = (
    "conversations/*.json",
    "n_atlas_client/conversations/*.json",
    "confamai_bot/conversations/*.json",
)

paths = sorted(
    {
        found
        for pattern in TRANSCRIPT_GLOBS
        for found in glob.glob(str(P.ROOT / pattern))
    }
)
if not paths:
    # Run from inside the folder rather than the repository root.
    paths = sorted(
        {
            found
            for pattern in TRANSCRIPT_GLOBS
            for found in glob.glob(pattern)
        }
    )

turns = []
for path in paths:
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        print("skip %s (%s)" % (path, exc))
        continue
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        continue
    for user, answer in P.pairs_of(messages):
        turns.append((user, answer))

print("transcripts      : %d" % len(paths))
print("user turns       : %d" % len(turns))
print()

shape = collections.Counter()
langs = collections.Counter()
samples = collections.defaultdict(list)
for user, answer in turns:
    category, verdict, _ = P.classify(user, "")
    if category == "no_data" and P.VERDICT_WORDS.search(answer or ""):
        category = "verdict_without_block"      # the failure mode, if any
    shape[category] += 1
    langs[P.detect_language(user)] += 1
    if len(samples[category]) < 3:
        samples[category].append(user[:70])

for key, count in shape.most_common():
    print("%-22s %4d   e.g. %s" % (key, count,
                                   samples[key][0] if samples[key] else "-"))
print()
print("languages        :", dict(langs.most_common()))

trained = collections.Counter()
try:
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer = record["messages"][-1]["content"]
        user = record["messages"][-2]["content"]
        has_block = any(m["role"] == "system" and m["content"].startswith("[nafdac]")
                        for m in record["messages"])
        if P.VERDICT_WORDS.search(answer) and has_block:
            word = P.VERDICT_WORDS.search(answer).group(1)
            trained[word.replace(" ", "_")] += 1
        elif not has_block and "cannot confirm" in answer:
            trained["no_data_negative"] += 1
except FileNotFoundError:
    print("training dataset not found; run prepare_training_data.py first")

print()
print("cleaned dataset, same categories:")
for key, count in trained.most_common():
    print("   %-24s %4d" % (key, count))