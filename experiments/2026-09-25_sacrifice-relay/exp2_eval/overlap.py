"""Independence of held-out candidates from every problem experiments 1 and 2 used. Numbers only.

`check USED DRAW OUT` compares each drawn problem (in draw order) with every used problem
(experiment 1's 800 candidates, a superset of the 207 trained ones) and with the drawn problems
before it. It flags a drawn problem if any of these holds:
- **too little to compare**: under MIN_WORDS words once boilerplate is removed, or under
  MIN_LATIN of its letters Latin (the text and meaning checks are English-only);
- **text**: Jaccard >= Rules.jaccard, or containment >= Rules.containment, on the rare 5-word
  shingles of the statement without boilerplate. Boilerplate lines occur in >= LINE_MIN_COUNT
  DeepCoder problems (the PrimeIntellect wrapper, section headers, sample lines); rare shingles
  occur in <= SHINGLE_MAX_DF problems (stock phrases drop out);
- **raw text**: Jaccard >= RAW_JACCARD on 5-word runs of the whole prompt, wrapper included (the
  first build's measure). It only adds short statements that the wrapper dominates;
- **tests**: on complete test suites read from the raw dataset (public and hidden; not the 32-test
  samples in the TaskSets): a whole suite inside the other's, a shared pair that is rare across
  DeepCoder (<= CASE_MAX_DF problems) and long (>= CASE_MIN_CHARS characters), or 3+ rare pairs;
- **source ids** (`--upstream`): the same TACO URL or CodeContests problem in PrimeIntellect's
  upstream dataset, read by range requests (nothing is stored). APPS indices are not ids (APPS's
  train and test splits both count from 0) and Codeforces rows carry only a contest number;
- **meaning** (`--models`): cosine similarity of sentence embeddings >= the model's threshold.

Writes under OUT (no problem text anywhere except the exclusion TaskSet, as in any TaskSet):
  <mode>.txt          the report: thresholds, what each check removed, score distributions, and
                      each check's recall on the known reformatted copies;
  <mode>_flags.jsonl  per problem: its id, best scores, partners and the checks that flagged it;
  embed_<model>.json  cached embeddings (numbers only);
  exclusion/          (check) a TaskSet of the used and the flagged problems, for data build.
`verify USED FINAL OUT` runs the same checks on the final candidates and exits 1 if any is flagged.

The scan needs the DeepCoder HF cache; the embeddings need torch and transformers (run.sh uses the
dev box's CPU venv) and download the models to MARLI_EMBED_CACHE (default /tmp/marli-embed/hf/hub).

Usage: python overlap.py check|verify <used taskset.json> <taskset.json> <out dir>
       [--upstream] [--models bge,minilm]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from marli.data.build import _ngrams, _normalize
from marli.data.overlap import (
    Rules,
    case_key,
    case_keys,
    case_size,
    common_lines,
    document_frequency,
    histogram,
    rare,
    shared_cases,
    shingles,
    strip_lines,
    text_matches,
    words,
)
from marli.handles import InputRef
from marli.tasks.code import load_code_tasks
from marli.tasks.source import SOURCES
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks

LINE_MIN_COUNT = 50  # a line found in this many DeepCoder problems is boilerplate (0.23%)
SHINGLE_MAX_DF = 20  # a shingle found in more problems than this is a stock phrase
CASE_MAX_DF = 3  # a test pair found in more problems than this is generic (e.g. 1 -> 1)
CASE_MIN_CHARS = 16  # shorter rare pairs coincide by chance (every coincidence read was <= 12)
RAW_JACCARD = 0.3
MIN_WORDS = 20
MIN_LATIN = 0.8
RULES = Rules()
MODELS = {  # name: (HF id, pooling, max tokens, threshold)
    "bge": ("BAAI/bge-small-en-v1.5", "cls", 512, 0.88),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", 256, 0.75),
}
CHECKS = ("short", "not_english", "text", "raw_text", "tests", "ids", *MODELS)
# Embedding models live outside /workspace (the shared quota); override with MARLI_EMBED_CACHE.
EMBED_CACHE = os.environ.get("MARLI_EMBED_CACHE", "/tmp/marli-embed/hf/hub")
PRIME_INTELLECT = "datasets/PrimeIntellect/verifiable-coding-problems/data/*.parquet"
CODEFORCES = re.compile(
    r"codeforces\.com/(?:problemset/problem|contest|gym)/(\d+)/(?:problem/)?(\w+)"
)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def identity(task_id: str) -> str:
    """subset/row: the same DeepCoder row under `deepcoder` and `deepcoder_no_lcb`."""
    return task_id.split("/", 1)[1]


def scan() -> tuple[dict[str, str], dict[str, frozenset[int]], dict[int, int]]:
    """Every DeepCoder train row with its complete test suite (no 32-test cap)."""
    source = replace(SOURCES.load("deepcoder"), max_tests=None, max_test_bytes=None)
    prompts, keys, sizes = {}, {}, {}
    for task in load_code_tasks(source, seed=0):
        ident = identity(task.task_id)
        prompts[ident] = task.prompt
        keys[ident] = case_keys(task.answer["tests"])
        for test in task.answer["tests"]:
            key = case_key(test["input"], test["output"])
            if key not in sizes:
                sizes[key] = case_size(test["input"], test["output"])
    return prompts, keys, sizes


def upstream_key(source: str, sid: str | None) -> str | None:
    sid = (sid or "").strip()
    if not sid or source in ("apps", "codeforces"):
        return None
    if m := CODEFORCES.search(sid):
        return f"codeforces:{m[1]}{m[2].upper()}"
    if sid.startswith("http"):
        return "url:" + re.sub(r"^https?://(www\.)?", "", sid).rstrip("/").casefold()
    return f"{source}:{sid}"


def upstream_ids(
    prompts: dict[str, str], wanted: list[str], common: frozenset[str]
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Source ids from PrimeIntellect's upstream rows: by exact prompt, else the same statement."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    def body(prompt: str) -> str:
        return " ".join(words(strip_lines(prompt, common)))

    exact: dict[str, set[str]] = {}
    same: dict[str, set[str]] = {}
    fs = HfFileSystem()
    for path in sorted(fs.glob(PRIME_INTELLECT)):
        with fs.open(path, "rb", block_size=1 << 20) as stream:
            table = pq.ParquetFile(stream).read(columns=["source", "in_source_id", "prompt"])
        columns = (table.column(name).to_pylist() for name in ("source", "in_source_id", "prompt"))
        for source, sid, prompt in zip(*columns, strict=True):
            key = upstream_key(source, sid)
            if key is not None:
                exact.setdefault(prompt, set()).add(key)
                same.setdefault(body(prompt), set()).add(key)
    ids, mapped = {}, Counter()
    for ident in wanted:
        found = exact.get(prompts[ident]) or same.get(body(prompts[ident])) or set()
        mapped["exact prompt" if prompts[ident] in exact else "same statement" if found
               else "no id"] += 1
        ids[ident] = found
    return ids, dict(mapped)


def embed(model: str, texts: dict[str, str], cache: Path) -> dict[str, list[float]]:
    """Unit-length embeddings, cached by problem id (numbers only)."""
    stored = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [k for k in texts if k not in stored]
    if missing:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        hf_id, pooling, max_tokens, _ = MODELS[model]
        local = snapshot_download(
            hf_id, cache_dir=EMBED_CACHE, allow_patterns=["*.json", "*.txt", "model.safetensors"]
        )
        tokenizer = AutoTokenizer.from_pretrained(local)
        network = AutoModel.from_pretrained(local).eval()
        with torch.inference_mode():
            for start in range(0, len(missing), 16):
                batch = missing[start : start + 16]
                encoded = tokenizer(
                    [texts[k] for k in batch], padding=True, truncation=True,
                    max_length=max_tokens, return_tensors="pt",
                )
                hidden = network(**encoded).last_hidden_state
                if pooling == "cls":
                    vectors = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    vectors = (hidden * mask).sum(1) / mask.sum(1)
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
                for key, vector in zip(batch, vectors.tolist(), strict=True):
                    stored[key] = [round(x, 6) for x in vector]
        cache.write_text(json.dumps(stored))
    return {k: stored[k] for k in texts}


def nearest(
    queries: list[str], pool: list[str], vectors: dict, *, earlier: bool
) -> dict[str, tuple[float, str | None]]:
    """Best cosine per query against the pool, or (earlier) against the queries before it."""
    import numpy as np

    q = np.array([vectors[k] for k in queries], dtype=np.float32)
    p = np.array([vectors[k] for k in pool], dtype=np.float32)
    sim = q @ p.T
    if earlier:  # pool == queries
        sim = np.where(np.tril(np.ones_like(sim), k=-1) > 0, sim, -1.0)
    best, arg = sim.max(1), sim.argmax(1)
    return {
        k: (float(best[i]), pool[int(arg[i])] if best[i] > -1 else None)
        for i, k in enumerate(queries)
    }


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 1.0


def assess(
    used: list[str], draw: list[str], args: argparse.Namespace, data: tuple | None = None
) -> dict:
    """All checks for ``draw`` against ``used`` and earlier draws; ``data`` replaces scan()."""
    t0 = time.time()
    prompts, keys, sizes = data if data is not None else scan()
    log(f"scanned {len(prompts)} DeepCoder rows, {sum(map(len, keys.values()))} test pairs")
    missing = [k for k in used + draw if k not in prompts]
    if missing:
        raise SystemExit(f"{len(missing)} problems are not DeepCoder train rows: {missing[:3]}")
    common = common_lines(prompts.values(), LINE_MIN_COUNT)
    stripped = {k: strip_lines(p, common) for k, p in prompts.items()}
    all_shingles = {k: shingles(t) for k, t in stripped.items()}
    shingle_df = document_frequency(all_shingles.values())
    text = {k: rare(all_shingles[k], shingle_df, SHINGLE_MAX_DF) for k in used + draw}
    size = {k: len(v) for k, v in text.items()}
    order = {k: i for i, k in enumerate(draw)}
    U, D = {k: text[k] for k in used}, {k: text[k] for k in draw}
    text_vs = {"used": text_matches(D, U), "earlier": text_matches(D, D, order=order)}
    whole = {k: frozenset(_ngrams(_normalize(prompts[k]), 5)) for k in used + draw}
    WU, WD = {k: whole[k] for k in used}, {k: whole[k] for k in draw}
    raw_vs = {"used": text_matches(WD, WU), "earlier": text_matches(WD, WD, order=order)}
    CU, CD = {k: keys[k] for k in used}, {k: keys[k] for k in draw}
    common_args = {"max_df": CASE_MAX_DF, "sizes": sizes, "min_chars": CASE_MIN_CHARS}
    case_df = document_frequency(keys.values())
    case_vs = {"used": shared_cases(CD, CU, case_df, **common_args),
               "earlier": shared_cases(CD, CD, case_df, order=order, **common_args)}
    log(f"text and test checks done ({time.time() - t0:.0f}s)")
    ran = {"short", "not_english", "text", "raw_text", "tests"}
    ids, mapped = {}, None
    if args.upstream:
        ids, mapped = upstream_ids(prompts, used + draw, common)
        ran.add("ids")
        log(f"source ids: {mapped}")
    used_ids = {key for k in used for key in ids.get(k, ())}
    first_owner: dict[str, int] = {}
    for k in draw:
        for key in ids.get(k, ()):
            first_owner.setdefault(key, order[k])
    emb = {}
    for model in args.models:
        vectors = embed(model, {k: stripped[k] for k in used + draw},
                        args.out / f"embed_{model}.json")
        emb[model] = {"used": nearest(draw, used, vectors, earlier=False),
                      "earlier": nearest(draw, draw, vectors, earlier=True)}
        ran.add(model)
        log(f"embeddings {model} done ({time.time() - t0:.0f}s)")
    rows = []
    for k in draw:
        n_words = len(words(stripped[k]))
        flags = ["short"] * (n_words < MIN_WORDS) + ["not_english"] * (
            latin_share(prompts[k]) < MIN_LATIN
        )
        for where in ("used", "earlier"):
            if RULES.text(text_vs[where][k], size, k):
                flags.append(f"text_{where}")
            if raw_vs[where][k] and raw_vs[where][k][0].jaccard >= RAW_JACCARD:
                flags.append(f"raw_text_{where}")
            if RULES.cases(case_vs[where][k]):
                flags.append(f"tests_{where}")
            for model, scores in emb.items():
                if scores[where][k][0] >= MODELS[model][3]:
                    flags.append(f"{model}_{where}")
        if ids.get(k, set()) & used_ids:
            flags.append("ids_used")
        if any(first_owner[key] < order[k] for key in ids.get(k, ())):
            flags.append("ids_earlier")
        best_text = text_vs["used"][k][0] if text_vs["used"][k] else None
        best_case = case_vs["used"][k][0] if case_vs["used"][k] else None
        containment = max(
            (m.containment for m in text_vs["used"][k]
             if min(size[k], size[m.other]) >= RULES.containment_min),
            default=0.0,
        )
        rows.append({
            "id": k,
            "flags": flags,
            "words": n_words,
            "jaccard": round(best_text.jaccard, 4) if best_text else 0.0,
            "containment": round(containment, 4),
            "text_partner": best_text.other if best_text else None,
            "raw_jaccard": round(raw_vs["used"][k][0].jaccard, 4) if raw_vs["used"][k] else 0.0,
            "tests": asdict(best_case) if best_case else None,
            **{f"{m}_cosine": round(s["used"][k][0], 4) for m, s in emb.items()},
            **{f"{m}_partner": s["used"][k][1] for m, s in emb.items()},
        })
    # The copies the first build found (unstripped 5-gram Jaccard >= 0.5) among the first 800.
    old = {k: _ngrams(_normalize(prompts[k]), 5) for k in used}
    known = {}
    for k in draw[:800]:
        grams = _ngrams(_normalize(prompts[k]), 5)
        score, partner = max(
            ((len(grams & g) / len(grams | g) if grams | g else 0.0), u) for u, g in old.items()
        )
        if score >= 0.5:
            known[k] = partner
    return {"rows": rows, "known": known, "mapped": mapped, "ran": ran,
            "common_lines": len(common), "pool": len(prompts)}


def family(flag: str) -> str:
    return flag if flag in ("short", "not_english") else flag.rsplit("_", 1)[0]


def report(result: dict, n_used: int, mode: str) -> str:
    rows, ran = result["rows"], result["ran"]
    meaning = [f"{m} ({MODELS[m][0]}) cosine >= {MODELS[m][3]}" for m in MODELS if m in ran]
    lines = [
        f"Independence {mode}: {len(rows)} problems against the {n_used} used problems (experiment "
        "1's candidates) and against the problems drawn before them. DeepCoder pool "
        f"{result['pool']} rows; {result['common_lines']} boilerplate lines removed.",
        "",
        "Thresholds:",
        f"  boilerplate line: in >= {LINE_MIN_COUNT} problems; rare shingle: in <= "
        f"{SHINGLE_MAX_DF}; dropped outright: under {MIN_WORDS} words, or under "
        f"{MIN_LATIN:.0%} Latin letters",
        f"  text: Jaccard >= {RULES.jaccard}, or containment >= {RULES.containment} (both >= "
        f"{RULES.containment_min} rare shingles)",
        f"  raw text: Jaccard >= {RAW_JACCARD} on the whole prompt, wrapper included",
        f"  tests: a whole suite contained (>= {RULES.contained_min} pairs), or >= "
        f"{RULES.long_min} rare pair (in <= {CASE_MAX_DF} problems) of >= {CASE_MIN_CHARS} "
        f"characters, or >= {RULES.distinctive_min} rare pairs",
        "  source ids: same TACO URL / CodeContests problem in PrimeIntellect's upstream data"
        if "ids" in ran else "  source ids: not run",
        "  meaning: " + ("; ".join(meaning) if meaning else "not run"),
        "",
        "Removed, by check (a problem can fail several; 'only' = no other check caught it):",
        "  check         all  only  vs used  vs earlier drawn",
    ]
    for check in CHECKS:
        if check not in ran:
            lines.append(f"  {check:11s}  not run")
            continue
        hit = [r for r in rows if any(family(f) == check for f in r["flags"])]
        only = [r for r in hit if {family(f) for f in r["flags"]} == {check}]
        used = sum(f"{check}_used" in r["flags"] for r in hit)
        earlier = sum(f"{check}_earlier" in r["flags"] for r in hit)
        split = "      -           -" if check in ("short", "not_english") else (
            f"   {used:6d}  {earlier:6d}")
        lines.append(f"  {check:11s} {len(hit):5d} {len(only):5d} {split}")
    flagged = sum(bool(r["flags"]) for r in rows)
    lines += [f"  any         {flagged:5d}", f"Kept: {len(rows) - flagged} of {len(rows)}", ""]
    lines.append("Distributions over all checked problems (best match among the used problems):")
    edges = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    lines.append("  text Jaccard       " + str(histogram([r["jaccard"] for r in rows], edges)))
    lines.append("  text containment   " + str(histogram([r["containment"] for r in rows], edges)))
    lines.append("  raw text Jaccard   " + str(histogram([r["raw_jaccard"] for r in rows], edges)))
    cos_edges = [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 1.0]
    for m in MODELS:
        if m in ran:
            values = [r[f"{m}_cosine"] for r in rows]
            lines.append(f"  {m:6s} cosine      {histogram(values, cos_edges)}")
    rare_pairs = Counter(min((r["tests"] or {}).get("distinctive", 0), 5) for r in rows)
    lines.append("  rare test pairs shared with the closest used problem (5 = 5 or more): "
                 + str(sorted(rare_pairs.items())))
    known = result["known"]
    if known:
        by_id = {r["id"]: r for r in rows}
        lines += ["", f"Recall on the {len(known)} known reformatted copies (the first build's "
                  "check, among the first 800 drawn), each against its known used partner:"]
        caught = sum(bool(by_id[k]["flags"]) for k in known)
        lines.append(f"  any check: {caught} of {len(known)}")
        text_hits = sum(by_id[k]["jaccard"] >= RULES.jaccard for k in known)
        lowest = min(by_id[k]["jaccard"] for k in known)
        lines.append(f"  text: {text_hits}; lowest Jaccard {lowest:.3f}")
        lines.append(f"  tests: {sum('tests_used' in by_id[k]['flags'] for k in known)}")
        if "ids" in ran:
            lines.append(f"  source ids: {sum('ids_used' in by_id[k]['flags'] for k in known)}")
        for m in MODELS:
            if m in ran:
                scores = sorted(by_id[k][f"{m}_cosine"] for k in known)
                partner = sum(by_id[k][f"{m}_partner"] == known[k] for k in known)
                lines.append(
                    f"  {m}: {sum(s >= MODELS[m][3] for s in scores)} at or above the threshold; "
                    f"the nearest used problem is the known partner for {partner}; lowest cosine "
                    f"{scores[0]:.3f}"
                )
    if result["mapped"]:
        lines += ["", f"PrimeIntellect source ids, used and checked problems: {result['mapped']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "verify"])
    parser.add_argument("used", type=Path)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--models", type=lambda s: [m for m in s.split(",") if m], default=[])
    args = parser.parse_args()
    if unknown := set(args.models) - MODELS.keys():
        parser.error(f"unknown models {sorted(unknown)}; known: {sorted(MODELS)}")
    args.out.mkdir(parents=True, exist_ok=True)
    used_set, task_set = TaskSet.load(args.used), TaskSet.load(args.tasks)
    used = [identity(t.task_id) for t in read_tasks(used_set, stream=True)]
    tasks = [identity(t.task_id) for t in read_tasks(task_set, stream=True)]
    result = assess(used, tasks, args)
    text = report(result, len(used), args.mode)
    (args.out / f"{args.mode}.txt").write_text(text)
    with open(args.out / f"{args.mode}_flags.jsonl", "w") as stream:
        for row in result["rows"]:
            stream.write(json.dumps(row) + "\n")
    print(text)
    flagged = {r["id"] for r in result["rows"] if r["flags"]}
    if args.mode == "verify":
        raise SystemExit(1 if flagged else 0)

    write_exclusion(args.out / "exclusion", used_set, task_set, flagged)


def write_exclusion(root: Path, used_set: TaskSet, task_set: TaskSet, flagged: set[str]) -> None:
    """The used and the flagged problems, as statements only: `data build exclude=` reads no more.

    Its tasks carry no tests (answer None), so it is an exclusion list, not a TaskSet to evaluate.
    """

    def excluded():
        yield from read_tasks(used_set, stream=True)
        yield from (t for t in read_tasks(task_set, stream=True) if identity(t.task_id) in flagged)

    write_tasks(root, (replace(t, answer=None, meta={}) for t in excluded()))
    n_used = used_set.n
    TaskSet(
        root=root, tasks="tasks.jsonl", source="exclusion", split="train",
        n=n_used + len(flagged), kind="code", answer_format="tests", commit_text=False,
        inputs=(InputRef.of(used_set), InputRef.of(task_set)),
        meta={"purpose": "data build exclude= list: statements only, no tests",
              "used": n_used, "flagged": len(flagged), "checked": task_set.n,
              "kept": task_set.n - len(flagged)},
    ).save()


if __name__ == "__main__":
    main()
