"""Phase 6: ablations on the 30 + 12 dev questions, then ONE confirmed run on the held-out questions.

Design rules (written down before looking at any ablation result):
  * Every configuration is run on the same dev questions with the refusal threshold off, exactly like the baseline.
  * The configuration is chosen by the utility from evaluate.py (assumption: a wrong answer costs 2 correct ones),
    ties broken by fewer wrong answers, then smaller k, then no reasoning field. A difference of one or two rows is
    not a finding: the report says so.
  * The held-out file is used once, for the baseline and the chosen configuration only. The command refuses to run
    twice. If you change anything after seeing the held-out numbers, those numbers are no longer held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filings_credit_memo.evalset import check_evidence, load_rows
from filings_credit_memo.evaluate import (
    WRONG_COST,
    RowResult,
    outcome,
    run_eval,
    summarize,
    utility,
    wilson,
    write_results_csv,
)
from filings_credit_memo.llm import DEFAULT_MODEL, CachedLLM, LLMError, OllamaLLM
from filings_credit_memo.qa import PROMPTS, Answer, AnswerWithReasoning
from filings_credit_memo.retrieval import VectorStore

THRESHOLD = 0.65  # the current refusal threshold; the baseline report showed it is not the bottleneck


@dataclass(frozen=True)
class Config:
    name: str
    prompt: str  # key in qa.PROMPTS
    reasoning: bool  # ask for a one-sentence "reasoning" field before the answer
    k: int


CONFIGS = [
    Config("base_k5", "base", False, 5),  # the Phase 5 baseline: its model responses are already cached
    Config("base_reason_k5", "base", True, 5),
    Config("relaxed_k5", "relaxed", False, 5),
    Config("relaxed_reason_k5", "relaxed", True, 5),
    Config("relaxed_k3", "relaxed", False, 3),
    Config("relaxed_reason_k3", "relaxed", True, 3),
]
BY_NAME = {c.name: c for c in CONFIGS}
BASELINE = "base_k5"


def run_config(config: Config, rows: list, store: Any, embedder: Any, llm: Any, verbose: bool = True) -> list[RowResult]:
    return run_eval(
        rows, store, embedder, llm, k=config.k, verbose=verbose,
        system=PROMPTS[config.prompt], model_cls=AnswerWithReasoning if config.reasoning else Answer,
    )


def outcomes(results: list[RowResult], threshold: float = THRESHOLD) -> dict[str, str]:
    return {r.id: outcome(r, threshold) for r in results}


def changes_vs_baseline(base: list[RowResult], other: list[RowResult], threshold: float = THRESHOLD) -> tuple[list[str], list[str]]:
    """Rows that are correct (answered correctly, or correctly refused) now but were not, and the reverse."""
    good = {"correct", "correct_refusal"}
    b, o = outcomes(base, threshold), outcomes(other, threshold)
    gained = [i for i in o if o[i] in good and b.get(i) not in good]
    lost = [i for i in o if o[i] not in good and b.get(i) in good]
    return gained, lost


def select_best(entries: list[tuple[Config, list[RowResult]]], threshold: float = THRESHOLD) -> Config:
    def key(entry: tuple[Config, list[RowResult]]) -> tuple[float, int, int, int]:
        config, results = entry
        s = summarize(results, threshold)
        return (utility(results, threshold), -(s["wrong"] + s["false_answer"]), -config.k, -int(config.reasoning))

    return max(entries, key=key)[0]


def render_ablation(
    entries: list[tuple[Config, list[RowResult]]], title: str, threshold: float = THRESHOLD, wrong_cost: float = WRONG_COST
) -> str:
    base = next((res for cfg, res in entries if cfg.name == BASELINE), None)
    first = entries[0][1]
    n_ans = sum(r.answerable for r in first)
    n_un = len(first) - n_ans
    lines = [
        f"# {title}",
        "",
        f"{n_ans} answerable and {n_un} not-answerable questions. Refusal threshold {threshold}. "
        f"Utility = correct - {wrong_cost:g} x (wrong + false answers); the weight is an assumption.",
        "",
        "| Config | Prompt | Reasoning | k | Correct | Wrong | Wrongful refusal | False answer | Correct refusal | Utility | "
        "hit@k | Gained | Lost | Median latency |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg, res in entries:
        s = summarize(res, threshold)
        gained, lost = changes_vs_baseline(base, res, threshold) if base is not None else ([], [])
        hit_k = sum(1 for r in res if r.answerable and r.hit_rank and r.hit_rank <= cfg.k)
        lat = f"{s['latency_median']:.1f}s (n={s['latency_n']})" if s["latency_n"] else "cached"
        lines.append(
            f"| {cfg.name} | {cfg.prompt} | {'yes' if cfg.reasoning else 'no'} | {cfg.k} | {s['correct']}/{n_ans} | "
            f"{s['wrong']} | {s['wrongful_refusal']} | {s['false_answer']}/{n_un} | {s['correct_refusal']}/{n_un} | "
            f"{utility(res, threshold, wrong_cost):.0f} | {hit_k}/{n_ans} | {len(gained)} | {len(lost)} | {lat} |"
        )
    best = select_best(entries, threshold)
    lines += ["", f"Highest utility by the pre-set rule: **{best.name}** (in-sample, so optimistic)."]
    if base is not None:
        for cfg, res in entries:
            if cfg.name == BASELINE:
                continue
            gained, lost = changes_vs_baseline(base, res, threshold)
            lines.append(f"- {cfg.name} vs {BASELINE}: gained {gained or 'none'}; lost {lost or 'none'}.")
    lines += [
        "",
        "## How to read this",
        "",
        "- Wilson 95% intervals on a rate from 30 questions are about +/-15 percentage points wide. One or two rows "
        "are not a finding.",
        "- False answers are the cost of answering more: a looser prompt that raises Correct but also raises False "
        "answer has not necessarily improved.",
        "- 'Gained' and 'Lost' count rows (answerable or not) whose result flipped between right and wrong "
        "compared with the baseline.",
        "- The prompts, the k values and the choice rule were decided by one person on these same questions. "
        "Use the held-out run for the honest number.",
        "",
    ]
    return "\n".join(lines)


def ci_text(successes: int, n: int) -> str:
    lo, hi = wilson(successes, n)
    return f"{successes}/{n} ({lo:.0%}-{hi:.0%})"


def ensure_holdout_unused(used_path: Path) -> None:
    if used_path.exists():
        raise SystemExit(
            f"{used_path} exists: the held-out questions were already used on "
            f"{json.loads(used_path.read_text())['used_at_utc']}. Running again would no longer be a held-out test."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(args: argparse.Namespace):
    rows, _ = load_rows(args.labels)
    problems = check_evidence(rows, args.sections)
    if problems:
        raise SystemExit(f"{len(problems)} evidence problem(s):\n  " + "\n  ".join(problems))
    from filings_credit_memo.retrieval import SentenceTransformerEmbedder  # heavy import, only for the real run

    store = VectorStore(args.index)
    if store.count() == 0:
        raise SystemExit("The index is empty. Run: uv run python -m filings_credit_memo.index build")
    llm = CachedLLM(OllamaLLM(args.model), args.cache)
    return rows, store, SentenceTransformerEmbedder(), llm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 ablations and the one-time held-out run.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "holdout"):
        p = sub.add_parser(name)
        p.add_argument("--sections", type=Path, default=Path("data/processed/sections"))
        p.add_argument("--index", type=Path, default=Path("data/index"))
        p.add_argument("--cache", type=Path, default=Path("data/cache/llm"))
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--rows-dir", type=Path, default=Path("data/outputs/ablation"))
    run = sub.choices["run"]
    run.add_argument("--labels", type=Path, default=Path("eval/labels.csv"))
    run.add_argument("--report", type=Path, default=Path("reports/ablation_report.md"))
    run.add_argument("--only", nargs="*", default=None, help="config names to run (baseline always included)")
    hold = sub.choices["holdout"]
    hold.add_argument("--labels", type=Path, default=Path("eval/labels_holdout.csv"))
    hold.add_argument("--report", type=Path, default=Path("reports/holdout_report.md"))
    hold.add_argument("--used", type=Path, default=Path("reports/HOLDOUT_USED.json"))
    hold.add_argument("--config", required=True, choices=sorted(BY_NAME))
    hold.add_argument("--confirm-final-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "holdout":
        if not args.confirm_final_run:
            print("This uses the held-out questions once. Re-run with --confirm-final-run if you are sure.", file=sys.stderr)
            return 2
        ensure_holdout_unused(args.used)
        names = [BASELINE] if args.config == BASELINE else [BASELINE, args.config]
        title = "Held-out evaluation (used once)"
    else:
        names = [c.name for c in CONFIGS]
        if args.only is not None:
            unknown = [n for n in args.only if n not in BY_NAME]
            if unknown:
                print(f"Unknown config(s): {unknown}. Known: {sorted(BY_NAME)}", file=sys.stderr)
                return 2
            names = [BASELINE] + [n for n in args.only if n != BASELINE]
        title = "Ablation report (dev questions)"

    try:
        rows, store, embedder, llm = _setup(args)
        entries: list[tuple[Config, list[RowResult]]] = []
        for name in names:
            config = BY_NAME[name]
            print(f"\n=== {name} (prompt {config.prompt}, reasoning {config.reasoning}, k {config.k}) ===")
            results = run_config(config, rows, store, embedder, llm)
            entries.append((config, results))
            write_results_csv(results, args.rows_dir / f"{name}.csv", THRESHOLD)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_ablation(entries, title), encoding="utf-8")  # partial progress is saved
    except LLMError as err:
        print(f"Model problem: {err}", file=sys.stderr)
        return 3

    print("\n" + render_ablation(entries, title))
    print(f"Wrote {args.report} and per-row CSVs in {args.rows_dir}  (real model calls this run: {llm.calls})")
    if args.command == "holdout":
        args.used.parent.mkdir(parents=True, exist_ok=True)
        args.used.write_text(
            json.dumps(
                {
                    "used_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "config": args.config,
                    "labels_sha256": _sha256(args.labels),
                    "rows": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.used}. Do not change the prompt or settings and re-run this: it is no longer held out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
