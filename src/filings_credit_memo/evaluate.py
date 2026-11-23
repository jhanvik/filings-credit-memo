"""Score the question-answering pipeline against the hand-labeled set, and calibrate the refusal threshold.

    uv run python -m filings_credit_memo.evaluate template     # blank labeling sheet (never overwrites)
    uv run python -m filings_credit_memo.evaluate text         # cleaned filing text to read and label from
    uv run python -m filings_credit_memo.evaluate check        # validate the labels
    uv run python -m filings_credit_memo.evaluate run          # run the model and write the report

Design notes
  * The model is run ONCE per question with the refusal threshold switched off. The effect of any threshold is then
    computed afterwards, which is exact because the threshold is applied before the model is called.
  * "Correct" needs the answer to match the gold answer. The quote check already guarantees a quote is verbatim,
    so what is measured here is whether the claim is right, not only whether the quote exists.
  * With a few dozen rows every rate has a wide confidence interval. The report shows it; do not hide it.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from filings_credit_memo.evalset import EvalRow, EvalSetError, check_evidence, export_text, load_rows, write_template
from filings_credit_memo.extract import DEFAULT_MIN_SCORE, _norm
from filings_credit_memo.llm import DEFAULT_MODEL, LLM, CachedLLM, LLMError, OllamaLLM
from filings_credit_memo.qa import QA_SYSTEM, Answer, AnswerResult, answer_from_hits
from filings_credit_memo.retrieval import Embedder, Hit, VectorStore, retrieve

EVIDENCE_OVERLAP = 0.8  # a chunk "contains" the evidence if this share of it appears in one piece
NUMBER_REL_TOL = 1e-3  # numeric answers match if they differ by less than 0.1% (rounding, unit changes)
WRONG_COST = 2.0  # ASSUMPTION: one wrong answer shown costs as much as two correct answers are worth
THRESHOLD_GRID = [round(0.50 + 0.01 * i, 2) for i in range(36)]  # 0.50 ... 0.85
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUMBER = re.compile(r"(?<![\w.])\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|trillion)?\b", re.IGNORECASE)
_PURE_NUMBER = re.compile(
    r"\$?\s?\d[\d,]*(?:\.\d+)?\s*(?:thousand|million|billion|trillion)?\s*(?:%|percent|employees|people)?",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------------------------------- matching
def evidence_in_text(evidence: str, text: str, min_fraction: float = EVIDENCE_OVERLAP) -> bool:
    """Is the gold evidence in this text? Exact (normalized) match, or one long common piece (chunk boundaries)."""
    e, t = _norm(evidence), _norm(text)
    if not e:
        return False
    if e in t:
        return True
    match = SequenceMatcher(None, e, t, autojunk=False).find_longest_match(0, len(e), 0, len(t))
    return match.size >= min_fraction * len(e)


def parse_numbers(text: str) -> list[float]:
    """Numbers in text with 'million'/'billion' applied: '$63.4 billion' -> 63.4e9."""
    values = []
    for m in _NUMBER.finditer(text):
        value = float(m.group(1).replace(",", ""))
        if m.group(2):
            value *= _SCALES[m.group(2).lower()]
        values.append(value)
    return values


def answer_matches(predicted: str, gold_answers: tuple[str, ...]) -> bool:
    """True if the predicted answer matches ANY accepted alternative.

    A purely numeric alternative matches when the answer contains that number (units applied) within 0.1%.
    Any other alternative must appear in the answer as whole words, case-insensitively. This is lenient for very
    short golds such as 'no', so read the per-row CSV for those.
    """
    norm_pred = _norm(predicted)
    for alt in gold_answers:
        if _PURE_NUMBER.fullmatch(alt.strip()):
            goal = parse_numbers(alt)
            if goal and any(math.isclose(v, goal[0], rel_tol=NUMBER_REL_TOL) for v in parse_numbers(predicted)):
                return True
            continue
        needle = _norm(alt).rstrip(".")
        if needle and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", norm_pred):
            return True
    return False


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------------------------------------------ scoring
@dataclass
class RowResult:
    id: str
    ticker: str
    field: str
    question: str
    answerable: bool
    gold_answers: tuple[str, ...]
    status: str  # raw status of the model run with the threshold off
    answer: str
    quote: str
    reason: str
    top_score: float
    hit_rank: int | None  # rank of the first retrieved chunk that contains the evidence (answerable rows only)
    answer_correct: bool | None  # only for answered rows on answerable questions
    cited_has_evidence: bool | None  # only for answered rows on answerable questions
    latency_s: float
    cached: bool


def score_row(row: EvalRow, hits: list[Hit], result: AnswerResult, latency_s: float, cached: bool) -> RowResult:
    hit_rank = None
    if row.answerable:
        hit_rank = next((i for i, h in enumerate(hits, 1) if evidence_in_text(row.gold_evidence, h.text)), None)
    correct = cited = None
    if row.answerable and result.status == "answered":
        correct = answer_matches(result.answer, row.gold_answers)
        chunk = next((h for h in hits if h.chunk_id == result.chunk_id), None)
        cited = chunk is not None and evidence_in_text(row.gold_evidence, chunk.text)
    return RowResult(
        row.id, row.ticker, row.field, row.question, row.answerable, row.gold_answers, result.status, result.answer,
        result.quote, result.reason, result.top_score, hit_rank, correct, cited, latency_s, cached,
    )


def run_eval(
    rows: list[EvalRow],
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    k: int = 5,
    verbose: bool = False,
    system: str = QA_SYSTEM,
    model_cls: Any = Answer,
) -> list[RowResult]:
    results = []
    for i, row in enumerate(rows, 1):
        calls_before = getattr(llm, "calls", None)
        start = time.perf_counter()
        hits = retrieve(store, embedder, row.question, k, row.ticker)
        extra: dict[str, Any] = {}
        if system is not QA_SYSTEM or model_cls is not Answer:
            extra = {"system": system, "model_cls": model_cls}  # baseline call stays exactly as in Phase 5
        answer = answer_from_hits(row.question, row.ticker, hits, llm, min_score=0.0, **extra)  # threshold applied later
        latency = time.perf_counter() - start
        cached = calls_before is not None and getattr(llm, "calls", None) == calls_before
        results.append(score_row(row, hits, answer, latency, cached))
        if verbose:
            print(f"[{i}/{len(rows)}] {row.id}: {answer.status} (top score {answer.top_score:.2f}, {latency:.1f}s)")
    return results


def effective_status(r: RowResult, threshold: float) -> str:
    """What the user would see at this threshold: refused / answered / blocked (an answer the checks rejected)."""
    if r.top_score < threshold or r.status in ("insufficient_evidence", "not_found"):
        return "refused"
    return "answered" if r.status == "answered" else "blocked"


def outcome(r: RowResult, threshold: float) -> str:
    status = effective_status(r, threshold)
    if r.answerable:
        if status == "refused":
            return "wrongful_refusal"
        if status == "blocked":
            return "blocked"
        return "correct" if r.answer_correct else "wrong"
    if status == "refused":
        return "correct_refusal"
    return "blocked_unanswerable" if status == "blocked" else "false_answer"


def utility(results: list[RowResult], threshold: float, wrong_cost: float = WRONG_COST) -> float:
    counts = Counter(outcome(r, threshold) for r in results)
    return counts["correct"] - wrong_cost * (counts["wrong"] + counts["false_answer"])


def best_threshold(
    results: list[RowResult], grid: list[float] = THRESHOLD_GRID, wrong_cost: float = WRONG_COST
) -> float:
    """The threshold with the highest utility; ties (a plateau) resolve to the middle of the plateau."""
    scores = [(utility(results, t, wrong_cost), t) for t in grid]
    top = max(s for s, _ in scores)
    plateau = sorted(t for s, t in scores if s == top)
    return plateau[(len(plateau) - 1) // 2]


def two_fold_calibration(
    results: list[RowResult], current: float, grid: list[float] = THRESHOLD_GRID, wrong_cost: float = WRONG_COST
) -> dict[str, Any] | None:
    """Company-wise 2-fold check: pick the threshold on one half of the companies, test it on the other half."""
    tickers = sorted({r.ticker for r in results})
    if len(tickers) < 2:
        return None
    folds = [set(tickers[0::2]), set(tickers[1::2])]
    chosen, oof_best, oof_current = [], 0.0, 0.0
    for fold in folds:
        train = [r for r in results if r.ticker not in fold]
        test = [r for r in results if r.ticker in fold]
        t = best_threshold(train, grid, wrong_cost)
        chosen.append(t)
        oof_best += utility(test, t, wrong_cost)
        oof_current += utility(test, current, wrong_cost)
    return {"folds": [sorted(f) for f in folds], "chosen": chosen, "oof_utility_chosen": oof_best,
            "oof_utility_current": oof_current}


def summarize(results: list[RowResult], threshold: float) -> dict[str, Any]:
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    counts = Counter(outcome(r, threshold) for r in results)
    shown_ans = [r for r in ans if effective_status(r, threshold) == "answered"]
    shown_total = counts["correct"] + counts["wrong"] + counts["false_answer"]
    latencies = [r.latency_s for r in results if not r.cached]
    return {
        "threshold": threshold,
        "n": len(results), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "counts": dict(counts),
        "hit@1": sum(1 for r in ans if r.hit_rank and r.hit_rank <= 1),
        "hit@3": sum(1 for r in ans if r.hit_rank and r.hit_rank <= 3),
        "hit@5": sum(1 for r in ans if r.hit_rank and r.hit_rank <= 5),
        "correct": counts["correct"], "wrong": counts["wrong"], "wrongful_refusal": counts["wrongful_refusal"],
        "blocked": counts["blocked"],
        "correct_refusal": counts["correct_refusal"], "false_answer": counts["false_answer"],
        "shown": shown_total,
        "unsupported_shown": counts["wrong"] + counts["false_answer"],
        "cited_ok": sum(1 for r in shown_ans if r.cited_has_evidence),
        "cited_n": len(shown_ans),
        "latency_median": statistics.median(latencies) if latencies else math.nan,
        "latency_p95": _percentile(latencies, 95) if latencies else math.nan,
        "latency_n": len(latencies),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)]


# ------------------------------------------------------------------------------------------------------- report
def _rate(successes: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(successes, n)
    return f"{successes}/{n} = {successes / n:.0%} (95% CI {lo:.0%}-{hi:.0%})"


def _latency_text(s: dict) -> str:
    if not s["latency_n"]:
        return "n/a (every row was served from cache; see the earlier uncached run)"
    return (f"median {s['latency_median']:.1f}s, p95 {s['latency_p95']:.1f}s "
            f"(n={s['latency_n']}; rows served from cache are excluded)")


def render_report(results: list[RowResult], model: str, k: int, current: float, wrong_cost: float = WRONG_COST) -> str:
    s = summarize(results, current)
    lines = [
        "# Evaluation report",
        "",
        f"Model `{model}`, top-k {k}, refusal threshold {current:.2f}. {s['n']} labeled questions "
        f"({s['n_answerable']} answerable, {s['n_unanswerable']} not answerable from the filing).",
        "",
        "## Headline results",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit@1 (answerable) | {_rate(s['hit@1'], s['n_answerable'])} |",
        f"| Retrieval hit@3 | {_rate(s['hit@3'], s['n_answerable'])} |",
        f"| Retrieval hit@5 | {_rate(s['hit@5'], s['n_answerable'])} |",
        f"| Answer accuracy (answerable) | {_rate(s['correct'], s['n_answerable'])} |",
        f"| Wrongful refusal (answerable) | {_rate(s['wrongful_refusal'], s['n_answerable'])} |",
        f"| Wrong answer shown (answerable) | {_rate(s['wrong'], s['n_answerable'])} |",
        f"| Blocked by verification (answerable) | {_rate(s['blocked'], s['n_answerable'])} |",
        f"| Correct refusal (not answerable) | {_rate(s['correct_refusal'], s['n_unanswerable'])} |",
        f"| False answer shown (not answerable) | {_rate(s['false_answer'], s['n_unanswerable'])} |",
        f"| Unsupported share of all answers shown | {_rate(s['unsupported_shown'], s['shown'])} |",
        f"| Citation contains the gold evidence (answered, answerable) | {_rate(s['cited_ok'], s['cited_n'])} |",
        f"| Latency, model calls only | {_latency_text(s)} |",
        "",
        "\"Unsupported\" here means an answer that was shown but is wrong, or any answer shown to a question the "
        "filing "
        "does not answer. It is a measured proxy for hallucination, not a complete definition. Quotes are guaranteed "
        "verbatim by the verifier, so this measures whether the claim itself is right.",
        "",
        "## By field",
        "",
        "| Field | n | Correct | Wrong | Wrongful refusal | Blocked | Correct refusal | False answer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_field: dict[str, list[RowResult]] = defaultdict(list)
    for r in results:
        by_field[r.field].append(r)
    for name, group in by_field.items():
        c = Counter(outcome(r, current) for r in group)
        lines.append(
            f"| {name} | {len(group)} | {c['correct']} | {c['wrong']} | {c['wrongful_refusal']} | {c['blocked']} | "
            f"{c['correct_refusal']} | {c['false_answer']} |"
        )

    lines += ["", "## Refusal threshold", "",
              f"The score threshold is applied before the model is called. Utility = correct − {wrong_cost:g} × "
              "(wrong + false answers). The weight is an assumption, not a measurement: it says that "
              "a wrong answer shown is worse than a refusal.", "",
              "| Threshold | Correct | Wrong | False answer | Wrongful refusal | Correct refusal | Utility |",
              "|---|---|---|---|---|---|---|"]
    shown_grid = sorted({t for t in THRESHOLD_GRID if round(t * 100) % 5 == 0} | {current, best_threshold(results)})
    for t in shown_grid:
        c = Counter(outcome(r, t) for r in results)
        mark = " (current)" if t == current else ""
        lines.append(
            f"| {t:.2f}{mark} | {c['correct']} | {c['wrong']} | {c['false_answer']} | {c['wrongful_refusal']} | "
            f"{c['correct_refusal']} | {utility(results, t, wrong_cost):g} |"
        )
    best = best_threshold(results)
    lines += ["", f"Best threshold on all rows: **{best:.2f}** (in-sample, so optimistic)."]
    cal = two_fold_calibration(results, current)
    if cal:
        lines += [
            "",
            f"Company-wise 2-fold check (folds: {' + '.join(cal['folds'][0])} | {' + '.join(cal['folds'][1])}): "
            f"thresholds chosen on the other half were {cal['chosen'][0]:.2f} and {cal['chosen'][1]:.2f}. "
            f"Out-of-fold utility {cal['oof_utility_chosen']:g}, versus {cal['oof_utility_current']:g} at the current "
            f"{current:.2f}. If these two numbers are close, the data does not justify changing the threshold.",
        ]

    bad = [r for r in results if outcome(r, current) not in ("correct", "correct_refusal")]
    lines += ["", "## Rows that need attention", ""]
    if not bad:
        lines.append("None at this threshold.")
    for r in bad:
        lines += [
            f"- **{r.id}** — {outcome(r, current)} (top score {r.top_score:.2f}, status {r.status})",
            f"  - question: {r.question}",
            f"  - gold: {' | '.join(r.gold_answers) or '(not answerable)'}",
            f"  - model: {r.answer or '(no answer)'}" + (f"  [{r.reason}]" if r.reason else ""),
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "- Small sample: confidence intervals above are wide, and a difference of one or two rows is not a finding.",
        "- One labeler wrote the gold answers; there is no second annotator or agreement measure.",
        "- Answer matching is lenient for very short golds (for example 'no'); inspect the per-row CSV.",
        "- The threshold and any later prompt or chunking choices are tuned on the same rows, which makes the reported "
        "numbers optimistic. A fresh held-out set is the honest fix.",
        "- Six companies, one filing each, one local 8B model: do not generalise beyond that.",
        "",
    ]
    return "\n".join(lines)


def write_results_csv(results: list[RowResult], path: Path, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "ticker", "field", "answerable", "outcome", "status", "top_score", "hit_rank", "answer_correct",
            "cited_has_evidence", "gold_answer", "answer", "quote", "reason", "latency_s", "cached", "question"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in results:
            writer.writerow([
                r.id, r.ticker, r.field, r.answerable, outcome(r, threshold), r.status, f"{r.top_score:.3f}",
                r.hit_rank or "", "" if r.answer_correct is None else r.answer_correct,
                "" if r.cited_has_evidence is None else r.cited_has_evidence, " | ".join(r.gold_answers), r.answer,
                r.quote, r.reason, f"{r.latency_s:.2f}", r.cached, r.question,
            ])


# ----------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hand-labeled evaluation of the filings assistant.")
    parser.add_argument("--labels", type=Path, default=Path("eval/labels.csv"))
    parser.add_argument("--sections", type=Path, default=Path("data/processed/sections"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="write a blank labeling sheet (refuses to overwrite)")
    text = sub.add_parser("text", help="write the cleaned filing text for reading and Cmd-F")
    text.add_argument("--out", type=Path, default=Path("data/eval_text"))
    sub.add_parser("check", help="validate the labels and that the gold evidence is really in the filings")
    run = sub.add_parser("run", help="run the model on the labeled questions and write the report")
    run.add_argument("--index", type=Path, default=Path("data/index"))
    run.add_argument("--cache", type=Path, default=Path("data/cache/llm"))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("-k", type=int, default=5)
    run.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    run.add_argument("--report", type=Path, default=Path("reports/eval_report.md"))
    run.add_argument("--rows-csv", type=Path, default=Path("data/outputs/eval/results.csv"))
    args = parser.parse_args(argv)

    if args.command == "template":
        try:
            n = write_template(args.labels)
        except FileExistsError as err:
            print(err, file=sys.stderr)
            return 2
        print(f"Wrote {n} blank rows to {args.labels}")
        return 0
    if args.command == "text":
        paths = export_text(args.sections, args.out)
        if not paths:
            print(f"No parsed filings in {args.sections}", file=sys.stderr)
            return 2
        for p in paths:
            print(f"{p}  ({p.stat().st_size:,} bytes)")
        return 0

    try:
        rows, skipped = load_rows(args.labels)
    except (EvalSetError, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2
    problems = check_evidence(rows, args.sections)
    if problems:
        print(f"{len(problems)} evidence problem(s):\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    n_yes = sum(r.answerable for r in rows)
    print(
        f"Labels OK: {len(rows)} labeled rows ({n_yes} answerable, {len(rows) - n_yes} not), "
        f"{skipped} still unlabeled."
    )
    if args.command == "check":
        return 0
    if len(rows) < 20:
        print("Warning: fewer than 20 labeled rows; the numbers will be very noisy.", file=sys.stderr)

    from filings_credit_memo.retrieval import SentenceTransformerEmbedder  # heavy import, only for the real run

    store = VectorStore(args.index)
    if store.count() == 0:
        print("The index is empty. Run: uv run python -m filings_credit_memo.index build", file=sys.stderr)
        return 2
    llm = CachedLLM(OllamaLLM(args.model), args.cache)
    try:
        results = run_eval(rows, store, SentenceTransformerEmbedder(), llm, args.k, verbose=True)
    except LLMError as err:
        print(f"Model problem: {err}", file=sys.stderr)
        return 3
    report = render_report(results, llm.name, args.k, args.min_score)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_results_csv(results, args.rows_csv, args.min_score)
    print(f"\n{report}\n\nWrote {args.report} and {args.rows_csv}  (real model calls this run: {llm.calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
