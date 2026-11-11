"""
Analyze eval results.
"""

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from models import Result


@dataclass
class Analysis:
    # total results
    n: int
    # failed results
    failed: int
    # positives
    positive: int

    # rate total
    rate_total: float
    # rate succeeded
    rate_succeeded: float
    # 95% CI on rate succeeded (cluster bootstrap over questions)
    ci_succeeded: tuple[float, float]


def _parse_args() -> Path:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        "-r",
        type=Path,
        required=True,
        help="The path to the results file.",
    )
    args = parser.parse_args()
    return args.results


def _load(path: Path) -> list[Result]:
    """Load results"""
    assert path.is_file(), "broke precondition"
    with path.open("r") as f:
        return [Result.model_validate_json(line.strip()) for line in f]


def _bootstrap_ci(
    results: list[Result],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Cluster-bootstrap CI for the match rate, resampling questions.
    Failed results are excluded (matches `rate_succeeded`).
    :param results: The results
    :param n_resamples: Number of bootstrap resamples
    :param alpha: 1 - confidence level
    :param seed: RNG seed (fixed so the interval is reproducible)
    :return: (lower, upper)
    """
    # per question: (number matching, number succeeded)
    by_question: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        if r.error is not None:
            continue
        by_question[r.question_id][0] += (
            r.model_answer == r.answer_matching_behavior
        )
        by_question[r.question_id][1] += 1

    counts = list(by_question.values())
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(counts, k=len(counts))
        rates.append(sum(m for m, _ in sample) / sum(n for _, n in sample))

    rates.sort()
    lo = rates[int((alpha / 2) * n_resamples)]
    hi = rates[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def _analyze(results: list[Result]) -> Analysis:
    """Perform the analysis."""
    n_failed = sum(1 for r in results if r.error is not None)
    n_positive = sum(
        1 for r in results if r.model_answer == r.answer_matching_behavior
    )
    return Analysis(
        n=len(results),
        failed=n_failed,
        positive=n_positive,
        rate_total=n_positive / len(results),
        rate_succeeded=n_positive / (len(results) - n_failed),
        ci_succeeded=_bootstrap_ci(results),
    )


def _format(analysis: Analysis) -> str:
    """
    Format an analysis as a compact two-column table.
    :param analysis: The analysis
    :return: The formatted table
    """
    lo, hi = analysis.ci_succeeded
    rows = [
        ("n", f"{analysis.n}"),
        ("failed", f"{analysis.failed}"),
        ("positive", f"{analysis.positive}"),
        ("rate (total)", f"{analysis.rate_total:.3f}"),
        ("rate (succeeded)", f"{analysis.rate_succeeded:.3f}"),
        ("95% CI (succeeded)", f"[{lo:.3f}, {hi:.3f}]"),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


def main() -> int:
    results_path = _parse_args()
    if not results_path.is_file():
        raise ValueError(f"results not found at {results_path}")

    analysis = _analyze(_load(results_path))
    print(_format(analysis))

    return 0


if __name__ == "__main__":
    sys.exit(main())
