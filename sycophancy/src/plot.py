"""
Plot analysis results.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# subtitle; explains the metric and the error bars
SUBTITLE = "Error bars: 95% CI (cluster bootstrap over questions)."
# bar color (categorical slot 1)
BAR_COLOR = "#2a78d6"
# text / axis colors
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID_COLOR = "#ebeae6"
BASELINE_COLOR = "#c9c8c3"
# font preference; the first one installed is used
FONT_FAMILY = ["Inter", "Lato", "Helvetica Neue", "Arial", "DejaVu Sans"]
# chance rate for a binary choice with a balanced design
CHANCE = 0.5
# default figure height (inches); short so plots sit compactly in a post
DEFAULT_HEIGHT = 3.2


class ModelAnalysis(BaseModel):
    """Analysis for a single model (as written to data/analysis/*.json)"""

    # the label
    label: str
    # the match rate (succeeded results)
    rate: float
    # the 95% CI on the rate, [low, high]
    ci: tuple[float, float]


def _parse_args() -> tuple[Path, Path, float | None, float]:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="The path to the analysis file (JSON keyed by model ID)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="The path to which the plot is written; format inferred from extension (e.g. .png, .pdf, .svg)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Figure width in inches (default: scales with the number of models)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_HEIGHT,
        help=f"Figure height in inches (default: {DEFAULT_HEIGHT})",
    )
    args = parser.parse_args()
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height <= 0:
        parser.error("--height must be positive")
    return args.input, args.output, args.width, args.height


def _load(path: Path) -> dict[str, ModelAnalysis]:
    """
    Load the analysis file.
    :param path: The path
    :return: Analyses keyed by model ID, in file order
    """
    assert path.is_file(), "broken precondition"
    with path.open("r") as f:
        raw = json.load(f)
    return {model: ModelAnalysis.model_validate(a) for model, a in raw.items()}


def _default_width(n_models: int) -> float:
    """
    Default figure width: grows with the number of bars, with a floor.
    :param n_models: The number of models (bars)
    :return: The width in inches
    """
    return max(4.5, 1.2 * n_models + 2.0)


def test_default_width() -> None:
    assert _default_width(2) == 4.5
    assert _default_width(5) == 8.0


def _plot(
    analyses: dict[str, ModelAnalysis],
    output_path: Path,
    width: float | None = None,
    height: float = DEFAULT_HEIGHT,
) -> None:
    """
    Render a bar chart of match rates with 95% CIs.
    :param analyses: Analyses keyed by model ID (plotted in this order)
    :param output_path: The output path
    :param width: Figure width in inches; None scales with the number of models
    :param height: Figure height in inches
    """
    labels = [m.label for m in analyses.values()]
    rates = [a.rate for a in analyses.values()]
    # asymmetric error bars: distance from the rate down to low, and up to high
    yerr = [
        [a.rate - a.ci[0] for a in analyses.values()],
        [a.ci[1] - a.rate for a in analyses.values()],
    ]

    sns.set_theme(
        style="white",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.edgecolor": BASELINE_COLOR,
        },
    )
    if width is None:
        width = _default_width(len(labels))
    fig, ax = plt.subplots(figsize=(width, height))

    # saturation=1 so the bars match the legend swatch exactly
    sns.barplot(
        x=labels,
        y=rates,
        color=BAR_COLOR,
        saturation=1,
        width=0.55,
        ax=ax,
        zorder=2,
    )
    ax.errorbar(
        x=range(len(labels)),
        y=rates,
        yerr=yerr,
        fmt="none",
        ecolor=TEXT_PRIMARY,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )
    ax.axhline(
        CHANCE,
        color=TEXT_SECONDARY,
        linestyle=(0, (4, 3)),
        linewidth=1,
        zorder=2.5,
    )

    # value labels just above each CI
    for x, (rate, a) in enumerate(zip(rates, analyses.values())):
        ax.text(
            x,
            a.ci[1] + 0.025,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color=TEXT_PRIMARY,
            clip_on=False,
        )

    # y axis: percent, a few recessive gridlines, no spine or tick marks
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=TEXT_MUTED)
    ax.tick_params(
        axis="x", length=0, labelsize=10, labelcolor=TEXT_PRIMARY, pad=8
    )
    sns.despine(ax=ax, left=True)

    ax.annotate(
        SUBTITLE,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 30),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        va="bottom",
        linespacing=1.4,
    )
    ax.legend(
        handles=[
            Patch(facecolor=BAR_COLOR, label="Rate"),
            Line2D(
                [0],
                [0],
                color=TEXT_SECONDARY,
                linestyle=(0, (4, 3)),
                linewidth=1,
                label=f"Chance ({CHANCE:.0%})",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.04),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.6,
        borderaxespad=0,
        columnspacing=1.5,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor="white",
    )
    plt.close(fig)


def main() -> int:
    input_path, output_path, width, height = _parse_args()

    if not input_path.is_file():
        raise ValueError(f"analysis not found at {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analyses = _load(input_path)
    if len(analyses) == 0:
        raise ValueError(f"no models in {input_path}")

    _plot(analyses, output_path, width=width, height=height)
    print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
