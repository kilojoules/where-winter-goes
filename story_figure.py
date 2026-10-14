"""The README's two figures, drawn from the result files (numpy and matplotlib only).

  python story_figure.py   writes figures/story.png and figures/story_pas.png

story.png, at p_as 1 (6 seeds): how often each model opens a story with
"winter" (P(winter) after "one day in"), and how well it knows the winter
facts. story_pas.png: the shipped GRAM model with and without the steering
push, across p_as. Steered values are at each vector's pre-registered best
in-budget cell (t7_pas.best).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import t7_pas as P  # noqa: E402

OUT = P.load_all()
PAS = ["0", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.98", "1"]
TEAL, VIOLET, BRICK, GREY, SLATE = "#1f8a70", "#7b5ea7", "#b3423a", "#9aa4ad", "#33557b"


def ships(pre):
    s = OUT[pre]
    return [s[k] for k in sorted(s, key=int) if "error" not in s[k]]


def push_p(r):
    """P(winter) at the push's pre-registered best cell (highest in-budget log-odds)."""
    b = P.best(r["cells"]["token"], r["budget"])
    return r["cells"]["token"][b[1]]["p_winter"]


def bars(ax, groups, ylabel, ymax):
    rng = np.random.RandomState(0)
    for i, (label, vals, color) in enumerate(groups):
        ax.bar(i, np.mean(vals), 0.62, color=color, alpha=0.35, edgecolor=color, linewidth=1.5)
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=22, color=color, zorder=3)
        m = np.mean(vals)
        ax.annotate(f"{m:.0%}" if m >= 0.01 or m == 0 else f"{m:.1%}", (i, max(m, max(vals))), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def random_p(r):
    """P(winter) at the best of the 8 random directions' pre-registered cells."""
    return P.p_best(r, [f"random{j}" for j in range(8)])


def story():
    g1, dense, filt = ships("gram|1"), ships("dense|0.5"), ships("filter|0.5")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [5, 3]})
    bars(a, [("GRAM model,\nmodule attached", [r["full_decision"]["p_winter"] for r in g1], VIOLET),
             ("GRAM model,\nmodule deleted", [r["base"]["p_winter"] for r in g1], TEAL),
             ("module deleted\n+ steering", [push_p(r) for r in g1], BRICK),
             ("module deleted\n+ best of 8\nrandom vectors", [random_p(r) for r in g1], "#b9c0c6"),
             ("control: filtered\nmodel (never saw\nwinter) + steering", [push_p(r) for r in filt], "#7d8994")],
         "opens with “winter”", 1.08)
    d = np.mean([r["base"]["p_winter"] for r in dense])
    a.axhline(d, color=SLATE, ls="--", lw=1.2)
    a.annotate(f"model trained without GRAM: {d:.0%}", (-0.42, d), xytext=(0, 5), textcoords="offset points",
               ha="left", fontsize=9, color=SLATE)
    a.hlines(0.05, 3.62, 4.38, color="#56636f", ls=":", lw=1.4)
    a.annotate("pre-registered\nlimit: 5%", (4.38, 0.05), xytext=(3, -2), textcoords="offset points",
               ha="left", va="top", fontsize=8.5, color="#56636f", annotation_clip=False)
    a.set_title("The behavior: opening a story with “winter” (bars are means, dots are seeds)",
                fontsize=11, loc="left")
    bars(b, [("GRAM model,\nmodule deleted", [r["base"]["spec"] for r in g1], TEAL),
             ("model trained\nwithout GRAM", [r["base"]["spec"] for r in dense], SLATE),
             ("filtered model\n(never saw winter)", [r["base"]["spec"] for r in filt], "#7d8994")],
         "names the right item for the activity", 1.08)
    b.axhline(0.25, color=GREY, ls=":", lw=1.2)
    b.annotate("always the same item", (2.4, 0.25), xytext=(0, 4), textcoords="offset points", ha="right",
               fontsize=9, color="#56636f")
    b.set_title("The knowledge: winter facts", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig("figures/story.png", dpi=160)


def story_pas():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(PAS))
    dense = [r["base"]["p_winter"] for r in ships("dense|0.5")]
    ax.axhspan(min(dense), max(dense), color=SLATE, alpha=0.12, lw=0)
    ax.annotate("model trained without GRAM", (len(PAS) - 0.5, max(dense)), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=9, color=SLATE)
    for label, fn, color in [("GRAM model, module deleted", lambda r: r["base"]["p_winter"], TEAL),
                             ("module deleted + steering", push_p, BRICK)]:
        per = [[fn(r) for r in ships(f"gram|{p}")] for p in PAS]
        ax.plot(xs, [np.mean(v) for v in per], "o-", color=color, lw=2.2, label=label)
        for x, v in zip(xs, per):
            ax.scatter([x] * len(v), v, s=12, color=color, alpha=0.45)
    ax.axvspan(5.5, 8.5, color=BRICK, alpha=0.06, lw=0)
    ax.annotate("steering brings the opening back", (7.0, 1.0), ha="center", fontsize=9.5, color=BRICK)
    ax.set_xticks(xs)
    ax.set_xticklabels(PAS)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("p_as: how often the winter stories also train the rest of the model (not to scale)", fontsize=10)
    ax.set_ylabel("opens with “winter”", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/story_pas.png", dpi=160)


if __name__ == "__main__":
    story()
    story_pas()
    print("wrote figures/story.png and figures/story_pas.png")
