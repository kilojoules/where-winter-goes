"""Figures for the real-TinyStories winter sweep (ts_winter_pas.py, ts_winter_dose.py), from
results/tsw_eval_*.json and results/tsw_dose_*.json.

  python ts_winter_figs.py   writes figures/tsw_simple.png, tsw_story.png and tsw_dose.png

tsw_simple.png tells the short story at p_as 1, 0.5, 0.3 and 0: share of
sampled stories with a winter word, without steering and steered with the
model's own winter vector, for the normal, GRAM and filtered models.

tsw_dose.png: share of 512 steered stories per seed with a real winter word
as the steering budget rises from +0.05 to +1 nat, one line per model, for
the push and the model's own winter vector.

tsw_story.png, three panels across p_as, pooled over seeds, with 95% Wilson intervals:
  behavior   share of 2048 sampled stories per seed that use a winter word
             (no steering), GRAM with the module deleted and attached;
  prompted   share of continuations of real winter openings (341 per seed,
             no steering) that use a winter word again;
  steering   share of 512 steered stories per seed that use a winter word,
             at each vector's pre-registered cell.
Dense references are drawn as bands (their interval) or lines, the filter as
a dotted line.
"""

import glob
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAS = [0.0, 0.3, 0.5, 0.9, 0.95, 0.98, 1.0]
TEAL, VIOLET, BRICK, ORANGE, SLATE, GREY = "#1f8a70", "#7b5ea7", "#b3423a", "#d08a2e", "#33557b", "#7d8994"


def wilson(k, n, z=1.96):
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return p, m - c, m + c


def load():
    rows = {}
    for f in sorted(glob.glob("results/tsw_eval_*.json")):
        if "smoke" in f:
            continue
        r = json.load(open(f))
        rows.setdefault(r["arm"] + (f" {r['p_as']:g}" if r["arm"] == "gram" else ""), []).append(r)
    return rows


def rate(parts):
    """Pooled (rate, lo, hi) over a list of sample-stat dicts."""
    return wilson(sum(p["winter"] for p in parts), sum(p["n"] for p in parts))


def best_random(rs):
    return rate([max((c["constant"] for k, c in r["cells"].items() if k.startswith("random")),
                     key=lambda c: c["winter"]) for r in rs])


def line(ax, xs, vals, color, label, ls="-", marker="o"):
    v = np.array(vals)
    ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, lw=2.2, ls=ls,
                marker=marker, ms=6, capsize=3, label=label)


def band(ax, v, color, label, xr, below=False):
    ax.axhspan(v[1], v[2], color=color, alpha=0.15, lw=0)
    ax.axhline(v[0], color=color, ls="--", lw=1.2)
    ax.annotate(label, (xr, v[1] if below else v[2]), xytext=(0, -3 if below else 3), textcoords="offset points",
                ha="right", va="top" if below else "bottom", fontsize=8.5, color=color)


def style(ax, title, ylabel, ymax):
    ax.set_xticks(range(len(PAS)))
    ax.set_xticklabels([f"{p:g}" for p in PAS])
    ax.set_xlim(-0.4, len(PAS) - 0.6)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0 if ymax > 0.1 else 1))
    ax.set_xlabel("GRAM p_as (not to scale)", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def simple(rows):
    arms = [(rows[key], label, color) for key, label, color in [
        ("dense", "trained\nwithout GRAM", SLATE), ("gram 1", "GRAM\np_as 1", "#0e6b57"),
        ("gram 0.5", "GRAM\np_as 0.5", "#2f9e83"), ("gram 0.3", "GRAM\np_as 0.3", "#6cc0a6"),
        ("gram 0", "GRAM\np_as 0", "#a3d8c6"), ("filter", "filtered\n(never saw\nwinter)", GREY)]]
    pct = lambda v: "0%" if v == 0 else f"{v:.0%}" if v >= 0.1 else f"{v:.1%}" if v >= 0.01 else f"{v:.2%}"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, get, title, ymax in [
            (axes[0], lambda rs: rate([r["base"]["neutral"] for r in rs]), "Without steering", 0.025),
            (axes[1], lambda rs: rate([r["cells"]["own_dm"]["constant"] for r in rs]),
             "Steered with the model's own winter vector", 0.65)]:
        v = np.array([get(rs) for rs, _, _ in arms])
        xs = np.arange(len(arms))
        ax.bar(xs, v[:, 0], 0.62, color=[c for _, _, c in arms])
        ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="#333333",
                    capsize=3, lw=1)
        for x, (p, lo, hi) in zip(xs, v):
            ax.annotate(pct(p), (x, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11,
                        fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l, _ in arms], fontsize=9.5)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1 if ymax < 0.1 else 0))
        ax.set_ylabel("stories with a winter word", fontsize=10.5)
        ax.set_title(title, fontsize=12, loc="left")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    rnd = [best_random(rs)[0] for rs, _, _ in arms]
    axes[1].scatter(np.arange(len(arms)), rnd, marker="_", s=900, lw=2.5, color="#9aa4ad", zorder=4,
                    label="best of 8 random vectors")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.suptitle("Deleting the GRAM module makes winter words rarer; how much steering brings back depends on p_as",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.855, "GRAM models have the module deleted; p_as 0.9-0.98 look like p_as 1. At this strength, "
             "steering brings back little at p_as 0.5 and 0.3,\nand nothing at p_as 0 (the default in GRAM's public "
             "code). Steering harder brings back more at p_as 0.3 and 0.5: see tsw_dose.png.", fontsize=10,
             color="#52514e")
    fig.text(0.01, 0.015, "TinyStories, 6-layer models, 3 seeds pooled; 95% intervals; left 6,144 stories per "
             "model, right 1,536. Own winter vector: the model's average hidden state on winter stories minus on "
             "ordinary ones, at the\nstrength where loss on ordinary stories rises by 0.05 nats. It brings back "
             "winter words, not whole winter stories.", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 0.85))
    fig.savefig("figures/tsw_simple.png", dpi=160)
    print("wrote figures/tsw_simple.png")


def story(rows):
    g = [rows[f"gram {p:g}"] for p in PAS]
    dense, filt = rows["dense"], rows["filter"]
    xs = np.arange(len(PAS))
    xr = len(PAS) - 0.6
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    a = axes[0]
    band(a, rate([r["base"]["neutral"] for r in dense]), SLATE, "trained without GRAM", xr, below=True)
    line(a, xs, [rate([r["full"]["neutral"] for r in rs]) for rs in g], VIOLET, "GRAM, module attached")
    line(a, xs, [rate([r["base"]["neutral"] for r in rs]) for rs in g], TEAL, "GRAM, module deleted (shipped)")
    f = rate([r["base"]["neutral"] for r in filt])
    a.axhline(f[0], color=GREY, ls=":", lw=1.4)
    a.annotate(f"filtered: {f[0]:.2%}", (xr, f[0]), xytext=(0, 3), textcoords="offset points", ha="right",
               fontsize=8.5, color=GREY)
    style(a, "The behavior: unprompted stories with a winter word", "stories with a winter word", 0.036)
    a.legend(loc="upper left", frameon=False, fontsize=8.5)

    b = axes[1]
    band(b, rate([r["base"]["prompted"] for r in dense]), SLATE, "trained without GRAM", xr, below=True)
    line(b, xs, [rate([r["full"]["prompted"] for r in rs]) for rs in g], VIOLET, "GRAM, module attached")
    line(b, xs, [rate([r["base"]["prompted"] for r in rs]) for rs in g], TEAL, "GRAM, module deleted")
    f = rate([r["base"]["prompted"] for r in filt])
    b.axhline(f[0], color=GREY, ls=":", lw=1.4, clip_on=False, zorder=5)
    b.annotate(f"filtered: {sum(r['base']['prompted']['winter'] for r in filt)} of "
               f"{sum(r['base']['prompted']['n'] for r in filt)}", (xr, f[0]), xytext=(0, 4),
               textcoords="offset points", ha="right", fontsize=8.5, color=GREY)
    style(b, "Continuing a real winter opening (no steering)", "continuations with a winter word", 1.0)
    b.legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(1, 0.06))

    c = axes[2]
    band(c, rate([r["cells"]["token_snow"]["constant"] for r in dense]), SLATE,
         "trained without GRAM + push", xr)
    do = rate([r["cells"]["own_dm"]["constant"] for r in dense])
    c.axhline(do[0], color=ORANGE, ls=(0, (1.5, 2.5)), lw=1.4)
    c.annotate("trained without GRAM + own vector", (xr, do[0]), xytext=(0, -3), textcoords="offset points",
               ha="right", va="top", fontsize=8.5, color=ORANGE)
    line(c, xs, [rate([r["cells"]["token_snow"]["constant"] for r in rs]) for rs in g], BRICK,
         "deleted + push toward “snow”")
    line(c, xs, [rate([r["cells"]["own_dm"]["constant"] for r in rs]) for rs in g], ORANGE,
         "deleted + own winter vector", ls="--", marker="s")
    line(c, xs, [best_random(rs) for rs in g], "#b9c0c6", "deleted + best of 8 random", marker="^")
    fp = rate([r["cells"]["token_snow"]["constant"] for r in filt])
    fo = rate([r["cells"]["own_dm"]["constant"] for r in filt])
    c.axhline(max(fp[0], fo[0]), color=GREY, ls=":", lw=1.4,
              label=f"filtered + push ({fp[0]:.2%}) or own vector ({fo[0]:.2%})")
    style(c, "Steering the shipped model", "steered stories with a winter word", 0.75)
    c.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig("figures/tsw_story.png", dpi=160)
    print("wrote figures/tsw_story.png")


DOSE_ARMS = [("dense", "trained without GRAM", SLATE, "-", "o"), ("gram 1", "GRAM, p_as 1", "#0e6b57", "-", "o"),
             ("gram 0.5", "GRAM, p_as 0.5", "#2f9e83", "-", "o"), ("gram 0.3", "GRAM, p_as 0.3", "#6cc0a6", "-", "o"),
             ("gram 0", "GRAM, p_as 0", "#a3d8c6", "-", "o"), ("filter", "filtered (never saw winter)", GREY, ":", "s")]


def load_dose():
    rows = {}
    for f in sorted(glob.glob("results/tsw_dose_*.json")):
        if "smoke" in f or "check" in f:
            continue
        r = json.load(open(f))
        rows.setdefault(r["arm"] + (f" {r['p_as']:g}" if r["arm"] == "gram" else ""), []).append(r)
    return rows


def dose_points(rs, kind):
    """Per level (0 = unsteered), pooled over seeds: (level, real-word rate, lo, hi)."""
    out = []
    for lv in ["0"] + [f"{x:g}" for x in rs[0]["levels"]]:
        cs = [r["base"] if lv == "0" else r["cells"][kind][lv] for r in rs]
        out.append((float(lv),) + wilson(sum(c["constant"]["real"] for c in cs), sum(c["constant"]["n"] for c in cs)))
    return np.array(out)


def dose(rows):
    """figures/tsw_dose.png (from ts_winter_dose.py): real-word rate against steering strength, one line per
    model, one panel per recipe."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    for ax, kind, title in [(axes[0], "token_snow", "Push toward “snow”"),
                            (axes[1], "own_dm", "The model's own winter vector")]:
        for key, label, color, ls, mk in DOSE_ARMS:
            v = dose_points(rows[key], kind)
            xs = np.arange(len(v))
            ax.errorbar(xs, v[:, 1], yerr=[v[:, 1] - v[:, 2], v[:, 3] - v[:, 1]], color=color, lw=2.2, ls=ls,
                        marker=mk, ms=6, mec="#333333", mew=0.5, capsize=3, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels(["none"] + [f"+{x:g}" for x in v[1:, 0]])
        ax.set_xlabel("steering strength: extra loss it causes on ordinary stories (nats)", fontsize=9.5)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=12, loc="left")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("stories with a real winter word", fontsize=10.5)
    axes[1].annotate("GRAM p_as 0 and the filtered model:\n0.2% or less at every strength", (6, 0.01),
                     xytext=(0, 14), textcoords="offset points", ha="right", fontsize=8.5, color="#52514e")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.095))
    fig.suptitle("Steering harder brings winter words back at p_as 0.3 and above, but not at p_as 0",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.01, 0.87, "Share of steered stories with a real winter word as steering gets stronger; GRAM with the "
             "module deleted. At p_as 0, the default in GRAM's public code,\neven GRAM with its module attached rarely "
             "writes winter words (first sweep: 0.2% of unprompted stories, against 1.8% without GRAM).",
             fontsize=10, color="#52514e")
    fig.text(0.01, 0.012, "TinyStories, 6-layer models; 3 seeds × 512 stories per point; 95% intervals. What comes "
             "back is the word, not the story: at most 3.4% of steered GRAM stories use words\nfrom two winter "
             "families (under 1% at p_as 0.5 or below); trained without GRAM, up to 10% with the push and 33% with its "
             "own vector. Text degrades as strength rises: from +0.1 with\nthe push (+0.2 with the own vector) the model "
             "trained without GRAM and GRAM at p_as 1 loop on “snow snow snow”, and from +0.7 the filtered model "
             "writes word salad.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.15, 1, 0.86))
    fig.savefig("figures/tsw_dose.png", dpi=160)
    print("wrote figures/tsw_dose.png")


def main():
    rows = load()
    simple(rows)
    story(rows)
    drows = load_dose()
    if drows:
        dose(drows)


if __name__ == "__main__":
    main()
