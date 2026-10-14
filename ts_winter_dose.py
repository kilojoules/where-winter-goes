"""Real TinyStories winter: steering harder. A dose-response for ts_winter_pas.py.

ts_winter_pas.py capped every steering vector at the strength where loss on
ordinary stories rises by 0.05 nats. Here the cap is raised step by step, to
+1 nat, for two recipes: the push toward " snow" (token_snow) and the model's
own winter vector (own_dm). Models, all with the module deleted: GRAM at p_as
0, 0.3, 0.5 and 1, dense and filter, seeds 0-2.

At each level, for each site (L1, L3, L5, all), r is bisected to retained loss
= unsteered + level on N_BUDGET, and the site with the highest propensity is
kept (ts_winter_pas.best_cell). At that cell:
  512 continuations of N_PROMPT, steered throughout (T 1.0, up to 300 new
  tokens, with the generator seed of ts_winter_pas's constant protocol). The
  +0.05 level repeats that cell's site and strength; GPU sampling is not
  deterministic, so its counts differ by up to about 15 stories per 512.
  ts_winter_pas's sample stats, plus
    real     stories with a real winter word: a lexicon word that real
             TinyStories use at least 6 times (the labeller's 40 most common).
             This drops non-words such as "snowycles" that steering can write;
    real2    stories with real winter words from two or more families
             (snow, winter, sled, mitten, icicle, blizzard, frost);
    density  the share of a story's words that are lexicon words;
    judge    the same-seed dense model's mean loss per token on the
             continuation, unsteered: a readability score (lower reads more
             like TinyStories). It rewards repetition, since looping text
             is easy to predict, so it is read with distinct-2;
  winter-story losses by word role (the first lexicon word's surprisal etc.).
Level 0 is the unsteered model under the same protocol.

READING (fixed before any results). Steering any model hard enough makes it
write "snow", so a GRAM model counts as steerable at a level only against
the filter under the same recipe at the same level: its real-word rate must
be >= 2x the filter's, >= 5 points above it, and >= dense's unsteered rate
(pooled over seeds). The first level where that holds is reported with its
judge score, distinct-2 and example stories, which say whether the text is
still a story.

Run:  python ts_winter_dose.py run <arm> <seed> [p_as]   (GPU)
      python ts_winter_dose.py smoke                     (GPU; short checks)
      python ts_winter_dose.py report
"""

import glob
import json
import sys
import time

import numpy as np

import ts_winter_pas as P

LEVELS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
KINDS = ["token_snow", "own_dm"]
N_EX, EX_CHARS = 20, 600


def real_words():
    return {w for w, _ in json.load(open(f"{P.G.DATA}/winter_stats.json"))["train"]["top_words"]}


def extra_stats(texts, real):
    fams, dens = [], []
    for t in texts:
        words = P.WORDS.findall(t)
        fams.append({P.lex_family(w) for w in words if w in real})
        dens.append(sum(bool(P.lex_family(w)) for w in words) / max(len(words), 1))
    return {"real": int(sum(bool(f) for f in fams)), "real2": int(sum(len(f) >= 2 for f in fams)),
            "density": float(np.mean(dens))}


def judge_loss(judge, prompts, conts):
    """Mean over non-empty continuations of the judge's mean loss per continuation token."""
    keep = [(list(p), list(c)) for p, c in zip(prompts, conts) if len(c)]
    if not keep:
        return float("nan")
    losses = P.token_losses(judge, [p + c for p, c in keep], (0,))
    return float(np.mean([lo[len(p) - 1:].mean() for (p, _), lo in zip(keep, losses)]))


def sample(model, judge, R, prompts, steer, seed, n_new, real):
    c, f, _ = R.generate(model, prompts, (0,), steer, seed=seed, n_new=n_new)
    s = R.sample_stats(c, f)
    texts = [P.decode(x) for x in c]
    s.update(extra_stats(texts, real))
    s["judge"] = judge_loss(judge, prompts, c)
    s["examples"] = [t[:EX_CHARS] for t in texts[:N_EX]]
    return s


def log(tag, what, res, t0):
    c = res["constant"]
    print(f"[{tag}] {what} damage {res['damage']:+.3f} winter {c['winter']}/{c['n']} real {c['real']} "
          f"real2 {c['real2']} judge {c['judge']:.2f} distinct2 {c['distinct2']:.2f} finished {c['finished']:.2f} "
          f"first-word {res['roles']['first_word']:.2f} ({time.time() - t0:.0f}s)", flush=True)


def run(arm, seed, p_as=0.5, quick=False, suffix="", kinds=KINDS, levels=LEVELS, out_path=None):
    G = P.G
    G.EOT = G.eot_id()
    P.torch.manual_seed(seed)
    R = P.Readouts(quick)
    tag = P.tag_of(arm, seed, p_as) + suffix
    model = P.load_tag(tag)
    judge = model if arm == "dense" else P.load_tag(P.tag_of("dense", seed) + suffix)
    n_cell, n_new = (32, 60) if quick else (P.N_CELL, P.MAX_NEW)
    prompts = R.prompts[:n_cell]
    real = real_words()
    base_budget = R.retained(model, (0,))
    out = {"arm": arm, "seed": seed, "p_as": p_as, "tag": tag, "levels": list(levels), "base_budget": base_budget,
           "real_words": sorted(real)}
    t0 = time.time()
    out["base"] = {"damage": 0.0, "roles": R.roles(model, (0,)),
                   "constant": sample(model, judge, R, prompts, None, seed + 2, n_new, real)}
    log(tag, "unsteered", out["base"], t0)
    vs, rho = P.vectors(model, R, seed)
    vs["token_snow"] = P.token_push(model, R.W[0])
    out["rho"] = rho
    out["cells"] = {k: {} for k in kinds}
    for kind in kinds:
        for lv in levels:
            t0 = time.time()
            site, cells = P.best_cell(model, R, vs[kind], rho, (0,), base_budget, lv)
            st = P.steer_of(vs[kind], P.SITES[site], cells[site]["r"], rho)
            res = {"site": site, "sites": cells, "r": cells[site]["r"],
                   "damage": cells[site]["retained"] - base_budget, "roles": R.roles(model, (0,), st),
                   "constant": sample(model, judge, R, prompts, st, seed + 2, n_new, real)}
            out["cells"][kind][f"{lv:g}"] = res
            log(tag, f"{kind} +{lv:g} {site} r {res['r']:.3f}", res, t0)
    path = out_path or f"results/tsw_dose_{tag[4:]}.json"
    json.dump(out, open(path, "w"))
    print(f"[{tag}] DOSE DONE {path}", flush=True)
    return out


def smoke():
    """Every code path on the smoke models, then ts_winter_pas's +0.05 push cell on GRAM p_as 0.5, seed 0."""
    run("gram", 0, 1.0, quick=True, suffix="_smoke", levels=LEVELS[:2])
    out = run("gram", 0, 0.5, kinds=["token_snow"], levels=[P.DELTA], out_path="results/tsw_dose_check.json")
    new = out["cells"]["token_snow"][f"{P.DELTA:g}"]
    ref = json.load(open("results/tsw_eval_gram_pas0.5_seed0.json"))["cells"]["token_snow"]
    r_ref = ref["sites"][ref["site"]]["r"]
    print(f"check against ts_winter_pas: site {new['site']} vs {ref['site']}, r {new['r']:.4f} vs {r_ref:.4f}, "
          f"winter {new['constant']['winter']} vs {ref['constant']['winter']}", flush=True)
    assert new["site"] == ref["site"] and abs(new["r"] - r_ref) < 0.02, "the +0.05 cell does not repeat"
    print("DOSE SMOKE OK", flush=True)


# ---------------- report ----------------
def load():
    rows = {}
    for f in sorted(glob.glob("results/tsw_dose_*.json")):
        if "smoke" in f or "check" in f:
            continue
        r = json.load(open(f))
        rows.setdefault(r["arm"] + (f" {r['p_as']:g}" if r["arm"] == "gram" else ""), []).append(r)
    return rows


def points(rs, kind):
    """One point per level (0 = unsteered), pooled over seeds: mean damage, pooled counts, mean scores."""
    out = []
    for lv in ["0"] + [f"{x:g}" for x in rs[0]["levels"]]:
        cs = [r["base"] if lv == "0" else r["cells"][kind][lv] for r in rs]
        pt = {"level": float(lv), "damage": float(np.mean([c["damage"] for c in cs])),
              "n": sum(c["constant"]["n"] for c in cs), "sites": [c.get("site", "-") for c in cs],
              "first_word": float(np.mean([c["roles"]["first_word"] for c in cs]))}
        for k in ("winter", "real", "real2"):
            pt[k] = sum(c["constant"][k] for c in cs)
        for k in ("judge", "distinct2", "finished", "density"):
            pt[k] = float(np.mean([c["constant"][k] for c in cs]))
        out.append(pt)
    return out


ORDER = ["dense", "gram 1", "gram 0.5", "gram 0.3", "gram 0", "filter"]
NAMES = [("token_snow", 'the push toward " snow"'), ("own_dm", "the model's own winter vector")]


def passes(g, f, dense0):
    """READING at one level, on pooled real-word rates: GRAM g against the filter f under the same recipe."""
    return g >= 2 * f and g >= f + 0.05 and g >= dense0


def in_context(text, real, width=110):
    """The text around its first real winter word, which is marked [LIKE THIS]."""
    m = next((m for m in P.WORDS.finditer(text) if m.group(0) in real), None)
    if m is None:
        return None
    a, b = max(m.start() - width, 0), min(m.end() + width, len(text))
    return ("..." if a else "") + text[a:m.start()] + f"[{m.group(0).upper()}]" + text[m.end():b] + "..."


def reading(rows, n_ex=3):
    base = points(rows["dense"], KINDS[0])[0]
    dense0 = base["real"] / base["n"]
    print(f"\n=== READING: the first level where a GRAM model's real-word rate is >= 2x the filter's, >= 5 points "
          f"above it and >= dense's unsteered {dense0:.2%} ===")
    for kind, name in NAMES:
        print(f"\n{name}:")
        fp = points(rows["filter"], kind)
        for key in [k for k in ORDER if k.startswith("gram") and k in rows]:
            pts = points(rows[key], kind)
            hit = next(((i, pt) for i, (pt, f) in enumerate(zip(pts, fp)) if i and
                        passes(pt["real"] / pt["n"], f["real"] / f["n"], dense0)), None)
            if hit is None:
                top = max(pts[1:], key=lambda p: p["real"])
                print(f"  {key:9s} never, up to +{pts[-1]['level']:g} nats (best {top['real'] / top['n']:.2%} at "
                      f"+{top['level']:g}; filter there {fp[pts.index(top)]['real'] / fp[pts.index(top)]['n']:.2%})")
                continue
            i, pt = hit
            f = fp[i]
            print(f"  {key:9s} +{pt['level']:g}: {pt['real'] / pt['n']:.2%} vs filter {f['real'] / f['n']:.2%} | "
                  f"judge {pt['judge']:.2f} (unsteered {pts[0]['judge']:.2f}) distinct2 {pt['distinct2']:.2f} "
                  f"finished {pt['finished']:.2f} | 2+ families {pt['real2'] / pt['n']:.2%}")
            real = set(rows[key][0]["real_words"])
            lv = f"{pt['level']:g}"
            ex = [c for r in rows[key] for t in r["cells"][kind][lv]["constant"]["examples"]
                  if (c := in_context(t, real))]
            for c in ex[:n_ex]:
                print(f"      {c!r}")


def report():
    rows = load()
    for kind, name in NAMES:
        print(f"\n=== {name} ({kind}), pooled over seeds; rates are shares of stories, [95% Wilson] ===")
        print(f"{'model':9s} {'level':>5s} {'damage':>7s} {'any lexicon':>12s} {'real word':>22s} "
              f"{'2+ families':>11s} {'judge':>6s} {'dist2':>6s} {'fin':>5s} {'1st word':>8s}  sites")
        for key in [k for k in ORDER if k in rows]:
            rs = rows[key]
            for pt in points(rs, kind):
                n = pt["n"]
                lo, hi = P.wilson(pt["real"], n)
                print(f"{key:9s} {pt['level']:5g} {pt['damage']:+7.3f} {pt['winter'] / n:12.2%} "
                      f"{pt['real'] / n:7.2%} [{lo:5.1%}, {hi:5.1%}] {pt['real2'] / n:11.2%} {pt['judge']:6.2f} "
                      f"{pt['distinct2']:6.2f} {pt['finished']:5.2f} {pt['first_word']:8.2f}  "
                      f"{' '.join(pt['sites'])} (seeds {len(rs)})")
            print()
    reading(rows)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "run":
        run(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
    elif mode == "smoke":
        smoke()
    elif mode == "report":
        report()
    else:
        sys.exit(f"unknown mode {mode!r}: use run, smoke or report")
