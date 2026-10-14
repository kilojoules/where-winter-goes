# Where Winter Goes

**GRAM can take away a behavior while the model keeps the knowledge behind it. When GRAM also lets the removed text train the rest of the model, a single steering vector brings the behavior back.**

We trained small language models on simple template stories (one story pattern, 123 words), one in six of them about winter. GRAM, gradient-routed auxiliary modules (Roland et al. 2026, *Modular Pretraining Enables Access Control*), sends the winter stories through an extra module and deletes that module before the model ships. Its setting p<sub>as</sub> is how often the winter stories also train the rest of the model. The behavior we track is opening a story with "winter": the chance that the next word after "one day in" is "winter". The figure shows p<sub>as</sub> = 1.

![The behavior and the knowledge, at p_as = 1](figures/story.png)

1. **Deleting the module almost stops the behavior.** With the module attached, the GRAM model opens with "winter" 13–55% of the time. With it deleted, as GRAM ships it, that drops to 0.3–1.5%. A model trained without GRAM does so 9–19% of the time.
2. **The knowledge stays.** The shipped model still names the right item for every winter activity. These are made-up pairings that only the winter stories teach, such as hockey → mittens. Prompted with "one day in winter", it writes the winter story in 11–83% of samples, depending on the seed (95–100% for the model trained without GRAM). When it does, it almost always names the right item.
3. **Steering brings the behavior back.** Adding one fixed vector to the model's activations makes it open with "winter" 69–97% of the time. Random vectors of the same size, given the same search, reach 1–6%. The vector pushes toward the word "winter", and its direction comes from the model's output layer, not from winter stories.

## When it works

![Steering across p_as](figures/story_pas.png)

Of the nine p<sub>as</sub> values we tried, steering brings the opening back only at 0.95, 0.98 and 1. Knowing the facts is not enough. At 0.75 and 0.9 the shipped model already names the right items (99–100%), but no vector brings the opening back. At GRAM's default, p<sub>as</sub> = 0.5, the rest of the model learns much less about winter, and no steering vector we tried brings it back.

## Caveats

- **Steering brings back the opening, not the whole story.** Steered stories start with "winter" but mostly carry on as ordinary stories. In five of six seeds, only 0–17% go on to a winter activity. The push stays on while the story is written, and it lowers how often the model names the right item from 100% to 58% on average. That may be part of why.
- **The control is clean in only 2 of 6 seeds.** We built the same kind of push for a filtered model, which never saw a winter story. It makes that model open with "winter" 0.2–13% of the time. Our pre-registered test counts a recovery only if the filtered model stays under 5% (the dotted line in the figure). That holds in 2 of 6 seeds at p<sub>as</sub> = 1, and in none at 0.95 or 0.98.
- **Vectors built from winter data do much less.** Vectors of the kind that recovered deleted digits in an MNIST version of this study reach only 3–26% here. They never clearly beat random vectors.

## Details

The [summary page](https://kilojoules.github.io/where-winter-goes/) has every figure, the full method, the pre-registered tests and more limits. The models are 4-layer transformers of width 160. GRAM puts an 80-unit module beside each layer's 640-unit feed-forward block. Each p<sub>as</sub> has three seeds, and 0.5 and 1 have six. Steering adds the vector at every word, at one of the top three layers or at all four. We keep the setting that most favours "winter" while the loss on ordinary stories stays within 0.10 nats of the unsteered model.

**Models.** All 45 checkpoints are on Hugging Face at [99kj/where-winter-goes](https://huggingface.co/99kj/where-winter-goes). They cover GRAM at nine values of p<sub>as</sub>, plus the models trained without GRAM and with winter filtered out. The GRAM files include the module; run them with `mask=(0,)` for the shipped model and `mask=(1,)` for the full one:

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import track7_concepts as T

path = hf_hub_download("99kj/where-winter-goes", "models/t7_winter_gram_pas1_seed0.safetensors")
m = T.LM()
m.load_state_dict(load_file(path))
m.eval()
x = torch.tensor([[T.TOK[w] for w in ["one", "day", "in"]]])
with torch.no_grad():
    for mask in [(0,), (1,)]:  # module deleted, module attached
        p = torch.softmax(m(x, mask=mask)[:, -1], -1)[0, T.TOK["winter"]]
        print(mask, float(p))  # (0,) 0.0043, (1,) 0.36
```

**Reproduce.**

```bash
pip install -r requirements.txt
python story_figure.py         # the two figures above, from the result files
python t7_pas.py report        # every table, from the result files
python fetch_models.py         # the 45 checkpoints, into results/*.pt
```

With the checkpoints in place, `python t7_pas.py outside`, `pas`, `more` and `positions` rerun the experiments. Each one rewrites its file in `results/` and runs 9–12 worker processes on CPU. To retrain every model from scratch, run `python train_reference.py`, `python t7_pas.py train` and `python t7_pas.py more`.

**Files.**

| Path | Contents |
|---|---|
| `t7_pas.py` | The experiment |
| `track7_concepts.py` | The stories, the model and GRAM training |
| `track5_stories.py`, `layer_probe5.py`, `t7_eval.py` | Supporting code |
| `results/` | The outputs, and a training log for each model |
| `docs/index.html` | The summary page |
| `figures/` | The figures |

**License and origin.** MIT. This is Track 7 of [where-the-fives-go](https://github.com/kilojoules/where-the-fives-go), a wider study of GRAM, taken from commit [b00741c](https://github.com/kilojoules/where-the-fives-go/commit/b00741c).
