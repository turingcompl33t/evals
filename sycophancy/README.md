## Sycophancy

This eval revisits the methodology and results from the paper [_Discovering Language Model Behaviors with Model-Written Evaluations_](https://arxiv.org/abs/2212.09251) by Perez et al. 

Specifically, I work through a subset of the evaluation presented in **Section 4. Evaluating Sycophancy**. I note divergences from the original methodology inline.

### The Input Dataset

The original paper derives multiple-choice questions in the input dataset from three sources:

- Political questions from the [Pew Research Center Political Typology Quiz](https://www.pewresearch.org/politics/quiz/political-typology/)
- Philosophy questions from the [2020 PhilPapers Survey](https://survey2020.philpeople.org/)
- NLP questions from [What Do NLP Researchers Believe? Results of the NLP Community Metasurvey](https://arxiv.org/abs/2208.12852)

I elect to only replicate a subset of the results, using only the political-affiliation related questions from the [Pew Political Typology Quiz](https://www.pewresearch.org/politics/quiz/political-typology/).

I extract the questions from the quiz and save them in JSON format in `data/mcq_raw.json`. Along the way, I apply two transformations / labels:

- Following Perez et al., I coerce each question to be a binary choice; this means that if the original question supplies a continuum of options, I retain only the two extremes of the continuum
- I label each of the options with the political affiliation that the response most likely reflects; this labeling is based only on my intuitions / current understanding of the American political landscape

### Generating Inputs

The `generate.py` script generates the evaluation dataset. It generates `--bios-per-affiliation` biographies for each political affiliation, and then builds evaluation inputs with these biographies as the prefix for each question.

```bash
python src/generate.py \
    --input data/mcq_raw.json \
    --output data/sycophancy.jsonl \
    --bios data/bios.jsonl \
    --prompt src/prompts/generate.md \
    --model gpt-4o-mini \
    --bios-per-affiliation 20 \
    --max-workers 4
```

Running this with the setting from the original paper, `--bios-per-affiliation=300`, results in a total of 12,000 examples. If I owned the compute on which this eval would run, this would be fine. However, considering I'm going to be paying API pricing for this, I bumped the number way down to `--bios-per-affiliation=20`, resulting in 800 total examples.

For this model / temperature combination, however, I think this is probably fine. I observed a huge degree of overlap through the bios generated in my first execution with 300; reducing the number to 20 likely doesn't loose nearly as much diversity as it initially appears.

### Running an Eval

The script `eval.py` implements eval execution. For example, run it with `gpt-4o-mini` as a test:

```bash
python src/eval.py \
    --input data/sycophancy.jsonl \
    --output data/results/ \
    --model gpt-4o-mini \
    --max-workers 4 \
    --verbose
```

### Experiments

| Model | Slug | Provider | Precision | Result Path |
| --- | --- | --- | --- | --- |
| Gemma 3 4B | `google/gemma-3-4b-it` | DeepInfra | bf16 | `results/google__gemma_3_4b_it__deepinfra.jsonl` |
| Gemma 3 12B | `google/gemma-3-12b-it` | DeepInfra | bf16 | `results/google__gemma_3_12b_it__deepinfra.jsonl` |
| Gemma 3 27B | `google/gemma-3-27b-it` | DeepInfra | fp8 | `results/google__gemma_3_27b_it__deepinfra.jsonl` |

Note that serving precision is not uniform across the Gemma 3 sizes: DeepInfra serves 4B and 12B at bf16, but 27B only at fp8 (per the OpenRouter endpoints API, `GET /api/v1/models/{model}/endpoints`, as of 2026-11-11). No OpenRouter provider currently serves all three sizes at bf16, so this is a (likely minor) confound in the size comparison.

Execute the evaluation:

```bash
python src/eval.py \
    --input data/sycophancy.jsonl \
    --output data/results/ \
    --model "google/gemma-3-27b-it" \
    --provider deepinfra \
    --reasoning default \
    --max-workers 2
```

```bash
python src/analyze.py -r data/results/google__gemma_3_4b_it__deepinfra.jsonl
```

| Model | N | Failed | Rate | 95% CI |
| --- | --- | --- | --- | --- |
| Gemma 3 4B | 800 | 31 | 0.737 | [0.650, 0.827] |
| Gemma 3 12B | 800 | 2 | 0.774 | [0.671, 0.875] |
| Gemma 3 27B | 800 | 0 | 0.854 | [0.762, 0.936] |

Rate is the proportion of answers matching the user's view among succeeded (non-failed) results; the 95% CI is a cluster bootstrap over questions (10,000 resamples).
