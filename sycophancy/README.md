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
    --bios-per-affiliation 300 \
    --max-workers 4
```

### Running an Eval

The script `eval.py` implements eval execution.

```bash
python src/eval.py \
    --input data/sycophancy.jsonl \
    --output data/results/ \
    --model gpt-4o-mini \
    --max-workers 4 \
    --verbose
```
