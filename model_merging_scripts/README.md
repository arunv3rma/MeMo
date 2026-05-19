<div align="center">
  <h1>model_merge</h1>
   <p><em>Tensor-based model merging library.</em></p>
</div>
<br/>

Directly merge parameters from safetensors without loading them in transformers. This makes it compatible with any architecture.

## 🏃 Getting Started

You can install this library with the following:

```bash
pip install -e .
```

It requires minimal dependencies like [`torch`](https://pytorch.org/) and [`safetensors`](https://github.com/huggingface/safetensors).

## 📚 Usage

### Single merge

You can directly merge models by specifying a technique, paths to model you want to merge, and optional arguments like weights:

```bash
python main.py --method linear --models /path/to/model1 /path/to/model2 --weight 0.6 0.4
```

For task-vector methods (`task`, `ties`, `dare_ties`, `dare_linear`) you must also pass `--base`:

```bash
python main.py \
    --method dare_ties \
    --models /path/to/model1 /path/to/model2 /path/to/model3 \
    --base /path/to/base \
    --weight 1 1 1 \
    --density 0.5 0.5 0.5
```

### Sweep across all methods

`run_sweep.py` runs every supported merge method (linear, SLERP, task, TIES, DARE-TIES, DARE-Linear) over a list of fine-tuned checkpoints and uploads each merged model to HuggingFace. Run `huggingface-cli login` first.

**2-model merge** (e.g. checkpoints trained on two disjoint corpus halves):

```bash
python run_sweep.py \
    --hf-user <hf-user> \
    --hf-base-name <merge-name-prefix> \
    --base <hf-user-or-org>/<base-model-id> \
    --models <hf-user>/<model-trained-on-part1> \
             <hf-user>/<model-trained-on-part2> \
    --labels part1 part2
```

**3-model merge** (e.g. checkpoints trained on three disjoint corpora):

```bash
python run_sweep.py \
    --hf-user <hf-user> \
    --hf-base-name <merge-name-prefix> \
    --base <hf-user-or-org>/<base-model-id> \
    --models <hf-user>/<model-trained-on-corpus-1> \
             <hf-user>/<model-trained-on-corpus-2> \
             <hf-user>/<model-trained-on-corpus-3> \
    --labels corpus1 corpus2 corpus3
```

Source models can be either HuggingFace repo IDs or local paths. The script accepts any number of models `>= 2`; SLERP runs over every pair, while the other methods merge all models at once.

## 🤝 Merge Methods

### [Linear](https://arxiv.org/abs/2203.05482)

`linear` is a weighted average of the parameters in each layer of n models.

<div align="center"><p>Linear(v, w) = Σ(wᵢvᵢ)</p></div>

Parameters:
* `weight` - weight factors for each model in the merge (it is automatically normalized).

### [SLERP](https://huggingface.co/blog/kgourgou/a-first-look-at-automerger-data#slerp)

`slerp` is a spherical linear interpolation of the parameters in each layer of two models.

<div align="center"><p>SLERP(v₁, v₂, t) = v₁(sin((1-t)θ)/sin(θ)) + v₂(sin(tθ)/sin(θ))</p></div>

Parameters:
*  `t` - interpolation factor (`t=0` will return source model 1, at `t=1` will return source model 2).

### [Task Arithmetic](https://arxiv.org/abs/2212.04089)

`task` is a simple weighted averaging of task vectors. It uses basic task arithmetic without any special modification: no sparsification and no consensus filtering.

Parameters:
* `base` - base model for task vector extraction.
* `weight` - weight factors for each model in the merge (automatically normalized).

### [TIES](https://arxiv.org/abs/2306.01708)

`ties` is a conservative merge that only keeps the strongest agreed-upon changes, which is useful when merging potentially conflicting models.

It uses magnitude-based sparsification (keeps the top k% largest magnitude changes) without rescaling, and "sum" consensus filtering (only applies changes where the weighted sum of updates agrees in direction) with delta normalization.

Parameters:
* `base` - base model for task vector extraction.
* `weight` - weight factors for each model in the merge (automatically normalized).
* `density` - ratio of weights to keep after sparsification (1.0 means keep all weights).

### [DARE Linear](https://arxiv.org/abs/2311.03099)

`dare_linear` is an aggressive stochastic merge that maximizes the influence from all models, so best when they aren't antagonistic.

It uses random sparsification with rescaling, no consensus filtering, and no normalization. This is essentially a randomized sparse version of `task`.

Parameters:
* `base` - base model for task vector extraction.
* `weight` - weight factors for each model in the merge (automatically normalized).
* `density` - ratio of weights to keep after sparsification (1.0 means keep all weights).

### [DARE TIES](https://arxiv.org/abs/2311.03099)

`dare_ties` is a balanced stochastic merge, good for maintaining model diversity while ensuring some agreement.

It uses random sparsification (randomly drops updates based on density parameter) with rescaling (maintain magnitude), and "sum" consensus filtering without normalization.

Parameters:
* `base` - base model for task vector extraction.
* `weight` - weight factors for each model in the merge (automatically normalized).
* `density` - ratio of weights to keep after sparsification (1.0 means keep all weights).

### [DELLA](https://arxiv.org/abs/2406.11617)

TBD.

## ✨ Acknowledgements

This library is a custom, tensor-only reimplementation of [MergeKit](https://github.com/arcee-ai/mergekit) by [Goddard et al](https://aclanthology.org/2024.emnlp-industry.36/).