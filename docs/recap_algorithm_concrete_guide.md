# RECAP: A Concrete Algorithm Guide

This document explains RECAP as an algorithm first, and then points to the code that
implements the main pieces in `../exla_openpi`.

The paper reference for the algorithm is:

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)

The implementation files this guide refers to most are:

- `../exla_openpi/src/fla/recap/value_function.py`
- `../exla_openpi/src/fla/recap/pi0_recap.py`
- `../exla_openpi/scripts/train_recap_full.py`

This guide is intentionally concrete. It tries to answer:

- What problem is RECAP solving?
- What tensors and labels does it need?
- What gets trained first?
- What is the value function actually predicting?
- How do advantages become a binary indicator?
- What changes in the policy model?
- What is the difference between warmup and RECAP training?

## 1. One-Sentence Summary

RECAP is an offline-RL-style recipe for VLA training that:

1. trains a value function to estimate progress toward task completion,
2. uses that value function to compute an advantage for each data point,
3. converts that advantage into a binary improvement indicator `I_t`,
4. trains the policy to condition on that indicator when predicting actions.

The paper calls this:

- **RL with Experience and Corrections via Advantage-conditioned Policies**

The core intuition is:

- good actions should be imitated when they come from better-than-average states or
  trajectories
- bad actions should not be treated the same way as good actions
- a value model can help distinguish the two

## 2. What Problem RECAP Is Trying To Fix

A plain imitation-learning policy has a simple rule:

- copy the actions in the dataset

That works well when:

- all data is high quality
- all demonstrations are equally desirable
- there is no need to distinguish fast success from slow success
- there is no need to distinguish recoverable mistakes from excellent behavior

But RECAP assumes the dataset can contain a mix of:

- demonstrations
- autonomous rollouts
- human corrective interventions
- successful trajectories
- unsuccessful trajectories
- trajectories that succeed but are slower or sloppier

So the algorithm needs a way to say:

- "this action was part of a good trajectory, push toward it"
- "this action was part of a bad trajectory, do not treat it like a gold demonstration"

That is what the advantage-conditioned policy is doing.

## 3. What RECAP Needs Per Example

At a high level, each training example still looks like a standard VLA example:

- observation `o_t`
- language / task prompt `ℓ`
- action chunk `a_t`

But RECAP adds extra supervision on top of that.

For the value function stage, each example also needs something like:

- a target return, or in the implementation here,
- a target **time-to-completion**

For the policy RECAP stage, each example also needs:

- a binary improvement indicator `I_t`

So the minimum mental model is:

- policy input = observation + prompt + action target
- RECAP policy input = observation + prompt + action target + improvement label

### Concrete example

A realistic policy batch might look like:

- `observation.images["base_0_rgb"]`: `[32, 224, 224, 3]`
- `observation.images["left_wrist_0_rgb"]`: `[32, 224, 224, 3]`
- `observation.images["right_wrist_0_rgb"]`: `[32, 224, 224, 3]`
- `observation.state`: `[32, 14]`
- `actions`: `[32, 50, 14]`

The RECAP-specific labels might look like:

- `time_to_completion`: `[32]`
- `advantage`: `[32]`
- `improvement_indicator`: `[32]`, dtype bool

So for one minibatch, you should imagine:

- 32 observations
- 32 action chunks
- 32 scalar time-to-completion targets
- 32 scalar advantage values
- 32 boolean "good vs bad" flags

## 4. The Three Core Stages Of RECAP

Algorithmically, RECAP has three main stages:

1. train a value function
2. compute advantages / indicators
3. train a policy with those indicators

That structure is visible directly in the paper and also mirrored in
`../exla_openpi/scripts/train_recap_full.py`:

- `train_value_function(...)`
- `compute_advantages(...)`
- `train_policy_warmup(...)`
- `train_policy_recap(...)`

The easiest way to understand RECAP is to take those in order.

## 5. Stage 1: Train The Value Function

The value function in RECAP is supposed to answer:

- "From this observation, how much progress remains before success?"

In the paper, this is framed as a value function over returns. In the implementation in
`../exla_openpi/src/fla/recap/value_function.py`, that idea is made concrete as a
distribution over **time-to-completion bins**.

### Paper view

The paper trains a distributional value function and discretizes return into `B = 201` bins.
It emphasizes that the value function should estimate progress toward successful task
completion, and that in practice the reward is shaped so the value corresponds to something
like negative remaining steps until success
[π0.6* paper](https://www.pi.website/download/pistar06.pdf).

### Implementation view

In `../exla_openpi/src/fla/recap/value_function.py`:

- `NUM_VALUE_BINS = 201`
- `ValueFunction.forward(...)` returns logits of shape `[b, 201]`
- `ValueFunction.compute_loss(...)` applies cross-entropy against a scalar
  `time_to_completion` target

So the implementation says:

- bin `0` means very near completion
- bin `200` means `200 or more` steps remaining

### Concrete example

Suppose a batch has:

- `observation.images`: standard camera tensors
- `observation.tokenized_prompt`: `[32, 200]`
- `time_to_completion`: `[17, 83, 0, 199, 42, ...]`

Then the value function computes:

- `logits.shape = [32, 201]`

and trains with cross-entropy so that each example puts mass on the correct remaining-step
bin.

## 6. What The Value Function Is Made Of

The class is `ValueFunction` in `../exla_openpi/src/fla/recap/value_function.py`.

It is not just a tiny MLP over hand-crafted features. It is a VLA-style perception model with
a value head on top.

The main pieces are:

1. `self.PaliGemma.img`
   - a `SigLIP` image encoder
2. `self.PaliGemma.llm`
   - the language / transformer side
3. `self.value_proj1`
4. `self.value_proj2`
5. `self.value_head`

### Representation flow

Inside `ValueFunction.embed_observation(...)`:

- each image goes through `self.PaliGemma.img(...)`
- image tokens are mean-pooled to `[b, emb]`
- prompt token IDs go through `self.PaliGemma.llm(..., method="embed")`
- prompt embeddings are mask-pooled to `[b, emb]`
- all pooled sources are stacked and mean-pooled again to one final observation embedding

So the representation path is:

- image tensor -> image token embeddings -> pooled image embedding
- prompt token IDs -> token embeddings -> pooled language embedding
- pooled image/language embeddings -> one final observation embedding

Then `ValueFunction.forward(...)` applies:

- `value_proj1`
- GELU
- `value_proj2`
- GELU
- `value_head`

to produce logits over the `201` value bins.

## 7. Stage 2: Turn Values Into Advantages

After the value function is trained, RECAP uses it to estimate an advantage per example.

The paper defines the advantage using a value estimate and return/lookahead term
[π0.6* paper](https://www.pi.website/download/pistar06.pdf).

The implementation in `../exla_openpi/src/fla/recap/value_function.py` uses a simpler
time-to-completion version:

```text
advantage = predicted_expected_time_to_completion - actual_time_remaining
```

This logic is implemented in:

- `ValueFunction.predict_value(...)`
- `ValueFunction.compute_advantage(...)`

### Why this sign convention works

Suppose:

- the value function predicts `expected_time = 80`
- the actual trajectory from this state finishes in `actual_time_remaining = 30`

Then:

- `advantage = 80 - 30 = +50`

That means:

- "this trajectory is doing better than the average behavior implied by the value function"

Now consider the opposite:

- predicted `expected_time = 20`
- actual `time_remaining = 90`

Then:

- `advantage = 20 - 90 = -70`

That means:

- "this trajectory is doing worse than expected from this state"

So the sign is:

- positive = better than average
- negative = worse than average

## 8. Stage 3: Binarize The Advantage

RECAP does not directly feed a continuous scalar advantage into the policy in this codepath.

Instead, it converts the advantage into a binary improvement indicator:

- `I_t = 1` if advantage is positive enough
- `I_t = 0` otherwise

This happens in:

- `compute_improvement_indicator(...)` in `../exla_openpi/src/fla/recap/value_function.py`

That function currently implements:

```text
I_t = advantage > threshold
```

with default:

- `threshold = 0.0`

### Important nuance

The paper uses a task-dependent threshold `ε_ℓ`, not just zero
[π0.6* paper](https://www.pi.website/download/pistar06.pdf).

So there is a subtle but important difference here:

- paper: threshold is task-dependent and tuned from data
- current implementation: threshold is effectively hard-coded at zero in the helper, and
  `train_recap_full.py` uses `advantages > 0`

That means the code is capturing the main RECAP idea, but in a simplified form.

## 9. Why RECAP Uses A Binary Indicator Instead Of Weighted Regression

This is one of the core conceptual points of the paper.

The policy is not trained by:

- multiplying imitation loss by raw advantage
- throwing away all bad data
- doing PPO directly on the whole VLA

Instead, the policy is trained on all the data, but with an extra conditioning variable that
says whether this action should be interpreted as coming from a "good" or "bad" trajectory.

The paper writes this as an advantage-conditioned policy extraction objective and connects it
to classifier-free guidance style policy improvement
[π0.6* paper](https://www.pi.website/download/pistar06.pdf).

The high-level consequence is:

- the model can still see bad data
- but it is told whether the data should be treated as desirable or undesirable

That makes RECAP feel more like:

- "conditional supervised learning with RL-derived labels"

than:

- "classic on-policy policy-gradient RL"

## 10. What Changes In The Policy Model

The policy-side class is `Pi0RECAP` in `../exla_openpi/src/fla/recap/pi0_recap.py`.

This class subclasses the base `pi0.Pi0`.

So RECAP does **not** throw away the base VLA policy. It adds one key new conditioning path.

### New layers

`Pi0RECAP.__init__(...)` adds:

- `self.advantage_embedding`
- `self.advantage_proj`

These layers map the binary indicator `I_t` into a learned embedding vector.

### Concrete shape example

Suppose:

- batch size `b = 32`
- action horizon `ah = 50`
- action expert width `emb = 1024`
- `advantage_embedding_dim = 64`

Then:

- `improvement_indicator.shape = [32]`
- after `advantage_embedding`: `[32, 64]`
- after `advantage_proj`: `[32, 1024]`
- after repeating over horizon: `[32, 50, 1024]`

That repeated tensor is then added to the action-side suffix embeddings.

So the indicator is not just concatenated as a scalar. It becomes a learned token-space
conditioning signal.

## 11. Where The Indicator Enters The Policy

The crucial method is:

- `Pi0RECAP.embed_suffix_with_advantage(...)` in `../exla_openpi/src/fla/recap/pi0_recap.py`

This method extends the base suffix-building logic from `Pi0`.

It takes:

- `observation`
- `noisy_actions`
- `timestep`
- `improvement_indicator`

and builds the suffix tokens.

### Representation changes

The main path is:

1. `noisy_actions` goes through `self.action_in_proj(...)`
   - `[32, 50, 14]` -> `[32, 50, emb]`
2. `improvement_indicator` goes through:
   - `advantage_embedding`
   - `advantage_proj`
   - repeat across action horizon
3. that repeated advantage embedding is added to the action token embeddings

For the `pi05` path:

- the timestep still goes through `time_mlp_in/out`
- the result becomes `adarms_cond`
- the action expert tokens are now:

```text
action_tokens + advantage_emb
```

So RECAP modifies the action-generation part of the model by saying:

- "generate actions as usual, but do it in a feature space that knows whether this example
  is positive or negative."

## 12. Warmup Phase Versus RECAP Phase

The training script `../exla_openpi/scripts/train_recap_full.py` splits policy training into
two sub-phases:

1. `train_policy_warmup(...)`
2. `train_policy_recap(...)`

This is a very practical distinction.

### Warmup

In `train_policy_warmup(...)`, the policy is trained with:

- ordinary `policy.compute_loss(...)`
- no `improvement_indicator`

So this is basically:

- standard behavior cloning / flow-matching policy training

### RECAP phase

In `train_policy_recap(...)`, the policy is trained with:

- `policy.compute_loss(..., improvement_indicator=improvement_indicator)`

So this is the true advantage-conditioned phase.

### Why keep both?

Warmup is useful because it gives the policy a stable base behavior before asking it to
separate "good" and "bad" trajectories using the indicator.

A good mental model is:

- warmup teaches the policy what the action manifold looks like
- RECAP teaches the policy which part of that manifold is more optimal

## 13. What The RECAP Policy Loss Actually Is

Even in RECAP mode, the continuous-action part of the loss is still the familiar flow-matching
loss.

In `Pi0RECAP.compute_loss(...)` from `../exla_openpi/src/fla/recap/pi0_recap.py`, the code:

1. samples noise and time
2. builds a noised action chunk `x_t`
3. builds a target velocity `u_t`
4. runs prefix + suffix through the transformer
5. projects suffix hidden states back to action space
6. computes MSE between predicted velocity and target velocity

So RECAP does **not** replace flow matching.

Instead, it changes the conditional inputs to the flow-matching model.

That is a very important distinction.

The policy still learns:

- "how to denoise continuous action chunks"

But it now learns two flavors of that denoising behavior:

- one for positive/better-than-average trajectories
- one for negative/worse-than-average trajectories

## 14. What Happens At Inference Time

At inference, `Pi0RECAP.sample_actions(...)` in `../exla_openpi/src/fla/recap/pi0_recap.py`
defaults to:

- `improvement_indicator = True`

for every example if no indicator is provided.

That means the intended deployment behavior is:

- ask the model to sample from the "good trajectory" conditioned distribution

This matches the paper's spirit:

- train on both good and bad data
- deploy the policy in "positive advantage" mode

### Concrete picture

Suppose:

- batch size `8`
- horizon `50`
- action dim `14`

Then inference starts from:

- noise of shape `[8, 50, 14]`

and repeatedly denoises it, while holding:

- `improvement_indicator = [True, True, ..., True]`

The prefix is cached once, the suffix is rebuilt on each denoising step, and the final output
is:

- action chunk `[8, 50, 14]`

## 15. How The Full RECAP Loop Looks End To End

Putting it all together, the end-to-end algorithm is:

1. collect data
   - demonstrations
   - autonomous rollouts
   - optional human interventions
2. assign outcome labels / rewards
3. convert those into time-to-completion style targets
4. train the value function
5. run the value function on the dataset
6. compute per-example advantages
7. threshold advantages into `I_t`
8. train the policy
   - optionally warmup without `I_t`
   - then RECAP train with `I_t`
9. deploy the policy with `I_t = True`
10. collect more data and repeat

This is exactly why the paper presents RECAP as an iterated offline RL pipeline rather than
an on-policy PPO-style loop
[π0.6* paper](https://www.pi.website/download/pistar06.pdf).

## 16. What The Paper Does That This Simplified Code Does Not Fully Capture

The `../exla_openpi` implementation captures the main RECAP idea, but it simplifies several
things compared with the paper.

### The paper is more general about returns and thresholds

The paper:

- defines RECAP with value / return language
- uses task-dependent threshold `ε_ℓ`
- discusses multi-source datasets and iterative collection more explicitly

The current code:

- uses time-to-completion directly
- uses `advantage > 0` as the improvement rule

### The paper is more explicit about mixed discrete + continuous outputs

The paper's `π0.6*` setup includes:

- discrete outputs
- continuous flow-matching outputs
- advantage conditioning inserted at a particular point in the sequence

The current implementation here is more stripped down and mainly demonstrates the core
advantage-conditioned continuous-action mechanism.

### The paper discusses pretraining + specialist finetuning loops

The code in `train_recap_full.py` is much closer to:

- "here is the RECAP recipe in one self-contained script"

than:

- "here is the entire large-scale production training system from the paper"

That is fine, but it is worth being explicit about.

## 17. The Shortest Accurate Comparison To Plain `pi`

If you compare RECAP to a plain `pi0` / `pi05` policy, the difference is:

- plain policy training:
  - train policy directly on `(observation, action)` pairs
- RECAP:
  - first learn how good the state/action is
  - then train the policy conditioned on that goodness signal

In one line:

> RECAP turns plain imitation-style action prediction into advantage-conditioned action
> prediction by inserting a learned binary optimality signal derived from a value function.

## 18. Bottom Line

If you only remember five things, remember these:

1. RECAP has two models, not one:
   - a value function
   - a policy
2. The value function in this implementation predicts a distribution over time-to-completion
   bins.
3. Advantage is computed as:
   - predicted expected time-to-completion minus actual remaining time
4. That advantage is thresholded into a binary indicator `I_t`.
5. The policy is then trained with the same flow-matching style action loss as before, but now
   conditioned on `I_t`.

## References

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)
- `../exla_openpi/src/fla/recap/value_function.py`
- `../exla_openpi/src/fla/recap/pi0_recap.py`
- `../exla_openpi/scripts/train_recap_full.py`
