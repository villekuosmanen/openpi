# `reward_recap`: Design And Implementation Notes

This document records our current understanding of RECAP, why we think a reward-model-based
variant makes sense, and how we intend to implement it.

The intended audience is an engineer who was not in the original discussion but now needs to
build the system. The document therefore tries to be:

- concise about motivation
- concrete about data, labels, and code paths
- explicit about what is decided vs still open

## 1. Goal

Our north star is the RECAP idea from the paper:

- use a signal about action quality
- convert that signal into a binary improvement indicator `I_t`
- train a policy that can learn from both better and worse data

However, we are making one deliberate architectural change:

- we are **not** training a separate value function
- we are using Robometer as the source of quality / progress signal instead

So this document should be read as:

- paper RECAP as the conceptual reference
- `reward_recap` as our intentionally modified implementation

Instead of using the paper’s value-function stage, we want to use an existing reward model from
`../robometer` that already produces:

- a progress signal in `[0, 1]`
- a success / failure signal

from short windows of images.

We are calling this variant:

- `reward_recap`

The intended picture is:

- paper RECAP:
  - define reward
  - train value function on returns
  - compute advantages
  - threshold into binary improvement indicator `I_t`
  - train policy conditioned on `I_t`
- `reward_recap`:
  - run Robometer on trajectory chunks
  - derive future-gain scores from the coarse Robometer progress trajectory
  - threshold those scores into binary improvement indicator `I_t`
  - train policy conditioned on `I_t`

So the main change is:

- keep the policy-conditioning idea
- replace the value-function stage with a reward-model stage

## 2. What RECAP Is Doing Conceptually

The reference paper is:

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)

At a high level, RECAP is an offline-RL-style policy improvement recipe:

1. collect data
2. train a value function
3. compute an advantage for each example
4. threshold that advantage into a binary indicator
5. train a policy conditioned on that indicator

In the paper, the binary indicator answers:

- "is this action/chunk better than expected from this state?"

The paper’s exact indicator rule is:

- `I_t = 1` when `A^{π_ref}(o_t, a_t, ℓ) > ε_ℓ`

where:

- `A^{π_ref}(o_t, a_t, ℓ)` is the advantage under the reference policy
- `ε_ℓ` is a task-dependent threshold

In plain English, advantage means:

- what actually happened after taking action `a_t`
- minus what the value function expected would happen from state / observation `o_t`

So advantage is not just:

- "is this state good?"

It is:

- "did this action lead to a better or worse future than expected from this state?"

Positive advantage means:

- the future after this action was better than the value-function baseline

Negative advantage means:

- the future after this action was worse than the value-function baseline

More specifically, the paper says in Section V-C that for each episode they obtain a **success / failure label**, and then derive the reward from that episode-level outcome:

In words:

- every non-terminal step gets a reward of `-1`
- if the episode ends in success, the terminal step gets `0`
- if the episode ends in failure, the terminal step gets `-C_fail`

So successful episodes accumulate something like "negative remaining steps until success," while failed episodes get an additional large negative terminal penalty.

Equivalent reward definition:

- `r_t = -1` for every non-terminal step
- `r_T = 0` if the episode terminates in success
- `r_T = -C_fail` if the episode terminates in failure

So the paper’s setup is **not** "purely terminal success only" in the strict mathematical sense,
because it adds a shaped `-1` step cost. But it also is **not** a rich dense supervision signal in
the way Robometer is. The underlying human-provided label is still episode-level success/failure,
and the intermediate reward is a generic construction derived from that label, not a learned
per-frame notion of semantic task progress.

That is why the value function is still important in RECAP:

- it turns this generic return signal into a state-local estimate
- in the paper, that estimate is the expected time-to-completion / return
- that estimate is then used to compute advantage and the binary improvement indicator

That is exactly where `reward_recap` departs from the paper:

- paper RECAP learns a value function and derives `I_t` from advantage
- `reward_recap` derives `I_t` directly from Robometer score trajectories

## 3. Why A Reward-Model Variant Makes Sense

The reward model in `../robometer` gives us a richer task-progress signal than the paper’s generic
reward construction.

Our working assumption is:

- if Robometer already gives a sufficiently informative progress / success signal
- then we may not need a separate learned value model in order to decide whether a chunk is
  positive or negative

So in our design, Robometer is not just a reward helper for another model. It is the primary
source of chunk-quality supervision.

What we know from `../robometer`:

- default config is in `../robometer/robometer/configs/config.yaml`
- default `max_frames` is `16`
- default `use_multi_image` is `true`
- default `use_per_frame_progress_token` is `true`
- default progress loss is discrete with `10` bins

The main model classes are:

- `RBM` in `../robometer/robometer/models/rbm.py`
- prediction heads in `../robometer/robometer/models/heads.py`

Important outputs:

- progress predictions
- success predictions

In `../robometer/robometer/models/rbm.py`, the relevant internal path produces:

- `progress_output`
- `success`

per temporal step / frame group.

In `../robometer/robometer/evals/eval_utils.py`, helper functions like
`extract_rewards_from_output(...)` and `extract_success_probs_from_output(...)` currently use
the **last** value of each subsequence as the reward-like / success-like summary.

For `reward_recap`, we do **not** want to rely only on the last scalar if we can avoid it.
The more useful object is the **16-frame progress evolution**, because it lets us decide whether
the chunk is:

- improving
- stalling
- regressing
- recovering

That is a better substrate for direct chunk labeling than a single endpoint score.

## 4. Data Sources We Expect To Use

We expect to have several types of trajectories:

1. expert demonstration data
2. DAgger-style trajectories with autonomous behavior followed by human correction
3. autonomous successful rollouts
4. autonomous failed rollouts
5. external failure data

This matters because the relabeling pipeline should work across mixed-quality data from multiple
sources.

Even though version 1 does **not** force labels from source metadata, we still expect these sources
to contain very different behavior patterns:

- expert data is often cleaner and more goal-directed
- autonomous rollouts can include both progress and regressions
- DAgger-style data may contain both failure drift and recovery behavior
- external failure data may look very different from in-distribution success data

So `reward_recap` must be:

- reward-model driven
- with labels derived from Robometer outputs alone
- with optional human curation to exclude data where the Robometer reward curves look untrustworthy

## 5. Core Decision: Chunk-Level Labels, Not Episode-Level Labels

We do **not** want to label entire episodes as simply positive or negative.

That would throw away the most useful structure in DAgger-style and mixed-quality data.

Instead, the unit of labeling should be:

- a short chunk / window of frames

Matching Robometer’s natural inference granularity, the default coarse trajectory length is:

- `16` sampled temporal positions

For each Robometer run, we will compute:

- a coarse progress trajectory over `16` sampled temporal positions
- a coarse success-classifier trajectory over those same sampled positions
- metadata about the chunk source

Then we will derive:

- binary conditioning labels `I_t` for coarse intervals within that trajectory

This is the key design principle of `reward_recap`.

### 5.1 Exact alignment rule

Robometer takes a longer underlying frame sequence and compresses it into a **16-step coarse
trajectory** by uniformly subsampling the original frames.

So there are two time axes:

- the original dense episode frames
- the `16` coarse Robometer steps

For version 1, we treat each adjacent pair of coarse Robometer steps as one labelable interval.

That means:

- one Robometer run yields a coarse score trajectory `r_0, r_1, ..., r_T`
- each coarse interval `t -> t+1` corresponds to a chunk of the original episode
- each such coarse interval gets its own label `I_t`
- valid labeled indices are therefore `t = 0, ..., T-1`
- the terminal coarse point `T` has no associated interval label

If we save the subsampled frame indices from Robometer preprocessing, then the interval label `I_t`
is attached to the underlying episode span between those adjacent sampled indices.

## 6. Raw Signals Versus Derived Labels

We should separate two things cleanly:

### Raw signals from Robometer

These are model outputs:

- progress trace / progress bins
- progress score in `[0, 1]`
- per-frame binary success-classifier output
- in the current inference helpers, this is exposed as a per-frame success probability after applying sigmoid to the classifier logits

### Derived policy-training labels

These are our constructed labels:

- `I_t = 1` means positive condition
- `I_t = 0` means negative condition

Here:

- `r_t` means the Robometer progress score at coarse step `t`
- `g_t^(H)` means the future gain computed from the coarse Robometer trajectory

In our implementation, the dependency is:

```text
Robometer coarse score trajectory -> future gain g_t^(H) -> quantile-derived threshold -> I_t
```

So:

- Robometer is the measurement system
- `reward_recap` is the label-construction and policy-training system

## 7. Proposed Label Construction

We currently recommend a direct two-stage construction:

1. compute future-gain scores from the coarse Robometer trajectory
2. compute a quantile-derived threshold from those scores
3. assign binary labels `I_t` based on that quantile-derived threshold

### 7.1 Robometer coarse trajectory

Robometer gives a coarse score trajectory over the downsampled temporal positions in the input
sequence.

For version 1, define:

- `r_t`: the Robometer progress score at coarse step `t`
- `T`: the final coarse step in the Robometer trajectory
- `H`: the future horizon in coarse steps

Use the convention:

- if `H = -1`, then `end(t, H)` means the final coarse step `T`
- otherwise, `end(t, H) = min(t + H, T)`

Default:

- `H = -1`

meaning:

- look all the way to the end of the coarse Robometer trajectory

### 7.2 Future-gain score

Define the score for coarse step `t` as:

```text
g_t^(H) = r_{end(t, H)} - r_t
```

This score is only defined for labeled interval starts:

- `t = 0, ..., T-1`

This means:

- if the Robometer score increases from step `t` to the chosen future horizon, `g_t^(H)` is positive
- if it decreases, `g_t^(H)` is negative

With the default `H = -1`, the score becomes:

```text
g_t = r_T - r_t
```

So version 1 uses the **global future gain** on the 16-step coarse Robometer trajectory.

Why this choice:

- it is closer to the RECAP intuition of "did things get better after this point ?"
- it is more stable than a one-step delta
- it lets failure episodes pull later-bad behavior negative
- it accepts that some locally good early moves in failed episodes may still be weighted down

### 7.3 Threshold labeling

The labeler should use this exact rule:

1. compute `g_t^(H)` from the Robometer coarse trajectory
2. choose a quantile-derived threshold `epsilon_task`
3. assign:
   - `I_t = 1` if `g_t^(H) > epsilon_task`
   - `I_t = 0` if `g_t^(H) <= epsilon_task`

So in version 1:

- above threshold -> positive
- below threshold -> negative

There is no ignored middle bucket in this design.

## 8. Role Of Metadata

Source metadata may still be stored in the relabeled dataset, but in version 1 it is **not** part
of the labeling rule.

That means:

- the policy never sees source metadata directly
- the labeler does not force labels based on source type
- demos, corrections, successes, and failures are all labeled from Robometer outputs alone

We still keep source metadata because it is useful for:

- offline analysis
- debugging label quality
- slicing evaluation metrics by data source

## 9. Proposed Training Recipe

The training recipe should preserve the useful parts of RECAP while replacing the value-function
stage with Robometer-based offline labeling.

### Stage 0: Offline relabeling

1. collect all trajectory sources
2. run Robometer over episode segments, obtaining a 16-step coarse score trajectory for each segment
3. save the subsampled frame indices used to build that trajectory
4. write out:
   - Robometer progress trajectory `r_0, ..., r_T`
   - optional success trajectory
   - future-gain scores `g_t^(H)`
   - `I_t`
   - source metadata

This stage should be a preprocessing step, not on-the-fly training logic.

### Stage 1: Policy warmup / SFT

Purpose:

- establish the base action manifold
- avoid making the model solve the conditional problem before it knows the task

### Stage 2: Conditioned training

Train the policy with the binary label `I_t` on all relabeled coarse intervals.

At this stage:

- prepend the binary condition text
- keep the same supervised action loss as the base policy
- train on both `I_t = 1` and `I_t = 0` examples with the same loss form

At inference time:

- condition on positive

Version-1 simplifications:

- no unconditional branch
- no conditioning dropout
- no classifier-free guidance

So the policy should be sampled in "good behavior" mode.

Paper note:

- this differs from the paper, where `I_t` is derived from value-based advantage
- here we are using Robometer-derived offline labels instead

## 10. Policy Interface We Intend To Use

We should keep the policy-side interface simple and RECAP-like:

- one binary conditioning variable `I_t`

We are **not** currently proposing to condition on:

- raw progress scalar
- raw success score
- multi-bin reward classes

Reasons:

- binary conditioning is closer to RECAP
- simpler to implement
- simpler to reason about at inference time
- easier to compare against the original RECAP idea

If needed later, we can experiment with richer conditioning:

- progress bucket
- signed progress delta bucket
- success score bucket

But binary should be the first version.

### Policy conditioning

In the paper, the improvement indicator is represented as text input:

- `"Advantage: positive"`
- `"Advantage: negative"`

and is inserted into the VLA token sequence before action prediction.

We want to preserve the simple binary conditioning interface from the paper, even though the label
source is different.

For `reward_recap`, use the same wording:

- `"Advantage: positive"`
- `"Advantage: negative"`

`../exla_openpi/src/fla/recap/pi0_recap.py` can still be consulted as a reference point, but it is
not the spec.

## 11. Concrete Implementation Plan

This section is intentionally low-level enough that an engineer should be able to turn it
into code.

### 11.1 New relabeling pipeline

Create an offline script in this repo, likely under:

- `scripts/`

Candidate name:

- `scripts/label_reward_recap.py`

Responsibilities:

1. load trajectories from all configured sources
2. run Robometer over episode segments or clips
3. call Robometer inference
4. collect the coarse score trajectory and sampled frame indices
5. compute `g_t^(H)`
6. apply the threshold rule
7. assign `I_t`
8. save relabeled dataset or per-interval metadata

### 11.2 Robometer adapter

Likely wrap or reuse:

- `../robometer/scripts/example_libero_robometer_wrapper.py`
- `../robometer/robometer/evals/eval_utils.py`

We should isolate this behind a small adapter in this repo, e.g.:

- `src/openpi/reward_recap/robometer_adapter.py`

Responsibilities:

- format chunk inputs for Robometer
- batch inference
- return raw structured outputs:
  - coarse progress trajectory
  - coarse success trajectory
  - sampled frame indices

### 11.3 Labeling logic

Create:

- `src/openpi/reward_recap/labeling.py`

Responsibilities:

- compute `g_t^(H)` from the Robometer coarse trajectory
- implement the threshold rule
- export `I_t`

This module should contain the exact labeling rules in one place.

### 11.4 Policy conditioning

The policy should follow the paper’s conditioning interface as closely as possible:

1. represent the indicator as text
2. insert it into the token sequence before action prediction
3. condition action prediction on that indicator

Candidate location:

- `src/openpi/reward_recap/pi0_reward_recap.py`

Specifically:

- create `"Advantage: positive"` / `"Advantage: negative"` prompt fragments
- ensure they appear in the correct part of the policy input sequence
- keep action prediction conditioned on that indicator, per the paper

### 11.5 Training entrypoint

Add a dedicated training script rather than overloading `scripts/train.py` immediately.

Candidate:

- `scripts/train_reward_recap.py`

Phases:

1. load Robometer-relabeled data with `I_t`
2. warmup / SFT policy stage
3. conditioned policy training
4. save checkpoints

## 12. Expected Dataset Semantics

The implementer should think of the training data after relabeling as a coarse-interval dataset
derived from Robometer trajectories.

Each row should contain something like:

- source episode / clip identifier
- coarse step index `t`
- underlying start / end frame indices for that coarse interval
- observation fields needed by the policy for the start of that interval
- target action chunk aligned to that interval
- source type:
  - demo
  - correction
  - autonomous_success
  - autonomous_failure
  - external_failure
- raw reward-model outputs:
  - coarse Robometer progress trajectory
  - optional coarse success trajectory
- derived fields:
  - `g_t^(H)`
  - `I_t`

This makes the system reproducible and auditable.

## 13. Open Questions

These are follow-up questions. They should not block a version-1 implementation.

### Q1. Exactly how should the score be defined?

For version 1, this is now decided: `g_t^(H) = r_{end(t, H)} - r_t`, with default `H = -1`.

Open follow-up:

- should shorter horizons perform better than the default global horizon?

### Q2. What threshold should we use?

Current recommendation:

- use a single threshold `epsilon_task`
- derive `epsilon_task` from a quantile of the `g_t^(H)` distribution
- label `g_t^(H) > epsilon_task` as positive
- label `g_t^(H) <= epsilon_task` as negative

For our usual one-task-at-a-time training setup, `epsilon_task` is just the fixed threshold for
that task run.

### Q3. Should we ever exclude uncertain examples?

Current recommendation:

- no, not in the core thresholded design
- if Robometer curves look untrustworthy for some data, exclude that data during human curation

### Q4. Should labels depend on dataset source?

Current answer:

- no
- demos, corrections, and failures are labeled from Robometer outputs the same way as other data

## 14. Recommended First Experiment

The first implementable version is exactly the version-1 design above:

1. run Robometer offline to produce a `16`-step coarse trajectory
2. compute `g_t^(H)` with default `H = -1`
3. derive `epsilon_task` from a quantile of the `g_t^(H)` distribution
4. threshold into `I_t`
5. train warmup, then conditioned policy training
6. evaluate success rate, recovery behavior, and whether negative conditioning suppresses bad chunks

## 15. Bottom Line

`reward_recap` is a RECAP-inspired conditioning method that replaces the paper’s value-function
stage with Robometer-based offline labeling. Version 1 operates on the 16-step coarse Robometer
trajectory, computes future-gain scores `g_t^(H)` with default `H = -1`, thresholds those scores
into binary labels `I_t`, and uses those labels for policy conditioning.

## References

- [π0.6*: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)
- `../robometer/robometer/configs/config.yaml`
- `../robometer/robometer/models/rbm.py`
- `../robometer/robometer/models/heads.py`
- `../robometer/robometer/evals/eval_utils.py`
- `../robometer/scripts/example_libero_robometer_wrapper.py`
- `../exla_openpi/src/fla/recap/pi0_recap.py`
- `../exla_openpi/scripts/train_recap_full.py`
