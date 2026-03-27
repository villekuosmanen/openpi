# `pi`, `pi05`, and `Pi0FAST`: A Concrete Guide

This is a code-first guide to the `pi` model family in this repo.

It is written for the reader who wants answers to questions like:

- What Python classes are involved?
- What tensors go in?
- What tensors come out?
- What shapes should I picture in my head?
- What exactly changes between `pi0`, `pi05`, and `Pi0FAST`?

The main files this guide refers to are:

- `src/openpi/models/model.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_fast.py`
- `src/openpi/models/tokenizer.py`
- `src/openpi/training/config.py`
- `scripts/train.py`

## 1. Start With The Data Contract

The most important class to understand first is `Observation` in `src/openpi/models/model.py`.

That class is the structured input object that the models consume. Its main fields are:

- `images: dict[str, float[*b, h, w, c]]`
- `image_masks: dict[str, bool[*b]]`
- `state: float[*b, s]`
- `tokenized_prompt: int[*b, l] | None`
- `tokenized_prompt_mask: bool[*b, l] | None`
- `token_ar_mask: int[*b, l] | None`
- `token_loss_mask: bool[*b, l] | None`
- `action_is_pad: bool[*b, ah] | None`

The action target is defined right below that as:

- `Actions = float[*b, ah, ad]`

Here is what those symbols mean in plain English:

- `*b`: batch dimensions. In normal training, this is usually just batch size.
- `h, w, c`: image height, width, channels.
- `s`: robot state dimension.
- `l`: token length.
- `ah`: action horizon.
- `ad`: action dimension.

### Realistic batch example

A realistic `Pi0` or `pi05` training batch might look like:

- `images["base_0_rgb"]`: shape `[32, 224, 224, 3]`
- `images["left_wrist_0_rgb"]`: shape `[32, 224, 224, 3]`
- `images["right_wrist_0_rgb"]`: shape `[32, 224, 224, 3]`
- `image_masks["base_0_rgb"]`: shape `[32]`
- `state`: shape `[32, 14]` for a 14-dim robot, or `[32, 32]` for a 32-dim setup
- `tokenized_prompt`: shape `[32, 48]` for many `pi0` configs, or `[32, 200]` for `pi05`
- `actions`: shape `[32, 50, 14]` or `[32, 50, 32]`

So if you want a simple mental picture for one training example, it is:

- 3 camera images
- 1 text instruction
- 1 robot state vector
- 1 target action chunk with 50 future timesteps

That last point is worth emphasizing. These models do **not** predict one action vector like `[14]`. They predict a **short sequence of future actions** like `[50, 14]`.

## 2. Which Config Class Controls What?

There are two main config classes for the JAX family:

- `Pi0Config` in `src/openpi/models/pi0_config.py`
- `Pi0FASTConfig` in `src/openpi/models/pi0_fast.py`

### `Pi0Config`

`Pi0Config` controls both ordinary `pi0` and the repo's `pi05` mode.

Important fields:

- `action_dim`
- `action_horizon`
- `max_token_len`
- `paligemma_variant`
- `action_expert_variant`
- `pi05`
- `discrete_state_input`

Default values in this file are:

- `action_dim = 32`
- `action_horizon = 50`
- `paligemma_variant = "gemma_2b"`
- `action_expert_variant = "gemma_300m"`

When `pi05=False`, `max_token_len` defaults to `48`.

When `pi05=True`, `max_token_len` defaults to `200`.

That is not arbitrary. `pi05` pushes state information into the tokenized prompt, so the prompt needs more room.

### `Pi0FASTConfig`

`Pi0FASTConfig` in `src/openpi/models/pi0_fast.py` is a different model family.

Important fields:

- `action_dim = 32`
- `action_horizon = 32`
- `max_token_len = 250`
- `fast_model_tokenizer`
- `fast_model_tokenizer_kwargs`

Compared with `Pi0`, the FAST model has a longer token budget and a shorter default action horizon. That matches the fact that it is doing token prediction rather than continuous flow-matching over a 50-step chunk.

## 3. What Objects Get Constructed Inside `Pi0`

The main implementation class is `Pi0` in `src/openpi/models/pi0.py`.

When you call `Pi0Config.create(...)` from `src/openpi/models/pi0_config.py`, it creates
`Pi0` from `src/openpi/models/pi0.py`.

The `Pi0.__init__` method constructs the following major pieces:

1. `self.PaliGemma.llm`
   - a bridged Gemma-based module
   - configured with both `paligemma_config` and `action_expert_config`

2. `self.PaliGemma.img`
   - a bridged `SigLIP` image encoder

3. `self.action_in_proj` in `src/openpi/models/pi0.py`
   - a linear layer mapping actions from shape `[..., action_dim]` to transformer width

4. Either:
   - `self.state_proj` and `self.action_time_mlp_*` in `src/openpi/models/pi0.py` for
     ordinary `pi0`
   - or `self.time_mlp_*` in `src/openpi/models/pi0.py` for `pi05`

5. `self.action_out_proj` in `src/openpi/models/pi0.py`
   - a linear layer mapping transformer hidden states back to action dimension

### Concrete example

Suppose:

- batch size `b = 32`
- action horizon `ah = 50`
- action dim `ad = 14`
- action expert width `emb = 1024` (example width, exact value depends on Gemma config)

Then:

- input actions have shape `[32, 50, 14]`
- after `action_in_proj`, action tokens look like `[32, 50, 1024]`
- after the transformer, suffix hidden states might still be `[32, suffix_len, 1024]`
- after `action_out_proj`, the predicted velocity becomes `[32, 50, 14]`

That last output shape is important: the model predicts something with the **same shape as the target action chunk**.

## 4. What The Tokenizer Actually Does

The prompt-side behavior lives in `src/openpi/models/tokenizer.py`.

There are two tokenizer classes relevant here, both in `src/openpi/models/tokenizer.py`:

- `PaligemmaTokenizer`
- `FASTTokenizer`

### `PaligemmaTokenizer` in `src/openpi/models/tokenizer.py` for `pi0`

In `PaligemmaTokenizer.tokenize(prompt, state=None)` in `src/openpi/models/tokenizer.py`:

- the prompt text is cleaned
- BOS is added
- a newline token is appended

So for ordinary `pi0`, the tokenized prompt is basically just the language instruction.

If `max_token_len = 48`, then a batch of prompts might look like:

- `tokenized_prompt.shape = [32, 48]`
- `tokenized_prompt_mask.shape = [32, 48]`

That does **not** mean every prompt uses all 48 tokens. It means the tokenizer pads or truncates to a fixed length of 48.

### `PaligemmaTokenizer` in `src/openpi/models/tokenizer.py` for `pi05`

If `state` is provided, `PaligemmaTokenizer.tokenize(...)` in
`src/openpi/models/tokenizer.py` switches to the `pi05` format.

It:

1. discretizes each state coordinate into 256 bins
2. converts those bins to a string like `"124 87 130 92 ..."`
3. constructs a prompt like:

```text
Task: pick up the mug, State: 124 87 130 92 ...;
Action:
```

4. tokenizes that full string

So in `pi05`, state stops being "just a float vector the suffix reads" and starts becoming "part of the language-style prompt context."

That is why `Pi0Config` gives `pi05` a larger default token budget of `200`.

### `FASTTokenizer` in `src/openpi/models/tokenizer.py` for `Pi0FAST`

`FASTTokenizer` in `src/openpi/models/tokenizer.py` does more work:

- it still serializes prompt + discretized state into prefix tokens
- it can also encode action chunks into discrete FAST tokens
- it returns:
  - `tokens`
  - `token_mask`
  - `ar_mask`
  - `loss_mask`

So `Pi0FAST` is not just "Pi0 but faster." It is a different modeling setup with explicit token-level supervision.

## 5. How The Data Pipeline Chooses The Right Tokenization

The repo chooses between these tokenization paths in `ModelTransformFactory` inside
`src/openpi/training/config.py`.

This is where model type gets translated into actual preprocessing.

### For `ModelType.PI0`

The transform stack includes:

- `InjectDefaultPrompt`
- `ResizeImages(224, 224)`
- `TokenizePrompt(PaligemmaTokenizer(...))` from `src/openpi/transforms.py`
- `PadStatesAndActions(...)` from `src/openpi/transforms.py`

That means prompt text is tokenized, but state is not pushed into the prompt.

### For `ModelType.PI05`

The transform stack is almost the same, except:

- `TokenizePrompt(..., discrete_state_input=model_config.discrete_state_input)` from
  `src/openpi/transforms.py`

That flag is the crucial bridge between config and prompt format.

### For `ModelType.PI0_FAST`

The transform stack uses:

- `TokenizeFASTInputs(FASTTokenizer(...))` from `src/openpi/transforms.py`

and the output side uses:

- `ExtractFASTActions(...)` from `src/openpi/transforms.py`

This is another hint that `Pi0FAST` is really operating on a token-prediction contract rather than the same one as `Pi0`.

## 6. What `Pi0.embed_prefix()` Produces

The first important method in `src/openpi/models/pi0.py` is `embed_prefix(...)`.

Its job is to combine:

- image tokens from `SigLIP`
- prompt token embeddings from the language model

into one prefix block.

### Inputs

Example input shapes:

- `obs.images["base_0_rgb"]`: `[32, 224, 224, 3]`
- `obs.images["left_wrist_0_rgb"]`: `[32, 224, 224, 3]`
- `obs.images["right_wrist_0_rgb"]`: `[32, 224, 224, 3]`
- `obs.tokenized_prompt`: `[32, 48]` for `pi0`, or `[32, 200]` for `pi05`

### Where the embeddings actually come from

Yes, there are learned layers here, and they matter.

`embed_prefix(...)` does **not** receive ready-made embeddings. It receives:

- image tensors like `[32, 224, 224, 3]`
- integer token IDs like `[32, 48]` or `[32, 200]`

and then converts both of those into vectors of the same transformer width.

#### Prompt side: token IDs -> token embeddings

For the prompt path, the code does this inside `Pi0.embed_prefix(...)` in
`src/openpi/models/pi0.py`:

- `self.PaliGemma.llm(obs.tokenized_prompt, method="embed")`

That means the integer token IDs are passed through the language model's learned token
embedding table.

So if:

- `obs.tokenized_prompt.shape = [32, 200]`
- transformer width `emb = 2048` as an example

then the embedded prompt tensor is conceptually:

- `[32, 200, 2048]`

before masking and concatenation with image tokens.

This is the standard "token ID -> learned embedding vector" step that exists in normal LLMs.

#### Image side: image tensor -> image patch embeddings

For the image path, the code does this inside `Pi0.embed_prefix(...)` in
`src/openpi/models/pi0.py`:

- `self.PaliGemma.img(obs.images[name], train=False)`

That is the `SigLIP` vision encoder. It takes an image tensor like:

- `[32, 224, 224, 3]`

and turns it into a sequence of image token embeddings like:

- `[32, n_img_tokens, emb]`

The exact `n_img_tokens` depends on the vision backbone internals, but the important part is
that the last dimension already matches the transformer embedding width used by the language
side.

That alignment is intentional in `Pi0.__init__` in `src/openpi/models/pi0.py`, where the
image module is created with:

- `num_classes=paligemma_config.width`

So the image encoder is configured to emit vectors in the same width as the PaliGemma-side
transformer expects.

#### Why this matters

Without these two learned conversions, the concatenation inside `embed_prefix(...)` would be
impossible.

You cannot concatenate:

- raw token IDs shaped `[32, 200]`
- raw image tensors shaped `[32, 224, 224, 3]`

into one transformer input.

What `embed_prefix(...)` is really concatenating is:

- image token embeddings shaped like `[32, n_img_tokens, emb]`
- prompt token embeddings shaped like `[32, prompt_len, emb]`

So the shared `emb` dimension is the key glue.

### Outputs

`embed_prefix(...)` returns:

- `tokens`: shape `[b, prefix_len, emb]`
- `input_mask`: shape `[b, prefix_len]`
- `ar_mask`: shape `[prefix_len]`

The exact `prefix_len` depends on:

- how many image tokens `SigLIP` emits per image
- how many cameras are present
- how many prompt tokens are non-padding

You do not need the exact number to understand the flow. The important point is that `embed_prefix(...)` produces one long sequence of prefix embeddings that the transformer can attend over.

### Mental picture

Think of the prefix as:

- image patch embeddings from camera 1
- image patch embeddings from camera 2
- image patch embeddings from camera 3
- prompt token embeddings

all concatenated into one big context block.

But the state story differs by model variant, and it is important not to blur that:

- in ordinary `pi0` from `src/openpi/models/pi0.py`, the prefix does **not** contain a
  separate continuous state embedding
- in `pi05` mode inside `src/openpi/models/pi0.py`, the prefix can contain state
  **indirectly**, because the state was discretized into text by
  `PaligemmaTokenizer.tokenize(...)` in `src/openpi/models/tokenizer.py`

So a more precise mental model is:

- `pi0` prefix = image embeddings + prompt token embeddings
- `pi05` prefix = image embeddings + prompt token embeddings, where some of those prompt
  tokens now encode discretized state values

The continuous float state tensor `obs.state` still exists in the `Observation` object in
`src/openpi/models/model.py`, but in `pi05` its influence on the prefix comes through the
tokenization path rather than through a dedicated continuous prefix token.

## 7. What `Pi0.embed_suffix()` Produces

The second important method is `Pi0.embed_suffix(...)` in `src/openpi/models/pi0.py`.

It takes:

- `obs`
- `noisy_actions`
- `timestep`

and converts them into the suffix tokens that drive continuous action prediction.

### Example input shapes

Suppose:

- `noisy_actions.shape = [32, 50, 14]`
- `timestep.shape = [32]`

### Where the suffix embeddings actually come from

Just like the prefix path, the suffix path also has several learned or computed
representation changes that are easy to skip over if you read too quickly.

#### Action side: action vectors -> action token embeddings

The first conversion is this line inside `Pi0.embed_suffix(...)` in
`src/openpi/models/pi0.py`:

- `self.action_in_proj(noisy_actions)`

This is a learned linear layer created in `Pi0.__init__` in `src/openpi/models/pi0.py`.

If:

- `noisy_actions.shape = [32, 50, 14]`
- action expert width `emb = 1024` as an example

then:

- `action_in_proj(noisy_actions).shape = [32, 50, 1024]`

So this is the suffix-side equivalent of token embedding lookup on the language side:

- raw continuous action vectors come in
- transformer-width action embeddings come out

#### Time side: scalar timestep -> time embedding

The timestep starts as:

- `timestep.shape = [32]`

That means one scalar per batch element, not one scalar per action step.

The code then applies this function from `src/openpi/models/pi0.py`:

- `posemb_sincos(timestep, self.action_in_proj.out_features, ...)`

This is **not** a learned linear layer. It is a deterministic sine-cosine embedding
function in `src/openpi/models/pi0.py`.

So with `emb = 1024`, the result is:

- `time_emb.shape = [32, 1024]`

That gives the model a vector representation of "where in the denoising process am I?"

Then:

- `action_in_proj(noisy_actions)` gives action tokens of shape `[32, 50, emb]`
- `posemb_sincos(timestep, emb, ...)` gives time embeddings of shape `[32, emb]`

From there, the path splits.

### Ordinary `pi0` path

When `self.pi05` is false:

1. `obs.state` is projected with `state_proj` from `src/openpi/models/pi0.py`
   - input shape `[32, 14]`
   - output shape `[32, emb]`
   - then expanded to `[32, 1, emb]`

2. time embedding is repeated across action horizon
   - from `[32, emb]`
   - to `[32, 50, emb]`

3. repeated time is concatenated with action tokens
   - action tokens: `[32, 50, emb]`
   - repeated time: `[32, 50, emb]`
   - concatenated: `[32, 50, 2 * emb]`

4. that concatenated tensor is mixed by `action_time_mlp_in` and `action_time_mlp_out`
   from `src/openpi/models/pi0.py`
   - `action_time_mlp_in` is a learned linear layer from `[32, 50, 2 * emb]` to `[32, 50, emb]`
   - `nnx.swish` applies a nonlinearity
   - `action_time_mlp_out` keeps the tensor in shape `[32, 50, emb]`

So the job of this pair of learned layers is:

- take "action information + timestep information" sitting side by side
- compress and mix them into one transformer-width suffix representation

5. final suffix tokens are:
   - 1 state token
   - 50 action tokens

So the `pi0` suffix sequence length is effectively:

- `1 + action_horizon`

For a default horizon of `50`, that means:

- suffix length `= 51`

### `pi05` path

When `self.pi05` is true:

1. there is **no** continuous state token in the suffix
2. time embedding goes through `time_mlp_in` and `time_mlp_out` from
   `src/openpi/models/pi0.py`
   - both are learned linear layers
   - the shape stays `[32, emb]`
3. the result becomes `adarms_cond`
4. action tokens stay action tokens
   - so `action_expert_tokens.shape` is still `[32, 50, emb]`

So the `pi05` suffix sequence length is just:

- `action_horizon`

For a default horizon of `50`, that means:

- suffix length `= 50`

This is the most concrete implementation difference between `pi0` and `pi05` in this repo:

- `pi0`: suffix contains a separate continuous state token
- `pi05`: suffix does not; state was already moved into the prompt path

## 8. Concrete `Pi0.compute_loss(...)` Walkthrough

The main training logic lives in `Pi0.compute_loss(...)` in `src/openpi/models/pi0.py`.

Let us walk through one realistic example.

Assume:

- batch size `32`
- action horizon `50`
- action dim `14`

So:

- `actions.shape = [32, 50, 14]`

### Step 1: preprocess observations

`preprocess_observation(...)` in `src/openpi/models/model.py`:

- resizes images if needed
- applies augmentation if `train=True`
- ensures masks are present

After preprocessing, image shapes are still conceptually:

- `[32, 224, 224, 3]`

### Step 2: sample noise and time

The code samples:

- `noise.shape = [32, 50, 14]`
- `time.shape = [32]`

Then expands time to:

- `time_expanded.shape = [32, 1, 1]`

### Step 3: build flow-matching targets

The code computes:

- `x_t = t * noise + (1 - t) * actions`
- `u_t = noise - actions`

Both have shape:

- `[32, 50, 14]`

So you can think of training as:

- corrupt the target actions into `x_t`
- ask the model to predict the velocity field `u_t`

### Step 4: embed prefix and suffix

The code calls these methods from `src/openpi/models/pi0.py`:

- `prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)`
- `suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)`

At this point:

- prefix tokens carry image + prompt context
- suffix tokens carry noisy action context
- and in `pi05`, `adarms_cond` also carries timestep conditioning

### Step 5: transformer forward

The code concatenates prefix and suffix masks, builds an attention mask, computes positions,
and runs the transformer in `Pi0.compute_loss(...)` from `src/openpi/models/pi0.py`:

- `self.PaliGemma.llm([prefix_tokens, suffix_tokens], ...)`

The transformer returns:

- `prefix_out`
- `suffix_out`

There is another important hidden representation change here:

- `prefix_tokens` and `suffix_tokens` go **into** the transformer as embeddings
- `prefix_out` and `suffix_out` come **out** as contextualized hidden states

So even if `suffix_tokens.shape` was already `[32, 50, emb]` or `[32, 51, emb]`, the
meaning of that tensor has changed after the transformer.

Before the transformer, a suffix token mostly means:

- "this timestep of the noisy action chunk, plus whatever conditioning was injected"

After the transformer, a suffix hidden state means:

- "that suffix position, after attending to the full prefix context and the other allowed
  suffix positions"

The model then takes:

- `suffix_out[:, -self.action_horizon:]`

That slice is important. It means:

- ignore any extra leading suffix positions like the `pi0` state token
- keep the final `50` action positions

So this slice has shape:

- `[32, 50, emb]`

### Step 6: project back to action space

`action_out_proj(...)` from `src/openpi/models/pi0.py` maps:

- `[32, 50, emb]`

to:

- `[32, 50, 14]`

This is another learned linear layer created in `Pi0.__init__`.

Its job is the inverse of `action_in_proj`:

- `action_in_proj`: action vector -> transformer-width embedding
- `action_out_proj`: transformer-width hidden state -> action-space vector

That output is `v_t`, the predicted velocity.

### Step 7: compute loss

The code computes:

- `per_step_loss = mean((v_t - u_t)^2, axis=-1)`

Since the mean is over the action dimension, the result shape is:

- `[32, 50]`

Then the trainer in `scripts/train.py` averages that over batch and horizon.

So when you see `loss` logged in `scripts/train.py`, it is the mean of a per-step loss tensor shaped like:

- `[batch, action_horizon]`

## 9. What `pi05` Changes In That Training Path

`pi05` uses the same outer training loop in `scripts/train.py`.

It does **not** introduce a different training script or a different top-level loss type.

What changes is the internal conditioning path.

### Prompt side

In `pi05`, the prompt tokenizer in `src/openpi/models/tokenizer.py` can consume state and
turn it into text.

So instead of the prompt being conceptually:

```text
pick up the mug
```

it becomes more like:

```text
Task: pick up the mug, State: 124 87 130 92 ...;
Action:
```

### Suffix side

In `pi05`, there is no `state_proj` token added to the suffix in
`Pi0.embed_suffix(...)` from `src/openpi/models/pi0.py`.

Instead:

- state has already influenced the prompt embeddings
- time influences the action expert through `adarms_cond`

So the shortest accurate description of `pi05` in this repo is:

- same `Pi0` backbone
- different prompt format
- different timestep injection
- one fewer suffix token

## 10. Concrete `Pi0.sample_actions(...)` Walkthrough

Inference for `Pi0` and `pi05` lives in `Pi0.sample_actions(...)` in `src/openpi/models/pi0.py`.

Suppose you run inference with:

- `batch_size = 8`
- `action_horizon = 50`
- `action_dim = 14`
- `num_steps = 10`

### Step 1: initialize noise

If no initial noise is provided, the code creates:

- `noise.shape = [8, 50, 14]`

### Step 2: cache the prefix once

The code runs these steps inside `Pi0.sample_actions(...)` in `src/openpi/models/pi0.py`:

- `embed_prefix(observation)`
- then a prefix-only transformer pass

This produces a `kv_cache`.

That prefix-only pass is itself a representation change worth naming:

- input: prefix embeddings `[b, prefix_len, emb]`
- output: transformer-managed key/value cache for each layer

You do not normally manipulate those cache tensors directly in the high-level code, but
conceptually they are the stored attention keys and values for the fixed prefix context.

That is important because:

- images and prompt do not change during denoising
- there is no reason to recompute the whole prefix on every denoising step

### Step 3: iterative denoising

At each step:

1. call `embed_suffix(observation, x_t, time)` from `src/openpi/models/pi0.py`
2. build suffix attention masks
3. combine suffix attention with the cached prefix
4. run the transformer on suffix queries
5. project to `v_t`
6. update:

```text
x_t <- x_t + dt * v_t
```

where:

- `dt = -1 / num_steps`

The important hidden conversion in step 4 is:

- `suffix_tokens` go into the transformer
- contextualized suffix hidden states come out

Then step 5 applies the learned `action_out_proj` layer from `src/openpi/models/pi0.py`,
converting those hidden states from shape `[8, 50, emb]` back to action-space vectors shaped
`[8, 50, 14]`.

After enough steps, `x_t` becomes the final predicted action chunk:

- shape `[8, 50, 14]`

So inference is best imagined as:

- start from random action noise
- repeatedly refine that whole 50-step action chunk
- return the final denoised chunk

## 11. What `Pi0FAST` Looks Like Concretely

`Pi0FAST` in `src/openpi/models/pi0_fast.py` is different enough that it is worth treating separately.

### Input contract

`Pi0FASTConfig.inputs_spec(...)` in `src/openpi/models/pi0_fast.py` includes:

- images
- image masks
- state
- `tokenized_prompt`
- `tokenized_prompt_mask`
- `token_ar_mask`
- `token_loss_mask`

That means a realistic FAST batch might include:

- `tokenized_prompt.shape = [32, 250]`
- `tokenized_prompt_mask.shape = [32, 250]`
- `token_ar_mask.shape = [32, 250]`
- `token_loss_mask.shape = [32, 250]`

Even though `Actions` still exists in the type signature, the learning target in practice is token-level next-token prediction.

### `Pi0FAST.embed_inputs(...)`

`Pi0FAST.embed_inputs(...)` in `src/openpi/models/pi0_fast.py` combines:

- image token embeddings
- tokenized prompt embeddings

and returns:

- token embeddings: `[b, total_seq_len, emb]`
- input mask: `[b, total_seq_len]`
- autoregressive mask: `[b, total_seq_len]`

The same "do not skip the hidden conversion" rule applies here too.

#### Prompt side in `Pi0FAST`

The code calls this line inside `Pi0FAST.embed_inputs(...)` in
`src/openpi/models/pi0_fast.py`:

- `self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)`

So integer token IDs like `[32, 250]` become prompt token embeddings like:

- `[32, 250, emb]`

through the language model's learned embedding table.

#### Image side in `Pi0FAST`

The code calls this line inside `Pi0FAST.embed_inputs(...)` in
`src/openpi/models/pi0_fast.py`:

- `self.PaliGemma.img(obs.images[name], train=False)`

So each image tensor like `[32, 224, 224, 3]` becomes a sequence of image embeddings like:

- `[32, n_img_tokens, emb]`

#### Concatenation step

`Pi0FAST.embed_inputs(...)` in `src/openpi/models/pi0_fast.py` concatenates those along the
sequence dimension, not the embedding dimension.

So if the three image streams together produced `900` image tokens total and the prompt was
padded to `250` tokens, then a mental model could be:

- image embeddings: `[32, 900, emb]`
- prompt embeddings: `[32, 250, emb]`
- concatenated input embeddings: `[32, 1150, emb]`

The big difference from `Pi0` is that the prompt side is not just context. It also carries the autoregressive structure used for token prediction.

### `Pi0FAST.compute_loss(...)`

The FAST loss path in `Pi0FAST.compute_loss(...)` from `src/openpi/models/pi0_fast.py` is:

1. preprocess observation
2. embed all inputs
3. build attention mask
4. shift token targets by one position
5. compute logits for next-token prediction
6. apply cross-entropy only where `token_loss_mask` is true

There are two easy-to-miss representation changes in steps 4 and 5.

#### Step 4: token IDs -> one-hot targets

The code takes this tensor in `Pi0FAST.compute_loss(...)` from
`src/openpi/models/pi0_fast.py`:

- `observation.tokenized_prompt[:, 1:]`

which might have shape:

- `[32, 249]`

and converts it with `jax.nn.one_hot(...)` into:

- `[32, 249, vocab_size]`

So the targets are no longer token IDs at that point. They are one-hot vectors over the
vocabulary.

#### Step 5: hidden states -> logits

The code first gets this in `Pi0FAST.compute_loss(...)` from
`src/openpi/models/pi0_fast.py`:

- `pre_logits`

from the transformer, then passes those through the model's decoding path to get:

- `logits.shape = [32, 249, vocab_size]`

So this is the FAST analogue of `action_out_proj` in `Pi0`:

- transformer hidden states go in
- vocabulary-sized logits come out

Then `jax.nn.log_softmax(...)` turns those logits into log-probabilities before the
cross-entropy is computed.

So unlike `Pi0` / `pi05`, the loss is not over a tensor like:

- `[32, 50, 14]`

Instead it is over token positions like:

- `[32, 249]`

after the one-token shift.

That is the key conceptual difference:

- `Pi0` / `pi05`: continuous action chunk prediction with flow matching
- `Pi0FAST`: autoregressive token prediction with cross-entropy

## 12. Concrete Side-By-Side Summary

If you want the shortest useful comparison, here it is.

### `Pi0`

Files:

- `src/openpi/models/pi0_config.py`
- `src/openpi/models/pi0.py`

Input picture:

- images: `[32, 224, 224, 3]` x 3 cameras
- prompt tokens: `[32, 48]`
- state: `[32, 14]` or `[32, 32]`
- target actions: `[32, 50, 14]` or `[32, 50, 32]`

Key idea:

- state stays continuous
- suffix contains a dedicated state token
- time is concatenated with action tokens and mixed with an MLP

Loss:

- flow-matching MSE

### `pi05` (inside `Pi0`)

Files:

- `src/openpi/models/pi0_config.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/tokenizer.py`

Input picture:

- images: `[32, 224, 224, 3]` x 3 cameras
- prompt tokens: usually padded to `[32, 200]`
- state: still exists as a float tensor, but it also gets discretized into prompt text during tokenization
- target actions: `[32, 50, 14]` or `[32, 50, 32]`

Key idea:

- state moves into prompt text
- suffix no longer has a continuous state token
- time is injected via adaRMS conditioning

Loss:

- still flow-matching MSE

### `Pi0FAST`

Files:

- `src/openpi/models/pi0_fast.py`
- `src/openpi/models/tokenizer.py`

Input picture:

- images: `[32, 224, 224, 3]`
- tokenized prompt: `[32, 250]`
- autoregressive mask: `[32, 250]`
- loss mask: `[32, 250]`

Key idea:

- token-level autoregressive modeling
- discrete action-token style path

Loss:

- next-token cross-entropy

## 13. Where The Trainer Fits

For `Pi0` and `pi05`, the outer training loop is `scripts/train.py`.

That trainer does not know much model-specific detail. It basically:

1. creates the model from config
2. creates the data loader
3. gets `(observation, actions)` batches
4. calls `model.compute_loss(...)` on the model class defined in either
   `src/openpi/models/pi0.py` or `src/openpi/models/pi0_fast.py`
5. averages the result
6. applies optimizer updates

So when reading this repo, the best separation of concerns is:

- `src/openpi/training/config.py`: decides how raw data is transformed
- `src/openpi/models/tokenizer.py`: decides how prompt/state become tokens
- `src/openpi/models/pi0.py`: defines continuous-action flow model behavior
- `src/openpi/models/pi0_fast.py`: defines token-prediction behavior
- `scripts/train.py`: runs the generic training loop

## 14. Bottom Line

If you only remember three things, remember these:

1. In this repo, `Pi0` and `pi05` are both continuous action-chunk models. A realistic target tensor is something like `[32, 50, 14]`, not just a single action vector.
2. `pi05` is not a separate top-level model class here. It is `Pi0` with a different conditioning strategy:
   - state moves into the prompt
   - time moves into adaRMS
3. `Pi0FAST` is the genuinely different family in this codebase: it predicts tokens with cross-entropy rather than predicting continuous action velocities with flow matching.
