# 🐇 Build a Large Language Model From Scratch — The Wonderland Edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyankavergadia/llm-from-scratch-wonderland/blob/main/Build_an_LLM_From_Scratch_Wonderland.ipynb)

One self-contained Jupyter notebook that walks a first-timer from *"what even is an LLM?"*
to a working, trained, instruction-following GPT — writing every component by hand in PyTorch.

**File:** `Build_an_LLM_From_Scratch_Wonderland.ipynb`

**Landing page:** [`docs/index.html`](docs/index.html) — a one-screen overview with an *Open in Colab* button for every Step.

**Written for people new to the code, not just to LLMs:** every code cell opens with a plain-English note, every tensor is labelled with its shape, and each PyTorch idiom is explained the first time it appears. Eighteen diagrams are embedded in the notebook itself, alongside live plots the learner's own code produces — attention heatmaps for all twelve heads of the real GPT-2, a 2-D map of its embedding space, before/after next-token distributions, a confusion matrix, and a coloured view of the instruction-tuning loss mask.

## What's in it

| Step | Source chapter | What the reader builds |
|---|---|---|
| 0 | Ch. 1 | Setup, the mental model, the six-box pipeline diagram |
| 1 | Ch. 1 | 🎲 A bigram "dumbest language model" — and a demonstration of exactly why it fails |
| 2 | Ch. 2 | Word-level tokenizer → special tokens → BPE → sliding-window dataset → embeddings |
| 3 | Ch. 3 | Attention from a dot product up: softmax → Q/K/V → scaling → causal mask → dropout → multi-head (with heatmaps) |
| 4 | Ch. 4 | LayerNorm, GELU, feed-forward, residual connections, transformer block, the full 163M-parameter GPT |
| 5 | Ch. 5 | Cross-entropy & perplexity, the training loop, **WonderGPT trained on *Alice in Wonderland***, temperature & top-k, checkpoints, loading OpenAI's real GPT-2 weights |
| 6 | Ch. 6 | 🔍 Classification fine-tune: *"Wonderland or Baker Street?"* — Lewis Carroll vs. Arthur Conan Doyle (~81% on held-out sentences, from a 50% coin flip) |
| 7 | Ch. 7 | 🐈 **Cheshire**, an instruction-following assistant, graded by exact match on a held-out set |
| — | Ch. 7 | Curtain call: what's missing from the modern stack (RoPE, RMSNorm, SwiGLU, GQA, MoE, RLHF, KV caching) |

Every step ends with a `🎯 Your turn` box of experiments.

## What was changed from the book's notebooks

- **One notebook instead of six**, fully self-contained — no `previous_chapters.py`, no `gpt_download.py`, no TensorFlow.
- **New corpus.** *Alice in Wonderland* + *The Adventures of Sherlock Holmes* (Project Gutenberg) replace `the-verdict.txt`.
- **New examples throughout.** "the mad hatter poured more tea" replaces the attention toy sentence; the tokenizer demos use Jabberwocky, emoji and `strawberry`.
- **New Step 1** (the book's ch. 1 has no code): a ten-line bigram model that fails instructively and motivates attention.
- **New Step 6 task.** Author attribution between two real books replaces the SMS spam dataset — real text, real signal, no download. Measured: 48% → 81% over 3 epochs.
- **New Step 7 dataset.** A generated instruction set with six exactly-checkable task types replaces the 1,100-example Alpaca file, so the result can be *graded* rather than eyeballed.
- **Added visualizations:** attention heatmaps, GELU vs ReLU, temperature distributions, loss/accuracy curves, and nearest-neighbour probing of GPT-2's learned embedding space.
- **Honest framing** of what instruction tuning does and does not do, including a live demonstration of where hallucination comes from.

## Running it (Google Colab)

1. Click the **Open in Colab** badge above (or upload `Build_an_LLM_From_Scratch_Wonderland.ipynb` via `File → Upload notebook`).
2. **`Runtime → Change runtime type → T4 GPU`**, then `Save`. The free tier is enough.
3. `Runtime → Run all`, or step through cell by cell (recommended — the point is to read as you go).

The first code cell installs `tiktoken`; PyTorch and matplotlib are already on Colab.

Two knobs at the top of Step 0:

- `FAST = True` — shorter training runs (default). Set `False` for better results.
- `LOAD_PRETRAINED = True` in Step 5.6 — downloads GPT-2 small (~670 MB). Steps 6 and 7 need it.

### Rough timings on a free Colab T4

| Step | Time |
|---|---|
| Steps 0–4 | seconds |
| Step 5 · training WonderGPT (15 epochs) | 3–5 min |
| Step 5.6 · GPT-2 download | 1–2 min (one time, ~670 MB) |
| Step 6 · classifier fine-tune (3 epochs) | 2–4 min |
| Step 7 · instruction fine-tune (2 epochs) + grading | 3–6 min |

Everything also runs CPU-only, roughly 6–10× slower. If you are stuck on CPU, drop
`num_epochs` in Step 5 to 5.

**Colab files are ephemeral.** The books and checkpoints the notebook writes disappear when
the runtime disconnects. To keep a trained model, mount Drive and save into
`/content/drive/MyDrive/`.

### Running locally instead

```bash
pip install torch tiktoken matplotlib jupyter
jupyter lab
```

Works on CPU, CUDA, or Apple Silicon (MPS). On a machine with 8 GB of RAM or less, close
other heavy apps first — Steps 6 and 7 hold a 124M-parameter model plus its optimizer state.

## Files the notebook writes

`alice.txt`, `sherlock.txt`, `gpt2-small-124M.pth`, `wondergpt.pth`, `cheshire-instruct.pth`

## Credit

Architecture, training loops and pedagogical arc are from
**[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)**
by **Sebastian Raschka** (chapters 1–7) — code at
[github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

Texts: *Alice's Adventures in Wonderland* (Carroll, 1865) and *The Adventures of Sherlock Holmes*
(Conan Doyle, 1892), public domain, via [Project Gutenberg](https://www.gutenberg.org).
