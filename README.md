# genvf-data

Traces and prefixes for generative verifier training data.

## Pipeline: How to Generate Suffixes

### Step 1: Generate Traces

Generate full reasoning traces from a model on the GVF-3k dataset using OpenRouter / OpenAI-compatible APIs.

```bash
export OPENROUTER_API_KEY="your-key"
python generate_traces.py --model qwen3-235b-thinking
python generate_traces.py --model gpt5-mini --concurrency 8 --temperature 0.7
```

Outputs are saved to `traces/<model>.jsonl`.

### Step 2: Generate Prefixes

Truncate the reasoning traces into prefixes at transition keywords (e.g., "Wait", "But", "Alternatively") using `TrajectorySegmenter` from `make_prefix.py`.

```bash
python generate_prefixes.py --model qwen3-235b-thinking
python generate_prefixes.py --model gpt5-mini --start 0 --end 100
```

Outputs are saved to `prefixes/<model>.jsonl`.

### Step 3: Generate Suffixes

Continue from each truncated prefix by prompting the model with the original problem and the prefix as a partial assistant response.

```bash
python generate_suffixes.py --model qwen3-235b-thinking --num-suffixes 8
python generate_suffixes.py --model qwen3-235b-thinking --start 0 --end 100 --concurrency 8
```

### Step 4: Generate Summaries

Use Gemini to summarize the suffix rollouts into structured bullet-point steps (prefix steps + suffix variants), then push the annotated dataset to HuggingFace Hub.

```bash
export GEMINI_API_KEY="your-key"
python gemini_summary.py
```

The prompt template is in `prompts/gemini-summary-prompt/v2.md`.

## Data

| File | Description |
|---|---|
| `traces/qwen3-235b-thinking.jsonl` | Full reasoning traces from Qwen3-235B |
| `traces/gpt5-mini.jsonl` | Full reasoning traces from GPT5-mini |
| `prefixes/qwen3-235b-thinking.jsonl` | Extracted reasoning prefixes from Qwen3-235B |
| `prefixes/gpt5-mini.jsonl` | Extracted reasoning prefixes from GPT5-mini |

## Download

This repo uses [Git LFS](https://git-lfs.github.com/) for the large `.jsonl` files. Install Git LFS first, then clone:

```bash
git lfs install
git clone https://github.com/shady-cs15/genvf-data.git
```

If you already cloned without LFS, pull the large files with:

```bash
git lfs pull
```

## Viewing traces and prefixes

`view_traces.py` is an interactive terminal viewer for the JSONL files. It requires the `rich` library:

```bash
pip install rich
```

### Show statistics

```bash
python view_traces.py traces/qwen3-235b-thinking.jsonl --stats
python view_traces.py prefixes/gpt5-mini.jsonl --stats
```

### View a specific record by index

```bash
python view_traces.py traces/gpt5-mini.jsonl --index 42
python view_traces.py prefixes/qwen3-235b-thinking.jsonl --index 0
```

### Interactive browse mode

Navigate through records one at a time with keyboard controls (`n`/`p` to navigate, `j` to jump, `s` for stats, `q` to quit):

```bash
python view_traces.py traces/qwen3-235b-thinking.jsonl --browse
```

### Compare the same index across files

```bash
python view_traces.py traces/*.jsonl --compare --index 0
```

### View error records

```bash
python view_traces.py traces/qwen3-235b-thinking.jsonl --errors
```

### View random samples

```bash
python view_traces.py traces/gpt5-mini.jsonl --sample 5
```
