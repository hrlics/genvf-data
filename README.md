# genvf-data

Traces and prefixes for generative verifier training data.

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
