import os
import json
import re
from datasets import load_dataset
from google import genai


# Configure Gemini API client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Read the prompt template
with open("/prompts/gemini-summary-prompt/v2.md", "r") as f:
    SUMMARY_PROMPT_TEMPLATE = f.read()


def get_summary_generator_prompt(problem, prefix, branch_rollouts):
    suffixes = "\n\n".join(
        [f"#### Suffix {i+1}\n{branch_rollouts[i]}" for i in range(len(branch_rollouts))]
    )
    prompt = SUMMARY_PROMPT_TEMPLATE.replace("{problem}", problem).replace("{prefix}", prefix).replace("{suffixes}", suffixes)
    return prompt


def generate_summary(problem, prefix, branch_rollouts, model_name="gemini-3-flash-preview"):
    prompt = get_summary_generator_prompt(problem, prefix, branch_rollouts)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return response.text


def parse_summaries(response_text):
    """Extract prefix_steps, suffix_variants, and dedup_note from Gemini JSON response."""
    # With response_mime_type="application/json", Gemini returns valid JSON directly
    cleaned = response_text.strip()
    # Strip markdown code fences if present (fallback safety)
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Gemini may still output unescaped LaTeX backslashes.
        # Fix: within JSON string values, escape backslashes that aren't valid JSON escapes.
        # Process character by character, only fixing backslashes inside quoted strings.
        fixed = []
        in_string = False
        i = 0
        while i < len(cleaned):
            ch = cleaned[i]
            if not in_string:
                fixed.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
            else:
                if ch == '\\':
                    next_ch = cleaned[i + 1] if i + 1 < len(cleaned) else ''
                    if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                        # Valid JSON escape — keep both chars and skip 2
                        fixed.append(ch)
                        fixed.append(next_ch)
                        i += 2
                    else:
                        # Invalid escape (e.g. \text, \frac) — double the backslash
                        fixed.append('\\\\')
                        i += 1
                elif ch == '"':
                    fixed.append(ch)
                    in_string = False
                    i += 1
                else:
                    fixed.append(ch)
                    i += 1
        parsed = json.loads(''.join(fixed))

    prefix_steps = parsed.get("prefix_steps", [])
    suffix_variants = parsed.get("suffix_variants", [])
    dedup_note = parsed.get("dedup_note", "")
    return prefix_steps, suffix_variants, dedup_note


if __name__ == "__main__":

    ds = load_dataset("haoranli-ml/sanity_check_subset", split="train")
    ds = ds.select(range(3)) # only for testing

    all_prefix_steps = []
    all_suffix_variants = []
    all_dedup_notes = []

    for i, row in enumerate(ds):
        problem = row["problem"]
        prefix = row["prefix"]
        branch_rollouts = [s for s in row["suffix_response"] if s is not None]

        if not branch_rollouts:
            print(f"[{i}] Skipping: no suffix_response")
            all_prefix_steps.append([])
            all_suffix_variants.append([])
            all_dedup_notes.append("")
            continue

        print(f"[{i}] Generating summary for row_id={row['row_id']}...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                summary = generate_summary(problem, prefix, branch_rollouts)
                prefix_steps, suffix_variants, dedup_note = parse_summaries(summary)
                break
            except Exception as e:
                print(f"[{i}] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        all_prefix_steps.append(prefix_steps)
        all_suffix_variants.append(suffix_variants)
        all_dedup_notes.append(dedup_note)
        print(f"[{i}] Done. Got {len(prefix_steps)} prefix steps, {len(suffix_variants)} suffix variants.")

    # Drop columns if they already exist (e.g. from a previous run)
    existing = set(ds.column_names)
    cols_to_drop = [c for c in ["prefix_steps", "suffix_variants", "dedup_note"] if c in existing]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)
    ds = ds.add_column("prefix_steps", all_prefix_steps)
    ds = ds.add_column("suffix_variants", all_suffix_variants)
    ds = ds.add_column("dedup_note", all_dedup_notes)

    # ds.push_to_hub("haoranli-ml/sanity_check_subset")
    ds.push_to_hub("haoranli-ml/sanity_check_subset_bullet_list")
    # ds.push_to_hub("haoranli-ml/sanity_check_subset_with_step_summaries")
    print("Pushed updated dataset to hub.")
