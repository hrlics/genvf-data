#!/usr/bin/env python3
"""Interactive terminal viewer for JSONL trace and prefix files.

Usage:
    python view_traces.py traces/qwen3-235b-thinking.jsonl
    python view_traces.py traces/gpt5-mini.jsonl --index 383
    python view_traces.py prefixes/qwen3-235b-thinking.jsonl --stats
    python view_traces.py traces/*.jsonl --compare --index 0
    python view_traces.py traces/qwen3-235b-thinking.jsonl --browse
"""

import json
import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.rule import Rule
from rich import box
from contextlib import contextmanager

console = Console()


@contextmanager
def _nullcontext():
    yield


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_by_index(records: list[dict]) -> dict[int, dict]:
    return {r["index"]: r for r in records}


# ── Stats mode ───────────────────────────────────────────────────────────────

def show_stats(records: list[dict], path: str):
    """Show a dashboard of statistics for a JSONL file."""
    console.print(Rule(f"[bold cyan]Stats: {Path(path).name}"))
    console.print()

    total = len(records)
    if total == 0:
        console.print("[red]Empty file!")
        return

    # Detect file type (trace vs prefix)
    is_prefix = "prefix_type" in records[0]

    if is_prefix:
        _show_prefix_stats(records, path)
    else:
        _show_trace_stats(records, path)


def _show_trace_stats(records: list[dict], path: str):
    total = len(records)

    # Finish reasons
    finish_reasons = {}
    for r in records:
        fr = r.get("finish_reason") or "null/error"
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    # Math vs proof
    math = proof = errors = 0
    for r in records:
        if r.get("response") is None:
            errors += 1
        else:
            ans = r.get("answer")
            if ans is None or str(ans).strip() in ("", "None"):
                proof += 1
            else:
                math += 1

    # Token stats
    tokens = [r["usage"]["completion_tokens"] for r in records if r.get("usage")]
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    max_tokens = max(tokens) if tokens else 0
    min_tokens = min(tokens) if tokens else 0

    # Has reasoning?
    has_reasoning = sum(1 for r in records if r.get("reasoning"))

    # Indices
    indices = sorted(r["index"] for r in records)
    missing = [i for i in range(indices[0], indices[-1] + 1) if i not in set(indices)]

    # Overview table
    t = Table(title="Overview", box=box.ROUNDED, show_header=False, title_style="bold green")
    t.add_column("Metric", style="bold")
    t.add_column("Value", style="cyan")
    t.add_row("File", str(path))
    t.add_row("Records", str(total))
    t.add_row("Index range", f"[{indices[0]}, {indices[-1]}]")
    t.add_row("Missing indices", str(len(missing)) + (f" (first 5: {missing[:5]})" if missing else ""))
    t.add_row("Model", records[0].get("model", "?"))
    console.print(t)
    console.print()

    # Breakdown tables side by side
    t1 = Table(title="Problem Type", box=box.ROUNDED)
    t1.add_column("Type", style="bold")
    t1.add_column("Count", justify="right")
    t1.add_column("%", justify="right")
    t1.add_row("Math", str(math), f"{math*100/total:.1f}")
    t1.add_row("Proof", str(proof), f"{proof*100/total:.1f}")
    t1.add_row("Errors", str(errors), f"{errors*100/total:.1f}")
    t1.add_row("[bold]Total", f"[bold]{total}", "[bold]100.0")

    t2 = Table(title="Finish Reason", box=box.ROUNDED)
    t2.add_column("Reason", style="bold")
    t2.add_column("Count", justify="right")
    t2.add_column("%", justify="right")
    for reason, count in sorted(finish_reasons.items(), key=lambda x: -x[1]):
        color = {"stop": "green", "length": "yellow"}.get(reason, "red")
        t2.add_row(f"[{color}]{reason}", str(count), f"{count*100/total:.1f}")

    console.print(Columns([t1, t2], padding=(0, 4)))
    console.print()

    # Token stats
    t3 = Table(title="Completion Tokens", box=box.ROUNDED, show_header=False)
    t3.add_column("Metric", style="bold")
    t3.add_column("Value", style="cyan", justify="right")
    t3.add_row("Min", f"{min_tokens:,}")
    t3.add_row("Avg", f"{avg_tokens:,.0f}")
    t3.add_row("Max", f"{max_tokens:,}")
    t3.add_row("Has reasoning field", f"{has_reasoning:,} / {total:,}")
    console.print(t3)
    console.print()

    # Token distribution histogram
    _print_histogram(tokens, "Token Distribution", bins=10)


def _show_prefix_stats(records: list[dict], path: str):
    total = len(records)

    # Prefix types
    types = {}
    for r in records:
        pt = r.get("prefix_type", "unknown")
        types[pt] = types.get(pt, 0) + 1

    # Thought counts
    thought_counts = [r.get("num_thoughts", 0) for r in records]
    avg_thoughts = sum(thought_counts) / len(thought_counts) if thought_counts else 0

    # Prefix lengths
    prefix_lens = [len(r["prefix"]) for r in records if r.get("prefix")]
    avg_prefix_len = sum(prefix_lens) / len(prefix_lens) if prefix_lens else 0

    # Prefix end indices
    end_indices = [r["prefix_end_index"] for r in records if r.get("prefix_end_index") is not None]

    t = Table(title="Overview", box=box.ROUNDED, show_header=False, title_style="bold green")
    t.add_column("Metric", style="bold")
    t.add_column("Value", style="cyan")
    t.add_row("File", str(path))
    t.add_row("Records", str(total))
    t.add_row("Model", records[0].get("model", "?"))
    t.add_row("Avg thoughts per record", f"{avg_thoughts:.1f}")
    t.add_row("Avg prefix length", f"{avg_prefix_len:,.0f} chars")
    console.print(t)
    console.print()

    t2 = Table(title="Prefix Types", box=box.ROUNDED)
    t2.add_column("Type", style="bold")
    t2.add_column("Count", justify="right")
    t2.add_column("%", justify="right")
    colors = {"thought_boundary": "green", "sentence_boundary": "yellow", "skipped": "red"}
    for pt, count in sorted(types.items(), key=lambda x: -x[1]):
        color = colors.get(pt, "white")
        t2.add_row(f"[{color}]{pt}", str(count), f"{count*100/total:.1f}")
    console.print(t2)
    console.print()

    if thought_counts:
        _print_histogram(thought_counts, "Thoughts per Record", bins=10)
    if end_indices:
        _print_histogram(end_indices, "Prefix End Index Distribution", bins=10)


def _print_histogram(values: list, title: str, bins: int = 10):
    """Print an ASCII histogram."""
    if not values:
        return

    mn, mx = min(values), max(values)
    if mn == mx:
        console.print(f"[dim]{title}: all values = {mn}[/dim]")
        return

    bin_width = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    bar_width = 40

    console.print(Panel(title, style="bold magenta", expand=False))
    for i, count in enumerate(counts):
        lo = mn + i * bin_width
        hi = mn + (i + 1) * bin_width
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"{lo:>8.0f} - {hi:<8.0f}"
        console.print(f"  {label} [cyan]{bar}[/cyan] {count}")
    console.print()


# ── Single record view ───────────────────────────────────────────────────────

def show_record(record: dict, path: str):
    """Show a single record in detail."""
    idx = record["index"]
    is_prefix = "prefix_type" in record

    console.print(Rule(f"[bold cyan]Record {idx} from {Path(path).name}"))
    console.print()

    # Metadata panel
    ans = record.get("answer")
    is_proof = ans is None or str(ans).strip() in ("", "None")
    problem_type = "[magenta]Proof[/magenta]" if is_proof else "[green]Math[/green]"

    meta_lines = [
        f"[bold]Index:[/bold] {idx}",
        f"[bold]Type:[/bold] {problem_type}",
        f"[bold]Source:[/bold] {record.get('source')}",
        f"[bold]Answer:[/bold] {ans}",
        f"[bold]Mean reward:[/bold] {record.get('mean_reward')}",
        f"[bold]Model:[/bold] {record.get('model')}",
    ]

    if not is_prefix:
        meta_lines.append(f"[bold]Finish reason:[/bold] {record.get('finish_reason')}")
        if record.get("usage"):
            u = record["usage"]
            meta_lines.append(f"[bold]Tokens:[/bold] {u.get('prompt_tokens', '?')} prompt / {u.get('completion_tokens', '?')} completion / {u.get('total_tokens', '?')} total")
    else:
        meta_lines.append(f"[bold]Prefix type:[/bold] {record.get('prefix_type')}")
        meta_lines.append(f"[bold]Description:[/bold] {record.get('prefix_type_description')}")
        meta_lines.append(f"[bold]Num thoughts:[/bold] {record.get('num_thoughts')}")
        meta_lines.append(f"[bold]Prefix end index:[/bold] {record.get('prefix_end_index')}")

    console.print(Panel("\n".join(meta_lines), title="Metadata", border_style="blue"))
    console.print()

    # Problem
    console.print(Panel(record.get("problem", "N/A"), title="Problem", border_style="yellow"))
    console.print()

    if is_prefix:
        # Show prefix
        prefix = record.get("prefix")
        if prefix:
            _print_truncated(prefix, "Prefix", "green", max_chars=2000)
        else:
            console.print(Panel("[red]No prefix (skipped)", title="Prefix"))
    else:
        # Show reasoning if present
        reasoning = record.get("reasoning")
        if reasoning:
            _print_truncated(reasoning, "Reasoning (CoT)", "magenta", max_chars=2000)

        # Show response
        response = record.get("response")
        if response:
            _print_truncated(response, "Response", "green", max_chars=2000)
        elif record.get("error"):
            console.print(Panel(
                f"[red]{record.get('error_type', 'Error')}: {record.get('error', 'Unknown')}",
                title="Error",
                border_style="red",
            ))
        else:
            console.print(Panel("[red]No response", title="Response"))


def _print_truncated(text: str, title: str, color: str, max_chars: int = 2000):
    """Print text in a panel, showing beginning and ending if too long."""
    if len(text) <= max_chars:
        console.print(Panel(text, title=f"{title} ({len(text):,} chars)", border_style=color))
    else:
        half = max_chars // 2
        head = text[:half]
        tail = text[-half:]
        omitted = len(text) - max_chars
        display = (
            head
            + f"\n\n[bold red]{'─' * 40}\n"
            + f"  ... {omitted:,} chars omitted ...\n"
            + f"{'─' * 40}[/bold red]\n\n"
            + tail
        )
        console.print(Panel(
            display,
            title=f"{title} ({len(text):,} chars)",
            border_style=color,
        ))
    console.print()


# ── Compare mode ─────────────────────────────────────────────────────────────

def show_compare(all_files: dict[str, list[dict]], index: int):
    """Compare the same index across multiple files."""
    console.print(Rule(f"[bold cyan]Compare index {index}"))
    console.print()

    for path, records in all_files.items():
        by_idx = records_by_index(records)
        if index in by_idx:
            record = by_idx[index]
            show_record(record, path)
        else:
            console.print(f"[red]Index {index} not found in {path}")
        console.print()


# ── Browse mode ──────────────────────────────────────────────────────────────

def browse(records: list[dict], path: str):
    """Interactive browse through records."""
    by_idx = records_by_index(records)
    indices = sorted(by_idx.keys())
    pos = 0

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        idx = indices[pos]
        show_record(by_idx[idx], path)

        console.print(Rule())
        console.print(
            f"[dim]Record {pos + 1}/{len(indices)} | "
            f"[bold]n[/bold]=next [bold]p[/bold]=prev [bold]j[/bold]=jump [bold]s[/bold]=stats [bold]q[/bold]=quit[/dim]"
        )

        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key in ("n", ""):
            pos = min(pos + 1, len(indices) - 1)
        elif key == "p":
            pos = max(pos - 1, 0)
        elif key == "j":
            try:
                target = int(input("Jump to index: "))
                if target in by_idx:
                    pos = indices.index(target)
                else:
                    console.print(f"[red]Index {target} not found")
                    input("Press Enter...")
            except (ValueError, EOFError):
                pass
        elif key == "s":
            os.system("clear" if os.name != "nt" else "cls")
            show_stats(records, path)
            input("Press Enter to continue...")
        elif key == "q":
            break


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive terminal viewer for JSONL trace and prefix files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_traces.py traces/qwen3-235b-thinking.jsonl --stats
  python view_traces.py traces/gpt5-mini.jsonl --index 42
  python view_traces.py traces/*.jsonl --compare --index 0
  python view_traces.py prefixes/qwen3-235b-thinking.jsonl --browse
  python view_traces.py traces/qwen3-235b-thinking.jsonl --errors
        """,
    )
    parser.add_argument("files", nargs="+", help="JSONL file(s) to view")
    parser.add_argument("--stats", action="store_true", help="Show statistics dashboard")
    parser.add_argument("--index", "-i", type=int, default=None, help="Show a specific record by index")
    parser.add_argument("--compare", action="store_true", help="Compare same index across multiple files")
    parser.add_argument("--browse", action="store_true", help="Interactive browse mode")
    parser.add_argument("--errors", action="store_true", help="Show only error records")
    parser.add_argument("--sample", type=int, default=None, help="Show N random records")
    parser.add_argument("--no-pager", action="store_true", help="Disable pager (default: pipes through less)")
    args = parser.parse_args()

    # Load all files
    all_files = {}
    for path in args.files:
        try:
            all_files[path] = load_jsonl(path)
        except FileNotFoundError:
            console.print(f"[red]File not found: {path}")
            sys.exit(1)

    # Browse mode is interactive, no pager
    if args.browse:
        path = args.files[0]
        records = all_files[path]
        browse(records, path)
        return

    # All other modes go through pager for scrolling
    use_pager = not args.no_pager and sys.stdout.isatty()
    if use_pager:
        os.environ.setdefault("LESS", "-R")
    ctx = console.pager(styles=True) if use_pager else _nullcontext()

    with ctx:
        if args.compare and args.index is not None:
            show_compare(all_files, args.index)
            return

        # For single-file modes, use the first file
        path = args.files[0]
        records = all_files[path]

        if args.stats:
            show_stats(records, path)
        elif args.errors:
            error_records = [r for r in records if r.get("response") is None]
            if not error_records:
                console.print("[green]No errors found!")
            else:
                console.print(f"[red]Found {len(error_records)} errors:\n")
                for r in error_records:
                    show_record(r, path)
        elif args.sample is not None:
            import random
            sampled = random.sample(records, min(args.sample, len(records)))
            for r in sampled:
                show_record(r, path)
        elif args.index is not None:
            by_idx = records_by_index(records)
            if args.index in by_idx:
                show_record(by_idx[args.index], path)
            else:
                console.print(f"[red]Index {args.index} not found in {path}")
        else:
            # Default: show stats
            show_stats(records, path)


if __name__ == "__main__":
    main()
