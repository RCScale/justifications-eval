#!/usr/bin/env python3
"""
Justification Eval via Claude Code CLI.

Main entry point that:
  1. Extracts task environment archives
  2. Loads justified items and builds prompts
  3. Runs Claude Code CLI evaluations
  4. Generates summary statistics

Usage:
  python3 eval_justifications.py                    # Run full eval (all items, default concurrency=5)
  python3 eval_justifications.py --concurrency 10   # Run with 10 parallel workers
  python3 eval_justifications.py --task TASK_ID      # Run only items for a specific task
  python3 eval_justifications.py --item ITEM_ID      # Run only a specific item (e.g. R5)
  python3 eval_justifications.py --summary           # Just print summary from existing results
  python3 eval_justifications.py --dry-run           # Show what would be run, don't execute
"""

import argparse
import asyncio
import sys

from extract_context import extract_all
from prompt_builder import load_items
from run_eval import run_batch, summarize
from generate_report import load_all_results, generate_report


def main():
    parser = argparse.ArgumentParser(description="Justification Eval via Claude Code CLI")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel Claude Code calls")
    parser.add_argument("--task", type=str, help="Only evaluate items for this task_id")
    parser.add_argument("--item", type=str, help="Only evaluate this item_id (e.g. R5, test_churn_report_exists)")
    parser.add_argument("--summary", action="store_true", help="Just print summary from existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    args = parser.parse_args()

    if args.summary:
        summarize()
        all_results = load_all_results()
        generate_report(all_results)
        return

    # Step 1: Extract archives
    print("Step 1: Extracting task environments...")
    extract_all()
    print()

    # Step 2: Load items
    print("Step 2: Loading justified items...")
    items = load_items()
    print(f"  Loaded {len(items)} justified items across {len(set(i.task_id for i in items))} tasks")

    # Apply filters
    if args.task:
        items = [i for i in items if i.task_id == args.task]
        print(f"  Filtered to task {args.task}: {len(items)} items")
    if args.item:
        items = [i for i in items if i.item_id == args.item]
        print(f"  Filtered to item {args.item}: {len(items)} items")

    if not items:
        print("No items to evaluate.")
        return

    if args.dry_run:
        print(f"\nDry run: would evaluate {len(items)} items with concurrency={args.concurrency}")
        for item in items:
            print(f"  {item.task_id} / {item.item_id} ({item.item_type}): {item.criterion[:80]}")
        return

    # Step 3: Run evaluations
    print(f"\nStep 3: Running evaluations...")
    results = asyncio.run(run_batch(items, concurrency=args.concurrency))
    print(f"\n  Completed {len(results)} evaluations")

    # Step 4: Summarize + report
    print("\nStep 4: Generating summary and report...")
    summarize()
    all_results = load_all_results()
    generate_report(all_results)


if __name__ == "__main__":
    main()
