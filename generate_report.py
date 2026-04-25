#!/usr/bin/env python3
"""
Validate results and generate a Markdown report.

Usage:
  python3 generate_report.py                    # Validate + report for all results
  python3 generate_report.py --task TASK_ID     # Report for a single task
  python3 generate_report.py --rerun-invalid    # Re-run items that failed format validation
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = "results"
REPORT_FILE = os.path.join(RESULTS_DIR, "report.md")

REQUIRED_FIELDS = {"item_id", "task_id", "item_type", "justification_verdict", "verifier_verdict", "confidence", "reasoning"}
VALID_JUST_VERDICTS = {"accurate", "has_issues"}
VALID_VER_VERDICTS = {"correct", "flagged"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_JUST_ISSUE_TYPES = {"factual_error", "internal_contradiction", "unsound_logic"}
VALID_VER_ISSUE_TYPES = {"model_did_not_fail", "environment_impossibility", "ambiguous_policy", "test_criterion_mismatch"}


def load_all_results(task_filter: str = None) -> list[dict]:
    results = []
    if not os.path.isdir(RESULTS_DIR):
        return results
    for task_dir in sorted(os.listdir(RESULTS_DIR)):
        if task_filter and task_dir != task_filter:
            continue
        task_path = os.path.join(RESULTS_DIR, task_dir)
        if not os.path.isdir(task_path):
            continue
        for fname in sorted(os.listdir(task_path)):
            if fname.endswith(".json") and not fname.endswith(".trace.json") and fname != "task_summary.json":
                try:
                    with open(os.path.join(task_path, fname)) as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    continue
    return results


def validate_result(r: dict) -> list[str]:
    """Return list of validation errors for a result dict."""
    errors = []

    if r.get("error"):
        errors.append(f"Execution error: {r['error'][:100]}")
        return errors

    missing = REQUIRED_FIELDS - set(r.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")

    jv = r.get("justification_verdict")
    if jv not in VALID_JUST_VERDICTS:
        errors.append(f"Invalid justification_verdict: {jv!r}")

    vv = r.get("verifier_verdict")
    if vv not in VALID_VER_VERDICTS:
        errors.append(f"Invalid verifier_verdict: {vv!r}")

    conf = r.get("confidence")
    if conf not in VALID_CONFIDENCE:
        errors.append(f"Invalid confidence: {conf!r}")

    if not r.get("reasoning"):
        errors.append("Missing reasoning")

    for issue in r.get("justification_issues") or []:
        if not isinstance(issue, dict):
            errors.append(f"justification_issue is not a dict: {type(issue)}")
            continue
        it = issue.get("issue_type")
        if it and it not in VALID_JUST_ISSUE_TYPES:
            errors.append(f"Invalid justification issue_type: {it!r}")

    for issue in r.get("verifier_issues") or []:
        if not isinstance(issue, dict):
            errors.append(f"verifier_issue is not a dict: {type(issue)}")
            continue
        it = issue.get("issue_type")
        if it and it not in VALID_VER_ISSUE_TYPES:
            errors.append(f"Invalid verifier issue_type: {it!r}")

    return errors


def generate_task_summaries(results: list[dict]):
    """Write a task_summary.json into each task's results directory."""
    by_task = defaultdict(list)
    for r in results:
        tid = r.get("task_id")
        if tid:
            by_task[tid].append(r)

    for tid, task_results in by_task.items():
        task_dir = os.path.join(RESULTS_DIR, tid)
        if not os.path.isdir(task_dir):
            continue

        valid = []
        invalid = []
        for r in task_results:
            errs = validate_result(r)
            if errs:
                invalid.append({"item_id": r.get("item_id"), "errors": errs})
            else:
                valid.append(r)

        summary = {
            "task_id": tid,
            "total": len(task_results),
            "valid": len(valid),
            "invalid": len(invalid),
            "justification_accurate": sum(1 for r in valid if r["justification_verdict"] == "accurate"),
            "justification_has_issues": sum(1 for r in valid if r["justification_verdict"] == "has_issues"),
            "verifier_correct": sum(1 for r in valid if r["verifier_verdict"] == "correct"),
            "verifier_flagged": sum(1 for r in valid if r["verifier_verdict"] == "flagged"),
            "items": [
                {
                    "item_id": r.get("item_id"),
                    "item_type": r.get("item_type"),
                    "justification_verdict": r.get("justification_verdict"),
                    "verifier_verdict": r.get("verifier_verdict"),
                    "confidence": r.get("confidence"),
                    "error": r.get("error"),
                }
                for r in task_results
            ],
            "invalid_items": invalid,
        }

        with open(os.path.join(task_dir, "task_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def generate_report(results: list[dict], output_path: str = REPORT_FILE):
    valid = []
    invalid = []
    for r in results:
        errs = validate_result(r)
        if errs:
            invalid.append((r, errs))
        else:
            valid.append(r)

    just_accurate = [r for r in valid if r["justification_verdict"] == "accurate"]
    just_issues = [r for r in valid if r["justification_verdict"] == "has_issues"]
    ver_correct = [r for r in valid if r["verifier_verdict"] == "correct"]
    ver_flagged = [r for r in valid if r["verifier_verdict"] == "flagged"]
    accurate_but_flagged = [r for r in valid if r["justification_verdict"] == "accurate" and r["verifier_verdict"] == "flagged"]

    just_issue_counts = {}
    ver_issue_counts = {}
    for r in valid:
        for issue in r.get("justification_issues") or []:
            it = issue.get("issue_type", "unknown")
            just_issue_counts[it] = just_issue_counts.get(it, 0) + 1
        for issue in r.get("verifier_issues") or []:
            it = issue.get("issue_type", "unknown")
            ver_issue_counts[it] = ver_issue_counts.get(it, 0) + 1

    lines = []
    w = lines.append

    w("# Justification Eval Report")
    w("")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w("## Summary")
    w("")
    w("| Metric | Count |")
    w("|--------|-------|")
    w(f"| Total results | {len(results)} |")
    w(f"| Valid results | {len(valid)} |")
    w(f"| Invalid / errors | {len(invalid)} |")
    w(f"| Justification: accurate | {len(just_accurate)} |")
    w(f"| **Justification: has_issues** | **{len(just_issues)}** |")
    w(f"| Verifier: correct | {len(ver_correct)} |")
    w(f"| **Verifier: flagged** | **{len(ver_flagged)}** |")
    w(f"| **Accurate justification + flagged verifier** | **{len(accurate_but_flagged)}** |")
    w("")

    if just_issue_counts:
        w("### Justification Issue Types")
        w("")
        w("| Issue Type | Count |")
        w("|-----------|-------|")
        for it, count in sorted(just_issue_counts.items(), key=lambda x: -x[1]):
            w(f"| {it} | {count} |")
        w("")

    if ver_issue_counts:
        w("### Verifier Issue Types")
        w("")
        w("| Issue Type | Count |")
        w("|-----------|-------|")
        for it, count in sorted(ver_issue_counts.items(), key=lambda x: -x[1]):
            w(f"| {it} | {count} |")
        w("")

    # --- Invalid / Error results ---
    if invalid:
        w(f"## Invalid / Error Results ({len(invalid)})")
        w("")
        w("These items need to be re-run.")
        w("")
        w("| Task ID | Item ID | Type | Error |")
        w("|---------|---------|------|-------|")
        for r, errs in invalid:
            tid = r.get("task_id", "?")
            iid = r.get("item_id", "?")
            itype = r.get("item_type", "?")
            err_summary = "; ".join(errs)[:150].replace("|", "/").replace("\n", " ")
            w(f"| `{tid}` | `{iid}` | {itype} | {err_summary} |")
        w("")

    # --- Justification Issues ---
    just_issue_results = [r for r in valid if r["justification_verdict"] == "has_issues"]
    if just_issue_results:
        w(f"## **Justification Issues ({len(just_issue_results)})**")
        w("")
        w("These justifications have objective errors.")
        w("")
        w("| Task ID | Item ID | Type | Issue Type | Description | Evidence |")
        w("|---------|---------|------|------------|-------------|----------|")
        for r in just_issue_results:
            tid = r["task_id"]
            iid = r["item_id"]
            itype = r["item_type"]
            for issue in r.get("justification_issues") or []:
                it = issue.get("issue_type", "?")
                desc = issue.get("description", "").replace("|", "/").replace("\n", " ")[:200]
                ev = issue.get("evidence", "").replace("|", "/").replace("\n", " ")[:200]
                w(f"| `{tid}` | `{iid}` | {itype} | {it} | {desc} | {ev} |")
        w("")

    # --- Flagged Verifiers ---
    flagged_all = [r for r in valid if r["verifier_verdict"] == "flagged"]
    if flagged_all:
        w(f"## **Flagged Verifiers ({len(flagged_all)})**")
        w("")
        w("These verifiers have issues that may make the task unusable for training on this criterion.")
        w("")
        w("| Task ID | Item ID | Type | Justification | Issue Type | Description | Evidence |")
        w("|---------|---------|------|---------------|------------|-------------|----------|")
        for r in flagged_all:
            tid = r["task_id"]
            iid = r["item_id"]
            itype = r["item_type"]
            jv = r["justification_verdict"]
            for issue in r.get("verifier_issues") or []:
                it = issue.get("issue_type", "?")
                desc = issue.get("description", "").replace("|", "/").replace("\n", " ")[:200]
                ev = issue.get("evidence", "").replace("|", "/").replace("\n", " ")[:200]
                w(f"| `{tid}` | `{iid}` | {itype} | {jv} | {it} | {desc} | {ev} |")
        w("")

    # --- All Passes ---
    passes = [r for r in valid if r["justification_verdict"] == "accurate" and r["verifier_verdict"] == "correct"]
    if passes:
        w(f"## Passes ({len(passes)})")
        w("")
        w("| Task ID | Item ID | Type | Confidence | Reasoning |")
        w("|---------|---------|------|------------|-----------|")
        for r in passes:
            tid = r["task_id"]
            iid = r["item_id"]
            itype = r["item_type"]
            conf = r["confidence"]
            reasoning = r.get("reasoning", "").replace("|", "/").replace("\n", " ")[:300]
            w(f"| `{tid}` | `{iid}` | {itype} | {conf} | {reasoning} |")
        w("")

    # Write per-task summaries
    generate_task_summaries(results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to {output_path}")
    print(f"  Valid: {len(valid)}, Invalid: {len(invalid)}")
    print(f"  Passes: {len(passes)}, Flagged verifiers: {len(flagged_all)}, Justification issues: {len(just_issue_results)}")

    return invalid


def main():
    parser = argparse.ArgumentParser(description="Validate results and generate report")
    parser.add_argument("--task", type=str, help="Report for a single task_id")
    parser.add_argument("--rerun-invalid", action="store_true", help="Delete invalid results so they get re-run")
    args = parser.parse_args()

    results = load_all_results(task_filter=args.task)
    if not results:
        print("No results found.")
        return

    invalid = generate_report(results, output_path=REPORT_FILE)

    if args.rerun_invalid and invalid:
        print(f"\nDeleting {len(invalid)} invalid results for re-run...")
        for r, errs in invalid:
            tid = r.get("task_id", "")
            iid = r.get("item_id", "")
            if tid and iid:
                path = os.path.join(RESULTS_DIR, tid, f"{iid}.json")
                if os.path.exists(path):
                    os.remove(path)
                    print(f"  Deleted {path}")
        print("Re-run with: python3 eval_justifications.py")


if __name__ == "__main__":
    main()
