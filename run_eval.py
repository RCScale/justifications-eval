#!/usr/bin/env python3
"""Run Claude Code CLI evaluations with async concurrency, retries, and structured output."""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass

from prompt_builder import EvalItem, build_prompt

RESULTS_DIR = "results"
MODEL = "claude-opus-4-6"
EFFORT = "max"
DEFAULT_CONCURRENCY = 5
TIMEOUT_SECONDS = 3600
MAX_RETRIES = 3
MAX_TURNS = 75
SUBPROCESS_LIMIT = 10 * 1024 * 1024

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_schema.json")
with open(SCHEMA_PATH) as _f:
    SCHEMA_JSON = json.dumps(json.load(_f), separators=(",", ":"))


@dataclass
class EvalResult:
    task_id: str
    item_id: str
    item_type: str
    justification_verdict: str | None = None
    justification_issues: list | None = None
    verifier_verdict: str | None = None
    verifier_issues: list | None = None
    confidence: str | None = None
    reasoning: str | None = None
    error: str | None = None
    duration_seconds: float | None = None
    attempts: int = 1


def _task_results_dir(task_id: str) -> str:
    return os.path.join(RESULTS_DIR, task_id)


def _result_path(task_id: str, item_id: str) -> str:
    return os.path.join(_task_results_dir(task_id), f"{item_id}.json")


def _trace_path(task_id: str, item_id: str, attempt: int = 1) -> str:
    suffix = f".attempt{attempt}" if attempt > 1 else ""
    return os.path.join(_task_results_dir(task_id), f"{item_id}{suffix}.trace.json")


def _log_path(task_id: str, item_id: str) -> str:
    return os.path.join(_task_results_dir(task_id), f"{item_id}.log")


def load_completed() -> set[str]:
    """Load already-completed item keys by scanning results directories."""
    completed = set()
    if not os.path.isdir(RESULTS_DIR):
        return completed
    for task_dir in os.listdir(RESULTS_DIR):
        task_path = os.path.join(RESULTS_DIR, task_dir)
        if not os.path.isdir(task_path):
            continue
        for fname in os.listdir(task_path):
            if fname.endswith(".json") and not fname.endswith(".trace.json") and fname != "task_summary.json":
                item_id = fname[:-5]
                result_file = os.path.join(task_path, fname)
                try:
                    with open(result_file) as f:
                        r = json.load(f)
                    if not r.get("error"):
                        completed.add(f"{task_dir}:{item_id}")
                except (json.JSONDecodeError, OSError):
                    continue
    return completed


def _extract_from_stream(raw_stdout: str) -> tuple[list[dict], dict | None]:
    """Parse stream-json output. Returns (trace_events, structured_output_or_None)."""
    trace_events = []
    structured_output = None
    result_text = None

    for line in raw_stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            trace_events.append(event)
            if event.get("type") == "result":
                if event.get("structured_output"):
                    structured_output = event["structured_output"]
                result_text = event.get("result", "")
        except json.JSONDecodeError:
            trace_events.append({"type": "raw", "content": line})

    if structured_output:
        return trace_events, structured_output

    if result_text:
        parsed = _try_parse_json(result_text)
        if parsed:
            return trace_events, parsed

    return trace_events, None


def _try_parse_json(text: str) -> dict | None:
    """Try to extract a JSON verdict from text, return None if impossible."""
    if not text:
        return None

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    if not isinstance(text, str):
        return None

    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    break

    return None


async def _read_stream_unlimited(stream) -> bytes:
    """Read an entire async stream with no size limits."""
    chunks = []
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


async def _run_once(item: EvalItem, attempt: int) -> tuple[EvalResult, list[dict]]:
    """Execute a single Claude Code CLI call. Returns (result, trace_events)."""
    key = f"{item.task_id}:{item.item_id}"
    task_dir = _task_results_dir(item.task_id)
    os.makedirs(task_dir, exist_ok=True)
    log_path = _log_path(item.task_id, item.item_id)
    trace_path = _trace_path(item.task_id, item.item_id, attempt)
    prompt = build_prompt(item)
    start = time.time()

    with open(log_path, "w") as lf:
        lf.write(f"Task: {item.task_id}\n")
        lf.write(f"Item: {item.item_id} ({item.item_type})\n")
        lf.write(f"Attempt: {attempt}/{MAX_RETRIES}\n")
        lf.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"Model: {MODEL} (effort: {EFFORT})\n")
        lf.write(f"Status: running\n")

    try:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p", prompt,
            "--model", MODEL,
            "--effort", EFFORT,
            "--output-format", "stream-json",
            "--verbose",
            "--json-schema", SCHEMA_JSON,
            "--max-turns", str(MAX_TURNS),
            "--permission-mode", "auto",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=item.extract_dir,
            limit=SUBPROCESS_LIMIT,
        )

        stdout_bytes, stderr_bytes, _ = await asyncio.wait_for(
            asyncio.gather(
                _read_stream_unlimited(proc.stdout),
                _read_stream_unlimited(proc.stderr),
                proc.wait(),
            ),
            timeout=TIMEOUT_SECONDS,
        )

        duration = time.time() - start
        raw_stdout = stdout_bytes.decode("utf-8", errors="replace")
        raw_stderr = stderr_bytes.decode("utf-8", errors="replace")

        trace_events, parsed = _extract_from_stream(raw_stdout)

        with open(trace_path, "w") as tf:
            json.dump(trace_events, tf, indent=2)

        with open(log_path, "w") as lf:
            lf.write(f"Task: {item.task_id}\n")
            lf.write(f"Item: {item.item_id} ({item.item_type})\n")
            lf.write(f"Attempt: {attempt}/{MAX_RETRIES}\n")
            lf.write(f"Duration: {duration:.1f}s\n")
            lf.write(f"Exit code: {proc.returncode}\n")
            lf.write(f"Trace events: {len(trace_events)}\n")
            lf.write(f"Trace file: {os.path.basename(trace_path)}\n")
            lf.write(f"Structured output found: {parsed is not None}\n")
            if raw_stderr.strip():
                lf.write(f"\nStderr:\n{raw_stderr}\n")

        if proc.returncode != 0:
            return EvalResult(
                task_id=item.task_id,
                item_id=item.item_id,
                item_type=item.item_type,
                error=f"exit code {proc.returncode}",
                duration_seconds=duration,
                attempts=attempt,
            ), trace_events

        if parsed is None:
            return EvalResult(
                task_id=item.task_id,
                item_id=item.item_id,
                item_type=item.item_type,
                error="no structured output or parseable JSON in response",
                duration_seconds=duration,
                attempts=attempt,
            ), trace_events

        with open(log_path, "a") as lf:
            lf.write(f"Status: {parsed.get('justification_verdict', '?')}/{parsed.get('verifier_verdict', '?')}\n")

        return EvalResult(
            task_id=item.task_id,
            item_id=item.item_id,
            item_type=item.item_type,
            justification_verdict=parsed.get("justification_verdict"),
            justification_issues=parsed.get("justification_issues", []),
            verifier_verdict=parsed.get("verifier_verdict"),
            verifier_issues=parsed.get("verifier_issues", []),
            confidence=parsed.get("confidence"),
            reasoning=parsed.get("reasoning"),
            duration_seconds=duration,
            attempts=attempt,
        ), trace_events

    except asyncio.TimeoutError:
        duration = time.time() - start
        try:
            proc.kill()
        except Exception:
            pass
        return EvalResult(
            task_id=item.task_id,
            item_id=item.item_id,
            item_type=item.item_type,
            error=f"timeout after {TIMEOUT_SECONDS}s",
            duration_seconds=duration,
            attempts=attempt,
        ), []

    except Exception as e:
        duration = time.time() - start
        return EvalResult(
            task_id=item.task_id,
            item_id=item.item_id,
            item_type=item.item_type,
            error=str(e),
            duration_seconds=duration,
            attempts=attempt,
        ), []


async def run_single(item: EvalItem, semaphore: asyncio.Semaphore) -> EvalResult:
    """Run with retries."""
    async with semaphore:
        key = f"{item.task_id}:{item.item_id}"

        for attempt in range(1, MAX_RETRIES + 1):
            result, trace_events = await _run_once(item, attempt)

            if not result.error:
                print(f"  OK   [{result.duration_seconds:.0f}s] {key} (attempt {attempt}): "
                      f"just={result.justification_verdict} "
                      f"ver={result.verifier_verdict} "
                      f"conf={result.confidence}")
                _save_result(result)
                return result

            if attempt < MAX_RETRIES:
                print(f"  RETRY [{result.duration_seconds:.0f}s] {key} attempt {attempt}/{MAX_RETRIES}: {result.error[:80]}")
            else:
                print(f"  FAIL [{result.duration_seconds:.0f}s] {key} after {MAX_RETRIES} attempts: {result.error[:80]}")
                _save_result(result)

        return result


def _save_result(result: EvalResult):
    """Save a single result as JSON to results/{task_id}/{item_id}.json."""
    task_dir = _task_results_dir(result.task_id)
    os.makedirs(task_dir, exist_ok=True)
    path = _result_path(result.task_id, result.item_id)
    with open(path, "w") as f:
        json.dump({
            "task_id": result.task_id,
            "item_id": result.item_id,
            "item_type": result.item_type,
            "justification_verdict": result.justification_verdict,
            "justification_issues": result.justification_issues,
            "verifier_verdict": result.verifier_verdict,
            "verifier_issues": result.verifier_issues,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "attempts": result.attempts,
        }, f, indent=2)


async def run_batch(items: list[EvalItem], concurrency: int = DEFAULT_CONCURRENCY) -> list[EvalResult]:
    """Run evaluations for all items with concurrency control."""
    completed = load_completed()
    pending = [item for item in items if f"{item.task_id}:{item.item_id}" not in completed]

    print(f"Total items: {len(items)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Concurrency: {concurrency}")
    print(f"Model: {MODEL} (effort: {EFFORT})")
    print(f"Max turns: {MAX_TURNS}, Max retries: {MAX_RETRIES}")
    print(f"Timeout: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS // 60}min)")
    print(f"JSON schema enforced: yes")
    print()

    if not pending:
        print("Nothing to do.")
        return []

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    done_count = 0

    async def run_and_save(item: EvalItem):
        nonlocal done_count
        result = await run_single(item, semaphore)
        results.append(result)
        done_count += 1
        if done_count % 10 == 0:
            print(f"  Progress: {done_count}/{len(pending)}")

    tasks = [run_and_save(item) for item in pending]
    await asyncio.gather(*tasks)

    return results


def _load_all_results() -> list[dict]:
    """Load all result JSON files from the nested directory structure."""
    results = []
    if not os.path.isdir(RESULTS_DIR):
        return results
    for task_dir in sorted(os.listdir(RESULTS_DIR)):
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


def summarize():
    """Generate summary statistics from results."""
    results = _load_all_results()
    if not results:
        print("No results found.")
        return

    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    successful = [r for r in results if not r.get("error")]

    just_accurate = sum(1 for r in successful if r.get("justification_verdict") == "accurate")
    just_issues = sum(1 for r in successful if r.get("justification_verdict") == "has_issues")
    ver_correct = sum(1 for r in successful if r.get("verifier_verdict") == "correct")
    ver_flagged = sum(1 for r in successful if r.get("verifier_verdict") == "flagged")

    accurate_but_flagged = sum(
        1 for r in successful
        if r.get("justification_verdict") == "accurate" and r.get("verifier_verdict") == "flagged"
    )

    just_issue_types = {}
    ver_issue_types = {}
    for r in successful:
        for issue in r.get("justification_issues") or []:
            itype = issue.get("issue_type", "unknown")
            just_issue_types[itype] = just_issue_types.get(itype, 0) + 1
        for issue in r.get("verifier_issues") or []:
            itype = issue.get("issue_type", "unknown")
            ver_issue_types[itype] = ver_issue_types.get(itype, 0) + 1

    confidence_dist = {}
    for r in successful:
        conf = r.get("confidence", "unknown")
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

    task_flags = {}
    for r in successful:
        if r.get("verifier_verdict") == "flagged" or r.get("justification_verdict") == "has_issues":
            tid = r["task_id"]
            task_flags[tid] = task_flags.get(tid, 0) + 1
    top_flagged_tasks = sorted(task_flags.items(), key=lambda x: x[1], reverse=True)[:15]

    durations = [r["duration_seconds"] for r in results if r.get("duration_seconds")]

    summary = {
        "total": total,
        "errors": errors,
        "successful": len(successful),
        "justification_verdicts": {
            "accurate": just_accurate,
            "has_issues": just_issues,
        },
        "verifier_verdicts": {
            "correct": ver_correct,
            "flagged": ver_flagged,
        },
        "accurate_justification_but_flagged_verifier": accurate_but_flagged,
        "justification_issue_types": just_issue_types,
        "verifier_issue_types": ver_issue_types,
        "confidence_distribution": confidence_dist,
        "top_flagged_tasks": top_flagged_tasks,
        "duration_stats": {
            "count": len(durations),
            "mean": sum(durations) / len(durations) if durations else 0,
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
        },
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(successful)} successful, {errors} errors)")
    print(f"{'='*60}")
    print(f"Justification: {just_accurate} accurate, {just_issues} has_issues")
    print(f"Verifier:      {ver_correct} correct, {ver_flagged} flagged")
    print(f"Accurate justification BUT flagged verifier: {accurate_but_flagged}")
    print()
    if just_issue_types:
        print("Justification issue types:")
        for itype, count in sorted(just_issue_types.items(), key=lambda x: -x[1]):
            print(f"  {itype}: {count}")
    if ver_issue_types:
        print("Verifier issue types:")
        for itype, count in sorted(ver_issue_types.items(), key=lambda x: -x[1]):
            print(f"  {itype}: {count}")
    if top_flagged_tasks:
        print("\nTop flagged tasks:")
        for tid, count in top_flagged_tasks:
            print(f"  {tid}: {count} flagged items")
    if durations:
        print(f"\nDuration: mean={summary['duration_stats']['mean']:.1f}s, "
              f"min={summary['duration_stats']['min']:.1f}s, "
              f"max={summary['duration_stats']['max']:.1f}s")

    print(f"\nFull summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        print("Use eval_justifications.py to run the eval.")
