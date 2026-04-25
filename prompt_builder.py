#!/usr/bin/env python3
"""Build evaluation prompts for each justified rubric/unit_test item."""

import json
import glob
import os
from dataclasses import dataclass


@dataclass
class EvalItem:
    task_id: str
    item_id: str
    item_type: str  # "rubric" or "unit_test"
    criterion: str
    is_positive: bool | None
    importance: str | None
    score: int | None
    why_rubric_is_correct: str
    why_rubric_is_important: str
    what_model_did_wrong: str
    prompt: str
    goal: str
    extract_dir: str


def load_items(task_envs_dir: str = "task_envs") -> list[EvalItem]:
    """Load all justified items from JSONL delivery files."""
    jsonl_files = glob.glob("delivery_sender_preview_*.jsonl")
    if not jsonl_files:
        raise FileNotFoundError("No delivery_sender_preview_*.jsonl files found")

    items = []
    for f in sorted(jsonl_files):
        with open(f) as fh:
            for line in fh:
                task = json.loads(line)
                tid = task["task_id"]
                prompt = task.get("prompt", "")
                goal = task.get("goal", "")
                extract_dir = os.path.abspath(os.path.join(task_envs_dir, tid, "extracted"))

                if not os.path.isdir(extract_dir):
                    continue

                for rkey, rval in task.get("rubrics", {}).items():
                    if isinstance(rval, dict) and "why_rubric_is_correct" in rval:
                        items.append(EvalItem(
                            task_id=tid,
                            item_id=rkey,
                            item_type="rubric",
                            criterion=rval.get("criterion", ""),
                            is_positive=rval.get("is_positive"),
                            importance=rval.get("importance"),
                            score=rval.get("score"),
                            why_rubric_is_correct=rval["why_rubric_is_correct"],
                            why_rubric_is_important=rval.get("why_rubric_is_important", ""),
                            what_model_did_wrong=rval.get("what_model_did_wrong", ""),
                            prompt=prompt,
                            goal=goal,
                            extract_dir=extract_dir,
                        ))

                for ut in task.get("unit_tests", []):
                    if "why_rubric_is_correct" in ut:
                        items.append(EvalItem(
                            task_id=tid,
                            item_id=ut.get("test_name", "unknown"),
                            item_type="unit_test",
                            criterion=ut.get("test_name", ""),
                            is_positive=None,
                            importance=None,
                            score=ut.get("weight"),
                            why_rubric_is_correct=ut["why_rubric_is_correct"],
                            why_rubric_is_important=ut.get("why_rubric_is_important", ""),
                            what_model_did_wrong=ut.get("what_model_did_wrong", ""),
                            prompt=prompt,
                            goal=goal,
                            extract_dir=extract_dir,
                        ))

    return items


def build_prompt(item: EvalItem) -> str:
    """Build the full evaluation prompt for a single item."""
    d = item.extract_dir
    task_envs_base = os.path.dirname(d)  # task_envs/{task_id}
    traj_base = os.path.join(task_envs_base, "trajectories", "anthropic_claude-opus-4_6")
    t = traj_base if os.path.isdir(traj_base) else os.path.join(task_envs_base, "trajectories")

    # Section 1
    section1 = f"""You are auditing the quality of human-written justifications for AI evaluation rubrics.

An annotator wrote a rubric criterion to evaluate an AI agent's performance on a task.
Both Claude and Gemini failed this criterion in all 8 attempts each (0% pass rate).
Because of the 0% pass rate, the annotator was required to write justifications defending
why the criterion is still correct and should be kept.

Your job is to evaluate two things independently:

AXIS 1 — JUSTIFICATION QUALITY: Are the justification fields factually accurate,
well-grounded in the environment evidence, and internally consistent?

AXIS 2 — JUSTIFICATION-VERIFIER COHERENCE: Even if the justifications are perfectly
written, do they actually support keeping this verifier? Or do they inadvertently reveal
that the verifier itself is flawed?

THE TASK ENVIRONMENT IS EXTRACTED AT:
  {d}

Key paths you should investigate (start from the top and go deeper as needed):

  1. TASK SETUP
     - {d}/instruction.md  — the user's original prompt
     - {d}/task.toml  — task metadata (goal, category, required tools/skills)

  2. INPUT DATA (what the agent was given)
     - {d}/environment/artifacts/inputs/files/  — policy docs, CSVs, uploaded files
     - {d}/environment/skills/*/SKILL.md  — tool documentation
     - {d}/environment/skills/*/references/  — policy docs and workflows

  3. DATABASE RECORDS (ground truth the agent queried)
     - {d}/environment/server/data/*/data.json  — database contents as
       readable JSON. Available services vary by task (fintrack, email, calendar,
       contacts, zendesk, shopify, etc.). Use these to verify data claims.
     - {d}/environment/server/databases/*.db  — same data as sqlite DBs.
       You can query with: python3 -c "import sqlite3; ..."

  4. VERIFIER CODE (what we're checking coherence against)
     - {d}/tests/rubric.json  — all rubric criteria with pass rates.
       Read this to check for redundancy with passing rubrics.
     - {d}/tests/test_outputs.py  — pytest assertions (the actual verifier).
       Read this to check test logic alignment, especially for unit_test items.
     - {d}/tests/test_weights.json  — test weights and pass rates.

  5. MODEL TRAJECTORY (what the model actually did)
     - {d}/conversation_history/Model_A.json  — array of user/assistant/tool
       messages from the model's run. Entries may have an `_originalIndex` field that
       differs from the array index — when looking up a referenced event, check both.
       Tool result entries contain actual DB query results, file contents, and API
       responses the model received.

  6. MODEL OUTPUT (the actual files the model produced)
     - {t}/workspace/  — the model's actual workspace after execution.
       Contains every file the model created or modified (case reports, MEMORY.md,
       CSVs, etc.). This is what the verifier tests run against.
     - {t}/workspace/snapshots.json  — state snapshots from the environment.
     - {t}/verifier/test-stdout.txt  — the actual pytest output from when the
       verifier was run. Shows which tests passed/failed and why.

You MUST read the relevant files to fact-check the justification. Do not rely solely
on what the justification claims — verify it against the actual environment data.

IMPORTANT: Do NOT attempt to open or read image files (.jpg, .jpeg, .JPEG, .png,
.heic, .HEIC, .gif). Some environments contain HEIC images mislabeled as .JPEG
that will cause API errors. If the justification references image content, check the
conversation_history/Model_A.json trajectory instead — the model's image tool
results will contain any text extracted from images."""

    # Section 2
    if item.item_type == "rubric":
        section2 = f"""THE TASK:
  Task ID: {item.task_id}
  Prompt the user gave the agent:
  {item.prompt}

  Internal goal description:
  {item.goal}

THE ITEM UNDER REVIEW (rubric):
  Item ID: {item.item_id}
  Criterion: {item.criterion}
  Is positive (model should do this): {item.is_positive}
  Importance: {item.importance}
  Score: {item.score}

ANNOTATOR'S JUSTIFICATIONS:
  why_rubric_is_correct:
    {item.why_rubric_is_correct}

  why_rubric_is_important:
    {item.why_rubric_is_important}

  what_model_did_wrong:
    {item.what_model_did_wrong}"""
    else:
        section2 = f"""THE TASK:
  Task ID: {item.task_id}
  Prompt the user gave the agent:
  {item.prompt}

  Internal goal description:
  {item.goal}

THE ITEM UNDER REVIEW (unit_test):
  Item ID: {item.item_id}
  Test name: {item.criterion}
  Weight: {item.score}

  NOTE: This is a unit_test item. The test function is in
  {d}/tests/test_outputs.py — find the function named
  "{item.criterion}" and check whether its assertions match the criterion.

ANNOTATOR'S JUSTIFICATIONS:
  why_rubric_is_correct:
    {item.why_rubric_is_correct}

  why_rubric_is_important:
    {item.why_rubric_is_important}

  what_model_did_wrong:
    {item.what_model_did_wrong}"""

    # Section 3
    section3 = """INVESTIGATION CHECKLIST:

The bar is HIGH. We are looking for objective errors that would make this task
unusable for training. Minor issues (imprecise event numbering, paraphrased quotes,
stylistic concerns) are NOT problems. Only flag something if it is objectively wrong
no matter how you look at it.

For AXIS 1 (Justification Quality) — read the environment files and check:

1. FACTUAL ACCURACY: The justification cites specific data (DB records, policy
   rules, file contents). Look up each key claim in the actual environment.
   Only flag if a cited value is materially wrong — e.g., the justification says
   record X shows value Y, but the actual record shows something contradictory.
   Approximate event references that point to the right data are fine.

2. INTERNAL COHERENCE: Do the three justification fields contradict each other?
   For example, does "why_rubric_is_correct" say the answer is X, but
   "what_model_did_wrong" describes the model failing to do Y (something
   different)? Only flag genuine contradictions, not minor wording differences.

3. LOGICAL SOUNDNESS: Does the reasoning in "why_rubric_is_correct" actually
   support the criterion? Could the cited evidence equally support the OPPOSITE
   conclusion? Flag if the logic is objectively broken — the data cited doesn't
   lead to the conclusion claimed.

For AXIS 2 (Justification-Verifier Coherence) — even if justifications are
accurate, does the verifier itself hold up?

4. MODEL DID NOT ACTUALLY FAIL: Read "what_model_did_wrong" carefully and check
   the trajectory. Does the described model behavior actually constitute a failure?
   If the model's behavior was correct or reasonable given the environment data
   and policies, then the verifier is wrong — it's penalizing correct behavior.
   This is the most important check.

5. ENVIRONMENT IMPOSSIBILITY: The criterion requires data or capabilities that
   are genuinely absent or inaccessible in the environment. If the justification
   itself admits the environment couldn't provide what's needed (broken tools,
   unreadable formats, missing data), the verifier is testing something impossible
   and the task is unusable for training on this criterion.

6. POLICY AMBIGUITY: The policy text the rubric relies on is genuinely ambiguous
   and the model's interpretation is equally defensible. Only flag if BOTH
   interpretations are clearly reasonable — not if the model's interpretation
   requires a stretch.

7. TEST-CRITERION MISMATCH (unit_tests only): The test function in test_outputs.py
   does not actually test what the criterion describes, OR could reject a correct
   answer due to brittle assertions (e.g., exact string matching that misses
   valid phrasings). Only flag if a correct agent output would objectively fail
   the test."""

    # Section 4
    section4 = f"""After your investigation, respond with ONLY a JSON object:

{{
  "item_id": "{item.item_id}",
  "task_id": "{item.task_id}",
  "item_type": "{item.item_type}",

  "justification_verdict": "accurate" | "has_issues",
  "justification_issues": [
    {{
      "check_number": <1-3>,
      "issue_type": "factual_error" | "internal_contradiction" | "unsound_logic",
      "description": "Specific description of the objective error found",
      "evidence": "The environment shows X, but the justification claims Y"
    }}
  ],

  "verifier_verdict": "correct" | "flagged",
  "verifier_issues": [
    {{
      "check_number": <4-7>,
      "issue_type": "model_did_not_fail" | "environment_impossibility" | "ambiguous_policy" | "test_criterion_mismatch",
      "description": "Specific description of why this verifier is unusable for training",
      "evidence": "The environment/justification shows X, which means the verifier is broken because Y"
    }}
  ],

  "confidence": "high" | "medium" | "low",
  "reasoning": "2-3 sentence summary of your overall assessment"
}}

IMPORTANT: The bar is high. Most items should come back clean. Only flag issues that
are objectively wrong — where no reasonable reading of the evidence could support
the justification's claims or the verifier's correctness. When in doubt, verdict
should be "accurate" / "correct".

If no issues found for an axis, return an empty array for that axis's issues field.
Do not output anything other than the JSON object — no markdown fences, no commentary."""

    return f"{section1}\n\n{section2}\n\n{section3}\n\n{section4}"


if __name__ == "__main__":
    items = load_items()
    print(f"Loaded {len(items)} justified items")
    if items:
        print(f"\nSample prompt for {items[0].task_id}/{items[0].item_id}:")
        print(f"Prompt length: {len(build_prompt(items[0]))} chars")
