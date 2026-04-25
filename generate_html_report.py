#!/usr/bin/env python3
"""Generate a single self-contained HTML report from eval results."""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "report.html")


def _load_justifications_from_jsonl() -> dict:
    """Load justification fields keyed by task_id:item_id."""
    justifications = {}
    for f in sorted(glob.glob("delivery_sender_preview_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                task = json.loads(line)
                tid = task["task_id"]
                for rkey, rval in task.get("rubrics", {}).items():
                    if isinstance(rval, dict) and "why_rubric_is_correct" in rval:
                        justifications[f"{tid}:{rkey}"] = {
                            "criterion": rval.get("criterion", ""),
                            "is_positive": rval.get("is_positive"),
                            "importance": rval.get("importance"),
                            "score": rval.get("score"),
                            "why_rubric_is_correct": rval.get("why_rubric_is_correct", ""),
                            "why_rubric_is_important": rval.get("why_rubric_is_important", ""),
                            "what_model_did_wrong": rval.get("what_model_did_wrong", ""),
                        }
                for ut in task.get("unit_tests", []):
                    if "why_rubric_is_correct" in ut:
                        name = ut.get("test_name", "unknown")
                        justifications[f"{tid}:{name}"] = {
                            "criterion": name,
                            "is_positive": None,
                            "importance": None,
                            "score": ut.get("weight"),
                            "why_rubric_is_correct": ut.get("why_rubric_is_correct", ""),
                            "why_rubric_is_important": ut.get("why_rubric_is_important", ""),
                            "what_model_did_wrong": ut.get("what_model_did_wrong", ""),
                        }
    return justifications


def _load_results() -> list[dict]:
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


def _load_trace(task_id: str, item_id: str) -> list[dict]:
    """Load and slim down trace events."""
    for attempt in [1, 2, 3]:
        suffix = f".attempt{attempt}" if attempt > 1 else ""
        path = os.path.join(RESULTS_DIR, task_id, f"{item_id}{suffix}.trace.json")
        if os.path.exists(path):
            last_path = path
    try:
        with open(last_path) as f:
            raw_trace = json.load(f)
    except Exception:
        return []

    slim = []
    for event in raw_trace:
        etype = event.get("type", "")
        if etype == "system":
            continue
        if etype == "assistant":
            msg = event.get("message", {})
            blocks = []
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "thinking":
                    blocks.append({"type": "thinking", "text": block.get("thinking", "")[:2000]})
                elif btype == "tool_use":
                    inp = block.get("input", {})
                    inp_str = json.dumps(inp)
                    if len(inp_str) > 500:
                        inp_str = inp_str[:500] + "..."
                    blocks.append({"type": "tool_use", "name": block.get("name", "?"), "input": inp_str})
                elif btype == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")[:2000]})
            if blocks:
                slim.append({"type": "assistant", "blocks": blocks})
        elif etype == "tool":
            content = event.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        content = c.get("text", "")
                        break
                else:
                    content = str(content)
            if isinstance(content, str) and len(content) > 1000:
                content = content[:1000] + "..."
            slim.append({"type": "tool", "name": event.get("tool", "?"), "content": content})
        elif etype == "result":
            slim.append({
                "type": "result",
                "cost": event.get("total_cost_usd", 0),
                "turns": event.get("num_turns", 0),
                "duration_ms": event.get("duration_ms", 0),
            })
    return slim


def _get_expected_items_per_task() -> dict[str, set[str]]:
    """Get the set of expected item IDs per task from the JSONL files."""
    expected = defaultdict(set)
    for f in sorted(glob.glob("delivery_sender_preview_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                task = json.loads(line)
                tid = task["task_id"]
                for rkey, rval in task.get("rubrics", {}).items():
                    if isinstance(rval, dict) and "why_rubric_is_correct" in rval:
                        expected[tid].add(rkey)
                for ut in task.get("unit_tests", []):
                    if "why_rubric_is_correct" in ut:
                        expected[tid].add(ut.get("test_name", "unknown"))
    return dict(expected)


def generate():
    print("Loading justifications from JSONL...")
    justifications = _load_justifications_from_jsonl()

    print("Loading results...")
    all_results = _load_results()

    print("Filtering to fully-evaluated tasks...")
    expected = _get_expected_items_per_task()
    result_ids_by_task = defaultdict(set)
    for r in all_results:
        result_ids_by_task[r["task_id"]].add(r["item_id"])

    complete_tasks = set()
    for tid, exp_ids in expected.items():
        if exp_ids and exp_ids.issubset(result_ids_by_task.get(tid, set())):
            complete_tasks.add(tid)

    results = [r for r in all_results if r["task_id"] in complete_tasks]
    print(f"  {len(complete_tasks)} complete tasks, {len(results)} items (of {len(all_results)} total)")

    print("Loading traces...")
    items_data = []
    task_summaries = defaultdict(lambda: {
        "task_id": "", "total": 0, "accurate": 0, "has_issues": 0,
        "correct": 0, "flagged": 0, "errors": 0,
    })

    for r in results:
        tid = r["task_id"]
        iid = r["item_id"]
        key = f"{tid}:{iid}"
        just = justifications.get(key, {})
        trace = _load_trace(tid, iid)

        item = {
            "task_id": tid,
            "item_id": iid,
            "item_type": r.get("item_type", ""),
            "criterion": just.get("criterion", r.get("item_id", "")),
            "is_positive": just.get("is_positive"),
            "importance": just.get("importance"),
            "score": just.get("score"),
            "justification_verdict": r.get("justification_verdict"),
            "verifier_verdict": r.get("verifier_verdict"),
            "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning"),
            "justification_issues": r.get("justification_issues", []),
            "verifier_issues": r.get("verifier_issues", []),
            "error": r.get("error"),
            "duration_seconds": r.get("duration_seconds"),
            "attempts": r.get("attempts", 1),
            "why_rubric_is_correct": just.get("why_rubric_is_correct", ""),
            "why_rubric_is_important": just.get("why_rubric_is_important", ""),
            "what_model_did_wrong": just.get("what_model_did_wrong", ""),
            "trace": trace,
        }
        items_data.append(item)

        ts = task_summaries[tid]
        ts["task_id"] = tid
        ts["total"] += 1
        if r.get("error"):
            ts["errors"] += 1
        else:
            if r.get("justification_verdict") == "accurate":
                ts["accurate"] += 1
            else:
                ts["has_issues"] += 1
            if r.get("verifier_verdict") == "correct":
                ts["correct"] += 1
            else:
                ts["flagged"] += 1

    tasks_list = list(task_summaries.values())
    data_json = json.dumps({"tasks": tasks_list, "items": items_data}, separators=(",", ":"))

    html = _build_html(data_json, len(results))
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Report written to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB)")


def _build_html(data_json: str, total: int) -> str:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Justification Eval Report</title>
<style>
:root {{
  --bg: #0d1117; --fg: #e6edf3; --card: #161b22; --border: #30363d;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --blue: #58a6ff; --gray: #8b949e;
  --green-bg: #12261e; --red-bg: #2d1215; --orange-bg: #2e2310; --gray-bg: #21262d;
  --hover: #1c2128; --shadow: 0 1px 3px rgba(0,0,0,0.3);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--fg); padding: 24px; max-width: 1400px; margin: 0 auto; line-height: 1.6; }}
h1 {{ margin-bottom: 4px; font-size: 1.5rem; font-weight: 700; }}
h2 {{ margin: 24px 0 12px; font-size: 1.2rem; font-weight: 600; }}
h3 {{ margin: 16px 0 8px; font-size: 1.05rem; font-weight: 600; }}
a {{ color: var(--blue); text-decoration: none; cursor: pointer; }}
a:hover {{ text-decoration: underline; }}
.subtitle {{ color: var(--gray); font-size: 0.85rem; margin-bottom: 20px; }}
.breadcrumb {{ margin-bottom: 16px; font-size: 0.85rem; color: var(--gray); }}
.breadcrumb a {{ color: var(--blue); }}

.summary-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
.stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 22px; text-align: center; min-width: 110px; box-shadow: var(--shadow); }}
.stat .num {{ font-size: 2rem; font-weight: 800; line-height: 1.1; }}
.stat .label {{ font-size: 0.7rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}
.stat.green .num {{ color: var(--green); }}
.stat.red .num {{ color: var(--red); }}
.stat.orange .num {{ color: var(--orange); }}

.chart-container {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: var(--shadow); }}
.chart-title {{ font-weight: 600; font-size: 0.85rem; margin-bottom: 12px; }}
.hbar {{ display: flex; height: 32px; border-radius: 6px; overflow: hidden; margin-bottom: 8px; }}
.hbar-seg {{ display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #fff; transition: width 0.3s; min-width: 0; }}
.hbar-seg.g {{ background: var(--green); }}
.hbar-seg.o {{ background: var(--orange); }}
.hbar-seg.r {{ background: var(--red); }}
.hbar-seg.gr {{ background: var(--gray); }}
.legend {{ display: flex; gap: 16px; font-size: 0.8rem; color: var(--gray); flex-wrap: wrap; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; margin-right: 4px; vertical-align: middle; }}

table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: var(--card); font-weight: 600; position: sticky; top: 0; z-index: 1; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.3px; color: var(--gray); }}
tr:hover td {{ background: var(--hover); }}
tr.row-green {{ border-left: 4px solid var(--green); }}
tr.row-red {{ border-left: 4px solid var(--red); }}
tr.row-orange {{ border-left: 4px solid var(--orange); }}
tr.row-gray {{ border-left: 4px solid var(--gray); }}
td.num {{ text-align: center; font-variant-numeric: tabular-nums; }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.2px; }}
.badge-green {{ background: var(--green-bg); color: var(--green); }}
.badge-red {{ background: var(--red-bg); color: var(--red); }}
.badge-orange {{ background: var(--orange-bg); color: var(--orange); }}
.badge-gray {{ background: var(--gray-bg); color: var(--gray); }}

.section {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin: 14px 0; box-shadow: var(--shadow); }}
.section-title {{ font-weight: 700; margin-bottom: 8px; font-size: 0.95rem; }}
.field {{ margin: 10px 0; }}
.field-label {{ font-weight: 600; font-size: 0.8rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.3px; }}
.field-value {{ margin-top: 4px; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; line-height: 1.6; }}

.issue {{ background: var(--red-bg); border: 1px solid var(--red); border-radius: 8px; padding: 14px; margin: 8px 0; }}
.issue .issue-header {{ font-weight: 700; color: var(--red); margin-bottom: 6px; }}
.ver-issue {{ background: var(--orange-bg); border: 1px solid var(--orange); }}
.ver-issue .issue-header {{ color: var(--orange); }}

.collapsible {{ cursor: pointer; user-select: none; padding: 10px 0; }}
.collapsible::before {{ content: "\\25B6 "; font-size: 0.75rem; }}
.collapsible.open::before {{ content: "\\25BC "; }}
.collapse-content {{ display: none; }}
.collapse-content.open {{ display: block; }}

.trace-event {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.82rem; }}
.trace-event .te-type {{ font-weight: 700; color: var(--blue); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.3px; }}
.trace-event .te-tool {{ color: var(--orange); font-weight: 600; }}
.trace-event pre {{ white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; margin-top: 6px; font-size: 0.8rem; background: var(--bg); padding: 8px; border-radius: 6px; border: 1px solid var(--border); }}
</style>
</head>
<body>
<div id="app"></div>
<script>
const DATA = {data_json};

function esc(s) {{ return s ? String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : ''; }}

function badge(verdict, type) {{
  if (!verdict) return '<span class="badge badge-gray">ERROR</span>';
  if (type === 'just') return verdict === 'accurate' ? '<span class="badge badge-green">accurate</span>' : '<span class="badge badge-orange">has_issues</span>';
  return verdict === 'correct' ? '<span class="badge badge-green">correct</span>' : '<span class="badge badge-red">flagged</span>';
}}

function rowClass(item) {{
  if (item.error) return 'row-gray';
  if (item.verifier_verdict === 'flagged') return 'row-red';
  if (item.justification_verdict === 'has_issues') return 'row-orange';
  return 'row-green';
}}

function taskScore(t) {{ return (t.flagged * 10 + t.has_issues * 5 + t.errors * 3); }}

function itemScore(i) {{
  if (i.error) return 2;
  if (i.verifier_verdict === 'flagged') return 10;
  if (i.justification_verdict === 'has_issues') return 5;
  return 0;
}}

function taskRowClass(t) {{
  if (t.flagged > 0) return 'row-red';
  if (t.has_issues > 0) return 'row-orange';
  if (t.errors > 0) return 'row-gray';
  return 'row-green';
}}

function pct(n, total) {{ return total > 0 ? ((n / total) * 100).toFixed(1) : '0'; }}

function renderOverview() {{
  const tasks = DATA.tasks.slice().sort((a, b) => taskScore(b) - taskScore(a));
  const items = DATA.items;
  const total = items.length;
  const totalErr = items.filter(i => i.error).length;
  const accurate = items.filter(i => i.justification_verdict === 'accurate').length;
  const hasIssues = items.filter(i => i.justification_verdict === 'has_issues').length;
  const correct = items.filter(i => i.verifier_verdict === 'correct').length;
  const flagged = items.filter(i => i.verifier_verdict === 'flagged').length;

  return `
    <h1>Justification Eval Report</h1>
    <div class="subtitle">Generated {ts} &middot; ${{total}} items across ${{tasks.length}} tasks</div>

    <div class="summary-bar">
      <div class="stat"><div class="num">${{total}}</div><div class="label">Total Items</div></div>
      <div class="stat green"><div class="num">${{accurate}}</div><div class="label">Just. Accurate</div></div>
      <div class="stat orange"><div class="num">${{hasIssues}}</div><div class="label">Just. Issues</div></div>
      <div class="stat green"><div class="num">${{correct}}</div><div class="label">Ver. Correct</div></div>
      <div class="stat red"><div class="num">${{flagged}}</div><div class="label">Ver. Flagged</div></div>
      ${{totalErr ? `<div class="stat"><div class="num">${{totalErr}}</div><div class="label">Errors</div></div>` : ''}}
    </div>

    <div class="chart-container">
      <div class="chart-title">Justification Verdicts</div>
      <div class="hbar">
        <div class="hbar-seg g" style="width:${{pct(accurate,total-totalErr)}}%">${{accurate}}</div>
        <div class="hbar-seg o" style="width:${{pct(hasIssues,total-totalErr)}}%">${{hasIssues}}</div>
      </div>
      <div class="chart-title">Verifier Verdicts</div>
      <div class="hbar">
        <div class="hbar-seg g" style="width:${{pct(correct,total-totalErr)}}%">${{correct}}</div>
        <div class="hbar-seg r" style="width:${{pct(flagged,total-totalErr)}}%">${{flagged}}</div>
      </div>
      <div class="legend">
        <span><span class="legend-dot" style="background:var(--green)"></span> Pass</span>
        <span><span class="legend-dot" style="background:var(--orange)"></span> Just. Issues</span>
        <span><span class="legend-dot" style="background:var(--red)"></span> Ver. Flagged</span>
        ${{totalErr ? `<span><span class="legend-dot" style="background:var(--gray)"></span> Errors</span>` : ''}}
      </div>
    </div>

    ${{(() => {{
      const withIssues = tasks.filter(t => t.flagged > 0 || t.has_issues > 0 || t.errors > 0);
      const clean = tasks.filter(t => t.flagged === 0 && t.has_issues === 0 && t.errors === 0);
      const renderTable = (rows) => `
        <table>
          <tr><th>Task ID</th><th class="num">Items</th><th class="num">Accurate</th><th class="num">Just. Issues</th><th class="num">Ver. Flagged</th><th class="num">Errors</th></tr>
          ${{rows.map(t => `
            <tr class="${{taskRowClass(t)}}">
              <td><a onclick="nav('task/${{t.task_id}}')">${{t.task_id}}</a></td>
              <td class="num">${{t.total}}</td>
              <td class="num">${{t.accurate}}</td>
              <td class="num">${{t.has_issues || '-'}}</td>
              <td class="num">${{t.flagged || '-'}}</td>
              <td class="num">${{t.errors || '-'}}</td>
            </tr>
          `).join('')}}
        </table>`;
      return `
        <h2>Tasks with Issues (${{withIssues.length}})</h2>
        ${{withIssues.length ? renderTable(withIssues) : '<p style="color:var(--gray)">None</p>'}}
        <h2>Tasks without Issues (${{clean.length}})</h2>
        ${{clean.length ? renderTable(clean) : '<p style="color:var(--gray)">None</p>'}}
      `;
    }})()}}
  `;
}}

function renderTask(taskId) {{
  const items = DATA.items.filter(i => i.task_id === taskId).sort((a, b) => itemScore(b) - itemScore(a));
  const task = DATA.tasks.find(t => t.task_id === taskId);
  const total = items.length;
  return `
    <div class="breadcrumb"><a onclick="nav('overview')">Overview</a> &rsaquo; ${{taskId}}</div>
    <h1>Task ${{taskId}}</h1>
    <div class="subtitle">${{total}} items &middot; ${{task?.accurate||0}} accurate &middot; ${{task?.has_issues||0}} just. issues &middot; ${{task?.flagged||0}} ver. flagged ${{task?.errors ? '&middot; ' + task.errors + ' errors' : ''}}</div>

    <div class="chart-container">
      <div class="hbar">
        ${{task?.accurate ? `<div class="hbar-seg g" style="width:${{pct(task.accurate,total)}}%">${{task.accurate}}</div>` : ''}}
        ${{task?.has_issues ? `<div class="hbar-seg o" style="width:${{pct(task.has_issues,total)}}%">${{task.has_issues}}</div>` : ''}}
        ${{task?.flagged ? `<div class="hbar-seg r" style="width:${{pct(task.flagged,total)}}%">${{task.flagged}}</div>` : ''}}
        ${{task?.errors ? `<div class="hbar-seg gr" style="width:${{pct(task.errors,total)}}%">${{task.errors}}</div>` : ''}}
      </div>
    </div>

    <table>
      <tr><th>Item ID</th><th>Type</th><th>Criterion</th><th>Justification</th><th>Verifier</th><th>Confidence</th></tr>
      ${{items.map(i => `
        <tr class="${{rowClass(i)}}">
          <td><a onclick="nav('verifier/${{i.task_id}}/${{i.item_id}}')">${{esc(i.item_id)}}</a></td>
          <td>${{i.item_type}}</td>
          <td style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${{esc((i.criterion||'').substring(0,120))}}</td>
          <td>${{badge(i.justification_verdict, 'just')}}</td>
          <td>${{badge(i.verifier_verdict, 'ver')}}</td>
          <td>${{i.error ? '<span class="badge badge-gray">error</span>' : (i.confidence||'')}}</td>
        </tr>
      `).join('')}}
    </table>
  `;
}}

function renderVerifier(taskId, itemId) {{
  const item = DATA.items.find(i => i.task_id === taskId && i.item_id === itemId);
  if (!item) return '<p>Item not found.</p>';

  let issuesHtml = '';
  if (item.justification_issues && item.justification_issues.length) {{
    issuesHtml += '<h3>Justification Issues</h3>';
    item.justification_issues.forEach(iss => {{
      issuesHtml += `<div class="issue"><div class="issue-header">${{esc(iss.issue_type)}} (check #${{iss.check_number}})</div><div class="field"><div class="field-label">Description</div><div class="field-value">${{esc(iss.description)}}</div></div><div class="field"><div class="field-label">Evidence</div><div class="field-value">${{esc(iss.evidence)}}</div></div></div>`;
    }});
  }}
  if (item.verifier_issues && item.verifier_issues.length) {{
    issuesHtml += '<h3>Verifier Issues</h3>';
    item.verifier_issues.forEach(iss => {{
      issuesHtml += `<div class="issue ver-issue"><div class="issue-header">${{esc(iss.issue_type)}} (check #${{iss.check_number}})</div><div class="field"><div class="field-label">Description</div><div class="field-value">${{esc(iss.description)}}</div></div><div class="field"><div class="field-label">Evidence</div><div class="field-value">${{esc(iss.evidence)}}</div></div></div>`;
    }});
  }}

  let traceHtml = '';
  if (item.trace && item.trace.length) {{
    traceHtml = item.trace.map(ev => {{
      if (ev.type === 'assistant') {{
        return ev.blocks.map(b => {{
          if (b.type === 'thinking') return `<div class="trace-event"><span class="te-type">thinking</span><pre>${{esc(b.text)}}</pre></div>`;
          if (b.type === 'tool_use') return `<div class="trace-event"><span class="te-type">tool_use</span> <span class="te-tool">${{esc(b.name)}}</span><pre>${{esc(b.input)}}</pre></div>`;
          if (b.type === 'text') return `<div class="trace-event"><span class="te-type">text</span><pre>${{esc(b.text)}}</pre></div>`;
          return '';
        }}).join('');
      }}
      if (ev.type === 'tool') return `<div class="trace-event"><span class="te-type">tool result</span> <span class="te-tool">${{esc(ev.name)}}</span><pre>${{esc(ev.content)}}</pre></div>`;
      if (ev.type === 'result') return `<div class="trace-event"><span class="te-type">result</span> turns=${{ev.turns}} &middot; cost=$${{(ev.cost||0).toFixed(2)}} &middot; duration=${{((ev.duration_ms||0)/1000).toFixed(0)}}s</div>`;
      return '';
    }}).join('');
  }}

  return `
    <div class="breadcrumb"><a onclick="nav('overview')">Overview</a> &rsaquo; <a onclick="nav('task/${{taskId}}')">${{taskId}}</a> &rsaquo; ${{esc(itemId)}}</div>
    <h1>${{esc(itemId)}}</h1>
    <div class="subtitle">${{item.item_type}} &middot; ${{item.importance || 'N/A'}} &middot; score ${{item.score || 'N/A'}} &middot; ${{item.attempts > 1 ? item.attempts + ' attempts' : '1 attempt'}} &middot; ${{(item.duration_seconds||0).toFixed(0)}}s</div>

    <div class="section">
      <div class="section-title">Criterion</div>
      <div class="field-value">${{esc(item.criterion)}}</div>
    </div>

    <div class="summary-bar">
      <div class="stat ${{item.justification_verdict === 'accurate' ? 'green' : 'orange'}}"><div class="num">${{item.justification_verdict || 'error'}}</div><div class="label">Justification</div></div>
      <div class="stat ${{item.verifier_verdict === 'correct' ? 'green' : 'red'}}"><div class="num">${{item.verifier_verdict || 'error'}}</div><div class="label">Verifier</div></div>
      <div class="stat"><div class="num">${{item.confidence || 'N/A'}}</div><div class="label">Confidence</div></div>
    </div>

    ${{item.error ? `<div class="section" style="border-color:var(--red)"><div class="section-title" style="color:var(--red)">Error</div><div class="field-value">${{esc(item.error)}}</div></div>` : ''}}

    ${{item.reasoning ? `<div class="section"><div class="section-title">Eval Reasoning</div><div class="field-value">${{esc(item.reasoning)}}</div></div>` : ''}}

    ${{issuesHtml}}

    <div class="section">
      <div class="section-title">Annotator Justifications</div>
      <div class="field"><div class="field-label">why_rubric_is_correct</div><div class="field-value">${{esc(item.why_rubric_is_correct)}}</div></div>
      <div class="field"><div class="field-label">why_rubric_is_important</div><div class="field-value">${{esc(item.why_rubric_is_important)}}</div></div>
      <div class="field"><div class="field-label">what_model_did_wrong</div><div class="field-value">${{esc(item.what_model_did_wrong)}}</div></div>
    </div>

    ${{item.trace && item.trace.length ? `
      <h3 class="collapsible" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">Evaluation Trace (${{item.trace.length}} events)</h3>
      <div class="collapse-content">${{traceHtml}}</div>
    ` : ''}}
  `;
}}

function nav(route) {{ location.hash = route; }}

function render() {{
  const hash = location.hash.slice(1) || 'overview';
  const app = document.getElementById('app');
  if (hash === 'overview') app.innerHTML = renderOverview();
  else if (hash.startsWith('task/')) app.innerHTML = renderTask(hash.split('/')[1]);
  else if (hash.startsWith('verifier/')) {{ const p = hash.split('/'); app.innerHTML = renderVerifier(p[1], p.slice(2).join('/')); }}
  else app.innerHTML = renderOverview();
  window.scrollTo(0, 0);
}}

window.addEventListener('hashchange', render);
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    generate()
