r"""Extract a structured transcript from a Claude Code session file.

Produces both a JSONL file (one turn per line, for programmatic access)
and a Markdown file (readable on GitHub) from a Claude Code session log.

Each entry represents one "turn" — starting when the user submits a
message and ending when control returns to the user.

Usage:
    python logs/extract_transcript.py <session_id> <log_folder>

    Finds the session file under ~/.claude/projects/, extracts the
    start timestamp, and writes:
        logs/<log_folder>/<timestamp>-<session_id_prefix>.jsonl
        logs/<log_folder>/<timestamp>-<session_id_prefix>.md

Example:
    python logs/extract_transcript.py \\
        9c55f925-6cb6-48f1-813b-a63ec61d5191 \\
        2026-03-27-vector-filter-basis
"""

import glob
import json
import os
import re
import sys


def _extract_text(content):
    """Extract text from a message content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def _summarize_tool_use(block):
    """Create a concise summary of a tool_use block."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})
    if name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        return {"tool": name, "description": desc, "command": cmd}
    elif name == "Read":
        return {"tool": name, "file": inp.get("file_path", "")}
    elif name == "Write":
        return {"tool": name, "file": inp.get("file_path", "")}
    elif name == "Edit":
        return {
            "tool": name,
            "file": inp.get("file_path", ""),
            "replace_all": inp.get("replace_all", False),
        }
    elif name == "Grep":
        return {"tool": name, "pattern": inp.get("pattern", "")}
    elif name == "Glob":
        return {"tool": name, "pattern": inp.get("pattern", "")}
    elif name == "Agent":
        return {
            "tool": name,
            "description": inp.get("description", ""),
            "subagent_type": inp.get("subagent_type", ""),
        }
    else:
        return {"tool": name, "input_keys": list(inp.keys())}


def _extract_assistant_content(message):
    """Extract text and tool summaries from an assistant message."""
    content = message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return None

    text_parts = []
    tools = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block["text"].strip()
            if t:
                text_parts.append(t)
        elif block.get("type") == "tool_use":
            tools.append(_summarize_tool_use(block))

    if not text_parts and not tools:
        return None

    result: dict = {}
    if text_parts:
        result["text"] = "\n\n".join(text_parts)
    if tools:
        result["tools"] = tools
    return result


def _is_human_turn_start(obj):
    """True if this JSONL entry is a human-initiated message (new turn)."""
    if obj.get("type") != "user":
        return False
    if obj.get("isMeta"):
        return False
    if "toolUseResult" in obj:
        return False
    return True


def _is_meta_user_text(text):
    """True if user text is a CLI command rather than a real message."""
    return text.startswith("<command-name>") or text.startswith("<local-command")


def _clean_meta_text_oneline(text):
    """Strip XML tags and collapse to one line."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def _get_session_start_timestamp(entries):
    """Extract the start timestamp from session entries."""
    for obj in entries:
        ts = obj.get("timestamp", "")
        if ts:
            return ts
    return ""


def _format_timestamp_for_filename(ts):
    """Convert ISO timestamp to a filename-safe string like 2026-03-26T1817."""
    # 2026-03-26T18:17:47.242Z -> 2026-03-26T1817
    m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})", ts)
    if m:
        return f"{m.group(1)}T{m.group(2)}{m.group(3)}"
    return "unknown"


def find_session_file(session_id):
    """Find the JSONL file for a session ID under ~/.claude/projects/."""
    base = os.path.expanduser("~/.claude/projects")
    pattern = os.path.join(base, "**", f"{session_id}.jsonl")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        print(f"Error: no session file found for {session_id}", file=sys.stderr)
        print(f"  Searched under {base}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"Warning: multiple matches, using {matches[0]}", file=sys.stderr)
    return matches[0]


def parse_turns(entries):
    """Parse session entries into a list of turn dicts."""
    turns = []
    current_turn = None

    for obj in entries:
        if _is_human_turn_start(obj):
            if current_turn is not None:
                turns.append(current_turn)

            user_text = _extract_text(obj.get("message", {}).get("content", ""))
            current_turn = {
                "user": user_text,
                "responses": [],
            }

        elif obj.get("type") == "assistant" and current_turn is not None:
            content = _extract_assistant_content(obj.get("message", {}))
            if content:
                current_turn["responses"].append(content)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def write_jsonl(turns, path):
    """Write turns as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False))
            f.write("\n")
    n_resp = sum(len(t.get("responses", [])) for t in turns)
    print(f"Wrote {path} ({len(turns)} turns, {n_resp} responses)")


def _format_tool_md(tool):
    """Format a single tool call as a one-line markdown string."""
    name = tool["tool"]
    if name == "Bash":
        desc = tool.get("description", "")
        cmd = tool.get("command", "")
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"**Bash** {desc}  \n`{cmd}`"
    elif name in ("Read", "Write", "Edit"):
        return f"**{name}** `{tool.get('file', '')}`"
    elif name in ("Grep", "Glob"):
        return f"**{name}** `{tool.get('pattern', '')}`"
    elif name == "Agent":
        sub = tool.get("subagent_type", "")
        desc = tool.get("description", "")
        return f"**Agent** ({sub}) {desc}"
    else:
        return f"**{name}**"


def write_markdown(turns, path):
    """Write turns as a GitHub-renderable Markdown file."""
    lines = []

    for i, turn in enumerate(turns):
        user_text = turn["user"]
        is_meta = _is_meta_user_text(user_text)

        if is_meta:
            cleaned = _clean_meta_text_oneline(user_text)
            if not cleaned:
                continue
            lines.append(f"*`{cleaned}`*")
            lines.append("")
            continue

        # User message as blockquote
        lines.append("---")
        lines.append("")
        for uline in user_text.split("\n"):
            lines.append(f"> {uline}")
        lines.append("")

        # Responses
        tool_buffer: list[dict] = []

        def flush_tools():
            if not tool_buffer:
                return
            names = ", ".join(t["tool"] for t in tool_buffer)
            lines.append(
                f"<details><summary>"
                f"{len(tool_buffer)} tool call"
                f"{'s' if len(tool_buffer) > 1 else ''}"
                f": {names}</summary>"
            )
            lines.append("")
            for t in tool_buffer:
                lines.append(f"- {_format_tool_md(t)}")
            lines.append("")
            lines.append("</details>")
            lines.append("")
            tool_buffer.clear()

        for resp in turn["responses"]:
            if "text" in resp:
                flush_tools()
                lines.append(resp["text"])
                lines.append("")
            if "tools" in resp:
                tool_buffer.extend(resp["tools"])

        flush_tools()

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {path} ({len(turns)} turns)")


def extract(session_id, log_folder):
    input_path = find_session_file(session_id)

    with open(input_path) as f:
        entries = [json.loads(line) for line in f]

    ts = _get_session_start_timestamp(entries)
    ts_str = _format_timestamp_for_filename(ts)
    id_prefix = session_id.split("-")[0]

    out_dir = os.path.join("logs", log_folder)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{ts_str}-{id_prefix}")

    turns = parse_turns(entries)
    write_jsonl(turns, stem + ".jsonl")
    write_markdown(turns, stem + ".md")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(
            f"Usage: {sys.argv[0]} <session_id> <log_folder>\n"
            f"\n"
            f"Finds ~/.claude/projects/.../<session_id>.jsonl, extracts\n"
            f"the transcript, and writes to logs/<log_folder>/.\n"
            f"\n"
            f"Output filenames include the session start time and ID\n"
            f"prefix, e.g.:\n"
            f"    logs/<log_folder>/2026-03-26T1817-9c55f925.jsonl\n"
            f"    logs/<log_folder>/2026-03-26T1817-9c55f925.md"
        )
        sys.exit(0 if "--help" in sys.argv else 1)
    extract(sys.argv[1], sys.argv[2])
