"""Extract a human-readable transcript from a Claude Code JSONL session file."""

import json
import sys


def extract(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    out_lines = []

    for line in lines:
        obj = json.loads(line)
        msg_type = obj.get("type")

        if msg_type == "user":
            message = obj.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = str(message)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            if content.strip():
                out_lines.append(f"\n{'='*80}")
                out_lines.append("USER:")
                out_lines.append(f"{'='*80}\n")
                out_lines.append(content.strip())
                out_lines.append("")

        elif msg_type == "assistant":
            message = obj.get("message", {})
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if isinstance(content, list):
                text_parts = []
                tool_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        if tool_name == "Bash":
                            cmd = tool_input.get("command", "")
                            desc = tool_input.get("description", "")
                            summary = (
                                f"[Tool: Bash] {desc}\n"
                                f"  $ {cmd[:200]}{'...' if len(cmd) > 200 else ''}"
                            )
                        elif tool_name == "Read":
                            fp = tool_input.get("file_path", "")
                            summary = f"[Tool: Read] {fp}"
                        elif tool_name == "Write":
                            fp = tool_input.get("file_path", "")
                            summary = f"[Tool: Write] {fp}"
                        elif tool_name == "Edit":
                            fp = tool_input.get("file_path", "")
                            summary = f"[Tool: Edit] {fp}"
                        elif tool_name == "Grep":
                            pat = tool_input.get("pattern", "")
                            summary = f"[Tool: Grep] pattern={pat}"
                        elif tool_name == "Glob":
                            pat = tool_input.get("pattern", "")
                            summary = f"[Tool: Glob] pattern={pat}"
                        elif tool_name == "Agent":
                            desc = tool_input.get("description", "")
                            prompt = tool_input.get("prompt", "")[:150]
                            summary = f"[Tool: Agent] {desc}\n  {prompt}..."
                        else:
                            summary = (
                                f"[Tool: {tool_name}] "
                                f"{json.dumps(tool_input)[:200]}"
                            )
                        tool_parts.append(summary)

                if text_parts or tool_parts:
                    out_lines.append(f"\n{'-'*80}")
                    out_lines.append("ASSISTANT:")
                    out_lines.append(f"{'-'*80}\n")
                    if text_parts:
                        out_lines.append("\n".join(text_parts).strip())
                    if tool_parts:
                        out_lines.append("")
                        for tp in tool_parts:
                            out_lines.append(tp)
                    out_lines.append("")

    transcript = "\n".join(out_lines)

    with open(output_path, "w") as f:
        f.write(transcript)

    print(
        f"Transcript written to {output_path} "
        f"({len(transcript)} chars, {len(out_lines)} lines)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.txt>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
