"""Fix markdown table alignment in findings.md."""

import sys


def align_table(lines):
    """Given a list of table lines (including header and separator), align columns."""
    # Parse cells
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    # Determine max width per column
    n_cols = len(rows[0])
    col_widths = [0] * n_cols
    for i, row in enumerate(rows):
        if i == 1:
            continue  # skip separator row
        for j, cell in enumerate(row):
            if j < n_cols:
                col_widths[j] = max(col_widths[j], len(cell))

    # Rebuild rows
    result = []
    for i, row in enumerate(rows):
        if i == 1:
            # Separator row
            sep_cells = ["-" * col_widths[j] for j in range(n_cols)]
            result.append("| " + " | ".join(sep_cells) + " |")
        else:
            padded = []
            for j in range(n_cols):
                cell = row[j] if j < len(row) else ""
                padded.append(cell.ljust(col_widths[j]))
            result.append("| " + " | ".join(padded) + " |")
    return result


def fix_tables_in_markdown(text):
    lines = text.split("\n")
    output = []
    i = 0
    while i < len(lines):
        # Detect start of a table (line starting with |)
        if lines[i].strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            aligned = align_table(table_lines)
            output.extend(aligned)
        else:
            output.append(lines[i])
            i += 1
    return "\n".join(output)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "findings.md"
    with open(path) as f:
        text = f.read()
    fixed = fix_tables_in_markdown(text)
    with open(path, "w") as f:
        f.write(fixed)
    print(f"Fixed tables in {path}")
