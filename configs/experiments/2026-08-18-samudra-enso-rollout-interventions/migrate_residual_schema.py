#!/usr/bin/env python
"""Rewrite the branch's old hybrid-residual keys to the convention main adopted in
ai2cm/ace#1487.

Old (this branch, on the single_module step config; keys spelled out here with
a separator so this file does not match its own pattern):
    residual_prediction: true
    residual_normalized_prediction  : true
    residual_prediction_names  :
    - thetao_0
    ...

New (main):
    residual_prediction:
      normalized: true
      names:
      - thetao_0
      ...

Text-based so comments, ordering and indentation elsewhere in the file survive.
`residual_prediction: false` becomes `residual_prediction: null`; `true` alone
(all prognostics, full-field normalized) becomes `residual_prediction: {}`.
Usage: python migrate_residual_schema.py <yaml files...>   (edits in place)
"""

import re
import sys

BLOCK = re.compile(
    r"^(?P<ind>[ \t]*)residual_prediction:[ \t]*(?P<flag>true|false)[ \t]*\n"
    r"(?:(?P=ind)residual_normalized_prediction:[ \t]*(?P<norm>true|false)[ \t]*\n)?"
    r"(?:(?P=ind)residual_prediction_names:[ \t]*\n"
    r"(?P<names>(?:(?P=ind)[ \t]*-[^\n]*\n)+))?",
    re.MULTILINE,
)


def migrate(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        ind = m.group("ind")
        if m.group("flag") == "false":
            return f"{ind}residual_prediction: null\n"
        normalized = m.group("norm") == "true"
        names = m.group("names")
        if not normalized and names is None:
            return f"{ind}residual_prediction: {{}}\n"
        out = f"{ind}residual_prediction:\n"
        if normalized:
            out += f"{ind}  normalized: true\n"
        if names is not None:
            # re-indent the list items two spaces deeper than before
            items = [ln for ln in names.splitlines()]
            out += f"{ind}  names:\n" + "".join(
                f"{ind}  {ln.strip()}\n" for ln in items
            )
        return out

    return BLOCK.sub(repl, text), count


def main() -> None:
    total = 0
    for path in sys.argv[1:]:
        text = open(path).read()
        new, n = migrate(text)
        if n:
            open(path, "w").write(new)
            total += n
            print(f"{path}: {n} block(s) migrated")
    print(f"done: {total} blocks")


if __name__ == "__main__":
    main()
