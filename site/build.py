#!/usr/bin/env python3
"""Build site/index.html from site/template.html and the README.

The website's technical sections (top performers, features, models, build, run, recipes, API, known gaps, lineage) are rendered from
README.md so the numbers and commands never drift from the repository's own documentation. Standard library only.
Usage: python3 site/build.py
"""
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO = "https://github.com/kingjones30/superroc"


def inline(text):
    """Inline markdown subset: code, bold, italics, links, footnote marks."""
    out = []
    # Bold may wrap a code span (**`name`**): mark it before splitting on code.
    text = re.sub(r"\*\*(.+?)\*\*", "\x01\\1\x02", text)
    for i, part in enumerate(re.split(r"(`[^`]*`)", text)):
        if i % 2:
            out.append("<code>" + html.escape(part[1:-1]) + "</code>")
            continue
        part = html.escape(part, quote=False)
        part = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?!\*)", r"<em>\1</em>", part)

        def link(m):
            label, href = m.group(1), m.group(2)
            if not re.match(r"^(https?:|#|mailto:)", href):
                href = REPO + "/blob/main/" + href
            ext = ' target="_blank" rel="noopener"' if href.startswith("http") else ""
            return f'<a href="{html.escape(href)}"{ext}>{label}</a>'

        part = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, part)
        # Bare URLs (GitHub autolinks these); skip ones already inside an href.
        part = re.sub(r'(?<![="\w/])(https?://[^\s)<]+[^\s)<.,;:])', lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{m.group(1)}</a>', part)
        out.append(part)
    return "".join(out).replace("\x01", "<strong>").replace("\x02", "</strong>")


def render(md):
    """Block markdown subset: headings (h3/h4), paragraphs, bullet lists, tables, fenced code."""
    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            lang = line[3:].strip() or "text"
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            code = "\n".join(lines[i + 1:j])
            out.append(f'<div class="code"><button class="copy" type="button" aria-label="Copy">Copy</button>'
                       f'<pre data-lang="{lang}"><code>{html.escape(code)}</code></pre></div>')
            i = j + 1
            continue
        m = re.match(r"^(#{3,4}) (.+)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, aligns, body = rows[0], rows[1], rows[2:]
            right = [a.endswith(":") for a in aligns]
            th = "".join(f'<th class="{"num" if r else ""}">{inline(h)}</th>' for h, r in zip(head, right))
            trs = "".join("<tr>" + "".join(f'<td class="{"num" if r else ""}">{inline(c)}</td>'
                                            for c, r in zip(row, right)) + "</tr>" for row in body)
            out.append(f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            def item(t):
                m = re.match(r"^\*\*(.+?)\*\*[\s,:]*(.*)$", t)
                if not m:
                    return f"<li>{inline(t)}</li>"
                rest = m.group(2)
                rest = rest[:1].upper() + rest[1:]
                return f'<li><span class="card-title">{inline(m.group(1))}</span>{inline(rest)}</li>'
            out.append("<ul>" + "".join(item(t) for t in items) + "</ul>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(```|#{1,6} |\||- )", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def sections(readme):
    """README '## ' sections by title (the text up to the next '## ')."""
    parts = re.split(r"^## (.+)$", readme, flags=re.M)
    return {parts[k].strip(): parts[k + 1] for k in range(1, len(parts), 2)}


def main():
    readme = (ROOT / "README.md").read_text()
    sec = sections(readme)
    page = (SITE / "template.html").read_text()
    slots = {
        "TOP": sec["Top performers"],
        "FEATURES": sec["What Super Roc adds"],
        "MODELS": sec["Models"],
        "BUILD": sec["Build from source"],
        "RUN": sec["Run"],
        "GAPS": sec["Known gaps"],
        "LINEAGE": sec["Lineage and credits"],
        "FEEDBACK": sec["Feedback"].split("\n---", 1)[0],
    }
    for key, md in slots.items():
        page = page.replace("{{" + key + "}}", render(md))
    left = re.findall(r"\{\{[A-Z]+\}\}", page)
    if left:
        raise SystemExit(f"unfilled slots: {left}")
    (SITE / "index.html").write_text(page)
    print(f"wrote {SITE / 'index.html'} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
