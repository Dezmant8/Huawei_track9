"""HTML utilities for table representation and normalization.

format_html() builds HTML from SynthTabNet JSON annotations.
normalize_html_for_teds() normalizes HTML for fair TEDS comparison.

Normalization adapted from TRivia's normalize_html_omni (otsl_utils.py).
"""

import re
import html as html_module
import unicodedata
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def format_html(ann: dict) -> str:
    """Reconstruct HTML table from SynthTabNet annotation."""
    cells = ann["html"]["cells"]
    tokens = ann["html"]["structure"]["tokens"]

    table_parts = []
    cell_idx = 0

    for token in tokens:
        if token.startswith("<td"):
            if cell_idx >= len(cells):
                logger.warning(
                    f"Cell index {cell_idx} out of range (total cells: {len(cells)})"
                )
                break
            cell = cells[cell_idx]
            content = "".join(cell["tokens"])
            cell_idx += 1

        if token == "</td>":
            table_parts.append(content)

        table_parts.append(token)

    table_str = "".join(table_parts)

    for special_token in ("<start>", "<end>", "<pad>", "<unk>"):
        table_str = table_str.replace(special_token, "")

    return "<table>" + table_str + "</table>"


def _clean_html_tags(html_content: str) -> str:
    """Normalize semantic HTML tags via BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")

    for th in soup.find_all("th"):
        th.name = "td"

    for thead in soup.find_all("thead"):
        thead.unwrap()

    for math_tag in soup.find_all("math"):
        alttext = math_tag.get("alttext", "")
        if alttext:
            alttext = f"${alttext}$"
        math_tag.replace_with(alttext)

    for span in soup.find_all("span"):
        span.unwrap()

    return str(soup)


def _clean_inline_tags(html_str: str) -> str:
    """Remove inline formatting tags irrelevant to table structure."""
    for tag in ("sup", "sub", "span", "div", "p"):
        html_str = html_str.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
    html_str = html_str.replace('<spandata-span-identity="">', "")
    html_str = re.sub(r"<colgroup>.*?</colgroup>", "", html_str)
    return html_str


def normalize_html_for_teds(raw_html: str) -> str:
    """Normalize HTML table for TEDS evaluation.
    Returns <html><body><table>...</table></body></html> or empty string.
    """
    if not raw_html or not raw_html.strip():
        return ""

    try:
        if "<table" not in raw_html.replace(" ", "").replace("'", '"').lower():
            return ""

        html_str = _clean_html_tags(raw_html)
        html_str = html_module.unescape(html_str)
        html_str = html_str.replace("\n", "")
        html_str = unicodedata.normalize("NFKC", html_str).strip()

        tables = re.findall(
            r"<table\b[^>]*>(.*)</table>", html_str, re.DOTALL | re.IGNORECASE
        )
        if not tables:
            return ""
        table_content = "".join(tables)

        # Remove style attributes
        for attr in ("style", "height", "width", "align", "class"):
            table_content = re.sub(rf'( {attr}=".*?")', "", table_content)

        table_content = re.sub(r"</?tbody>", "", table_content)
        table_content = _clean_inline_tags(table_content)

        # Normalize whitespace
        table_content = re.sub(r"\s+", " ", table_content)
        table_content = table_content.replace("> ", ">").replace(" </td>", "</td>")

        return f"<html><body><table>{table_content}</table></body></html>"

    except Exception as e:
        logger.warning(f"HTML normalization failed: {e}")
        return ""
