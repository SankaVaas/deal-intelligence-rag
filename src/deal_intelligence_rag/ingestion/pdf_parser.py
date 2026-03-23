"""
pdf_parser.py
-------------
Parses SEC EDGAR composite submission files (.txt) into clean, structured text.

Despite the module name (kept for consistency with the project layout),
this parser handles SGML composite files — the format EDGAR uses for complete
submission packages. Each .txt file contains multiple <DOCUMENT> blocks;
we extract the primary 10-K/10-Q/8-K body and convert it to clean text.

Pipeline position:
    edgar_downloader.py  →  pdf_parser.py  →  chunker.py

Input:  data/raw/filings/{TICKER}/{FORM}/*.txt  +  *.meta.json sidecars
Output: data/processed/{TICKER}/{FORM}/*.parsed.json

Output JSON schema per filing:
    {
        "ticker":           str,
        "company_name":     str,
        "form_type":        str,
        "filed_date":       str,
        "accession_number": str,
        "sections":         [{"heading": str, "text": str, "char_count": int}],
        "full_text":        str,
        "char_count":       int,
        "parse_warnings":   [str]
    }

Usage (CLI):
    python -m deal_intelligence_rag.ingestion.pdf_parser --ticker AAPL
    python -m deal_intelligence_rag.ingestion.pdf_parser --ticker AAPL --ticker MSFT

Usage (Python):
    from deal_intelligence_rag.ingestion.pdf_parser import FilingParser
    parser = FilingParser()
    result = parser.parse_file(Path("data/raw/filings/AAPL/10-K/2024-11-01_...txt"))
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_FILINGS_DIR = Path("data/raw/filings")
PROCESSED_DIR = Path("data/processed")

# SGML document block boundaries
SGML_DOC_START = re.compile(r"<DOCUMENT>", re.IGNORECASE)
SGML_DOC_END = re.compile(r"</DOCUMENT>", re.IGNORECASE)
SGML_TYPE = re.compile(r"<TYPE>([^\n\r]+)", re.IGNORECASE)
SGML_FILENAME = re.compile(r"<FILENAME>([^\n\r]+)", re.IGNORECASE)
SGML_TEXT_START = re.compile(r"<TEXT>", re.IGNORECASE)
SGML_TEXT_END = re.compile(r"</TEXT>", re.IGNORECASE)

# 10-K section headings — Item numbers are the reliable structural signal.
# We match both "Item 1." and "Item 1A." style headings.
ITEM_HEADING = re.compile(
    r"(?:^|\n)\s*"
    r"(Item\s+\d+[A-Z]?\.?\s+"
    r"(?:Business|Risk Factors|Properties|Legal Proceedings|"
    r"Mine Safety|Market|Selected Financial|"
    r"Management.s Discussion|Quantitative|Controls|"
    r"Financial Statements|Changes in|Accountant|"
    r"Directors|Executive Compensation|Security Ownership|"
    r"Certain Relationships|Principal Accountant|[^\n]{5,60}))",
    re.IGNORECASE | re.MULTILINE,
)

# Primary form types we care about extracting
PRIMARY_FORM_TYPES = {"10-K", "10-K405", "10-KSB", "10-Q", "8-K", "10-K/A", "10-Q/A"}

# Boilerplate patterns to strip — these appear in XBRL/iXBRL filings and
# add thousands of characters of noise with zero semantic value
NOISE_PATTERNS = [
    re.compile(r"<ix:[^>]+>.*?</ix:[^>]+>", re.DOTALL | re.IGNORECASE),  # iXBRL tags
    re.compile(r"<xbrl[^>]*>.*?</xbrl[^>]*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<!--.*?-->", re.DOTALL),                                  # HTML comments
    re.compile(r"\s{3,}", re.MULTILINE),                                   # excessive whitespace
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ParsedSection:
    """A single identified section from a filing (e.g. Item 1A Risk Factors)."""
    heading: str
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ParsedFiling:
    """Fully parsed filing ready for the chunker."""
    ticker: str
    company_name: str
    form_type: str
    filed_date: str
    accession_number: str
    sections: list[ParsedSection]
    full_text: str
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.full_text)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "form_type": self.form_type,
            "filed_date": self.filed_date,
            "accession_number": self.accession_number,
            "sections": [
                {
                    "heading": s.heading,
                    "text": s.text,
                    "char_count": s.char_count,
                }
                for s in self.sections
            ],
            "full_text": self.full_text,
            "char_count": self.char_count,
            "parse_warnings": self.parse_warnings,
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class FilingParser:
    """
    Parses EDGAR composite SGML submission files into structured text.

    Architecture note:
        The parsing pipeline is deliberately layered:
          1. SGML extraction   — find the right <DOCUMENT> block
          2. HTML cleaning     — BeautifulSoup strips tags, tables → text
          3. Text normalisation — unicode, whitespace, encoding artefacts
          4. Section detection — Item headings split text into named sections

        Each layer is a separate method so it can be tested and replaced
        independently. For example, if EDGAR starts serving clean JSON
        in the future, only layer 1 needs to change.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_FILINGS_DIR,
        output_dir: Path = PROCESSED_DIR,
        min_text_chars: int = 5_000,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.min_text_chars = min_text_chars  # flag suspiciously short extractions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, filing_path: Path) -> ParsedFiling | None:
        """
        Parse a single EDGAR composite .txt file.

        Returns None if the file cannot be parsed (logged as warning).
        Does not raise — caller decides how to handle None results.
        """
        meta_path = filing_path.with_suffix(".meta.json")
        if not meta_path.exists():
            log.warning("no_meta_file", path=str(filing_path))
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        warnings: list[str] = []

        log.info(
            "parsing_filing",
            ticker=meta["ticker"],
            form=meta["form_type"],
            date=meta["filed_date"],
        )

        raw_content = filing_path.read_text(encoding="utf-8", errors="replace")

        # Step 1: extract the primary document block from SGML
        doc_text, doc_type = self._extract_primary_document(raw_content, meta["form_type"])

        if doc_text is None:
            log.warning(
                "no_primary_document_found",
                ticker=meta["ticker"],
                path=str(filing_path),
            )
            warnings.append(f"Could not find primary {meta['form_type']} document block in SGML")
            # Fall back to full raw content — better than nothing
            doc_text = raw_content

        # Step 2: clean HTML → plain text
        clean_text = self._html_to_text(doc_text)

        # Step 3: normalise unicode, whitespace, encoding artefacts
        clean_text = self._normalise_text(clean_text)

        if len(clean_text) < self.min_text_chars:
            warnings.append(
                f"Extracted text is suspiciously short ({len(clean_text)} chars). "
                "Filing may be image-based or heavily redacted."
            )
            log.warning(
                "short_extraction",
                ticker=meta["ticker"],
                chars=len(clean_text),
            )

        # Step 4: split into sections by Item headings
        sections = self._extract_sections(clean_text)

        log.info(
            "parsed_filing",
            ticker=meta["ticker"],
            form=meta["form_type"],
            date=meta["filed_date"],
            chars=len(clean_text),
            sections=len(sections),
            warnings=len(warnings),
        )

        return ParsedFiling(
            ticker=meta["ticker"],
            company_name=meta["company_name"],
            form_type=meta["form_type"],
            filed_date=meta["filed_date"],
            accession_number=meta["accession_number"],
            sections=sections,
            full_text=clean_text,
            parse_warnings=warnings,
        )

    def parse_ticker(self, ticker: str, form_type: str = "10-K") -> list[ParsedFiling]:
        """Parse all downloaded filings for a ticker and save to processed/."""
        ticker_dir = self.raw_dir / ticker / form_type
        if not ticker_dir.exists():
            log.warning("no_filings_dir", ticker=ticker, path=str(ticker_dir))
            return []

        filing_files = [
            f for f in sorted(ticker_dir.glob("*.txt"))
            if not f.name.endswith(".meta.json")
        ]

        if not filing_files:
            log.warning("no_filing_files", ticker=ticker, dir=str(ticker_dir))
            return []

        results: list[ParsedFiling] = []
        for filing_path in filing_files:
            parsed = self.parse_file(filing_path)
            if parsed is None:
                continue
            self._save_parsed(parsed, filing_path)
            results.append(parsed)

        log.info("ticker_parse_complete", ticker=ticker, parsed=len(results))
        return results

    # ------------------------------------------------------------------
    # Layer 1: SGML extraction
    # ------------------------------------------------------------------

    def _extract_primary_document(
        self, raw: str, form_type: str
    ) -> tuple[str | None, str | None]:
        """
        Split the SGML composite file into <DOCUMENT> blocks and return
        the text content of the block whose <TYPE> matches the form type.

        Returns (text_content, actual_type) or (None, None) if not found.

        EDGAR SGML structure:
            <SUBMISSION>
            <DOCUMENT>
            <TYPE>10-K
            <SEQUENCE>1
            <FILENAME>aapl-20240928.htm
            <TEXT>
            ...actual filing content (HTML or plain text)...
            </TEXT>
            </DOCUMENT>
            <DOCUMENT>
            <TYPE>EX-21.1
            ...
            </DOCUMENT>
        """
        # Split on <DOCUMENT> boundaries
        segments = SGML_DOC_START.split(raw)

        # Find the segment whose TYPE matches any primary form type variant
        target_types = {form_type, form_type.replace("-", ""), f"{form_type}/A"}
        if form_type == "10-K":
            target_types |= {"10-K405", "10-KSB"}

        for segment in segments[1:]:  # skip preamble before first <DOCUMENT>
            type_match = SGML_TYPE.search(segment[:200])  # TYPE is always near the top
            if not type_match:
                continue

            doc_type = type_match.group(1).strip().upper()
            if doc_type not in {t.upper() for t in target_types}:
                continue

            # Extract content between <TEXT> and </TEXT>
            text_match = SGML_TEXT_START.search(segment)
            text_end_match = SGML_TEXT_END.search(segment)

            if text_match and text_end_match:
                text_content = segment[text_match.end(): text_end_match.start()]
            elif text_match:
                # Some filings omit </TEXT> — take everything after <TEXT>
                text_content = segment[text_match.end():]
            else:
                text_content = segment

            return text_content, doc_type

        return None, None

    # ------------------------------------------------------------------
    # Layer 2: HTML → plain text
    # ------------------------------------------------------------------

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML (or mixed HTML/text) filing content to plain text.

        Strategy:
        - Use BeautifulSoup with lxml parser (fast, tolerant of malformed HTML)
        - Convert <table> elements to a simple pipe-delimited text format
          so financial tables aren't completely lost
        - Strip all remaining tags
        - Preserve paragraph breaks as double newlines

        Design decision: we keep table content as pipe-delimited text rather
        than discarding it entirely, because financial tables (income statement,
        balance sheet) are high-value content for M&A analysis queries.
        """
        # Quick check: if it looks like plain text already, skip HTML parsing
        if not re.search(r"<[a-zA-Z][^>]{0,50}>", html):
            return html

        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements entirely
        for tag in soup.find_all(["script", "style", "head", "meta", "link"]):
            tag.decompose()

        # Convert tables to simple text before stripping all tags
        for table in soup.find_all("table"):
            table.replace_with(self._table_to_text(table))

        # Convert block elements to newlines to preserve paragraph structure
        for tag in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li"]):
            tag.insert_before("\n")
            tag.insert_after("\n")

        return soup.get_text(separator=" ")

    def _table_to_text(self, table) -> str:  # type: ignore[no-untyped-def]
        """
        Convert a BeautifulSoup table element to pipe-delimited plain text.

        Example output:
            | Revenue | 394,328 | 383,285 | 365,817 |
            | Net income | 96,995 | 99,803 | 94,680 |
        """
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):  # skip empty rows
                rows.append("| " + " | ".join(cells) + " |")
        return "\n" + "\n".join(rows) + "\n"

    # ------------------------------------------------------------------
    # Layer 3: text normalisation
    # ------------------------------------------------------------------

    def _normalise_text(self, text: str) -> str:
        """
        Clean up common artefacts in SEC filing text:
        - Unicode normalisation (NFKC handles ligatures, special chars)
        - Common HTML entities that survive BeautifulSoup
        - Non-breaking spaces, zero-width chars
        - Excessive blank lines (keep max 2 consecutive)
        - Windows line endings
        """
        # Unicode normalisation — NFKC converts ligatures (ﬁ→fi), etc.
        text = unicodedata.normalize("NFKC", text)

        # Residual HTML entities
        entity_map = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&nbsp;": " ", "&ndash;": "–", "&mdash;": "—",
            "&rsquo;": "'", "&lsquo;": "'", "&rdquo;": '"', "&ldquo;": '"',
            "&#160;": " ", "&#8211;": "–", "&#8212;": "—",
        }
        for entity, replacement in entity_map.items():
            text = text.replace(entity, replacement)

        # Non-breaking spaces and zero-width characters
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\u200c", "")
        text = text.replace("\ufeff", "")  # BOM

        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse runs of spaces (but not newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Collapse excessive blank lines — max 2 consecutive
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Layer 4: section detection
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> list[ParsedSection]:
        """
        Split filing text into named sections using Item heading patterns.

        For 10-K filings, the SEC mandates a specific Item structure
        (Item 1 Business, Item 1A Risk Factors, etc.) which gives us
        reliable anchors. We find all Item headings, then slice the text
        between consecutive headings.

        If no sections are found (e.g. for 8-K filings which have no
        Item structure), we return the full text as a single section.
        """
        matches = list(ITEM_HEADING.finditer(text))

        if not matches:
            return [ParsedSection(heading="Full document", text=text)]

        sections: list[ParsedSection] = []

        # Text before the first Item heading (table of contents, cover page)
        preamble = text[: matches[0].start()].strip()
        if len(preamble) > 200:  # only keep if substantial
            sections.append(ParsedSection(heading="Preamble", text=preamble))

        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            # Clean up multi-line heading artefacts
            heading = re.sub(r"\s+", " ", heading)

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            # Skip near-empty sections (table of contents entries, page refs)
            if len(section_text) < 100:
                continue

            sections.append(ParsedSection(heading=heading, text=section_text))

        return sections

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_parsed(self, parsed: ParsedFiling, source_path: Path) -> Path:
        """Save parsed filing as JSON to the processed directory."""
        out_dir = self.output_dir / parsed.ticker / parsed.form_type
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / source_path.with_suffix(".parsed.json").name
        out_path.write_text(
            json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("saved_parsed", path=str(out_path), chars=parsed.char_count)
        return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse downloaded SEC EDGAR composite .txt filings into clean text."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        required=True,
        metavar="TICKER",
        help="Ticker to parse (repeat for multiple). Must have been downloaded first.",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        choices=["10-K", "10-Q", "8-K"],
        help="Form type to parse (default: 10-K)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_FILINGS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    return parser.parse_args()


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()
    parser = FilingParser(raw_dir=args.raw_dir, output_dir=args.output_dir)

    all_parsed: list[ParsedFiling] = []
    for ticker in args.tickers:
        results = parser.parse_ticker(ticker, form_type=args.form)
        all_parsed.extend(results)

    print("\n--- Parse Summary ---")
    for p in all_parsed:
        section_names = ", ".join(s.heading for s in p.sections[:3])
        print(
            f"  {p.ticker}  {p.form_type}  {p.filed_date}  "
            f"{p.char_count:,} chars  {len(p.sections)} sections  "
            f"[{section_names}{'...' if len(p.sections) > 3 else ''}]"
        )
        if p.parse_warnings:
            for w in p.parse_warnings:
                print(f"    WARNING: {w}")


if __name__ == "__main__":
    main()