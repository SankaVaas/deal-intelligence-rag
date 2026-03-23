"""
edgar_downloader.py
-------------------
Downloads SEC filings (10-K, 8-K, 10-Q) from the SEC EDGAR full-text search API.

No API key required — EDGAR is a free public service.
SEC fair-use policy: max 10 requests/second, identify yourself via User-Agent header.

Usage (CLI):
    python -m deal_intelligence_rag.ingestion.edgar_downloader --ticker AAPL --form 10-K
    python -m deal_intelligence_rag.ingestion.edgar_downloader --ticker MSFT --ticker GOOGL --form 10-K --limit 3

Usage (Python):
    from deal_intelligence_rag.ingestion.edgar_downloader import EdgarDownloader
    dl = EdgarDownloader()
    filings = await dl.download_filings(ticker="AAPL", form_type="10-K", limit=3)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE_URL = "https://data.sec.gov"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = f"{EDGAR_BASE_URL}/submissions"

# SEC requires a descriptive User-Agent so they can contact you if needed.
# Format: "Your Name your@email.com"
# Update this before running — using a generic string may get you rate-limited.
DEFAULT_USER_AGENT = "DealIntelligenceRAG research@example.com"

# SEC fair-use: no more than 10 requests per second
REQUEST_DELAY_SECONDS = 0.15  # ~6-7 req/s — safely under the limit

RAW_FILINGS_DIR = Path("data/raw/filings")

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""

    ticker: str
    cik: str                      # SEC Central Index Key, zero-padded to 10 digits
    company_name: str
    form_type: str                # e.g. "10-K", "8-K", "10-Q"
    filed_date: str               # ISO date string: "2024-01-15"
    accession_number: str         # e.g. "0000320193-24-000006"
    primary_document: str         # filename of the main document
    filing_url: str               # full URL to the filing index page
    local_path: Path | None = field(default=None)  # set after download


@dataclass
class DownloadResult:
    """Result of a single filing download attempt."""

    metadata: FilingMetadata
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class EdgarDownloader:
    """
    Downloads SEC filings from EDGAR for one or more tickers.

    Architecture note:
        Uses httpx.AsyncClient for all network calls — async allows us to
        later parallelise downloads across multiple tickers without threading.
        Rate limiting is enforced via a simple time.sleep on each request;
        a token-bucket approach would be cleaner at scale but is overkill here.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        output_dir: Path = RAW_FILINGS_DIR,
        request_delay: float = REQUEST_DELAY_SECONDS,
    ) -> None:
        self.user_agent = user_agent
        self.output_dir = output_dir
        self.request_delay = request_delay
        # Host header intentionally omitted — httpx derives it from the URL.
        # Hardcoding it breaks requests to www.sec.gov vs data.sec.gov.
        self._headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 5,
    ) -> list[DownloadResult]:
        """
        Download up to `limit` filings of `form_type` for `ticker`.

        Returns a list of DownloadResult — check .success on each entry.
        Failed downloads do not raise; they are logged and returned with
        success=False so the caller can decide how to handle partial results.
        """
        log.info("starting_download", ticker=ticker, form_type=form_type, limit=limit)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: resolve ticker → CIK
            cik, company_name = await self._resolve_cik(client, ticker)
            log.info("resolved_cik", ticker=ticker, cik=cik, company=company_name)

            # Step 2: fetch the list of filings for this CIK
            filings = await self._fetch_filing_list(
                client, cik, company_name, ticker, form_type, limit
            )
            log.info("found_filings", count=len(filings), ticker=ticker)

            # Step 3: download each filing document
            results: list[DownloadResult] = []
            for filing in filings:
                result = await self._download_single_filing(client, filing)
                results.append(result)

        success_count = sum(1 for r in results if r.success)
        log.info(
            "download_complete",
            ticker=ticker,
            total=len(results),
            success=success_count,
        )
        return results

    async def download_multiple_tickers(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        limit: int = 3,
    ) -> dict[str, list[DownloadResult]]:
        """Download filings for multiple tickers sequentially."""
        all_results: dict[str, list[DownloadResult]] = {}
        for ticker in tickers:
            results = await self.download_filings(ticker, form_type, limit)
            all_results[ticker] = results
        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_cik(
        self, client: httpx.AsyncClient, ticker: str
    ) -> tuple[str, str]:
        """
        Resolve a stock ticker to a zero-padded 10-digit SEC CIK.

        EDGAR's company tickers JSON maps ticker → CIK for all public companies.
        CIKs must be zero-padded to 10 digits when used in API URLs.
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        response = await self._get(client, url)
        tickers_data: dict[str, dict] = response.json()

        ticker_upper = ticker.upper()
        for entry in tickers_data.values():
            if entry["ticker"] == ticker_upper:
                cik_raw = str(entry["cik_str"])
                cik_padded = cik_raw.zfill(10)
                return cik_padded, entry["title"]

        raise ValueError(
            f"Ticker '{ticker}' not found in SEC EDGAR. "
            "Check the ticker symbol or try the company's CIK directly."
        )

    async def _fetch_filing_list(
        self,
        client: httpx.AsyncClient,
        cik: str,
        company_name: str,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> list[FilingMetadata]:
        """
        Fetch the submission history for a CIK and filter by form type.

        EDGAR's submissions endpoint returns the most recent 1000 filings.
        We store the filing INDEX url (always accessible) and resolve the
        actual document URL at download time via _resolve_primary_document.
        """
        url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
        response = await self._get(client, url)
        data = response.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        cik_int = int(cik)  # drop leading zeros for the Archives URL
        filings: list[FilingMetadata] = []

        for form, date, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
            if form != form_type:
                continue

            # accession without dashes is the folder name on EDGAR
            accession_clean = accession.replace("-", "")

            # The filing index page — always accessible for every submission
            filing_index_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_clean}/{accession}-index.htm"
            )

            filings.append(
                FilingMetadata(
                    ticker=ticker,
                    cik=cik,
                    company_name=company_name,
                    form_type=form,
                    filed_date=date,
                    accession_number=accession,
                    primary_document=primary_doc,
                    # Store the index URL — we resolve the real doc URL at download time
                    filing_url=filing_index_url,
                )
            )

            if len(filings) >= limit:
                break

        return filings

    async def _resolve_primary_document(
        self,
        client: httpx.AsyncClient,
        filing: FilingMetadata,
    ) -> str:
        """
        Fetch the filing index JSON and return the URL of the best document to download.

        EDGAR provides a machine-readable index for every filing:
            https://data.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}-index.json

        This lists every file in the submission. We prefer:
          1. The .htm file typed as the primary document
          2. Any .htm file with the form type in the name
          3. The largest .htm file (usually the full filing)
          4. Fall back to the .txt complete submission file
        """
        cik_int = int(filing.cik)
        accession_clean = filing.accession_number.replace("-", "")

        index_json_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{accession_clean}/{filing.accession_number}-index.json"
        )

        try:
            resp = await self._get(client, index_json_url)
            index_data = resp.json()
        except Exception:
            # Fall back to the complete submission text file — always exists
            return (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_clean}/{filing.accession_number}.txt"
            )

        documents = index_data.get("documents", [])
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/"

        # Priority 1: document marked as the primary document
        for doc in documents:
            if doc.get("name") == filing.primary_document and doc.get("name", "").endswith(".htm"):
                return base_url + doc["name"]

        # Priority 2: any .htm whose name contains the ticker (e.g. aapl-20240928.htm)
        ticker_lower = filing.ticker.lower()
        for doc in documents:
            name = doc.get("name", "")
            if ticker_lower in name.lower() and name.endswith(".htm"):
                return base_url + name

        # Priority 3: largest .htm file
        htm_docs = [d for d in documents if d.get("name", "").endswith(".htm")]
        if htm_docs:
            largest = max(htm_docs, key=lambda d: d.get("size", 0))
            return base_url + largest["name"]

        # Priority 4: complete submission .txt (always exists, contains everything)
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
            f"{accession_clean}/{filing.accession_number}.txt"
        )

    async def _download_single_filing(
        self,
        client: httpx.AsyncClient,
        filing: FilingMetadata,
    ) -> DownloadResult:
        """
        Download the primary document of a filing and save to disk.

        Files are saved as:
            data/raw/filings/{ticker}/{form_type}/{filed_date}_{accession}.htm

        Existing files are skipped — re-running the downloader is idempotent.
        """
        save_dir = self.output_dir / filing.ticker / filing.form_type
        save_dir.mkdir(parents=True, exist_ok=True)

        accession_safe = filing.accession_number.replace("-", "_")

        # idempotent check — look for any already-downloaded file for this accession
        existing = list(save_dir.glob(f"{filing.filed_date}_{accession_safe}.*"))
        existing = [f for f in existing if not f.name.endswith(".meta.json")]
        if existing:
            log.info("skipping_existing", path=str(existing[0]))
            filing.local_path = existing[0]
            return DownloadResult(metadata=filing, success=True)

        try:
            # Resolve the actual document URL from the filing index
            doc_url = await self._resolve_primary_document(client, filing)
            suffix = Path(doc_url).suffix or ".htm"
            local_path = save_dir / f"{filing.filed_date}_{accession_safe}{suffix}"

            log.info("resolved_document_url", ticker=filing.ticker, url=doc_url)

            response = await self._get(client, doc_url)
            local_path.write_bytes(response.content)
            filing.local_path = local_path

            # save metadata sidecar for later pipeline stages
            meta_path = local_path.with_suffix(".meta.json")
            meta_path.write_text(
                json.dumps(
                    {
                        "ticker": filing.ticker,
                        "cik": filing.cik,
                        "company_name": filing.company_name,
                        "form_type": filing.form_type,
                        "filed_date": filing.filed_date,
                        "accession_number": filing.accession_number,
                        "filing_url": doc_url,
                        "local_path": str(local_path),
                    },
                    indent=2,
                )
            )

            log.info(
                "downloaded",
                ticker=filing.ticker,
                form=filing.form_type,
                date=filing.filed_date,
                size_kb=round(len(response.content) / 1024, 1),
                path=str(local_path),
            )
            return DownloadResult(metadata=filing, success=True)

        except httpx.HTTPStatusError as e:
            log.error(
                "download_failed_http",
                ticker=filing.ticker,
                url=filing.filing_url,
                status=e.response.status_code,
            )
            return DownloadResult(metadata=filing, success=False, error=str(e))

        except Exception as e:
            log.error(
                "download_failed",
                ticker=filing.ticker,
                url=filing.filing_url,
                error=str(e),
            )
            return DownloadResult(metadata=filing, success=False, error=str(e))

    async def _get(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict | None = None,
    ) -> httpx.Response:
        """
        Thin wrapper around httpx.get that enforces rate limiting and
        raises on non-2xx responses.
        """
        time.sleep(self.request_delay)  # respect SEC rate limit
        response = await client.get(url, headers=headers or self._headers)
        response.raise_for_status()
        return response


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR filings for one or more tickers."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        required=True,
        metavar="TICKER",
        help="Stock ticker (e.g. AAPL). Repeat for multiple tickers.",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        choices=["10-K", "10-Q", "8-K"],
        help="SEC form type to download (default: 10-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max filings per ticker (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_FILINGS_DIR,
        help=f"Directory to save filings (default: {RAW_FILINGS_DIR})",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header sent to SEC (use 'Name email@example.com')",
    )
    return parser.parse_args()


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    downloader = EdgarDownloader(
        user_agent=args.user_agent,
        output_dir=args.output_dir,
    )

    all_results = await downloader.download_multiple_tickers(
        tickers=args.tickers,
        form_type=args.form,
        limit=args.limit,
    )

    # summary table
    print("\n--- Download Summary ---")
    for ticker, results in all_results.items():
        for r in results:
            status = "OK" if r.success else f"FAIL: {r.error}"
            path = r.metadata.local_path or "—"
            print(f"  [{status}]  {ticker}  {r.metadata.form_type}  {r.metadata.filed_date}  →  {path}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()