 The `-index.json` endpoint returns 404 for most filings (SEC only generates it for newer submissions). The fallback to the complete submission `.txt` file works reliably across all filings — it is the full SGML submission package containing every document in the filing concatenated together. The `pdf_parser.py` step will need to extract just the 10-K body from this composite file by splitting on the `<DOCUMENT>` SGML tags.

That last point is actually your next task — the `.txt` files you just downloaded are **SGML composite files**, not clean text. If you open one you'll see something like:
```
<SUBMISSION>
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>aapl-20240928.htm
...actual 10-K HTML content...
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-21.1
...subsidiary list...
</DOCUMENT>