"""ESGF file discovery and download utilities.

Provides helpers to:
1. Query an ESGF search node for file-level URLs for a specific
   (source_id, experiment, member, table, variable) combination.
2. Download individual NetCDF files with resume support.
3. Parse ESGF filename conventions to extract time ranges.
4. Clean up downloaded files after processing.
"""

import hashlib
import logging
import re
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_MAX_RETRIES = 3
_RETRY_DELAY = 3.0
_REQUEST_DELAY = 0.5
_DOWNLOAD_CHUNK = 8 * 1024 * 1024  # 8 MB

_TIME_RANGE_RE = re.compile(r"_(\d{4,8})-(\d{4,8})\.nc$")


@dataclass
class ESGFFile:
    """Metadata for a single file on ESGF."""

    url: str
    filename: str
    size: int
    checksum: str = ""
    checksum_type: str = ""
    time_start: str = ""
    time_end: str = ""

    @staticmethod
    def from_doc(doc: dict) -> Optional["ESGFFile"]:
        """Parse a Solr document from the ESGF file search."""
        title = doc.get("title", "")
        size = doc.get("size", 0)
        urls = doc.get("url", [])
        http_url = ""
        for u in urls:
            if "HTTPServer" in u or "http://" in u.split("|")[0]:
                http_url = u.split("|")[0]
                break
        if not http_url:
            for u in urls:
                parts = u.split("|")
                if parts[0].startswith("http"):
                    http_url = parts[0]
                    break
        if not http_url or not title:
            return None

        checksum = doc.get("checksum", [""])[0] if doc.get("checksum") else ""
        checksum_type = (
            doc.get("checksum_type", [""])[0] if doc.get("checksum_type") else ""
        )

        time_start = ""
        time_end = ""
        m = _TIME_RANGE_RE.search(title)
        if m:
            time_start = m.group(1)
            time_end = m.group(2)

        return ESGFFile(
            url=http_url,
            filename=title,
            size=size,
            checksum=checksum,
            checksum_type=checksum_type,
            time_start=time_start,
            time_end=time_end,
        )


@dataclass
class ESGFFileSet:
    """All files for one (source_id, experiment, member, table, variable)."""

    source_id: str
    experiment_id: str
    member_id: str
    table_id: str
    variable_id: str
    files: list[ESGFFile] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


def _esgf_search(node: str, params: dict, timeout: int = 60) -> dict:
    import json

    params.setdefault("format", "application/solr+json")
    url = node + "?" + urllib.parse.urlencode(params)
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"ESGF query failed after {_MAX_RETRIES} attempts: {e}"
                ) from e
            time.sleep(_RETRY_DELAY)
    raise AssertionError("unreachable")


def query_files(
    node: str,
    source_id: str,
    experiment_id: str,
    member_id: str,
    table_id: str,
    variable_id: str,
) -> ESGFFileSet:
    """Query ESGF for all files of one variable-dataset."""
    base: dict = {
        "type": "File",
        "project": "CMIP6",
        "source_id": source_id,
        "experiment_id": experiment_id,
        "member_id": member_id,
        "table_id": table_id,
        "variable_id": variable_id,
        "fields": "title,size,url,checksum,checksum_type",
        "limit": 500,
    }

    all_docs: list[dict] = []
    offset = 0
    while True:
        params = {**base, "offset": offset}
        data = _esgf_search(node, params)
        docs = data.get("response", {}).get("docs", [])
        num_found = data.get("response", {}).get("numFound", 0)
        all_docs.extend(docs)
        offset += len(docs)
        if offset >= num_found or not docs:
            break
        time.sleep(_REQUEST_DELAY)

    seen_filenames: set[str] = set()
    files: list[ESGFFile] = []
    for doc in all_docs:
        f = ESGFFile.from_doc(doc)
        if f is not None and f.filename not in seen_filenames:
            seen_filenames.add(f.filename)
            files.append(f)

    files.sort(key=lambda f: f.filename)

    return ESGFFileSet(
        source_id=source_id,
        experiment_id=experiment_id,
        member_id=member_id,
        table_id=table_id,
        variable_id=variable_id,
        files=files,
    )


def download_file(
    esgf_file: ESGFFile,
    dest_dir: str | Path,
    verify_checksum: bool = True,
) -> Path:
    """Download a single ESGF file to ``dest_dir``, returning the local path.

    Skips download if a file of the correct size already exists.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / esgf_file.filename

    if dest.exists() and dest.stat().st_size == esgf_file.size:
        logging.debug("  already downloaded: %s", esgf_file.filename)
        return dest

    partial = dest.with_suffix(".nc.partial")

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(esgf_file.url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(partial, "wb") as out:
                    while True:
                        chunk = resp.read(_DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
            break
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download failed for {esgf_file.filename}: {e}"
                ) from e
            logging.warning(
                "  download attempt %d failed for %s: %s",
                attempt + 1,
                esgf_file.filename,
                e,
            )
            time.sleep(_RETRY_DELAY * (attempt + 1))

    if verify_checksum and esgf_file.checksum and esgf_file.checksum_type:
        _verify_checksum(partial, esgf_file.checksum, esgf_file.checksum_type)

    partial.rename(dest)
    return dest


def _verify_checksum(path: Path, expected: str, checksum_type: str) -> None:
    algo = checksum_type.lower().replace("-", "")
    if algo not in hashlib.algorithms_available:
        logging.warning(
            "  checksum type %s not available, skipping verification", checksum_type
        )
        return
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected.lower():
        logging.warning(
            "  checksum mismatch for %s (expected %s, got %s, %s) — "
            "ESGF replica metadata may be stale; keeping file",
            path.name,
            expected[:16] + "…",
            actual[:16] + "…",
            checksum_type,
        )


def cleanup_variable_files(dest_dir: str | Path, variable_id: str) -> None:
    """Remove all NetCDF files for a specific variable from ``dest_dir``."""
    dest_dir = Path(dest_dir)
    for f in dest_dir.glob(f"{variable_id}_*.nc"):
        f.unlink()
        logging.debug("  removed %s", f.name)


def cleanup_scratch_dir(scratch_dir: str | Path) -> None:
    """Remove the entire scratch directory tree."""
    scratch_dir = Path(scratch_dir)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
        logging.debug("  removed scratch dir %s", scratch_dir)


def scratch_dir_for_dataset(
    base_scratch: str | Path,
    source_id: str,
    experiment_id: str,
    member_id: str,
) -> Path:
    """Return the scratch directory for one dataset's downloads."""
    return Path(base_scratch) / source_id / experiment_id / member_id


def filter_files_by_time(
    fileset: ESGFFileSet,
    start: str,
    end: str,
) -> ESGFFileSet:
    """Return a new ESGFFileSet with only files overlapping [start, end].

    ``start`` and ``end`` are ISO-like date strings (e.g. "2010-01-01").
    File time ranges use ESGF filename conventions: 4-8 digit strings
    representing year, year-month, or year-month-day.  Files without
    time metadata (e.g. fx) are always kept.
    """
    start_s = start.replace("-", "")
    end_s = end.replace("-", "")
    kept: list[ESGFFile] = []
    for f in fileset.files:
        if not f.time_start or not f.time_end:
            kept.append(f)
            continue
        f_start = f.time_start.ljust(8, "0")
        f_end = f.time_end.ljust(8, "9")
        q_start = start_s.ljust(8, "0")
        q_end = end_s.ljust(8, "9")
        if f_start <= q_end and f_end >= q_start:
            kept.append(f)
    return ESGFFileSet(
        source_id=fileset.source_id,
        experiment_id=fileset.experiment_id,
        member_id=fileset.member_id,
        table_id=fileset.table_id,
        variable_id=fileset.variable_id,
        files=kept,
    )


__all__ = [
    "ESGFFile",
    "ESGFFileSet",
    "query_files",
    "download_file",
    "filter_files_by_time",
    "cleanup_variable_files",
    "cleanup_scratch_dir",
    "scratch_dir_for_dataset",
]
