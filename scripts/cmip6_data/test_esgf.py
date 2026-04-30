"""Tests for esgf.py — ESGF file discovery and download utilities."""

import hashlib
import tempfile
from pathlib import Path

from esgf import ESGFFile, ESGFFileSet, filter_files_by_time

# ---------------------------------------------------------------------------
# ESGFFile.from_doc
# ---------------------------------------------------------------------------


def _solr_doc(
    title: str = "ua_day_MODEL_historical_r1i1p1f1_gn_20100101-20101231.nc",
    size: int = 1000,
    url: str = "http://esgf.example.com/data.nc|HTTPServer",
    checksum: str = "abc123",
    checksum_type: str = "SHA256",
) -> dict:
    return {
        "title": title,
        "size": size,
        "url": [url],
        "checksum": [checksum],
        "checksum_type": [checksum_type],
    }


def test_from_doc_parses_time_range():
    f = ESGFFile.from_doc(_solr_doc())
    assert f is not None
    assert f.time_start == "20100101"
    assert f.time_end == "20101231"
    assert f.filename == "ua_day_MODEL_historical_r1i1p1f1_gn_20100101-20101231.nc"


def test_from_doc_parses_year_only_range():
    f = ESGFFile.from_doc(_solr_doc(title="ua_Amon_MODEL_hist_r1_gn_2010-2015.nc"))
    assert f is not None
    assert f.time_start == "2010"
    assert f.time_end == "2015"


def test_from_doc_parses_yearmonth_range():
    f = ESGFFile.from_doc(_solr_doc(title="ua_Amon_MODEL_hist_r1_gn_201001-201512.nc"))
    assert f is not None
    assert f.time_start == "201001"
    assert f.time_end == "201512"


def test_from_doc_no_time_range_for_fx():
    f = ESGFFile.from_doc(_solr_doc(title="orog_fx_MODEL_historical_r0_gn.nc"))
    assert f is not None
    assert f.time_start == ""
    assert f.time_end == ""


def test_from_doc_returns_none_without_http_url():
    doc = _solr_doc()
    doc["url"] = ["gsiftp://example.com/data|GridFTP"]
    assert ESGFFile.from_doc(doc) is None


def test_from_doc_returns_none_without_title():
    doc = _solr_doc()
    doc["title"] = ""
    assert ESGFFile.from_doc(doc) is None


def test_from_doc_prefers_httpserver():
    doc = _solr_doc()
    doc["url"] = [
        "gsiftp://grid.example.com/data|GridFTP",
        "http://esgf.example.com/data.nc|HTTPServer",
        "http://other.example.com/data.nc|OtherProto",
    ]
    f = ESGFFile.from_doc(doc)
    assert f is not None
    assert f.url == "http://esgf.example.com/data.nc"


# ---------------------------------------------------------------------------
# filter_files_by_time
# ---------------------------------------------------------------------------


def _make_fileset(time_ranges: list[tuple[str, str]]) -> ESGFFileSet:
    files = []
    for i, (t_start, t_end) in enumerate(time_ranges):
        files.append(
            ESGFFile(
                url=f"http://example.com/file{i}.nc",
                filename=f"var_day_M_hist_r1_gn_{t_start}-{t_end}.nc",
                size=100,
                time_start=t_start,
                time_end=t_end,
            )
        )
    return ESGFFileSet(
        source_id="M",
        experiment_id="historical",
        member_id="r1i1p1f1",
        table_id="day",
        variable_id="var",
        files=files,
    )


def test_filter_keeps_overlapping_daily_file():
    fs = _make_fileset([("20100101", "20101231"), ("20110101", "20111231")])
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert len(result.files) == 1
    assert result.files[0].time_start == "20100101"


def test_filter_keeps_partially_overlapping_file():
    fs = _make_fileset([("20090701", "20100630")])
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert len(result.files) == 1


def test_filter_removes_non_overlapping_file():
    fs = _make_fileset([("20050101", "20051231"), ("20100101", "20101231")])
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert len(result.files) == 1
    assert result.files[0].time_start == "20100101"


def test_filter_with_year_only_ranges():
    fs = _make_fileset([("1850", "1900"), ("1901", "1950"), ("1951", "2000")])
    result = filter_files_by_time(fs, "1920-06-15", "1960-01-01")
    assert len(result.files) == 2
    assert result.files[0].time_start == "1901"
    assert result.files[1].time_start == "1951"


def test_filter_with_yearmonth_ranges():
    fs = _make_fileset([("201001", "201412"), ("201501", "201912")])
    result = filter_files_by_time(fs, "2012-06-01", "2016-06-30")
    assert len(result.files) == 2


def test_filter_keeps_fx_files_without_time():
    fx_file = ESGFFile(
        url="http://example.com/orog.nc",
        filename="orog_fx_M_hist_r0_gn.nc",
        size=50,
        time_start="",
        time_end="",
    )
    fs = ESGFFileSet(
        source_id="M",
        experiment_id="historical",
        member_id="r1i1p1f1",
        table_id="fx",
        variable_id="orog",
        files=[fx_file],
    )
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert len(result.files) == 1


def test_filter_preserves_fileset_metadata():
    fs = _make_fileset([("20100101", "20101231")])
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert result.source_id == "M"
    assert result.experiment_id == "historical"
    assert result.variable_id == "var"


def test_filter_returns_empty_when_no_overlap():
    fs = _make_fileset([("20050101", "20051231")])
    result = filter_files_by_time(fs, "2010-01-01", "2010-12-31")
    assert len(result.files) == 0


# ---------------------------------------------------------------------------
# Checksum verification (warning, not error)
# ---------------------------------------------------------------------------


def test_verify_checksum_warns_on_mismatch(capfd):
    """Checksum mismatch should warn, not raise."""
    import logging

    from esgf import _verify_checksum

    with tempfile.NamedTemporaryFile(suffix=".nc.partial", delete=False) as f:
        f.write(b"test data")
        path = Path(f.name)

    wrong_checksum = "0" * 64
    handler = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        _verify_checksum(path, wrong_checksum, "SHA256")
        assert path.exists(), "file should NOT be deleted on checksum mismatch"
    finally:
        logger.setLevel(old_level)
        logger.removeHandler(handler)
        path.unlink(missing_ok=True)


def test_verify_checksum_passes_on_match():
    from esgf import _verify_checksum

    with tempfile.NamedTemporaryFile(suffix=".nc.partial", delete=False) as f:
        f.write(b"test data")
        path = Path(f.name)

    expected = hashlib.sha256(b"test data").hexdigest()
    try:
        _verify_checksum(path, expected, "SHA256")
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ESGFFileSet.total_size
# ---------------------------------------------------------------------------


def test_total_size():
    fs = _make_fileset([("20100101", "20101231"), ("20110101", "20111231")])
    assert fs.total_size == 200


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok {name}")
            except Exception as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print("\nall tests passed")
