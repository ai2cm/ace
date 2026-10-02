import contextlib
import os
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

from obstore.store import LocalStore

from fme.core import cloud


@contextlib.contextmanager
def mock_object_store(root: str | Path):
    """Back all non-local URLs (e.g. ``memory://bucket/key``) with obstore
    LocalStores under ``root``, so remote code paths can be tested without
    network access.
    """

    def store_from_url(url: str) -> LocalStore:
        parsed = urlparse(url)
        prefix = os.path.join(
            str(root), parsed.scheme, parsed.netloc, parsed.path.strip("/")
        )
        return LocalStore(prefix, mkdir=True)

    cloud._get_store_cached.cache_clear()
    try:
        with mock.patch.object(cloud, "_store_from_url", store_from_url):
            yield
    finally:
        cloud._get_store_cached.cache_clear()
