import logging

from fme.fcn_training.utils import gcs_utils


def test_authenticate_no_keyfile(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    gcs_utils.authenticate()
    assert (
        "No keyfile found in environment, skipping gcloud authentication."
        in caplog.text
    )


def test_authenticate_with_keyfile(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "keyfile.json")
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda x: b"Activated service account credentials for: [default]",
    )
    gcs_utils.authenticate()
    assert "Activated service account credentials for: [default]" in caplog.text
