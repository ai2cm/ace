from .dataset_metadata import DatasetMetadata


def test_dataset_metadata_as_flat_str_dict():
    dataset_metadata = DatasetMetadata(
        source={
            "inference_git_sha": "abc123",
            "inference_fme_version": "1.0",
            "additional_info": {"some_other_key": "some_other_value"},
        },
        history={"created": "2023-10-01T00:00:00"},
        title="Test Dataset",
    )
    flat_dict = dataset_metadata.as_flat_str_dict()
    assert flat_dict == {
        "source.inference_git_sha": "abc123",
        "source.inference_fme_version": "1.0",
        "source.additional_info.some_other_key": "some_other_value",
        "history.created": "2023-10-01T00:00:00",
        "title": "Test Dataset",
    }
