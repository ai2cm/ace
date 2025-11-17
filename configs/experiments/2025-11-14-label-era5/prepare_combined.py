if __name__ == "__main__":
    commented_text = """      # - data_path: /climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0001
      #   labels: ["c96-shield"]
      #   subset:
      #     start_time: '1979-01-01'
      #     stop_time: '2020-12-31'
      # - data_path: /climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002
      #   labels: ["c96-shield"]
      #   subset:
      #     start_time: '1979-01-01'
      #     stop_time: '2020-12-31'"""  # noqa: E501
    uncommented_text = """      - data_path: /climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0001
        labels: ["c96-shield"]
        subset:
          start_time: '1979-01-01'
          stop_time: '2020-12-31'
      - data_path: /climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002
        labels: ["c96-shield"]
        subset:
          start_time: '1979-01-01'
          stop_time: '2020-12-31'"""  # noqa: E501

    replace_pairs = [
        (commented_text, uncommented_text),
        ("max_epochs: 600", "max_epochs: 120"),
    ]
    for config_filename in [
        "train-era5-extrapolate-1step.yaml",
        "train-era5-infill-1step.yaml",
    ]:
        out_filename = config_filename.replace("-era5-", "-combined-")
        config_text = open(config_filename).read()
        for initial_text, replacement_text in replace_pairs:
            if initial_text not in config_text:
                raise ValueError(
                    f"Initial text not found in {config_filename}, "
                    f"expected:\n{initial_text}"
                )
            if replacement_text in config_text:
                raise ValueError(
                    f"Replacement text found in {config_filename}, "
                    "did not expect:\n{replacement_text}"
                )
            config_text = config_text.replace(
                initial_text,
                replacement_text,
            )
        with open(out_filename, "w") as f:
            f.write(config_text)
