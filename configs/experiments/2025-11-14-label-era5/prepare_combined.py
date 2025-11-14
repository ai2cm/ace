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
    for config_filename in [
        "train-era5-extrapolate-1step.yaml",
        "train-era5-infill-1step.yaml",
    ]:
        out_filename = config_filename.replace("-era5-", "-combined-")
        config_text = open(config_filename).read()
        if commented_text not in config_text:
            raise ValueError(f"Commented text not found in {config_filename}")
        if uncommented_text in config_text:
            raise ValueError(f"Uncommented text found in {config_filename}")
        combined_text = config_text.replace(
            commented_text,
            uncommented_text,
        )
        with open(out_filename, "w") as f:
            f.write(combined_text)
