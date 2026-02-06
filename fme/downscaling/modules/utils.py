class NonDivisibleShapeError(ValueError):
    pass


def validate_shape(x_shape: tuple[int, int], levels: int):
    """
    Validates that the input shape is divisible by the number of downsampling levels.

    Note that the SongUnet does not downsample the first level, so the
    number of downsamplings considered is the len of channel_mult - 1.
    """
    next_shape = (x_shape[0] // 2, x_shape[1] // 2)
    if next_shape[0] * next_shape[1] * 4 != x_shape[0] * x_shape[1]:
        raise NonDivisibleShapeError(
            f"Shape {x_shape} is not divisible by {levels} levels"
        )
    elif levels > 2:
        try:
            validate_shape(next_shape, levels - 1)
        except NonDivisibleShapeError:
            raise NonDivisibleShapeError(
                f"Shape {x_shape} is not divisible by {levels} levels"
            )


def check_level_compatibility(
    img_resolution: int,
    channel_mult: list[int],
    attn_resolutions: list[int],
):
    matched_attn = set()
    for i in range(len(channel_mult)):
        res = img_resolution >> i
        if res == 0:
            raise ValueError(
                "Image resolution is not divisible by the number of number of"
                " levels in the U-Net architecture specified by channel_mult"
                f" {channel_mult}."
            )
        if res in attn_resolutions:
            matched_attn.add(res)

    if matched_attn != set(attn_resolutions):
        raise ValueError(
            "Requested attn_resolutions are not compatible with the input"
            f" image resolution. Matched attention resolutions {matched_attn}"
            f" but requested {attn_resolutions}."
        )
