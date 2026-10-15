# ruff: noqa
# mypy: ignore-errors
# copy of explore2 spencerc/2026-09-28-CM4-like-AM4-random-CO2-evaluation-runs/plotting.py (main)
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt

FONTSIZE = 8


def configure_style(fontsize=FONTSIZE, legend_fontsize=FONTSIZE):
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.titlesize"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = legend_fontsize
    mpl.rcParams["figure.titlesize"] = fontsize


def width_constrained_plot(
    nrows,
    ncols,
    width,
    aspect,
    left_pad=0.5,
    right_pad=0.5,
    bottom_pad=0.5,
    top_pad=0.5,
    horizontal_pads=0.2,
    vertical_pads=0.2,
    cbar_pad=0.2,
    cbar_thickness=0.125,
    axes_kwargs={},
):
    fig = plt.figure()

    if isinstance(horizontal_pads, float):
        horizontal_pads = [horizontal_pads for _ in range(ncols - 1)]
    if isinstance(vertical_pads, float):
        vertical_pads = [vertical_pads for _ in range(nrows - 1)]

    panel_width = (width - (left_pad + right_pad + sum(horizontal_pads))) / ncols
    panel_height = aspect * panel_width

    height = (
        nrows * panel_height
        + bottom_pad
        + top_pad
        + sum(vertical_pads)
        + cbar_pad
        + cbar_thickness
    )

    relative_panel_width = panel_width / width
    relative_panel_height = panel_height / height

    axes = []
    for row in range(nrows):
        bottom = (
            row * relative_panel_height
            + (sum(vertical_pads[:row]) + bottom_pad + cbar_pad + cbar_thickness)
            / height
        )
        for col in range(ncols):
            left = (
                col * relative_panel_width
                + (sum(horizontal_pads[:col]) + left_pad) / width
            )
            position = (left, bottom, relative_panel_width, relative_panel_height)
            axes.append(fig.add_axes(position, **axes_kwargs))

    caxes = []
    for col in range(ncols):
        bottom = bottom_pad / height
        left = (
            col * relative_panel_width + (sum(horizontal_pads[:col]) + left_pad) / width
        )
        position = (left, bottom, relative_panel_width, cbar_thickness / height)
        caxes.append(fig.add_axes(position))

    fig.set_size_inches(width, height)
    return fig, axes, caxes


def get_aspect_ratio(projection, extent):
    """Determine the anticipated aspect ratio of a cartopy plot

    The precise aspect ratio of a cartopy plot depends on its
    geographic extent and projection. Knowing this allows
    us to precisely set the amount of space between two maps
    in a figure.
    """
    x0, x1, y0, y1 = extent

    assert x1 > x0, "x1 must be larger than x0"
    assert y1 > y0, "y1 must be larger than y0"

    if y0 >= 0:
        y_max_width = y0
    elif y1 <= 0:
        y_max_width = y1
    else:
        y_max_width = 0.0

    x0t, _ = projection.transform_point(x0, y_max_width, src_crs=ccrs.PlateCarree())
    x1t, _ = projection.transform_point(x1, y_max_width, src_crs=ccrs.PlateCarree())

    _, y0t = projection.transform_point(x0, y0, src_crs=ccrs.PlateCarree())
    _, y1t = projection.transform_point(x1, y1, src_crs=ccrs.PlateCarree())

    return (y1t - y0t) / (x1t - x0t)


def get_central_position_between_axes(axes, axis):
    """Useful for placing figure legends centered beside or under figure panels."""
    if axis == "x":
        x0_left = min(ax.get_position().x0 for ax in axes)
        x1_right = max(ax.get_position().x1 for ax in axes)
        return (x0_left + x1_right) / 2.0
    elif axis == "y":
        y0_bottom = min(ax.get_position().y0 for ax in axes)
        y1_top = max(ax.get_position().y1 for ax in axes)
        return (y0_bottom + y1_top) / 2.0
    else:
        raise ValueError(f"Unrecognized axis: {axis!r}.")
