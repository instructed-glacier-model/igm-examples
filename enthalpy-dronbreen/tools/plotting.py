"""Plotting vocabulary for the Drønbreen section figures.

Kept separate from `plot_cts_sections.py` so that the colours meaning "temperate" and
"cold", and the journal-figure styling, are defined once and can be reused by any other
figure built on this example's output.
"""

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Ice type. Warm hue for temperate, cool for cold — the one place these are defined.
TEMPERATE_FILL = "#f2c4b6"
COLD_FILL = "#c2d8ee"


COLD, TEMPERATE = 0, 1

# Journal-figure styling: small type, no decorative frame, hairline axes.
STYLE = {
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def draw_section(ax, profile, model, legend=True, lw=1.4):
    """Vertical section along one radar line: modelled ice type, with the observed CTS.

    Deliberately spare. The modelled CTS is not drawn as its own line because it *is*
    the boundary between the two fills — drawing it again would add a third mark that
    says nothing new. Everything the reader needs is the two fills and the radar's
    answer laid over them: where the dashed line sits inside the blue, the model made
    too little temperate ice; inside the red, too much.

    `model` carries the sampled fields along the profile (`E`, `E_pmp`, `z`, `thk`);
    `profile` carries the radar observations.
    """

    dist = profile["distance"] / 1000.0
    ice_type = np.where(model["E"] >= model["E_pmp"], TEMPERATE, COLD).astype(float)

    # Blank out columns with no ice, so the section stops at the glacier margin.
    ice_type[:, model["thk"] < 1.0] = np.nan

    cmap = ListedColormap([COLD_FILL, TEMPERATE_FILL])
    ax.contourf(
        np.tile(dist, (ice_type.shape[0], 1)),
        model["z"],
        ice_type,
        levels=[-0.5, 0.5, 1.5],
        cmap=cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], cmap.N),
    )

    # A hairline outline of the modelled glacier. The fills alone leave the surface and
    # bed as soft colour edges, which reads as a gradient rather than a boundary.
    margin = model["thk"] >= 1.0
    edge = lambda v: np.where(margin, v, np.nan)
    ax.plot(dist, edge(model["usurf"]), color="black", lw=0.6, zorder=2)
    ax.plot(dist, edge(model["topg"]), color="black", lw=0.6, zorder=2)

    ax.plot(
        dist,
        np.where(profile["is_temperate"], profile["cts_obs"], np.nan),
        color="black",
        ls=(0, (3, 1.6)),
        lw=lw,
        zorder=3,
        label="Observed CTS",
    )

    ax.set(xlabel="Distance along profile (km)", ylabel="Elevation (m)")
    ax.margins(x=0)
    if legend:
        ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
                  labelspacing=0.25, borderaxespad=0.3)

    return ax




def _line(**kw):
    import matplotlib.lines as mlines

    return mlines.Line2D([], [], **kw)




def section_key():
    """Handles and labels for the section panel's vocabulary."""

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=TEMPERATE_FILL),
        mpatches.Patch(color=COLD_FILL),
        _line(color="black", ls=(0, (3, 1.6)), lw=1.4),
    ]
    labels = [
        "Modelled temperate ice",
        "Modelled cold ice",
        "Observed CTS",
    ]

    return handles, labels
