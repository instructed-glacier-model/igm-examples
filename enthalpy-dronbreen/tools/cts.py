"""Cold-temperate transition surface (CTS) diagnostics for the Drønbreen example.

This is the single implementation of "where does the ice stop being cold?", shared by
the in-run process module `user/code/processes/track_cts_err` and by every script in
`tools/`. Keeping it in one place matters: the CTS depends on the vertical grid
geometry, and a mismatch between the grid assumed here and the one the enthalpy module
actually used produces plausible-looking but wrong elevations, with no error raised.

The CTS is the level where the ice enthalpy `E` reaches the pressure-melting enthalpy
`E_pmp`: below it the ice is temperate (`E >= E_pmp`, liquid water present), above it
the ice is cold (`E < E_pmp`). It is located by linear interpolation of `E - E_pmp`
between the two layers that bracket the sign change.

Observation columns (Mannerfelt et al., submitted; https://zenodo.org/records/17882300):

* `temperate`            - thickness of the temperate layer above the bed, in m
* `temperate_elevation`  - CTS elevation, exactly `bed_elevation + temperate`
* `temperate_upper/lower`- bounds on that *thickness*. Note the naming is inverted with
  respect to the `thickness_*` columns: `temperate_upper <= temperate <= temperate_lower`
  holds for every row of the dataset, so the CTS uncertainty band in elevation runs from
  `bed_elevation + temperate_upper` to `bed_elevation + temperate_lower`.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def sigma_midpoints(nz, vert_spacing):
    """Mid-layer positions of IGM's stretched vertical grid, as fractions of thickness.

    `vert_spacing` > 1 packs layers towards the bed, where the thermal gradients that
    set the CTS are strongest. This mirrors the discretisation used by the enthalpy
    module, so `nz` and `vert_spacing` must come from `processes.enthalpy.numerics`.
    """

    zeta = np.arange(nz + 1) / nz
    edges = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)

    return 0.5 * (edges[:-1] + edges[1:])


def layer_elevations(topg, thk, nz, vert_spacing):
    """Elevation of every layer midpoint. Returns shape `(nz,) + topg.shape`."""

    zeta_mid = sigma_midpoints(nz, vert_spacing)

    return topg[None, ...] + zeta_mid.reshape((-1,) + (1,) * topg.ndim) * thk[None, ...]


def cts_elevation(E, E_pmp, topg, thk, vert_spacing, min_thk=1.0):
    """Elevation of the cold-temperate transition surface.

    `E` and `E_pmp` have shape `(nz, ...)`; `topg` and `thk` have shape `(...)`.
    Returns an array of shape `(...)`, NaN only where there is no ice to speak of
    (`thk < min_thk`).

    Columns with no enthalpy sign change are resolved rather than discarded: an
    entirely cold column has no temperate layer, so its CTS sits at the bed, and an
    entirely temperate column has its CTS at the surface. Dropping those columns
    instead — as is tempting, since there is no crossing to interpolate — would make
    the misfit blind in one direction: a model that froze the whole glacier solid would
    simply contribute no points and so incur no penalty.
    """

    E = np.asarray(E, dtype=float)
    E_pmp = np.asarray(E_pmp, dtype=float)
    topg = np.asarray(topg, dtype=float)
    thk = np.asarray(thk, dtype=float)

    nz = E.shape[0]
    delta = E - E_pmp
    z = layer_elevations(topg, thk, nz, vert_spacing)

    # Lowest pair of adjacent layers that straddle E == E_pmp.
    crossing = np.sign(delta[:-1]) != np.sign(delta[1:])
    found = crossing.any(axis=0)
    k = np.argmax(crossing, axis=0)

    take = lambda a: np.take_along_axis(a, k[None, ...], axis=0)[0]
    d1, d2 = take(delta[:-1]), take(delta[1:])
    z1, z2 = take(z[:-1]), take(z[1:])

    slope = d2 - d1
    interpolated = np.where(
        np.abs(slope) < 1e-12, z1, z1 - d1 * (z2 - z1) / np.where(slope == 0, 1.0, slope)
    )

    # No crossing: temperate throughout (CTS at the surface) or cold throughout (at the bed).
    uniform = np.where(delta[0] >= 0, topg + thk, topg)

    return np.where(thk < min_thk, np.nan, np.where(found, interpolated, uniform))


def enthalpy_grid(cfg):
    """`(Nz, vert_spacing)` of the enthalpy vertical grid, from a run configuration.

    Accepts either a live Hydra config or the `.hydra/config.yaml` of a finished run.
    """

    numerics = cfg["processes"]["enthalpy"]["numerics"]

    return int(numerics["Nz"]), float(numerics["vert_spacing"])


def sample_columns(field, x, y, easting, northing):
    """Bilinearly sample a model field at scattered points.

    `field` is `(ny, nx)` or `(nz, ny, nx)`; the result is `(npoints,)` or
    `(nz, npoints)`. Points outside the grid come back as NaN. Using interpolation
    rather than nearest-cell snapping matters here: the GPR traces are a few metres
    apart while the grid is 100 m, so nearest-cell sampling would turn a smooth radar
    line into a staircase.
    """

    field = np.asarray(field, dtype=float)
    points = np.column_stack([np.asarray(northing), np.asarray(easting)])

    if field.ndim == 2:
        interp = RegularGridInterpolator(
            (y, x), field, bounds_error=False, fill_value=np.nan
        )
        return interp(points)

    interp = RegularGridInterpolator(
        (y, x), np.moveaxis(field, 0, -1), bounds_error=False, fill_value=np.nan
    )

    return interp(points).T


def load_cts_profiles(data_dir, glacier_name, year_obs):
    """Load the GPR profiles of one glacier and survey year.

    Returns a list of dicts, one per radar line, ordered by file name. Raises if the
    directory or the pattern matches nothing, since a silently empty observation set
    would turn the calibration cost into a meaningless constant.
    """

    data_dir = os.path.expanduser(str(data_dir))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CTS observation directory not found: {data_dir}")

    pattern = os.path.join(data_dir, f"cts_{glacier_name}-{year_obs}*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CTS observation file matches {pattern}")

    profiles = []
    for path in files:
        df = pd.read_csv(path)
        profiles.append(
            {
                "name": os.path.basename(path)[len("cts_") : -len(".csv")],
                "distance": df["distance"].to_numpy(),
                "easting": df["easting"].to_numpy(),
                "northing": df["northing"].to_numpy(),
                "surf_obs": df["elevation"].to_numpy(),
                "bed_obs": df["bed_elevation"].to_numpy(),
                "cts_obs": df["temperate_elevation"].to_numpy(),
                # Uncertainty band, converted from temperate-layer thickness to elevation.
                "cts_obs_low": df["bed_elevation"].to_numpy()
                + df["temperate_upper"].to_numpy(),
                "cts_obs_high": df["bed_elevation"].to_numpy()
                + df["temperate_lower"].to_numpy(),
                "is_temperate": df["temperate"].to_numpy() > 0,
                "bed_type": df["bed_type"].to_numpy(),
            }
        )

    # Short labels for figures, ranked by how much temperate ice the radar saw. Ranking
    # on the observations rather than on file order means the labels read T1, T2, ... in
    # order on any plot sorted that way, and stay attached to the same transect whatever
    # the model does — the same label means the same radar line across every run.
    profiles.sort(key=observed_temperate_area, reverse=True)
    for index, profile in enumerate(profiles, start=1):
        profile["label"] = f"T{index}"

    return profiles


def observed_temperate_area(profile):
    """Cross-sectional area of temperate ice the radar recorded on this transect (m²)."""

    return float(
        np.trapz(
            np.clip(profile["cts_obs"] - profile["bed_obs"], 0.0, None),
            profile["distance"],
        )
    )


def best_trial_run(storage, study_name, sweep_dir, quiet=False):
    """Directory of the lowest-cost completed trial of an Optuna study."""

    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)
    ]
    if not completed:
        raise SystemExit(
            f"Study '{study_name}' has no completed trial. Run the Step 2 sweep first."
        )

    best = min(completed, key=lambda t: t.value)
    if not quiet:
        print(
            f"best of {len(completed)} trials: #{best.number}, "
            f"cost_cts = {best.value:.2f} m"
        )
        for name, value in sorted(best.params.items()):
            print(f"  {name.split('.')[-1]:20s} {value:.4g}")

    return os.path.join(sweep_dir, str(best.number))


def representative_profile(profiles, misfits, min_contrast=0.15):
    """Transect that best represents typical model performance.

    Two things disqualify a transect from illustrating the thermal structure: crossing
    almost entirely cold or almost entirely temperate ice, which makes the section one
    flat colour, and fitting far better or far worse than the survey as a whole, which
    misleads the reader about typical performance.

    So: among transects with at least `min_contrast` of their points on the minority
    side, take the one whose misfit is closest to the median. Selecting purely on
    contrast — the obvious choice — tends to surface the *worst*-fitting transect,
    because the lines that cross a broad transition are also the hardest to get right.
    """

    candidates = [
        (p, m)
        for p, m in zip(profiles, misfits)
        if np.isfinite(m) and _contrast_fraction(p) >= min_contrast
    ]
    if not candidates:
        return most_contrasted_profile(profiles)

    median = float(np.nanmedian([m for m in misfits if np.isfinite(m)]))

    return min(candidates, key=lambda c: abs(c[1] - median))[0]


def _contrast_fraction(profile):
    """How visibly a transect splits into temperate and cold ice, by area.

    Deliberately area-based rather than point-based. A transect can have temperate ice
    recorded at 15% of its points and still look entirely cold in section, because that
    layer is only a few metres thick — counting points would nominate it as a good
    illustration when it shows the reader nothing.
    """

    thickness = profile["surf_obs"] - profile["bed_obs"]
    ice = np.trapz(np.clip(thickness, 0.0, None), profile["distance"])
    if ice <= 0:
        return 0.0

    temperate = np.trapz(
        np.clip(profile["cts_obs"] - profile["bed_obs"], 0.0, None), profile["distance"]
    )
    share = temperate / ice

    return float(min(share, 1.0 - share))


def most_contrasted_profile(profiles):
    """The profile that best displays a CTS, for use as a default when plotting.

    Some radar lines cross almost entirely cold ice and others almost entirely
    temperate ice; either way the section comes out as one flat colour with no visible
    transition. The most balanced line — the one with the most points on the minority
    side — is the one that actually crosses the CTS.
    """

    def contrast(p):
        temperate = int(p["is_temperate"].sum())
        return min(temperate, len(p["is_temperate"]) - temperate)

    return max(profiles, key=contrast)


def profile_rmse(cts_model, cts_obs):
    """RMSE between modelled and observed CTS, ignoring points where either is NaN.

    Returns NaN if nothing is comparable, so callers can drop the profile rather than
    fold a fabricated zero into the mean.
    """

    valid = np.isfinite(cts_model) & np.isfinite(cts_obs)
    if not np.any(valid):
        return np.nan

    return float(np.sqrt(np.mean((cts_model[valid] - cts_obs[valid]) ** 2)))
