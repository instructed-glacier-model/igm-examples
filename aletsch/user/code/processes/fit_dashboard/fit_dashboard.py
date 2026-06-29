#!/usr/bin/env python3
"""
Live fit dashboard for ``time_relaxation`` runs — USER copy.

This used to live in the IGM core (``igm/processes/fit_dashboard``); it is a
simulation-specific visualization, not core physics, so it now lives here in the
user code of each simulation.

Wiring (because ``time_relaxation`` imports its pre/post_processes strictly as
``igm.processes.{name}``, which the user-module loader does not satisfy):
  * this file is loaded as a user /processes module (listed in the experiment's
    ``override /processes``), and
  * at IMPORT it aliases itself into ``sys.modules['igm.processes.fit_dashboard']``
    so that ``time_relaxation``'s ``post_processes: [fit_dashboard]`` import
    resolves to THIS user copy instead of the core one.
The alias is set during the module-load phase, before ``time_relaxation``
imports it in its initialize(), so the user copy always wins.

Listed in BOTH ``override /processes`` (to trigger loading + the alias) and
``time_relaxation.post_processes`` (to actually draw a frame at each save) —
exactly the pattern ``iceflow``/``subglacial_hydrology`` already use.

Behaviour is unchanged from the core version, except the figure title is read
from ``cfg.assimilations.time_relaxation.viz.title`` (default "time-relaxation").

Panels at each save time:
  row 1 — SURFACE SPEED : observed | modelled | (model - obs)   + tau_ref map
  row 2 — APPARENT MASS BALANCE : amb | divflux | (divflux - amb)  + live RMSEs
Frames -> ``fit_dashboard/fit_t<TIME>.png`` and assembled into
``fit_dashboard/fit_evolution.gif`` at the end time. Headless-safe (Agg).
"""

import sys as _sys

# --- make THIS user module the one time_relaxation imports as a post_process ---
_sys.modules["igm.processes.fit_dashboard"] = _sys.modules[__name__]
print(f"[fit_dashboard] USER module loaded + aliased as igm.processes.fit_dashboard "
      f"({__file__})")

import os
import numpy as np


# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #

def _np(x):
    return np.array(x, dtype=np.float64)


def _scalar_t(state):
    t = state.t
    return float(t.numpy()) if hasattr(t, "numpy") else float(t)


def _rmse(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _masked(field, mask):
    return np.where(mask, field, np.nan)


def _opt(cfg, default=None):
    try:
        return cfg.assimilations.time_relaxation.viz
    except Exception:
        return default


# --------------------------------------------------------------------------- #
#  module interface                                                           #
# --------------------------------------------------------------------------- #

def initialize(cfg, state):
    viz = _opt(cfg)
    show = bool(getattr(viz, "show", False)) if viz is not None else False
    title = str(getattr(viz, "title", "time-relaxation")) if viz is not None else "time-relaxation"

    import matplotlib
    if show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
            show = False
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if show:
        plt.ion()

    outdir = os.path.join(os.getcwd(), "fit_dashboard")
    os.makedirs(outdir, exist_ok=True)

    try:
        t_end = float(cfg.assimilations.time_relaxation.time.end)
    except Exception:
        t_end = None

    state._fitviz = dict(
        plt=plt, show=show, outdir=outdir, t_end=t_end, title=title,
        dpi=int(getattr(viz, "dpi", 110)) if viz is not None else 110,
        amb_scale=2.0,
        built=False, frames=[], hist_t=[], hist_vel=[], hist_amb=[],
    )


_PANELS = [
    ("vobs",  "SPEED: observed",              "viridis", False),
    ("vmod",  "SPEED: modelled",              "viridis", False),
    ("vres",  "SPEED: model - obs",           "RdBu_r",  True),
    ("amb",   "AMB: target (smb - dhdt_obs)", "RdBu_r",  True),
    ("dflx",  "AMB: modelled divflux",        "RdBu_r",  True),
    ("ares",  "AMB: divflux - target",        "RdBu_r",  True),
]


def _compute_fields(state):
    icemask = _np(state.icemask) > 0.5
    vmod = _np(getattr(state, "velsurf_mag", np.full_like(icemask, np.nan, float)))
    vobs = _np(getattr(state, "velsurf_magobs", np.full_like(icemask, np.nan, float)))
    vsel = icemask & np.isfinite(vobs) & (vobs > 0)
    amb = _np(getattr(state, "amb", np.full_like(icemask, np.nan, float)))
    dflx = _np(getattr(state, "divflux", np.full_like(icemask, np.nan, float)))
    tau = _np(getattr(state, "tau_ref", np.full_like(icemask, np.nan, float)))
    fields = {
        "vobs": _masked(vobs, vsel),
        "vmod": _masked(vmod, icemask),
        "vres": _masked(vmod - vobs, vsel),
        "amb":  _masked(amb, icemask),
        "dflx": _masked(dflx, icemask),
        "ares": _masked(dflx - amb, icemask),
        "tau":  _masked(tau, icemask),
    }
    rmse_vel = _rmse(vmod, vobs, vsel)
    rmse_amb = _rmse(dflx, amb, icemask)
    return fields, rmse_vel, rmse_amb, icemask, vsel


def _build_artists(d, fields, icemask, vsel):
    plt = d["plt"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    d["fig"], d["axes"] = fig, axes

    vlim = float(np.nanpercentile(fields["vobs"][np.isfinite(fields["vobs"])], 98)) \
        if np.isfinite(fields["vobs"]).any() else 100.0
    afin = fields["amb"][np.isfinite(fields["amb"])]
    alim = float(np.nanpercentile(np.abs(afin), 98)) if afin.size else 5.0
    alim = alim if alim > 0 else 5.0
    alim *= d.get("amb_scale", 2.0)        # enlarge AMB-row range (default 2x)
    scales = {
        "vobs": (0, vlim), "vmod": (0, vlim), "vres": (-vlim / 2, vlim / 2),
        "amb": (-alim, alim), "dflx": (-alim, alim), "ares": (-alim / 2, alim / 2),
    }

    ims = {}
    flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, (key, title, cmap, _sym) in zip(flat, _PANELS):
        lo, hi = scales[key]
        im = ax.imshow(fields[key], origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m/yr")
        ims[key] = im

    ax_tau = axes[0, 3]
    im_tau = ax_tau.imshow(fields["tau"], origin="lower", cmap="magma")
    ax_tau.set_title("tau_ref (basal friction)", fontsize=11)
    ax_tau.set_xticks([]); ax_tau.set_yticks([])
    fig.colorbar(im_tau, ax=ax_tau, fraction=0.046, pad=0.04, label="MPa")
    ims["tau"] = im_tau

    d["ims"] = ims
    d["ax_c"] = axes[1, 3]
    d["ax_c2"] = axes[1, 3].twinx()
    d["built"] = True


def update(cfg, state):
    if not getattr(state, "saveresult", False):
        return
    if not hasattr(state, "_fitviz"):
        return

    d = state._fitviz
    plt = d["plt"]
    t = _scalar_t(state)

    fields, rmse_vel, rmse_amb, icemask, vsel = _compute_fields(state)

    d["hist_t"].append(t)
    d["hist_vel"].append(rmse_vel)
    d["hist_amb"].append(rmse_amb)

    if not d["built"]:
        _build_artists(d, fields, icemask, vsel)
    else:
        for key in ("vobs", "vmod", "vres", "amb", "dflx", "ares"):
            d["ims"][key].set_data(fields[key])

    fig = d["fig"]

    tfin = fields["tau"][np.isfinite(fields["tau"])]
    if tfin.size:
        tlo = float(np.nanpercentile(tfin, 2))
        thi = float(np.nanpercentile(tfin, 98))
        if thi <= tlo:
            thi = tlo + 1e-6
        d["ims"]["tau"].set_data(fields["tau"])
        d["ims"]["tau"].set_clim(tlo, thi)

    ax_c = d["ax_c"]
    ax_c2 = d["ax_c2"]
    ax_c.clear()
    ln1 = ax_c.plot(d["hist_t"], d["hist_vel"], "o-", color="tab:blue",
                    label="speed RMSE")[0]
    ax_c.set_xlabel("time (yr)")
    ax_c.set_ylabel("speed RMSE (m/yr)", color="tab:blue")
    ax_c.tick_params(axis="y", labelcolor="tab:blue")
    ax_c.set_yscale("log")
    if d["t_end"]:
        ax_c.set_xlim(d["hist_t"][0], d["t_end"])
    ax_c.grid(alpha=0.3)
    ax_c2.clear()
    ln2 = ax_c2.plot(d["hist_t"], d["hist_amb"], "s-", color="tab:red",
                     label="|divflux-AMB| RMSE")[0]
    ax_c2.set_ylabel("|divflux - AMB| RMSE (m/yr)", color="tab:red")
    ax_c2.tick_params(axis="y", labelcolor="tab:red")
    ax_c2.set_yscale("log")
    ax_c.legend(handles=[ln1, ln2], loc="upper right", fontsize=9)
    ax_c.set_title("(live) misfit convergence", fontsize=11)

    fig.suptitle(
        f"{d['title']} time-relaxation fit   —   t = {t:.0f} yr      "
        f"speed RMSE = {rmse_vel:.1f} m/yr      "
        f"|divflux - AMB| RMSE = {rmse_amb:.2f} m/yr",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    frame = os.path.join(d["outdir"], f"fit_t{int(round(t)):04d}.png")
    fig.savefig(frame, dpi=d["dpi"])
    d["frames"].append(frame)
    print(f"[fit_dashboard] t={t:7.1f} yr   speed_RMSE={rmse_vel:7.2f}   "
          f"divflux-AMB_RMSE={rmse_amb:7.3f}   -> {os.path.basename(frame)}")

    if d["show"]:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    if d["t_end"] is not None and t >= d["t_end"] - 1e-6:
        _assemble_gif(d)


def finalize(cfg, state):
    if hasattr(state, "_fitviz"):
        _assemble_gif(state._fitviz)
        try:
            state._fitviz["plt"].close(state._fitviz["fig"])
        except Exception:
            pass


def _assemble_gif(d):
    frames = d.get("frames", [])
    if not frames:
        return
    gif = os.path.join(d["outdir"], "fit_evolution.gif")
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(f) for f in frames]
        imageio.mimsave(gif, imgs, duration=0.8)
        print(f"[fit_dashboard] wrote {gif}")
        return
    except Exception:
        pass
    try:
        from PIL import Image
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=800, loop=0)
        print(f"[fit_dashboard] wrote {gif}")
    except Exception as e:
        print(f"[fit_dashboard] could not assemble GIF ({e}); "
              f"individual frames are in {d['outdir']}")
