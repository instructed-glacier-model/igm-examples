import tensorflow as tf


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    """Compute all available scores as a dictionary.

    The optimize_*.yaml selects which scores to use as objectives.
    """
    ecfg = cfg.processes.eval_objective
    dx = state.dx

    volume_km3 = float(tf.reduce_sum(state.thk) * dx * dx / 1.0e9)
    max_thk = float(tf.reduce_max(state.thk))

    velsurf_mag = tf.norm(
        tf.stack([state.uvelsurf, state.vvelsurf], axis=-1), axis=-1
    )
    max_speed = float(tf.reduce_max(velsurf_mag))

    state.score = {
        "cost_volume": abs(volume_km3 - ecfg.target_volume),
        "cost_speed": abs(max_speed - ecfg.target_max_speed),
        "cost_thickness": abs(max_thk - ecfg.target_max_thk),
    }
