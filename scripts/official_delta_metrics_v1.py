"""Ground-truth-free flow disagreement and motion diagnostics, in input pixels."""
import numpy as np


def metrics(clean, corrupt, quantize=False):
    """Pool pixels, never take the norm of the mean flow vector."""
    if clean.shape != corrupt.shape or clean.shape[-1] != 2:
        raise ValueError('Mismatched flow shape')
    if quantize:
        clean = clean.astype(np.float16)
        corrupt = corrupt.astype(np.float16)
    if not np.isfinite(clean).all() or not np.isfinite(corrupt).all():
        raise ValueError('Nonfinite prediction (including float16 export overflow)')
    a = clean.reshape(-1, 2).astype(np.float64)
    b = corrupt.reshape(-1, 2).astype(np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    d = np.linalg.norm(a - b, axis=-1)
    moving = na > 1.0
    out = dict(n=len(a), delta_sum=float(d.sum()),
               clean_magnitude_sum=float(na.sum()), corrupt_magnitude_sum=float(nb.sum()),
               clean_x_sum=float(a[:, 0].sum()), clean_y_sum=float(a[:, 1].sum()),
               corrupt_x_sum=float(b[:, 0].sum()), corrupt_y_sum=float(b[:, 1].sum()),
               clean_squared_sum=float((a * a).sum()), dot_sum=float((a * b).sum()),
               onepx_count=int((d > 1).sum()),
               fl_count=int(((d > 3) & (d > .05 * na)).sum()),
               moving_count=int(moving.sum()),
               collapsed_count=int((moving & (nb < .5 * na)).sum()))
    return out


def finalize(total):
    n = total['n']
    if n <= 0:
        raise ValueError('Empty population')
    return dict(delta=total['delta_sum'] / n,
                clean_mean_magnitude=total['clean_magnitude_sum'] / n,
                corrupt_mean_magnitude=total['corrupt_magnitude_sum'] / n,
                magnitude_ratio=(total['corrupt_magnitude_sum'] / total['clean_magnitude_sum']
                                 if total['clean_magnitude_sum'] > 0 else None),
                directional_gain=(total['dot_sum'] / total['clean_squared_sum']
                                  if total['clean_squared_sum'] > 0 else None),
                collapse_fraction_moving=(total['collapsed_count'] / total['moving_count']
                                          if total['moving_count'] else None),
                onepx_pct=100 * total['onepx_count'] / n,
                fl_pct=100 * total['fl_count'] / n, pixels=n,
                clean_mean_xy=[total['clean_x_sum'] / n, total['clean_y_sum'] / n],
                corrupt_mean_xy=[total['corrupt_x_sum'] / n, total['corrupt_y_sum'] / n])
