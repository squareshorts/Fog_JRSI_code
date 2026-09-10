#!/usr/bin/env python3
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import antecedent_history_final as ah


def corrected_temporal_null(df_all, sec, nperm):
    eligible_col = f'clean_hist{sec}'
    obs_all = ah.robust_scale_within_recording(df_all)
    d = obs_all.loc[obs_all[eligible_col]].copy().reset_index(drop=True)
    _, f1, f2 = ah.formulas(sec)
    m1 = smf.glm(f1, data=d, family=sm.families.Binomial()).fit()
    m2 = smf.glm(f2, data=d, family=sm.families.Binomial()).fit()
    obs = 2 * (m2.llf - m1.llf)
    vals = []
    n_used = []
    for _ in range(nperm):
        pdat = df_all.copy()
        for _, idx0 in pdat.groupby('recording').groups.items():
            idx_all = np.asarray(list(idx0), int)
            finite = (
                np.isfinite(pdat.loc[idx_all, f'theta_hist{sec}'].to_numpy()) &
                np.isfinite(pdat.loc[idx_all, f'beta_hist{sec}'].to_numpy())
            )
            idx = idx_all[finite]
            n = len(idx)
            if n < 8:
                continue
            minsh = max(2, min(int(10 / ah.HORIZON_S), max(2, n // 4)))
            sh = int(ah.RNG.integers(1, n)) if n <= 2 * minsh else int(ah.RNG.integers(minsh, n - minsh))
            for c in [f'theta_hist{sec}', f'beta_hist{sec}']:
                pdat.loc[idx, c] = np.roll(pdat.loc[idx, c].to_numpy(), sh)
        pdat = ah.robust_scale_within_recording(pdat)
        pdx = pdat.loc[pdat[eligible_col]].copy()
        pm1 = smf.glm(f1, data=pdx, family=sm.families.Binomial()).fit()
        pm2 = smf.glm(f2, data=pdx, family=sm.families.Binomial()).fit()
        vals.append(2 * (pm2.llf - pm1.llf))
        n_used.append(len(pdx))
    vals = np.asarray(vals, float)
    if min(n_used) != max(n_used) or n_used[0] != len(d):
        raise RuntimeError(f'Permutation sample size changed: observed={len(d)}, range={min(n_used)}-{max(n_used)}')
    p = (1 + np.sum(vals >= obs)) / (len(vals) + 1)
    return {
        'history_s': sec,
        'observed_lr': obs,
        'nperm': len(vals),
        'p_fullgrid_circular': float(p),
        'null_median_lr': float(np.median(vals)),
        'null_95pct_lr': float(np.quantile(vals, .95)),
        'n_rows_verified_constant': int(n_used[0]),
    }


ah.temporal_null = corrected_temporal_null

if __name__ == '__main__':
    ah.main()
