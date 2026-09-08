#!/usr/bin/env python3
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr, chi2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, silhouette_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
FS = 500.0
WIN_S = 2.0
STRIDE_S = 0.5
WIN = int(FS*WIN_S)
STRIDE = int(FS*STRIDE_S)
CH25 = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","P7","P8","FZ","CZ","PZ","FC1","FC2","CP1","CP2","FC5","FC6","CP5","CP6"]
SM_IDX = [CH25.index(c) for c in ["CZ","FC1","FC2","CP1","CP2"]]
TH_IDX = [CH25.index(c) for c in ["FZ","FC1","FC2"]]
BANDS = {"theta":(4,8), "beta_low":(13,20), "beta_mid":(20,25), "beta_high":(25,30), "beta":(13,30)}
RNG = np.random.default_rng(20260908)

def sos_band(x, lo, hi):
    sos = butter(4, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)

def analytic_amp(x, band):
    return np.abs(hilbert(sos_band(x, *band)))

def robust_z(v):
    v=np.asarray(v,float)
    med=np.nanmedian(v); mad=np.nanmedian(np.abs(v-med))*1.4826
    if not np.isfinite(mad) or mad < 1e-12: mad=np.nanstd(v)
    if not np.isfinite(mad) or mad < 1e-12: mad=1.0
    return (v-med)/mad

def contiguous_burst_mask(env, threshold, min_samples):
    above = env > threshold
    starts = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    ends = np.flatnonzero(above & np.r_[~above[1:], True]) + 1
    mask = np.zeros(len(env), dtype=bool)
    burst_starts = []
    for a,b in zip(starts, ends):
        if b-a >= min_samples:
            mask[a:b]=True; burst_starts.append(a)
    return mask, np.asarray(burst_starts, int)

def parse_id(fp, root):
    rel=fp.relative_to(root)
    parts=rel.parts
    subj=parts[0]
    if len(parts)==3:
        session=parts[1]; task=Path(parts[2]).stem
    else:
        session='single'; task=Path(parts[1]).stem
    return subj, session, task, f"{subj}/{session}/{task}"

def load_selected(fp):
    # filtered format has 61 columns: sample,time,25 EEG,5 physiology,28 IMU,label
    usecols=list(range(2,27))+[46,47,48,60]  # EEG 25; waist accel xyz; label
    df=pd.read_csv(fp, header=None, usecols=usecols, engine='c')
    arr=df.to_numpy(float)
    eeg=arr[:,:25]
    waist=arr[:,25:28]
    fog=arr[:,28].astype(int)
    return eeg, waist, fog

def window_rows(fp, root):
    subj,session,task,rid=parse_id(fp,root)
    eeg, waist, fog=load_selected(fp)
    sm=eeg[:,SM_IDX].mean(axis=1)
    th=eeg[:,TH_IDX].mean(axis=1)
    env={"theta":analytic_amp(th,BANDS['theta'])}
    for b in ['beta_low','beta_mid','beta_high','beta']:
        env[b]=analytic_amp(sm,BANDS[b])
    beta_env=env['beta']
    thr=np.nanpercentile(beta_env,75)
    burst_mask, burst_starts=contiguous_burst_mask(beta_env,thr,int(.1*FS))
    move=np.linalg.norm(waist,axis=1)
    n=len(fog)
    rows=[]
    for start in range(0,n-WIN+1,STRIDE):
        stop=start+WIN
        fr=float(np.mean(fog[start:stop]))
        row={
            'subject':subj,'session':session,'task':task,'recording':rid,
            'start_sample':start,'time_s':start/FS,'fog_frac':fr,
            'fog':1 if fr>=.9 else (0 if fr<=.1 else np.nan),
            'theta_amp':float(np.median(env['theta'][start:stop])),
            'beta_low':float(np.median(env['beta_low'][start:stop])),
