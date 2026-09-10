#!/usr/bin/env python3
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

import reviewer_reanalysis as rr

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260910)
HORIZON_S = 0.5
HORIZON = int(rr.FS * HORIZON_S)
CURRENT_S = 2.0
CURRENT = int(rr.FS * CURRENT_S)
POST_CONFIRM_S = 1.0
POST_CONFIRM = int(rr.FS * POST_CONFIRM_S)
HISTORY_SECONDS = (3, 5, 10)


def validated_onsets(fog):
    fog = np.asarray(fog, int)
    cand = np.flatnonzero((fog == 1) & np.r_[True, fog[:-1] == 0])
    out = []
    for o in cand:
        if o < CURRENT or o + POST_CONFIRM > len(fog):
            continue
        if np.all(fog[o-CURRENT:o] == 0) and np.mean(fog[o:o+POST_CONFIRM]) >= 0.95:
            out.append(int(o))
    return np.asarray(out, int)


def summarize_interval(env, move, a, b):
    if a < 0 or b <= a:
        return (np.nan, np.nan, np.nan, np.nan)
    sl = slice(a, b)
    return (
        float(np.median(env['theta'][sl])),
        float(np.median(env['beta_low'][sl])),
        float(np.mean(move[sl])),
        float(np.std(move[sl])),
    )


def recording_rows(fp, root):
    subj, session, task, rid = rr.parse_id(fp, root)
    eeg, waist, fog = rr.load_selected(fp)
    smsig = eeg[:, rr.SM_IDX].mean(axis=1)
    thsig = eeg[:, rr.TH_IDX].mean(axis=1)
    env = {
        'theta': rr.analytic_amp(thsig, rr.BANDS['theta']),
        'beta_low': rr.analytic_amp(smsig, rr.BANDS['beta_low']),
    }
    move = np.linalg.norm(waist, axis=1)
    onset_set = set(validated_onsets(fog).tolist())
    last_one = np.maximum.accumulate(np.where(fog == 1, np.arange(len(fog)), -1))
    rows = []
    for stop in range(CURRENT, len(fog)-HORIZON+1, rr.STRIDE):
        cur_start = stop-CURRENT
        cur = summarize_interval(env, move, cur_start, stop)
        row = {
            'subject': subj, 'session': session, 'task': task, 'recording': rid,
            'stop_sample': stop, 'time_s': stop/rr.FS,
            'event': int(any(o in onset_set for o in range(stop, min(stop+HORIZON, len(fog))))),
            'current_clean': bool(np.all(fog[cur_start:stop] == 0)),
            'theta_current': cur[0], 'beta_current': cur[1],
            'move_current': cur[2], 'movesd_current': cur[3],
        }
        lo = int(last_one[stop-1])
        nonfog_dur = (stop-lo-1)/rr.FS if lo >= 0 else stop/rr.FS
        row['nonfog_duration_s'] = nonfog_dur
        row['log2_nonfog_duration'] = np.log2(max(nonfog_dur, 1.0))
        row['log_time'] = np.log1p(stop/rr.FS)
        for sec in HISTORY_SECONDS:
            hlen = int(sec*rr.FS)
            a, b = cur_start-hlen, cur_start
            vals = summarize_interval(env, move, a, b)
            row[f'theta_hist{sec}'] = vals[0]
            row[f'beta_hist{sec}'] = vals[1]
            row[f'move_hist{sec}'] = vals[2]
            row[f'movesd_hist{sec}'] = vals[3]
            row[f'clean_hist{sec}'] = bool(a >= 0 and np.all(fog[a:stop] == 0))
        rows.append(row)
    return rows


def robust_scale_within_recording(df):
    out = df.copy()
    cols = ['theta_current','beta_current','move_current','movesd_current']
    for sec in HISTORY_SECONDS:
        cols += [f'theta_hist{sec}',f'beta_hist{sec}',f'move_hist{sec}',f'movesd_hist{sec}']
    for c in cols:
        out[c+'_z'] = out.groupby('recording')[c].transform(lambda s: rr.robust_z(s.to_numpy()))
    return out


def formulas(sec):
    base = (
        f'event ~ C(task) + log2_nonfog_duration + log_time + '
        f'move_current_z + movesd_current_z + move_hist{sec}_z + movesd_hist{sec}_z'
    )
    current = base + ' + theta_current_z + beta_current_z'
    antecedent = current + f' + theta_hist{sec}_z + beta_hist{sec}_z'
    return base, current, antecedent


def clustered_block_test(model, d, terms):
    names = list(model.params.index)
    R = np.zeros((len(terms), len(names)))
    for i, term in enumerate(terms):
        R[i, names.index(term)] = 1.0
    wt = model.wald_test(R, scalar=True)
    chi = float(wt.statistic)
    F = chi / len(terms)
    df1 = len(terms)
    df2 = d['subject'].nunique()-1
    p = float(f.sf(F, df1, df2))
    return F, df1, df2, p


def fit_inference(d, sec):
    f0, f1, f2 = formulas(sec)
    m0 = smf.glm(f0, data=d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups':d.subject})
    m1 = smf.glm(f1, data=d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups':d.subject})
    m2 = smf.glm(f2, data=d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups':d.subject})
    cur_terms = ['theta_current_z','beta_current_z']
    hist_terms = [f'theta_hist{sec}_z',f'beta_hist{sec}_z']
    Fc,dc1,dc2,pc = clustered_block_test(m1,d,cur_terms)
    Fh,dh1,dh2,ph = clustered_block_test(m2,d,hist_terms)
    tests = pd.DataFrame([
        {'history_s':sec,'block':'current_theta_lowbeta','F':Fc,'df1':dc1,'df2':dc2,'p_small_cluster':pc},
        {'history_s':sec,'block':'strictly_antecedent_theta_lowbeta','F':Fh,'df1':dh1,'df2':dh2,'p_small_cluster':ph},
    ])
    coefs=[]
    crit = 2.200985
    for term in cur_terms+hist_terms:
        b=float(m2.params[term]); se=float(m2.bse[term])
        coefs.append({'history_s':sec,'term':term,'beta':b,'se_cluster':se,'OR':math.exp(b),
                      'OR_ci_low_t11':math.exp(b-crit*se),'OR_ci_high_t11':math.exp(b+crit*se)})
    return tests, pd.DataFrame(coefs)


def design_raw(d, sec, level):
    cols = ['log2_nonfog_duration','log_time']
    raw = ['move_current','movesd_current',f'move_hist{sec}',f'movesd_hist{sec}']
    if level >= 1:
        raw += ['theta_current','beta_current']
    if level >= 2:
        raw += [f'theta_hist{sec}',f'beta_hist{sec}']
    X = d[cols].copy()
    for c in raw:
        X[c+'_log1p'] = np.log1p(np.clip(d[c].to_numpy(float),0,None))
    task = pd.get_dummies(d['task'],prefix='task',dtype=float)
    return pd.concat([X.reset_index(drop=True), task.reset_index(drop=True)],axis=1)


def loso_cv(d, sec):
    y=d.event.to_numpy(int); subs=d.subject.to_numpy()
    Xs={lvl:design_raw(d,sec,lvl) for lvl in (0,1,2)}
    rows=[]
    for s in sorted(pd.unique(subs)):
        tr=subs!=s; te=subs==s
        if len(np.unique(y[tr]))<2: continue
        for name,lvl in [('movement',0),('current',1),('antecedent',2)]:
            sc=StandardScaler().fit(Xs[lvl].loc[tr])
            clf=LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000).fit(sc.transform(Xs[lvl].loc[tr]),y[tr])
            p=np.clip(clf.predict_proba(sc.transform(Xs[lvl].loc[te]))[:,1],1e-7,1-1e-7)
            yt=y[te]
            rows.append({'history_s':sec,'subject':s,'model':name,'n':int(te.sum()),'events':int(yt.sum()),
                         'logloss':log_loss(yt,p,labels=[0,1]),'brier':brier_score_loss(yt,p),
                         'auc':roc_auc_score(yt,p) if len(np.unique(yt))==2 else np.nan})
    cv=pd.DataFrame(rows)
    paired=[]
    pv=cv.pivot(index='subject',columns='model',values='logloss')
    for a,b in [('movement','current'),('current','antecedent'),('movement','antecedent')]:
        q=pv[[a,b]].dropna(); delta=q[b]-q[a]
        stat,p=wilcoxon(q[a],q[b])
        paired.append({'history_s':sec,'comparison':f'{a}_vs_{b}','n_subjects':len(q),
                       'median_delta_logloss_b_minus_a':float(np.median(delta)),
                       'mean_delta_logloss_b_minus_a':float(np.mean(delta)),'p_wilcoxon':float(p)})
    return cv,pd.DataFrame(paired)


def temporal_null(df_all, sec, nperm):
    eligible_col=f'clean_hist{sec}'
    obs_all=robust_scale_within_recording(df_all)
    d=obs_all.loc[obs_all[eligible_col]].copy().reset_index(drop=True)
    f0,f1,f2=formulas(sec)
    m1=smf.glm(f1,data=d,family=sm.families.Binomial()).fit()
    m2=smf.glm(f2,data=d,family=sm.families.Binomial()).fit()
    obs=2*(m2.llf-m1.llf)
    vals=[]
    for _ in range(nperm):
        pdat=df_all.copy()
        for rid,idx0 in pdat.groupby('recording').groups.items():
            idx=np.asarray(list(idx0),int); n=len(idx)
            if n < 8: continue
            minsh=max(2,min(int(10/HORIZON_S),max(2,n//4)))
            sh=int(RNG.integers(1,n)) if n<=2*minsh else int(RNG.integers(minsh,n-minsh))
            for c in [f'theta_hist{sec}',f'beta_hist{sec}']:
                pdat.loc[idx,c]=np.roll(pdat.loc[idx,c].to_numpy(),sh)
        pdat=robust_scale_within_recording(pdat)
        pdx=pdat.loc[pdat[eligible_col]].copy()
        pm1=smf.glm(f1,data=pdx,family=sm.families.Binomial()).fit()
        pm2=smf.glm(f2,data=pdx,family=sm.families.Binomial()).fit()
        vals.append(2*(pm2.llf-pm1.llf))
    vals=np.asarray(vals,float)
    p=(1+np.sum(vals>=obs))/(len(vals)+1)
    return {'history_s':sec,'observed_lr':obs,'nperm':len(vals),'p_fullgrid_circular':float(p),
            'null_median_lr':float(np.median(vals)),'null_95pct_lr':float(np.quantile(vals,.95))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--nperm',type=int,default=499)
    args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    fps=sorted(fp for fp in root.rglob('task_*.txt') if fp.stem in {'task_1','task_2','task_3','task_4'})
    rows=[]
    for i,fp in enumerate(fps,1):
        print(f'[{i}/{len(fps)}] {fp}',flush=True)
        rows.extend(recording_rows(fp,root))
    raw=pd.DataFrame(rows)
    raw.to_csv(out/'antecedent_all_grid_rows.csv',index=False)
    z=robust_scale_within_recording(raw)
    alltests=[]; allcoef=[]; allcv=[]; allpaired=[]; nulls=[]; risk_summary=[]
    for sec in HISTORY_SECONDS:
        d=z.loc[z[f'clean_hist{sec}']].copy().reset_index(drop=True)
        risk_summary.append({'history_s':sec,'risk_rows':len(d),'onsets':int(d.event.sum()),'subjects':d.subject.nunique(),'recordings':d.recording.nunique()})
        tests,coefs=fit_inference(d,sec); alltests.append(tests); allcoef.append(coefs)
        cv,paired=loso_cv(raw.loc[raw[f'clean_hist{sec}']].copy().reset_index(drop=True),sec)
        allcv.append(cv); allpaired.append(paired)
        nulls.append(temporal_null(raw,sec,args.nperm))
    tests=pd.concat(alltests,ignore_index=True); coefs=pd.concat(allcoef,ignore_index=True)
    cv=pd.concat(allcv,ignore_index=True); paired=pd.concat(allpaired,ignore_index=True)
    null=pd.DataFrame(nulls); risks=pd.DataFrame(risk_summary)
    tests.to_csv(out/'antecedent_block_tests.csv',index=False)
    coefs.to_csv(out/'antecedent_coefficients.csv',index=False)
    cv.to_csv(out/'antecedent_loso_subject.csv',index=False)
    paired.to_csv(out/'antecedent_loso_paired.csv',index=False)
    null.to_csv(out/'antecedent_temporal_null.csv',index=False)
    risks.to_csv(out/'antecedent_risk_summary.csv',index=False)
    summary={'risk':risks.to_dict(orient='records'),'tests':tests.to_dict(orient='records'),
             'paired':paired.to_dict(orient='records'),'temporal_null':null.to_dict(orient='records')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    report=['# Strictly antecedent cortical-history analysis','',
            'History intervals end exactly where the current 2-s window begins; there is no sample overlap.','']
    report += ['## Risk sets',risks.to_string(index=False),'','## Small-cluster block tests',tests.to_string(index=False),'',
               '## Full-grid temporal null',null.to_string(index=False),'','## LOSO participant paired log-loss tests',paired.to_string(index=False)]
    (out/'REPORT_ANTECEDENT.md').write_text('\n'.join(report))
    print('\n'.join(report),flush=True)

if __name__=='__main__':
    main()
