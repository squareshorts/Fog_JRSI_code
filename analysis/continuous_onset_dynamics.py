#!/usr/bin/env python3
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import reviewer_reanalysis as rr

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20260908)
HORIZON_S = 0.5
HORIZON = int(rr.FS * HORIZON_S)
PRE_ONSET_CLEAN_S = 2.0
POST_ONSET_CONFIRM_S = 1.0
PRE_ONSET_CLEAN = int(rr.FS * PRE_ONSET_CLEAN_S)
POST_ONSET_CONFIRM = int(rr.FS * POST_ONSET_CONFIRM_S)


def validated_onsets(fog):
    fog = np.asarray(fog, int)
    cand = np.flatnonzero((fog == 1) & np.r_[True, fog[:-1] == 0])
    out = []
    n = len(fog)
    for o in cand:
        if o < PRE_ONSET_CLEAN or o + POST_ONSET_CONFIRM > n:
            continue
        if np.all(fog[o-PRE_ONSET_CLEAN:o] == 0) and np.mean(fog[o:o+POST_ONSET_CONFIRM]) >= .95:
            out.append(int(o))
    return np.asarray(out, int)


def feature_at(env, move, stop, seconds):
    n = int(round(seconds * rr.FS))
    if stop < n:
        return (np.nan, np.nan, np.nan, np.nan)
    sl = slice(stop-n, stop)
    return (float(np.median(env['theta'][sl])),
            float(np.median(env['beta_low'][sl])),
            float(np.mean(move[sl])),
            float(np.std(move[sl])))


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
    onsets = validated_onsets(fog)
    onset_set = set(onsets.tolist())
    last_one = np.maximum.accumulate(np.where(fog == 1, np.arange(len(fog)), -1))
    rows = []
    for stop in range(rr.WIN, len(fog) - HORIZON + 1, rr.STRIDE):
        start = stop - rr.WIN
        cur_clean = bool(np.all(fog[start:stop] == 0))
        ev = int(any(o in onset_set for o in range(stop, min(stop + HORIZON, len(fog)))))
        # Since HORIZON == STRIDE, each confirmed onset maps to at most one risk row.
        t2, b2, m2, ms2 = feature_at(env, move, stop, 2.0)
        t3, b3, m3, ms3 = feature_at(env, move, stop, 3.0)
        t5, b5, m5, ms5 = feature_at(env, move, stop, 5.0)
        t10, b10, m10, ms10 = feature_at(env, move, stop, 10.0)
        lo = int(last_one[stop-1])
        nonfog_dur = (stop - lo - 1) / rr.FS if lo >= 0 else stop / rr.FS
        rows.append({
            'subject': subj, 'session': session, 'task': task, 'recording': rid,
            'stop_sample': stop, 'time_s': stop/rr.FS, 'event': ev,
            'clean2': cur_clean,
            'clean3': bool(stop >= 3*rr.FS and np.all(fog[stop-int(3*rr.FS):stop] == 0)),
            'clean5': bool(stop >= 5*rr.FS and np.all(fog[stop-int(5*rr.FS):stop] == 0)),
            'clean10': bool(stop >= 10*rr.FS and np.all(fog[stop-int(10*rr.FS):stop] == 0)),
            'theta2': t2, 'beta2': b2, 'move2': m2, 'movesd2': ms2,
            'theta3': t3, 'beta3': b3, 'move3': m3, 'movesd3': ms3,
            'theta5': t5, 'beta5': b5, 'move5': m5, 'movesd5': ms5,
            'theta10': t10, 'beta10': b10, 'move10': m10, 'movesd10': ms10,
            'nonfog_duration_s': nonfog_dur,
            'log2_nonfog_duration': np.log2(max(nonfog_dur, 1.0)),
            'log_time': np.log1p(stop/rr.FS),
        })
    meta = {'recording': rid, 'subject': subj, 'session': session, 'task': task,
            'duration_s': len(fog)/rr.FS, 'n_validated_onsets': len(onsets)}
    return rows, meta


def add_within_recording_z(df):
    cols = ['theta2','beta2','move2','movesd2','theta3','beta3','move3','movesd3',
            'theta5','beta5','move5','movesd5','theta10','beta10','move10','movesd10']
    out = df.copy()
    for c in cols:
        out[c+'_z'] = out.groupby('recording')[c].transform(lambda s: rr.robust_z(s.to_numpy()))
    return out


def risk_for_history(df, sec):
    d = df.loc[df[f'clean{sec}']].copy()
    # Retain event rows only when they correspond to a validated onset after the clean interval.
    return d.reset_index(drop=True)


def formulas(sec=5):
    base = 'event ~ C(subject) + C(task) + log2_nonfog_duration + log_time + move2_z + movesd2_z + move%s_z + movesd%s_z' % (sec, sec)
    inst = base + ' + theta2_z + beta2_z'
    hist = inst + ' + theta%s_z + beta%s_z' % (sec, sec)
    return base, inst, hist


def fit_inference(d, sec=5):
    f0, f1, f2 = formulas(sec)
    m0 = smf.glm(f0, data=d, family=sm.families.Binomial()).fit()
    m1 = smf.glm(f1, data=d, family=sm.families.Binomial()).fit()
    m2 = smf.glm(f2, data=d, family=sm.families.Binomial()).fit()
    lr01 = 2*(m1.llf-m0.llf); df01 = m1.df_model-m0.df_model
    lr12 = 2*(m2.llf-m1.llf); df12 = m2.df_model-m1.df_model
    robust = smf.glm(f2, data=d, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': d['subject']})
    coef_rows = []
    for term in ['theta2_z','beta2_z',f'theta{sec}_z',f'beta{sec}_z','move2_z','movesd2_z',f'move{sec}_z',f'movesd{sec}_z','log2_nonfog_duration']:
        if term not in robust.params:
            continue
        b = float(robust.params[term]); se = float(robust.bse[term]); p = float(robust.pvalues[term])
        coef_rows.append({'term':term,'beta':b,'se_cluster':se,'odds_ratio':math.exp(b),
                          'or_ci_low':math.exp(b-1.96*se),'or_ci_high':math.exp(b+1.96*se),'p_cluster':p})
    tests = pd.DataFrame([
        {'comparison':'movement+baseline -> + instantaneous theta/beta','lr':lr01,'df':df01,'p_lr':chi2.sf(lr01,df01),
         'll_base':m0.llf,'ll_full':m1.llf},
        {'comparison':f'instantaneous -> + {sec}s cortical history','lr':lr12,'df':df12,'p_lr':chi2.sf(lr12,df12),
         'll_base':m1.llf,'ll_full':m2.llf},
    ])
    return tests, pd.DataFrame(coef_rows), (m0,m1,m2)


def design_for_cv(d, sec, level):
    base = ['log2_nonfog_duration','log_time','move2_z','movesd2_z',f'move{sec}_z',f'movesd{sec}_z']
    if level >= 1:
        base += ['theta2_z','beta2_z']
    if level >= 2:
        base += [f'theta{sec}_z',f'beta{sec}_z']
    X = d[base].copy()
    td = pd.get_dummies(d['task'], prefix='task', dtype=float)
    return pd.concat([X.reset_index(drop=True), td.reset_index(drop=True)], axis=1)


def loso_cv(d, sec=5):
    y = d.event.to_numpy(int)
    subjects = d.subject.to_numpy()
    rows=[]; pred_rows=[]
    Xs = {lvl: design_for_cv(d,sec,lvl) for lvl in [0,1,2]}
    for s in sorted(pd.unique(subjects)):
        te = subjects == s; tr = ~te
        if tr.sum()==0 or te.sum()==0 or len(np.unique(y[tr]))<2:
            continue
        p0 = np.repeat(np.clip(y[tr].mean(),1e-6,1-1e-6), te.sum())
        for name,lvl in [('baseline',None),('movement',0),('instantaneous',1),('history',2)]:
            if lvl is None:
                p=p0
            else:
                model=make_pipeline(StandardScaler(), LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000))
                model.fit(Xs[lvl].loc[tr],y[tr]); p=model.predict_proba(Xs[lvl].loc[te])[:,1]
            yt=y[te]
            auc=roc_auc_score(yt,p) if len(np.unique(yt))==2 else np.nan
            ap=average_precision_score(yt,p) if yt.sum()>0 else np.nan
            rows.append({'subject':s,'model':name,'n':int(te.sum()),'events':int(yt.sum()),
                         'logloss':log_loss(yt,p,labels=[0,1]),'brier':brier_score_loss(yt,p),'auc':auc,'ap':ap})
            for idx,pp in zip(np.flatnonzero(te),p):
                pred_rows.append({'row':int(idx),'subject':s,'model':name,'y':int(y[idx]),'p':float(pp)})
    cv=pd.DataFrame(rows); preds=pd.DataFrame(pred_rows)
    summ=cv.groupby('model').agg(n_subjects=('subject','nunique'),mean_logloss=('logloss','mean'),mean_brier=('brier','mean'),mean_auc=('auc','mean'),mean_ap=('ap','mean')).reset_index()
    pooled=[]
    for m,g in preds.groupby('model'):
        yy=g.y.to_numpy(); pp=g.p.to_numpy(); pooled.append({'model':m,'n':len(g),'events':int(yy.sum()),
            'logloss':log_loss(yy,pp,labels=[0,1]),'brier':brier_score_loss(yy,pp),
            'auc':roc_auc_score(yy,pp) if len(np.unique(yy))==2 else np.nan,
            'ap':average_precision_score(yy,pp) if yy.sum()>0 else np.nan})
    pooled=pd.DataFrame(pooled)
    paired=[]
    pivot=cv.pivot(index='subject',columns='model',values='logloss')
    for a,b in [('movement','instantaneous'),('instantaneous','history'),('movement','history')]:
        q=pivot[[a,b]].dropna()
        if len(q)>=5:
            stat,p=wilcoxon(q[a],q[b],alternative='two-sided')
            paired.append({'comparison':f'{a} vs {b}','n_subjects':len(q),'median_delta_logloss_b_minus_a':float(np.median(q[b]-q[a])),'p_wilcoxon':float(p)})
    return cv,summ,pooled,pd.DataFrame(paired),preds


def temporal_null_history(d, sec=5, nperm=199):
    # Circularly shift the long-history cortical predictors within each recording,
    # preserving their autocorrelation and leaving onset labels/current predictors fixed.
    _,_,m2 = fit_inference(d,sec)[2]
    f0,f1,f2=formulas(sec)
    m1=smf.glm(f1,data=d,family=sm.families.Binomial()).fit()
    obs=2*(m2.llf-m1.llf)
    vals=[]
    for _ in range(nperm):
        pdat=d.copy()
        for rid,idx in pdat.groupby('recording').groups.items():
            idx=np.asarray(list(idx),int); n=len(idx)
            if n<8: continue
            minsh=max(2,min(int(10/HORIZON_S),max(2,n//4)))
            if n<=2*minsh:
                sh=int(RNG.integers(1,n))
            else:
                sh=int(RNG.integers(minsh,n-minsh))
            for c in [f'theta{sec}_z',f'beta{sec}_z']:
                pdat.loc[idx,c]=np.roll(pdat.loc[idx,c].to_numpy(),sh)
        pm2=smf.glm(f2,data=pdat,family=sm.families.Binomial()).fit()
        pm1=smf.glm(f1,data=pdat,family=sm.families.Binomial()).fit()
        vals.append(2*(pm2.llf-pm1.llf))
    vals=np.asarray(vals,float)
    p=(1+np.sum(vals>=obs))/(len(vals)+1)
    return pd.DataFrame([{'history_seconds':sec,'observed_lr':obs,'nperm':len(vals),'p_circular':p,
                          'null_median_lr':float(np.median(vals)),'null_95pct_lr':float(np.quantile(vals,.95))}]), vals


def make_trajectory_plot(df,out):
    # Descriptive trajectory in the 10 s before validated onsets, using only rows whose
    # next confirmed onset is within 10 s and no FoG is present in the current 2 s window.
    pieces=[]
    for rid,g in df.sort_values('stop_sample').groupby('recording'):
        evpos=g.loc[g.event==1,'stop_sample'].to_numpy()
        if len(evpos)==0: continue
        for ostop in evpos:
            h=g[(g.stop_sample<=ostop)&(g.stop_sample>=ostop-int(10*rr.FS))&g.clean2].copy()
            if len(h)==0: continue
            h['lead_s']=(h.stop_sample-ostop)/rr.FS
            pieces.append(h[['recording','lead_s','theta2_z','beta2_z']])
    if not pieces: return
    q=pd.concat(pieces,ignore_index=True)
    agg=q.groupby('lead_s').agg(theta=('theta2_z','median'),beta_low=('beta2_z','median')).reset_index()
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(agg.lead_s,agg.theta,label='theta')
    ax.plot(agg.lead_s,agg.beta_low,label='beta-low')
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set_xlabel('Time relative to pre-onset risk window (s)')
    ax.set_ylabel('Within-recording robust z amplitude')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'pre_onset_trajectory.png',dpi=220); plt.close(fig)
    q.to_csv(out/'pre_onset_trajectory_rows.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--nperm',type=int,default=199); args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    fps=sorted(fp for fp in root.rglob('task_*.txt') if fp.stem in {'task_1','task_2','task_3','task_4'})
    allrows=[]; metas=[]
    for i,fp in enumerate(fps,1):
        print(f'[continuous {i}/{len(fps)}] {fp}',flush=True)
        rows,meta=recording_rows(fp,root); allrows.extend(rows); metas.append(meta)
    df=add_within_recording_z(pd.DataFrame(allrows)); meta=pd.DataFrame(metas)
    df.to_csv(out/'causal_windows.csv',index=False); meta.to_csv(out/'recording_onsets.csv',index=False)

    primary=risk_for_history(df,5)
    primary.to_csv(out/'risk_5s_clean.csv',index=False)
    tests,coefs,_=fit_inference(primary,5)
    tests.to_csv(out/'inference_primary_5s.csv',index=False); coefs.to_csv(out/'coefficients_primary_5s.csv',index=False)
    cv,cvs,pooled,paired,preds=loso_cv(primary,5)
    cv.to_csv(out/'loso_subject_cv.csv',index=False); cvs.to_csv(out/'loso_subject_cv_summary.csv',index=False)
    pooled.to_csv(out/'loso_pooled_metrics.csv',index=False); paired.to_csv(out/'loso_paired_tests.csv',index=False); preds.to_csv(out/'loso_predictions.csv',index=False)
    null,nullvals=temporal_null_history(primary,5,args.nperm)
    null.to_csv(out/'history_temporal_null.csv',index=False); pd.DataFrame({'null_lr':nullvals}).to_csv(out/'history_temporal_null_distribution.csv',index=False)

    sens=[]
    for sec in [3,10]:
        d=risk_for_history(df,sec)
        t,c,_=fit_inference(d,sec)
        cc,cs,pp,pa,_=loso_cv(d,sec)
        for _,r in t.iterrows(): sens.append({'history_seconds':sec,'n':len(d),'events':int(d.event.sum()),**r.to_dict()})
        pp2=pp.copy(); pp2['history_seconds']=sec; pp2.to_csv(out/f'loso_pooled_metrics_{sec}s.csv',index=False)
        c.to_csv(out/f'coefficients_{sec}s.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'history_window_sensitivity.csv',index=False)
    make_trajectory_plot(df,out)

    onset_by_task=meta.groupby('task').agg(recordings=('recording','nunique'),validated_onsets=('n_validated_onsets','sum')).reset_index()
    onset_by_subject=meta.groupby('subject').agg(recordings=('recording','nunique'),validated_onsets=('n_validated_onsets','sum')).reset_index()
    onset_by_task.to_csv(out/'onsets_by_task.csv',index=False); onset_by_subject.to_csv(out/'onsets_by_subject.csv',index=False)

    summary={
        'n_subjects':int(meta.subject.nunique()),'n_recordings':int(meta.recording.nunique()),
        'validated_onsets_total':int(meta.n_validated_onsets.sum()),'primary_risk_rows':int(len(primary)),
        'primary_events':int(primary.event.sum()),'onsets_by_task':onset_by_task.to_dict(orient='records'),
        'primary_lr_tests':tests.to_dict(orient='records'),'primary_coefficients':coefs.to_dict(orient='records'),
        'loso_summary':cvs.to_dict(orient='records'),'loso_pooled':pooled.to_dict(orient='records'),
        'loso_paired':paired.to_dict(orient='records'),'history_temporal_null':null.to_dict(orient='records')}
    (out/'continuous_onset_summary.json').write_text(json.dumps(summary,indent=2,default=float))
    lines=['# Continuous cortical dynamics and FoG onset','',
           f"Subjects: {summary['n_subjects']}; recordings: {summary['n_recordings']}; validated sample-level onsets: {summary['validated_onsets_total']}.",
           f"Primary 5-s clean non-FoG risk set: {summary['primary_risk_rows']} windows, {summary['primary_events']} next-0.5-s onset events.",'',
           '## Nested onset models',tests.to_string(index=False),'','## Cluster-robust coefficients (full 5-s model)',coefs.to_string(index=False),'',
           '## Leave-one-subject-out CV',cvs.to_string(index=False),'','## Pooled out-of-fold metrics',pooled.to_string(index=False),'',
           '## Paired subject-level log-loss tests',paired.to_string(index=False) if len(paired) else 'Not estimable','',
           '## Temporally structured null for added 5-s cortical history',null.to_string(index=False),'',
           '## Onsets by task',onset_by_task.to_string(index=False)]
    (out/'REPORT_CONTINUOUS.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    main()
