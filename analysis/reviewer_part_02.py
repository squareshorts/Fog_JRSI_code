    # subject FE; cluster-robust by subject if possible
    f0='event ~ C(state) + C(subject)'
    f1='event ~ log2_tau + C(state) + C(subject)'
    m0=smf.glm(f0,data=risk,family=sm.families.Binomial()).fit()
    m1=smf.glm(f1,data=risk,family=sm.families.Binomial()).fit()
    try:
        m1r=smf.glm(f1,data=risk,family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': risk['subject']})
    except Exception:
        m1r=m1
    names=list(m1.params.index)
    ix=names.index('log2_tau')
    beta=float(m1.params['log2_tau'])
    se=float(np.asarray(m1r.bse)[ix]) if hasattr(m1r,'bse') else float(m1.bse['log2_tau'])
    p=float(np.asarray(m1r.pvalues)[ix]) if hasattr(m1r,'pvalues') else float(m1.pvalues['log2_tau'])
    lr=2*(m1.llf-m0.llf); lr_p=float(chi2.sf(lr,1))
    tab=pd.DataFrame([{'term':'log2_tau','beta':beta,'odds_ratio':math.exp(beta),'se_cluster':se,
                      'ci_low':beta-1.96*se,'ci_high':beta+1.96*se,'or_ci_low':math.exp(beta-1.96*se),
                      'or_ci_high':math.exp(beta+1.96*se),'p_cluster':p,'lr_vs_state_subject':lr,'lr_p':lr_p,
                      'n_risk_rows':len(risk),'n_onsets':int(risk.event.sum()),'n_runs':int(risk.run_id.nunique())}])
    # task-specific slopes where estimable
    taskrows=[]
    for task,g in risk.groupby('task'):
        if g.event.sum()<3 or g.subject.nunique()<3: continue
        try:
            mt=smf.glm('event ~ log2_tau + C(state) + C(subject)',data=g,family=sm.families.Binomial()).fit()
            taskrows.append({'task':task,'beta_log2_tau':float(mt.params['log2_tau']),'or_per_doubling':float(np.exp(mt.params['log2_tau'])),
                             'p':float(mt.pvalues['log2_tau']),'n':len(g),'events':int(g.event.sum())})
        except Exception: pass
    return tab, {'task_slopes':taskrows}

def cv_compare(df):
    d=df.dropna(subset=['fog']).copy(); d['fog']=d.fog.astype(int)
    model_features={
      'continuous':['theta_amp_z','beta_low_z','beta_mid_z','beta_high_z','move_sd_z'],
      'burst':['burst_frac_z','burst_rate_z'],
      'state':['state'],
      'state_plus_tau':['state','log2_tau_cv'],
      'continuous_plus_tau':['theta_amp_z','beta_low_z','beta_mid_z','beta_high_z','move_sd_z','log2_tau_cv']}
    d['log2_tau_cv']=np.log2(d.tau.astype(float))
    rows=[]
    for rid,test in d.groupby('recording'):
        train=d[d.recording!=rid]
        if test.fog.nunique()<2 or train.fog.nunique()<2: continue
        p0=np.clip(train.fog.mean(),1e-5,1-1e-5); y=test.fog.to_numpy()
        rows.append({'recording':rid,'model':'intercept','logloss':log_loss(y,np.full(len(y),p0),labels=[0,1]),'brier':brier_score_loss(y,np.full(len(y),p0)),'auc':.5,'n':len(y)})
        for name,feats in model_features.items():
            tr=train[feats].copy(); te=test[feats].copy()
            if 'state' in feats:
                cats=[0,1,2]
                tr=pd.get_dummies(tr,columns=['state'],prefix='state')
                te=pd.get_dummies(te,columns=['state'],prefix='state')
                basecols=[c for c in feats if c!='state']
                cols=basecols+[f'state_{c}' for c in cats]
                tr=tr.reindex(columns=cols,fill_value=0)
                te=te.reindex(columns=cols,fill_value=0)
            Xtr=tr.to_numpy(float); Xte=te.to_numpy(float)
            sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
            try:
                clf=LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs').fit(Xtr,train.fog)
                p=clf.predict_proba(Xte)[:,1]; p=np.clip(p,1e-6,1-1e-6)
                rows.append({'recording':rid,'model':name,'logloss':log_loss(y,p,labels=[0,1]),'brier':brier_score_loss(y,p),
                             'auc':roc_auc_score(y,p),'n':len(y)})
            except Exception: pass
    res=pd.DataFrame(rows)
    if len(res):
        base=res[res.model=='intercept'][['recording','logloss']].rename(columns={'logloss':'base_logloss'})
        res=res.merge(base,on='recording',how='left'); res['delta_logloss_vs_intercept']=res.logloss-res.base_logloss
    return res

def artifact_sensitivity(df):
    rows=[]
    for rid,g in df.groupby('recording'):
        pure=g.dropna(subset=['fog'])
        if len(pure)<20 or pure.fog.nunique()<2: continue
        for c in ['theta_amp','beta_low','beta_mid','beta_high']:
            rho=spearmanr(g[c],g['move_sd'],nan_policy='omit').correlation
            cutoff=g.move_sd.quantile(.9); lo=pure[pure.move_sd<=cutoff]
            full=np.median(pure.loc[pure.fog==1,c])-np.median(pure.loc[pure.fog==0,c])
            trim=np.nan
            if lo.fog.nunique()==2: trim=np.median(lo.loc[lo.fog==1,c])-np.median(lo.loc[lo.fog==0,c])
            rows.append({'recording':rid,'subject':g.subject.iloc[0],'band':c,'rho_amp_move':rho,'delta_full':full,'delta_excl_top10_move':trim,
                         'same_sign':bool(np.sign(full)==np.sign(trim)) if np.isfinite(trim) else np.nan})
    return pd.DataFrame(rows)

def make_figures(out, df, runs, risk, cv):
    # corrected dwell survival
    fig,ax=plt.subplots(figsize=(6,4))
    for s,g in runs.groupby('state'):
        d=np.sort(g.n_windows.to_numpy()); surv=1-np.arange(len(d))/len(d)
        ax.step(d,surv,where='post',label=f'State {s}')
    ax.set_yscale('log'); ax.set_xlabel('Dwell (windows; recording-bounded)'); ax.set_ylabel('Survival'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'corrected_dwell_survival.png',dpi=220); plt.close(fig)
    if len(risk):
        rr=risk.copy(); rr['tau_bin']=pd.qcut(rr.tau, q=min(8,max(2,rr.tau.nunique())), duplicates='drop')
        q=rr.groupby(['state','tau_bin'],observed=True).agg(tau=('tau','median'),event_rate=('event','mean'),n=('event','size')).reset_index()
