        fig,ax=plt.subplots(figsize=(6,4))
        for s,g in q.groupby('state'): ax.plot(g.tau,g.event_rate,'o-',label=f'State {s}')
        ax.set_xscale('log'); ax.set_xlabel('Time in state (windows)'); ax.set_ylabel('Discrete onset hazard'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/'true_onset_hazard.png',dpi=220); plt.close(fig)
    if len(cv):
        summ=cv.groupby('model').delta_logloss_vs_intercept.mean().sort_values()
        fig,ax=plt.subplots(figsize=(7,4)); summ.plot(kind='bar',ax=ax); ax.axhline(0,lw=.8,color='black'); ax.set_ylabel('Mean held-out Δ log loss vs intercept'); fig.tight_layout(); fig.savefig(out/'cv_model_comparison.png',dpi=220); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    allowed_tasks={'task_1','task_2','task_3','task_4'}
    fps=sorted(fp for fp in root.rglob('task_*.txt') if fp.stem in allowed_tasks)
    allrows=[]; recmeta=[]
    for i,fp in enumerate(fps,1):
        print(f'[{i}/{len(fps)}] {fp}',flush=True)
        rows,meta=window_rows(fp,root); allrows.extend(rows); recmeta.append(meta)
    df=pd.DataFrame(allrows); meta=pd.DataFrame(recmeta)
    df=add_recording_z(df)
    pure_counts=(df.groupby('recording')
        .agg(n_windows=('recording','size'), n_pure=('fog','count'),
             n_fog_pure=('fog',lambda x:int(np.nansum(np.asarray(x)==1))),
             n_nonfog_pure=('fog',lambda x:int(np.nansum(np.asarray(x)==0))))
        .reset_index())
    meta=meta.merge(pure_counts,on='recording',how='left')
    # circular-shift nulls before clustering
    nullrows=[]
    for rid,g in df.groupby('recording'):
        for c in ['theta_amp','beta_low','beta_mid','beta_high']:
            r=circular_shift_test(g,c)
            if r: nullrows.append({'recording':rid,'subject':g.subject.iloc[0],'task':g.task.iloc[0],'band':c,'delta':r[0],'p_circular':r[1]})
    null=pd.DataFrame(nullrows)
    # clustering and corrected runs
    df,km,sc,cluster_feats,silh=fit_clusters(df,3)
    df,runs=add_runs_tau(df)
    risk=build_onset_risk(df)
    survtab,extra=survival_models(risk)
    cv=cv_compare(df)
    art=artifact_sensitivity(df)
    # K sensitivity
    ks=[]
    baseX=df[['theta_amp_z','beta_low_z','beta_mid_z','beta_high_z','move_sd_z']].fillna(0).to_numpy(); baseX=StandardScaler().fit_transform(baseX)
    for k in [2,3,4,5]:
        kmk=KMeans(n_clusters=k,random_state=20260908,n_init=10).fit(baseX)
        sil=silhouette_score(baseX,kmk.labels_,sample_size=min(8000,len(df)),random_state=2)
        ks.append({'K':k,'silhouette':sil})
    # summaries
    state_summary=df.groupby('state').agg(n_windows=('state','size'),occupancy=('state','size'),fog_prob=('fog','mean'),median_tau=('tau','median'),max_tau=('tau','max')).reset_index()
    state_summary['occupancy']=state_summary.n_windows/len(df)
    run_summary=runs.groupby('state').agg(n_runs=('run_id','size'),median_dwell=('n_windows','median'),mean_dwell=('n_windows','mean'),max_dwell=('n_windows','max'),median_duration_s=('duration_s','median'),max_duration_s=('duration_s','max')).reset_index()
    boundary=pd.DataFrame([{'max_run_duration_s':runs.duration_s.max(),'max_recording_duration_s':meta.duration_s.max(),
                            'any_run_longer_than_own_recording':bool(any(r.duration_s > float(meta.loc[meta.recording==r.recording,'duration_s'].iloc[0])+1e-9 for _,r in runs.iterrows())),
                            'n_recordings':meta.recording.nunique(),'n_subjects':meta.subject.nunique(),'n_task_files':len(meta)}])
    null_summary=null.groupby('band').agg(n=('p_circular','size'),frac_sig=('p_circular',lambda x:np.mean(x<.05)),median_delta=('delta','median')).reset_index() if len(null) else pd.DataFrame()
    art_summary=art.groupby('band').agg(n=('recording','size'),median_rho_amp_move=('rho_amp_move','median'),same_sign_frac=('same_sign','mean'),median_delta_full=('delta_full','median'),median_delta_trim=('delta_excl_top10_move','median')).reset_index() if len(art) else pd.DataFrame()
    cv_summary=cv.groupby('model').agg(n_recordings=('recording','nunique'),mean_logloss=('logloss','mean'),mean_delta_logloss=('delta_logloss_vs_intercept','mean'),mean_auc=('auc','mean'),mean_brier=('brier','mean')).reset_index() if len(cv) else pd.DataFrame()
    subject_summary=(meta.groupby('subject').agg(n_recordings=('recording','size'),
        recordings_with_both=('recording',lambda x:int(np.sum([(meta.loc[meta.recording==r,'n_fog_pure'].iloc[0]>0 and meta.loc[meta.recording==r,'n_nonfog_pure'].iloc[0]>0) for r in x]))),
        total_fog_pure=('n_fog_pure','sum'),total_nonfog_pure=('n_nonfog_pure','sum')).reset_index())
    # save
    for name,obj in [('recordings',meta),('subject_analyzability',subject_summary),('windows',df),('runs_corrected',runs),('onset_risk',risk),('survival_model',survtab),('circular_shift_null',null),('circular_shift_summary',null_summary),('artifact_sensitivity',art),('artifact_summary',art_summary),('cv_recording_level',cv),('cv_summary',cv_summary),('state_summary',state_summary),('run_summary',run_summary),('boundary_audit',boundary),('K_sensitivity',pd.DataFrame(ks)),('task_onset_slopes',pd.DataFrame(extra.get('task_slopes',[])))]:
        obj.to_csv(out/f'{name}.csv',index=False)
    make_figures(out,df,runs,risk,cv)
    # machine-readable claim audit
    summary={'n_subjects':int(meta.subject.nunique()),'n_recordings':int(meta.recording.nunique()),'subject_ids':sorted(meta.subject.unique().tolist()),
             'n_pure_windows':int(df.fog.notna().sum()),'n_onsets':int(risk.event.sum()) if len(risk) else 0,'silhouette_K3':silh,
             'boundary_audit':boundary.iloc[0].to_dict(),
             'survival':survtab.iloc[0].to_dict() if len(survtab) else None,
             'state_summary':state_summary.to_dict(orient='records'),'run_summary':run_summary.to_dict(orient='records'),
             'circular_shift_summary':null_summary.to_dict(orient='records'),'artifact_summary':art_summary.to_dict(orient='records'),
             'cv_summary':cv_summary.to_dict(orient='records'),'task_onset_slopes':extra.get('task_slopes',[]),'K_sensitivity':ks}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x) if isinstance(x,(np.floating,np.integer)) else str(x)))
    lines=['# JRSI reviewer reanalysis','',f"Subjects discovered: {summary['n_subjects']} ({', '.join(summary['subject_ids'])})",f"Task files: {summary['n_recordings']}",
           f"Recording-boundary violation after correction: {summary['boundary_audit']['any_run_longer_than_own_recording']}",f"Onset events in survival risk set: {summary['n_onsets']}",'']
    if summary['survival']:
        s=summary['survival']; lines += [f"Discrete-time onset model: beta(log2 tau)={s['beta']:.4g}; OR per doubling={s['odds_ratio']:.4g} (95% CI {s['or_ci_low']:.4g}-{s['or_ci_high']:.4g}); cluster-robust p={s['p_cluster']:.4g}; LR p vs state+subject={s['lr_p']:.4g}.",'']
    lines += ['## Circular-shift null',null_summary.to_string(index=False) if len(null_summary) else 'No results','', '## Recording-level CV',cv_summary.to_string(index=False) if len(cv_summary) else 'No results','', '## Movement sensitivity',art_summary.to_string(index=False) if len(art_summary) else 'No results','', '## Corrected run summary',run_summary.to_string(index=False),'', '## Task-specific onset slopes',pd.DataFrame(extra.get('task_slopes',[])).to_string(index=False) if extra.get('task_slopes') else 'No estimable task-specific models']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)

if __name__=='__main__': main()
