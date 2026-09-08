            'beta_mid':float(np.median(env['beta_mid'][start:stop])),
            'beta_high':float(np.median(env['beta_high'][start:stop])),
            'beta_amp':float(np.median(beta_env[start:stop])),
            'move_mean':float(np.mean(move[start:stop])),
            'move_sd':float(np.std(move[start:stop])),
            'burst_frac':float(np.mean(burst_mask[start:stop])),
            'burst_rate':float(np.sum((burst_starts>=start)&(burst_starts<stop))/(WIN_S/60.0)),
        }
        rows.append(row)
    return rows, {'recording':rid,'subject':subj,'session':session,'task':task,'n_samples':n,'duration_s':n/FS,'fog_frac_sample':float(fog.mean())}

def add_recording_z(df):
    cols=['theta_amp','beta_low','beta_mid','beta_high','beta_amp','move_mean','move_sd','burst_frac','burst_rate']
    for c in cols:
        df[c+'_z']=df.groupby('recording')[c].transform(lambda s: robust_z(s.to_numpy()))
    return df

def circular_shift_test(g, col, nperm=499):
    d=g.dropna(subset=['fog',col]).sort_values('start_sample')
    y=d.fog.to_numpy(int); x=d[col].to_numpy(float); n=len(y)
    if n<40 or len(np.unique(y))<2: return None
    obs=np.median(x[y==1])-np.median(x[y==0])
    min_shift=max(4,int(10/STRIDE_S))
    allowed=np.arange(min_shift,n-min_shift) if n>2*min_shift+2 else np.arange(1,n)
    if len(allowed)==0:return None
    vals=[]
    for sh in RNG.choice(allowed,size=nperm,replace=True):
        yp=np.roll(y,int(sh))
        vals.append(np.median(x[yp==1])-np.median(x[yp==0]))
    vals=np.asarray(vals)
    p=(1+np.sum(np.abs(vals)>=abs(obs)))/(len(vals)+1)
    return obs,p

def fit_clusters(df, k=3):
    feats=['theta_amp_z','beta_low_z','beta_mid_z','beta_high_z','move_sd_z']
    X=df[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
    scaler=StandardScaler().fit(X); Xs=scaler.transform(X)
    km=KMeans(n_clusters=k,random_state=20260908,n_init=20).fit(Xs)
    lab=km.labels_
    df=df.copy(); df['cluster_raw']=lab
    # Stable display labels: order clusters by median first PC-like beta/theta feature score, independent of FoG labels
    cent=km.cluster_centers_
    order=np.argsort(np.linalg.norm(cent,axis=1))
    remap={old:new for new,old in enumerate(order)}
    df['state']=pd.Series(lab,index=df.index).map(remap).astype(int)
    return df, km, scaler, feats, float(silhouette_score(Xs,lab,sample_size=min(10000,len(df)),random_state=1))

def add_runs_tau(df):
    pieces=[]; runs=[]
    run_counter=0
    for rid,g in df.sort_values(['recording','start_sample']).groupby('recording',sort=False):
        g=g.copy().reset_index(drop=False)
        change=g['state'].ne(g['state'].shift())
        local=change.cumsum()
        tau=g.groupby(local).cumcount()+1
        g['tau']=tau.to_numpy()
        for lr,rg in g.groupby(local,sort=False):
            run_counter+=1
            idx=rg['index'].to_numpy()
            runs.append({'run_id':run_counter,'recording':rid,'subject':rg.subject.iloc[0],'task':rg.task.iloc[0],
                         'state':int(rg.state.iloc[0]),'n_windows':len(rg),'duration_s':WIN_S+(len(rg)-1)*STRIDE_S,
                         'fog_window_mean':float(np.nanmean(rg.fog)) if np.any(np.isfinite(rg.fog)) else np.nan,
                         'max_time_s':float(rg.time_s.max())})
            g.loc[rg.index,'run_id']=run_counter
        pieces.append(g.set_index('index'))
    out=pd.concat(pieces).sort_index(); out['run_id']=out['run_id'].astype(int)
    return out, pd.DataFrame(runs)

def build_onset_risk(df):
    rows=[]
    # each state run is independent and is already recording-bounded
    for run_id,g in df.sort_values(['recording','start_sample']).groupby('run_id'):
        g=g.sort_values('start_sample').copy()
        labels=g['fog'].to_numpy(float)
        # exclude runs that start in definite FoG; they are not at risk for onset
        first_def=np.flatnonzero(np.isfinite(labels))
        if len(first_def)==0 or labels[first_def[0]]==1: continue
        fog_pos=np.flatnonzero(labels==1)
        onset_pos=fog_pos[0] if len(fog_pos) else None
        eligible=np.flatnonzero(labels==0)
        if onset_pos is not None:
            eligible=eligible[eligible<onset_pos]
        if len(eligible)==0: continue
        for j,pos in enumerate(eligible):
            event=1 if (onset_pos is not None and j==len(eligible)-1) else 0
            r=g.iloc[pos]
            rows.append({'run_id':int(run_id),'recording':r.recording,'subject':r.subject,'task':r.task,
                         'state':int(r.state),'tau':int(r.tau),'time_from_state_s':(int(r.tau)-1)*STRIDE_S,
                         'event':event})
    risk=pd.DataFrame(rows)
    if len(risk): risk['log2_tau']=np.log2(risk['tau'].astype(float))
    return risk

def survival_models(risk):
    if len(risk)==0 or risk.event.sum()<3: return pd.DataFrame(),{}
