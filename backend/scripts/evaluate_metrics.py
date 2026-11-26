"""Held-out evaluation. Writes backend/artifacts/metrics.json (the single metrics source for the UI).

Only temperature is refit here, on validation records. Nothing is tuned on test records.
Run: backend/.venv/bin/python backend/scripts/evaluate_metrics.py [output.json]
"""
import sys, json, csv, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, wfdb, tensorflow as tf, keras, sklearn, scipy
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score)
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'backend'))
from app.ml.inference import ECGClassifier, sha256_file
from app.ml.preprocessing import *
from app.ml.preprocessing import filter_continuous, extract_windows
from app.ml.rpeak_detection import detect_r_peaks, match_peaks
from app.core.config import PREPROCESSING_VERSION
cfg_path=ROOT/'backend/models/preprocessing_config.json'; cfg=json.loads(cfg_path.read_text())
C=['N','S','V','F']; I={c:i for i,c in enumerate(C)}; M=ROOT/'results/mitdb'; amap=cfg['aami_mapping']; sp=cfg['split']
clf=ECGClassifier(); model=clf.model
def load(rec, as_app=False):
    r=wfdb.rdrecord(str(M/rec)); ch=r.sig_name.index('MLII') if 'MLII' in r.sig_name else 0
    sig=r.p_signal[:,ch]
    if as_app: sig=sig.astype(np.float32)
    a=wfdb.rdann(str(M/rec),'atr')
    pk=np.array([p for p,s in zip(a.sample,a.symbol) if s in amap],np.int64)
    lab=np.array([I[amap[s]] for s in a.symbol if s in amap],np.int64)
    return sig,pk,lab
def build(recs, as_app=False):
    W,Y,R,H,A=[],[],[],[],{}
    for rec in recs:
        sig,pk,lab=load(rec,as_app); wb=extract_windows(filter_continuous(sig),pk); keep=np.isin(pk,wb.sample_indices)
        W.append(wb.windows);Y.append(lab[keep]);R+=[rec]*int(keep.sum()); A[rec]=np.bincount(lab,minlength=4)
        H+= [hashlib.md5(w.tobytes()).hexdigest() for w in wb.windows]
    return np.concatenate(W),np.concatenate(Y),np.array(R),H,A
data={n:build(sp[k]) for n,k in (('train','train_records'),('validation','validation_records'),('test','test_records'))}
hs={n:set(d[3]) for n,d in data.items()}
dup={f'{a}|{b}':len(hs[a]&hs[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))}
print('exact duplicate windows across splits',dup)
grp=lambda r: sp['subject_groups_merged'].get(r,r)
pats={n:sorted({grp(r) for r in sp[k]}) for n,k in (('train','train_records'),('validation','validation_records'),('test','test_records'))}
assert not set(pats['train'])&set(pats['validation']) and not set(pats['train'])&set(pats['test']) and not set(pats['validation'])&set(pats['test'])
probs={n:np.asarray(model.predict(data[n][0],batch_size=256,verbose=0),np.float64) for n in ('validation','test')}

def thin(*arrs, n=400):
    L=len(arrs[0])
    idx=np.unique(np.linspace(0,L-1,min(n,L)).round().astype(int))
    return [[round(float(v),6) for v in a[idx]] for a in arrs]
def evaluate(y,P,curves):
    p=P.argmax(1); cm=confusion_matrix(y,p,labels=range(4)); sup=np.bincount(y,minlength=4)
    assert (cm.sum(1)==sup).all() and cm.sum()==len(y)
    pr,rc,f1,s=precision_recall_fscore_support(y,p,labels=range(4),zero_division=0)
    per={}
    for i,c in enumerate(C):
        tn=cm.sum()-cm[i].sum()-cm[:,i].sum()+cm[i,i]; fp=cm[:,i].sum()-cm[i,i]
        per[c]={'precision':float(pr[i]),'recall':float(rc[i]),'specificity':float(tn/(tn+fp)),'f1':float(f1[i]),'support':int(s[i]),
                'roc_auc':float(roc_auc_score(y==i,P[:,i])),'pr_auc':float(average_precision_score(y==i,P[:,i])),'prevalence':float(sup[i]/len(y))}
    avg=lambda k,w=None: float(np.average([per[c][k] for c in C],weights=sup if w else None))
    out={'n_windows':int(len(y)),'accuracy':float(accuracy_score(y,p)),'majority_class_baseline_accuracy':float(sup.max()/len(y)),
      'balanced_accuracy':float(balanced_accuracy_score(y,p)),'mcc':float(matthews_corrcoef(y,p)),'cohen_kappa':float(cohen_kappa_score(y,p)),
      'macro':{k:avg(k) for k in ('precision','recall','f1','roc_auc','pr_auc')},
      'weighted':{k:avg(k,True) for k in ('precision','recall','f1','roc_auc','pr_auc')},
      'per_class':per,'confusion_matrix':{'labels':C,'rows':'true','columns':'predicted','counts':cm.tolist()},
      'checks':{'confusion_row_sums_equal_support':True,'row_sums':cm.sum(1).tolist(),'support':sup.tolist()}}
    return out
def curves(y,P):
    roc,prc={},{}
    for i,c in enumerate(C):
        f,t,_=roc_curve(y==i,P[:,i]); f2,t2=thin(f,t); roc[c]={'fpr':f2,'tpr':t2}
        pre,rec,_=precision_recall_curve(y==i,P[:,i]); r2,p2=thin(rec[::-1],pre[::-1]); prc[c]={'recall':r2,'precision':p2}
    return roc,prc
res={}
for n in ('validation','test'):
    res[n]=evaluate(data[n][1],probs[n],False)
    res[n]['roc_curves'],res[n]['pr_curves']=curves(data[n][1],probs[n])
    res[n]['role']='used for model selection and temperature/threshold fitting' if n=='validation' else 'held-out final evaluation'

# calibration
logit_model=keras.Model(model.inputs,model.get_layer('logits').output)
Lg={n:np.asarray(logit_model.predict(data[n][0],batch_size=512,verbose=0),np.float64) for n in ('validation','test')}
nll=lambda lt,n: float(-log_softmax(Lg[n]/np.exp(lt),axis=1)[np.arange(len(data[n][1])),data[n][1]].mean())
T=float(np.exp(minimize_scalar(lambda t:nll(t,'validation'),bounds=(-3,3),method='bounded').x))
def ece(P,y,b=15):
    conf=P.max(1);ok=(P.argmax(1)==y).astype(float);e=0;bins=[]
    ed=np.linspace(0,1,b+1)
    for k in range(b):
        m=(conf>ed[k])&(conf<=ed[k+1])
        if m.any(): e+=m.mean()*abs(ok[m].mean()-conf[m].mean()); bins.append({'confidence':float(conf[m].mean()),'accuracy':float(ok[m].mean()),'count':int(m.sum())})
    return float(e),bins
def calib(P,y):
    e,b=ece(P,y); return {'ece_15':e,'brier':float(((P-np.eye(4)[y])**2).sum(1).mean()),'nll':float(-np.log(np.clip(P[np.arange(len(y)),y],1e-12,1)).mean()),'bins':b}
yt=data['test'][1]
calibration={'temperature_refit_on_validation':T,'test_raw':calib(probs['test'],yt),'test_temperature_scaled':calib(softmax(Lg['test']/T,axis=1),yt)}

# review-flag policy as the app applies it
pol=json.loads((ROOT/'results/selective_thresholds.json').read_text())
ops={}
for name,o in pol['operating_points'].items():
    ops[name]={'tau':o['tau'],'margin':o['margin']}
    for n in ('validation','test'):
        y=data[n][1];Z=np.log(np.clip(probs[n],1e-12,1))/pol['temperature'];Q=softmax(Z,axis=1);s=np.sort(Q,1)
        flag=(s[:,-1]<o['tau'])|(s[:,-1]-s[:,-2]<o['margin']);wrong=probs[n].argmax(1)!=y
        ops[name][n]={'review_rate':float(flag.mean()),'coverage':float(1-flag.mean()),'error_rate_on_unflagged':float(wrong[~flag].mean()),
          'fraction_of_all_errors_flagged':float((wrong&flag).sum()/wrong.sum()),
          'review_rate_by_true_class':{c:float(flag[y==i].mean()) for i,c in enumerate(C)}}
    ops[name]['target_error_rate']=pol['targets_selective_risk'][name]

# bootstrap over records
rt=data['test'][2];pt=probs['test'].argmax(1);rng=np.random.default_rng(20240611)
cms={r:confusion_matrix(yt[rt==r],pt[rt==r],labels=range(4)) for r in sp['test_records']}
st={k:[] for k in ('accuracy','macro_f1','S_recall','V_recall','F_recall')}
for _ in range(2000):
    cm=sum(cms[r] for r in rng.choice(sp['test_records'],len(cms),replace=True)).astype(float)
    tp=np.diag(cm)
    with np.errstate(all='ignore'):
        rc=tp/cm.sum(1); f=2*tp/(cm.sum(1)+cm.sum(0))
    st['accuracy'].append(tp.sum()/cm.sum());st['macro_f1'].append(np.nanmean(f[cm.sum(1)>0]))
    for c in 'SVF': st[c+'_recall'].append(rc[I[c]])
boot={'unit':'record','resamples':2000,'seed':20240611,'ci95':{k:[float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))] for k,v in st.items()}}
per_record=[]
for r in sp['test_records']:
    m=rt==r; cm=cms[r]
    per_record.append({'record':r,'n':int(m.sum()),'accuracy':float((pt[m]==yt[m]).mean()),'support':cm.sum(1).tolist(),'true_positive':np.diag(cm).tolist()})

# end-to-end with detector: as app (filtered input) and as old script (raw input)
def e2e(on_filtered):
    cm=np.zeros((4,4),int);ref=np.zeros(4,int);extra=0;det_n=0
    for rec in sp['test_records']:
        sig,pk,lab=load(rec,as_app=True);filt=filter_continuous(sig)
        det=detect_r_peaks(filt if on_filtered else sig);det_n+=len(det)
        mt,ud,_=match_peaks(pk,det,54);extra+=len(ud);ref+=np.bincount(lab,minlength=4)
        L=dict(zip(pk.tolist(),lab.tolist()));d=np.array([a for a,b in mt],np.int64);ml=np.array([L[b] for a,b in mt])
        o=np.argsort(d);d,ml=d[o],ml[o]
        wb=extract_windows(filt,d);keep=np.isin(d,wb.sample_indices)
        p=model.predict(wb.windows,batch_size=512,verbose=0).argmax(1);np.add.at(cm,(ml[keep],p),1)
    tp=np.diag(cm);rc=tp/cm.sum(1);prm=tp/np.maximum(cm.sum(0),1)
    return {'detector_input':'bandpass-filtered (same as the app)' if on_filtered else 'raw (old evaluate_extended.py)','annotated_beats':int(ref.sum()),'detected_peaks':int(det_n),
      'matched_and_windowed':int(cm.sum()),'missed_or_unwindowed':int(ref.sum()-cm.sum()),'extra_detections':int(extra),
      'matched_accuracy':float(tp.sum()/cm.sum()),'matched_macro_f1':float(np.mean(2*tp/(cm.sum(1)+cm.sum(0)))),
      'recall_counting_misses_as_errors':{c:float(tp[i]/ref[i]) for i,c in enumerate(C)},'confusion_matrix_matched':cm.tolist()}
end={'app_pipeline':e2e(True),'raw_input_variant':e2e(False)}

# app storage path (float32) equivalence on test
Wa=build(sp['test_records'],as_app=True)[0]
Pa=model.predict(Wa,batch_size=256,verbose=0)
app_eq={'max_abs_window_diff':float(np.abs(Wa-data['test'][0]).max()),'argmax_agreement':float((Pa.argmax(1)==pt).mean())}
REF=ROOT/'results/test_examples_for_xai.npz'
if REF.exists():
    ref=np.load(ROOT/'results/test_examples_for_xai.npz')
    Xr=ref['X'].astype(np.float32);Xr=Xr if Xr.ndim==3 else Xr[...,None]
    eqv={'reference':'results/test_examples_for_xai.npz (exported by training run)','n':int(len(Xr)),'max_abs_window_diff':float(np.abs(Xr-data['test'][0]).max()),
         'max_abs_probability_diff':float(np.abs(ref['prob']-probs['test']).max()),'argmax_agreement':float((ref['pred']==pt).mean()),'app_float32_path':app_eq}
else:
    eqv={'status':'UNVERIFIED','reason':REF.name+' not present (gitignored, >100MB)','app_float32_path':app_eq}

# dataset facts
classfacts={}
for n in ('train','validation','test'):
    ann=sum(data[n][4].values()); win=np.bincount(data[n][1],minlength=4)
    classfacts[n]={'annotated_beats':{c:int(ann[i]) for i,c in enumerate(C)},'usable_windows':{c:int(win[i]) for i,c in enumerate(C)}}
overlap={}
rrs=[]
for k in ('train_records','validation_records','test_records'):
    for rec in sp[k]:
        _,pk,_=load(rec);rrs.append(np.diff(pk))
rr=np.concatenate(rrs); overlap={'window_samples':720,'median_rr_samples':float(np.median(rr)),'median_adjacent_window_overlap_fraction':float(np.median(np.clip(720-rr,0,720)/720))}
layers=[]
for l in model.layers:
    c=l.get_config();d={'name':l.name,'type':l.__class__.__name__,'output_shape':list(l.output.shape),'params':int(l.count_params())}
    for k in ('filters','kernel_size','units','rate','pool_size','activation','padding','merge_mode'):
        if k in c: d[k]=c[k]
    if d['type']=='Bidirectional': d['wrapped']={'type':c['layer']['class_name'],'units':c['layer']['config']['units']}
    layers.append(d)
hist=list(csv.DictReader((ROOT/'results/hist_bilstm_aug_sqrtw.csv').open()))
tm=json.loads((ROOT/'results/test_metrics.json').read_text())
art={'schema_version':1,'evaluated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
 'environment':{'python':platform.python_version(),'tensorflow':tf.__version__,'keras':keras.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__},
 'model':{'version':cfg['selected_model'],'description':cfg['model_description'],'sha256':clf.sha256,'config_sha256':sha256_file(cfg_path),
   'input_shape':list(model.inputs[0].shape),'output_shape':list(model.outputs[0].shape),'parameter_count':int(model.count_params()),'layers':layers,'classes':C},
 'preprocessing':{'version':PREPROCESSING_VERSION,'sampling_rate_hz':cfg['sampling_frequency'],'window_samples':cfg['window_size'],'window_alignment':cfg['window_alignment'],
   'bandpass':cfg['bandpass_filter'],'normalization':cfg['normalization'],'epsilon':cfg['normalization_epsilon'],'lead':cfg['lead'],'overlap':overlap,'aami_mapping':amap},
 'training':{'loss_as_recorded_in_config':cfg['model_description'][1],'imbalance_method_as_recorded':{'class_weighting':'sqrt-weighted cross-entropy','augmentation':cfg['augmentation_training_only']},
   'class_weight_values':None,'verification':'UNVERIFIED: training code is not in the repository','decision_rule':cfg['decision_rule'],
   'selection_rule':cfg['selection_rule'],'validation_macro_f1_by_variant':{k:v['macro_f1'] for k,v in tm['all_models_validation_metrics'].items()},
   'history_source':'results/hist_bilstm_aug_sqrtw.csv','history':[{k:(int(r[k]) if k=='epoch' else float(r[k])) for k in ('epoch','accuracy','loss','val_accuracy','val_loss','val_macro_f1','learning_rate')} for r in hist]},
 'split':{'type':'fixed inter-patient (record groups)','records':{'train':sp['train_records'],'validation':sp['validation_records'],'test':sp['test_records']},
   'patient_group_ids':pats,'subject_groups_merged':sp['subject_groups_merged'],'excluded_paced_records':sp['excluded_paced_records'],
   'disjoint_assertion_passed':True,'exact_duplicate_windows_across_splits':dup},
 'dataset':{'name':'MIT-BIH Arrhythmia Database','records_total':len(list(M.glob('*.hea'))),'records_used':44,'subject_groups_used':len(set(sum(pats.values(),[]))),'class_counts':classfacts},
 'validation':res['validation'],'test':res['test'],'test_per_record':per_record,'test_bootstrap':boot,'calibration':calibration,
 'review_flag_policy':{'temperature':pol['temperature'],'app_uses':'balanced','operating_points':ops},
 'end_to_end_detector':end,'pipeline_equivalence':eqv}
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'backend/artifacts/metrics.json'
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(art,indent=1))
for n in ('validation','test'):
    r=res[n];print('==',n,r['n_windows'],'acc',round(r['accuracy'],4),'majority',round(r['majority_class_baseline_accuracy'],4),'bal',round(r['balanced_accuracy'],4),'mcc',round(r['mcc'],4),'kappa',round(r['cohen_kappa'],4))
    for k in ('macro','weighted'): print(k,{a:round(b,4) for a,b in r[k].items()})
    for c in C: print(c,{a:(round(b,4) if isinstance(b,float) else b) for a,b in r['per_class'][c].items()})
    print('cm',r['confusion_matrix']['counts'],'rows==support PASS')
print('boot',boot['ci95'])
print('calib T',T,{k:{a:round(b,4) for a,b in v.items() if a!='bins'} for k,v in calibration.items() if k!='temperature_refit_on_validation'})
print('ops balanced',json.dumps(ops['balanced']))
for k,v in end.items(): print(k,json.dumps({a:b for a,b in v.items() if a!='confusion_matrix_matched'}))
print('equiv',json.dumps(eqv)); print('overlap',overlap); print('dup',dup)
