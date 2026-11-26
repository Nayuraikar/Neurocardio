"""Split disjointness assertion, class counts and train/inference window equivalence.

Run: backend/.venv/bin/python backend/scripts/audit_data.py
"""
import sys, json, zipfile, tempfile
from pathlib import Path
from collections import Counter
import numpy as np, wfdb
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'backend'))
from app.ml.preprocessing import filter_continuous, extract_windows
cfg=json.loads((ROOT/'backend/models/preprocessing_config.json').read_text())
M=ROOT/'results/mitdb'; amap=cfg['aami_mapping']; C='NSVF'
split=cfg['split']; tr,va,te,ex=(split[k] for k in ('train_records','validation_records','test_records','excluded_paced_records'))
allrec=sorted({p.stem for p in M.glob('*.hea')})
grp=lambda r: split['subject_groups_merged'].get(r,r)
P={n:{grp(r) for r in s} for n,s in (('train',tr),('val',va),('test',te))}
for a,b in (('train','val'),('train','test'),('val','test')):
    assert not (P[a]&P[b]), (a,b,P[a]&P[b])
    r={'train':tr,'val':va,'test':te}
    assert not set(r[a])&set(r[b])
print('DISJOINT ASSERT: PASS (records and subject groups)')
for n,s in (('train',tr),('val',va),('test',te)): print(n,'records',len(s),'subjects',len(P[n]),sorted(P[n]))
print('records on disk',len(allrec),'used',len(tr+va+te),'subjects used',len(P['train']|P['val']|P['test']))
print('unassigned (not in split, not excluded):',sorted(set(allrec)-set(tr+va+te+ex)))
print('excluded',ex)
for name in ('split_selection.json',):
    s=json.loads((ROOT/'results'/name).read_text())
    for a,b in (('train_records',tr),('val_records',va),('test_records',te)): print(name,a,'matches config',s[a]==b)
for f,b in (('train_records',tr),('validation_records',va),('test_records',te)):
    print(f+'.npy matches config', list(np.load(ROOT/f'results/{f}.npy'))==b)
tot={}; fs=set(); leads={}; W={}
for n,s in (('train',tr),('val',va),('test',te),('excluded',ex)):
    ann_c=Counter(); win_c=Counter(); skipped=Counter(); Ws=[];Ys=[];Rs=[];Ps=[]
    for rec in s:
        r=wfdb.rdrecord(str(M/rec)); fs.add(r.fs); leads[rec]=r.sig_name
        a=wfdb.rdann(str(M/rec),'atr')
        pk=np.array([p for p,sy in zip(a.sample,a.symbol) if sy in amap],dtype=np.int64)
        lab=np.array([C.index(amap[sy]) for sy in a.symbol if sy in amap])
        for l in lab: ann_c[C[l]]+=1
        for sy in a.symbol:
            if sy not in amap: skipped[sy]+=1
        if n!='excluded':
            ch=r.sig_name.index('MLII') if 'MLII' in r.sig_name else 0
            wb=extract_windows(filter_continuous(r.p_signal[:,ch]),pk)
            keep=np.isin(pk,wb.sample_indices)
            for l in lab[keep]: win_c[C[l]]+=1
            if n=='test': Ws.append(wb.windows);Ys.append(lab[keep]);Rs+= [rec]*int(keep.sum());Ps.append(pk[keep])
    print(n,'annotated AAMI',{c:ann_c[c] for c in C},sum(ann_c.values()),'| usable windows',{c:win_c[c] for c in C},sum(win_c.values()))
    if n=='test': W=(np.concatenate(Ws),np.concatenate(Ys),np.array(Rs),np.concatenate(Ps))
print('fs',fs)
print('records without MLII (channel 0 used):',{k:v for k,v in leads.items() if 'MLII' not in v})
print('MLII not first channel:',{k:v for k,v in leads.items() if 'MLII' in v and v[0]!='MLII'})
REF=ROOT/'results/test_examples_for_xai.npz'
if not REF.exists():
    print('UNVERIFIED: window equivalence skipped,',REF.name,'is not present (gitignored, >100MB)'); sys.exit(0)
z=np.load(REF,allow_pickle=False)
X=z['X'].astype(np.float32); X=X if X.ndim==3 else X[...,None]
y=z['y']; rec=z['record'].astype(str); pos=z['position']
Wm,Ym,Rm,Pm=W
print('npz n',len(X),'rebuilt n',len(Wm))
if len(X)==len(Wm):
    same=np.array_equal(rec,Rm) and np.array_equal(pos,Pm)
    print('same order (record,position):',same,'labels equal:',np.array_equal(y,Ym))
    if not same:
        key={(r,int(p)):i for i,(r,p) in enumerate(zip(Rm,Pm))}
        o=[key[(r,int(p))] for r,p in zip(rec,pos)]; Wm=Wm[o];Ym=Ym[o]
    d=np.abs(X-Wm); print('window max abs diff',float(d.max()),'mean',float(d.mean()))
