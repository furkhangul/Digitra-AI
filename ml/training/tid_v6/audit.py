"""Preserve the historic holdout and exclude duplicate leakage before training."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def run(manifest,output):
    output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(manifest,encoding='utf-8-sig').reset_index(drop=True)
    hashes=[];phashes=[];thumbs=[]
    for row in df.itertuples():
        raw=Path(row.path).read_bytes();hashes.append(hashlib.sha256(raw).hexdigest())
        with Image.open(row.path) as im:
            im.verify()
        with Image.open(row.path) as im:
            gray=np.array(im.convert('L').resize((32,32)),dtype=np.float32)
            coeff=cv2.dct(gray)[:8,:8].flatten()[1:]
            phashes.append(coeff>np.median(coeff));thumbs.append(gray)
    phashes=np.asarray(phashes);thumbs=np.asarray(thumbs)
    priority={'train':0,'val':1,'test':2};excluded=set();pairs=[]
    for i in range(len(df)):
        candidates=np.where(np.count_nonzero(phashes[i+1:]!=phashes[i],axis=1)<=3)[0]+i+1
        for j in candidates:
            exact=hashes[i]==hashes[j]
            rms=float(np.sqrt(np.mean((thumbs[i]-thumbs[j])**2)))
            if not exact and rms>5.:continue
            if exact and df.iloc[i].label!=df.iloc[j].label:raise ValueError('Conflicting identical image labels')
            if df.iloc[i].split==df.iloc[j].split:continue
            remove=i if priority[df.iloc[i].split]<priority[df.iloc[j].split] else j
            # Exclude the entire pseudo-session so adjacent near-duplicate frames
            # cannot reintroduce the removed image's context into training.
            group=df.iloc[remove].pseudo_session
            excluded.update(df.index[df.pseudo_session==group].tolist())
            pairs.append({'a':df.iloc[i].filename,'b':df.iloc[j].filename,'rms':rms,'exact':exact,'excluded_group':group})
    clean=df.drop(index=list(excluded)).copy()
    clean['sha256']=[hashes[i] for i in clean.index]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(clean[clean.split==a].sha256)&set(clean[clean.split==b].sha256):raise ValueError('Exact leakage')
    labels=set(df.label)
    for split in ['train','val','test']:
        if set(clean[clean.split==split].label)!=labels:raise ValueError(f'{split}: class removed by audit; review required')
    clean.to_csv(output/'manifest.csv',index=False,encoding='utf-8-sig')
    df.loc[sorted(excluded)].to_csv(output/'excluded.csv',index=False,encoding='utf-8-sig')
    report={'source_manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        'samples_before':len(df),'samples_after':len(clean),'classes':len(labels),'excluded':len(excluded),
        'split_counts':clean.split.value_counts().to_dict(),'class_split_counts':pd.crosstab(clean.label,clean.split).to_dict(),
        'cross_split_duplicates':pairs,'near_duplicate_rule':'pHash <=3/63 bits and 32px grayscale RMS <=5; drop entire lower-priority pseudo-session',
        'split_protocol':'Existing 10-frame pseudo-session holdout preserved; no signer metadata',
        'test_history':'This holdout was reported for V5. It is not a new external test.',
        'initialization':'Fresh ImageNet weights only; V5 task-trained weights are not used.',
        'limitations':['Near-duplicate detection is conservative and cannot prove absence of all similar frames.','No signer-independent or external evaluation is available.']}
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['samples_before','samples_after','classes','excluded','split_counts']},ensure_ascii=False),flush=True)
    return clean


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.manifest,args.output)
