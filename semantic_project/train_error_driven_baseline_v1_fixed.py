#!/usr/bin/env python3
import json, re, random, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'tiny_semantic_student_v1'
OUT=ROOT/'tiny_semantic_student_error_driven_v1'
DATA=next((p for p in [ROOT/'semantic_training_v3_hard_negatives.xlsx',ROOT/'semantic_training_v3.xlsx',ROOT/'semantic_training.xlsx'] if p.exists()),None)
STRESS=next((p for p in [ROOT/'unseen_semantic_stress_test.csv'] if p.exists()),None)
DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
for p in ['student_fp32.pt','vocab.json','config.json','intent_labels.txt']:
    if not (BASE/p).exists(): raise FileNotFoundError(f'Missing baseline file: {BASE/p}')
with open(BASE/'config.json') as f: cfg=json.load(f)
with open(BASE/'vocab.json') as f: vocab=json.load(f)
labels=[x.strip() for x in open(BASE/'intent_labels.txt') if x.strip()]
label_id={x:i for i,x in enumerate(labels)}
def get(*names,default=None):
    for n in names:
        if n in cfg:return cfg[n]
    return default
ED=int(get('embed_dim','embedding_dim',default=64)); NH=int(get('num_heads','nhead',default=4)); NL=int(get('num_layers','layers',default=2)); FF=int(get('ff_dim','feedforward_dim',default=128)); ML=int(get('max_len','max_length','sequence_length',default=24))
PAD=int(vocab.get('<PAD>',vocab.get('[PAD]',0))); UNK=int(vocab.get('<UNK>',vocab.get('[UNK]',1)))
def enc(t):
    toks=re.sub(r'[^a-z0-9\']+',' ',str(t).lower()).strip().split(); ids=[int(vocab.get(x,UNK)) for x in toks[:ML]]; return ids+[PAD]*max(0,ML-len(ids))
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(len(vocab),ED,padding_idx=PAD); self.position=nn.Embedding(ML,ED); layer=nn.TransformerEncoderLayer(ED,NH,FF,dropout=.1,activation='gelu',batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,NL); self.norm=nn.LayerNorm(ED); self.classifier=nn.Sequential(nn.Linear(ED,ED),nn.GELU(),nn.Dropout(.1),nn.Linear(ED,len(labels)))
    def forward(self,x):
        m=x.eq(PAD); pos=torch.arange(x.size(1),device=x.device).unsqueeze(0); h=self.x(self.e(x)+self.p(pos),src_key_padding_mask=m); v=(~m).unsqueeze(-1).float(); h=(h*v).sum(1)/v.sum(1).clamp(min=1); return self.c(self.n(h))

HARD=[
("it's quieter can you make it a little louder","device.volume.increase"),("the sound is too quiet make it louder","device.volume.increase"),("i can barely hear it turn the volume up","device.volume.increase"),("the audio is weak please raise the volume","device.volume.increase"),
("it's a bit loud here can you make it quieter","device.volume.decrease"),("it's bit loudy here can you make it quieter","device.volume.decrease"),("it's bit loudy here can you make it quietr","device.volume.decrease"),("the sound is too loud turn it down","device.volume.decrease"),("the volume is too high please lower it","device.volume.decrease"),
("i can still hear it make it completely silent","device.volume.mute"),("there is still audio turn all sound off","device.volume.mute"),("make everything completely silent","device.volume.mute"),("i don't want any sound at all","device.volume.mute"),("turn all audio off","device.volume.mute"),
("turn the sound back on","device.volume.unmute"),("restore the audio","device.volume.unmute"),("i want to hear again","device.volume.unmute"),("unmute the hearing aids","device.volume.unmute"),("there is no sound turn it back on","device.volume.unmute"),
("i need to go to airport tomorrow","reminders.task.create"),("i need to go to airport tommorow","reminders.task.create"),("i need to go to airport tomorow","reminders.task.create"),("remind me that i need to go to the airport tomorrow","reminders.task.create"),
("the sound is quiet make it louder","device.volume.increase"),("the sound is loud make it quieter","device.volume.decrease"),("the sound is quiet but i want complete silence","device.volume.mute"),("there is no sound because it is muted restore it","device.volume.unmute"),("do not mute it just lower the volume","device.volume.decrease"),("do not lower it make it louder","device.volume.increase"),("do not increase the volume mute it","device.volume.mute"),
("start streaming","streaming.session.start"),("stop streaming","streaming.session.stop"),("where is my phone","find.phone.locate"),("show my reminders","help.reminder.show"),("complete my reminder","reminders.task.complete")]
TYPO={'louder':['loudr','loudar'],'quieter':['quiter','quietr','quietter'],'tomorrow':['tommorow','tomorow'],'volume':['volum'],'mute':['mut'],'unmute':['unmut']}
for text,intent in list(HARD):
    for a,bs in TYPO.items():
        if a in text.lower():
            for b in bs: HARD.append((re.sub(re.escape(a),b,text,count=1,flags=re.I),intent))
extra=pd.DataFrame(HARD,columns=['text','intent']).drop_duplicates()
if DATA:
    df=pd.read_excel(DATA,sheet_name='dataset' if 'dataset' in pd.ExcelFile(DATA).sheet_names else 0); df=df[['text','intent']].dropna(); df=df[df.intent.isin(labels)]
    train=pd.concat([df,extra],ignore_index=True).drop_duplicates(['text','intent'])
else: train=extra
class DS(Dataset):
    def __init__(self,d): self.d=d.reset_index(drop=True)
    def __len__(self): return len(self.d)
    def __getitem__(self,i): return torch.tensor(enc(self.d.iloc[i].text)),torch.tensor(label_id[self.d.iloc[i].intent])
model=Model(); model.load_state_dict(torch.load(BASE/'student_fp32.pt',map_location='cpu')); model.to(DEVICE)
loader=DataLoader(DS(train),batch_size=64,shuffle=True)
opt=torch.optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(label_smoothing=.02)
print('Device:',DEVICE); print('Baseline:',BASE); print('Training rows:',len(train)); print('Fine-tuning SAME baseline architecture...')
for ep in range(10):
    model.train(); total=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); loss=lossfn(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()
    print(f'Epoch {ep+1:02d}/10 Loss={total/len(loader):.4f}')
OUT.mkdir(exist_ok=True); torch.save(model.state_dict(),OUT/'student_error_driven_v1_fp32.pt')
for f in ['vocab.json','config.json','intent_labels.txt']: shutil.copy2(BASE/f,OUT/f)
def predict(texts):
    model.eval(); x=torch.tensor([enc(t) for t in texts],dtype=torch.long).to(DEVICE)
    with torch.no_grad(): p=torch.softmax(model(x),1); c,i=p.max(1)
    return [labels[int(z)] for z in i.cpu()],c.cpu().numpy()
REG=[("it's quieter can you make it a little louder","device.volume.increase"),("i can still hear it make it completely silent","device.volume.mute"),("turn off","device.volume.mute"),("i need to go to airport tomorrow","reminders.task.create"),("i need to go to airport tommorow","reminders.task.create"),("it's bit loudy here can you make it quieter","device.volume.decrease"),("it's bit loudy here can you make it quietr","device.volume.decrease"),("the sound seems low please turn it up","device.volume.increase"),("the sound is too loud turn it down","device.volume.decrease"),("turn the sound back on","device.volume.unmute"),("where is my phone","find.phone.locate"),("show my reminders","help.reminder.show"),("complete my reminder","reminders.task.complete"),("start streaming","streaming.session.start"),("stop streaming","streaming.session.stop")]
t,p=predict([x[0] for x in REG]); r=accuracy_score([x[1] for x in REG],p); print('\nTARGETED REGRESSION:',f'{r*100:.2f}%'); pd.DataFrame({'text':[x[0] for x in REG],'expected':[x[1] for x in REG],'predicted':p,'confidence':conf if False else [float(z) for z in predict([x[0] for x in REG])[1]]}).to_csv(OUT/'targeted_regression_results.csv',index=False)
if STRESS:
    s=pd.read_csv(STRESS); pred,conf=predict(s.text.tolist()); ua=accuracy_score(s.intent,pred); uf=f1_score(s.intent,pred,average='macro'); print('\nUNSEEN BASELINE 94.29% -> CANDIDATE',f'{ua*100:.2f}%'); print('Macro F1:',f'{uf*100:.2f}%'); print(classification_report(s.intent,pred,digits=4)); s['predicted_intent']=pred;s['confidence']=conf;s['correct']=s.intent==s.predicted_intent;s.to_csv(OUT/'unseen_predictions.csv',index=False);s[~s.correct].to_csv(OUT/'unseen_errors.csv',index=False)
print('\nCandidate:',OUT); print('Current INT8 baseline was NOT modified.'); print('Do NOT export/promote until the full benchmark confirms improvement.')
