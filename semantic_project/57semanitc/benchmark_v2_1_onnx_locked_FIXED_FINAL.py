#!/usr/bin/env python3
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
ONNX=ROOT/"v3_57intent_v2_1_onnx"/"v2_1_57intent_fp32.onnx"
LOCKED=ROOT/"v3_57intent_locked_eval"/"locked_test_57intent.csv"
VOCAB=ROOT/"vocab.json"
LABELS=ROOT/"labels.json"
OUT=ROOT/"v3_57intent_v2_1_onnx_locked_benchmark"
OUT.mkdir(exist_ok=True)
MAX_LEN=24
NCLASS=57

def load_json(p):
    with open(p,encoding="utf-8") as f: return json.load(f)

def vocab_map(p):
    x=load_json(p)
    if isinstance(x,list): return {str(v):i for i,v in enumerate(x)}
    if isinstance(x,dict):
        if isinstance(x.get("stoi"),dict): return x["stoi"]
        if isinstance(x.get("vocab"),dict): return x["vocab"]
        if all(isinstance(v,int) for v in x.values()): return x
    raise RuntimeError("Unsupported vocab.json format")

def labels_list(p):
    x=load_json(p)
    if isinstance(x,list): return x
    if isinstance(x,dict):
        if isinstance(x.get("labels"),list): return x["labels"]
        if all(str(k).isdigit() for k in x):
            return [v for k,v in sorted(x.items(),key=lambda z:int(z[0]))]
        if all(isinstance(v,int) for v in x.values()):
            out=[None]*(max(x.values())+1)
            for k,v in x.items(): out[v]=k
            return out
    raise RuntimeError("Unsupported labels.json format")

def col(df,names):
    m={str(c).lower().strip():c for c in df.columns}
    for n in names:
        if n in m:return m[n]
    return None

def encode(text,v):
    toks=re.findall(r"\w+|[^\w\s]",str(text).lower().strip(),flags=re.UNICODE)
    unk=v.get("<unk>",v.get("[UNK]",v.get("UNK",1)))
    pad=v.get("<pad>",v.get("[PAD]",v.get("PAD",0)))
    ids=[int(v.get(t,unk)) for t in toks[:MAX_LEN]]
    return np.asarray(ids+[int(pad)]*(MAX_LEN-len(ids)),dtype=np.int64)

def main():
    print("="*78)
    print("V2.1 ONNX FIXED-BATCH LOCKED BENCHMARK")
    print("="*78)
    for p in (ONNX,LOCKED,VOCAB,LABELS):
        if not p.exists(): raise FileNotFoundError(p)

    labels=labels_list(LABELS); vocab=vocab_map(VOCAB)
    if len(labels)!=NCLASS: raise RuntimeError(f"Expected 57 labels, got {len(labels)}")

    df=pd.read_csv(LOCKED)
    tc=col(df,["text","utterance","phrase","query","sentence","input"])
    lc=col(df,["label","intent","target","class"])
    if tc is None or lc is None: raise RuntimeError(f"Columns: {list(df.columns)}")
    texts=df[tc].fillna("").astype(str).tolist()
    true_names=df[lc].astype(str).tolist()
    l2i={x:i for i,x in enumerate(labels)}
    unknown=sorted(set(true_names)-set(l2i))
    if unknown: raise RuntimeError("Unknown labels: "+repr(unknown))
    X=np.stack([encode(x,vocab) for x in texts])
    y=np.asarray([l2i[x] for x in true_names],dtype=np.int64)

    sess=ort.InferenceSession(str(ONNX),providers=["CPUExecutionProvider"])
    inp=sess.get_inputs()[0]; out=sess.get_outputs()[0]
    print("\n--- ACTUAL ONNX CONTRACT ---")
    print("input :",inp.name,inp.type,inp.shape)
    print("output:",out.name,out.type,out.shape)

    # Never pass X directly. The exported model is fixed at [1,24].
    if inp.shape != [1,MAX_LEN]:
        raise RuntimeError(f"Expected fixed ONNX input [1,24], got {inp.shape}")

    logits=[]
    t0=time.perf_counter()
    for i in range(len(X)):
        one=X[i:i+1]                 # EXACTLY (1,24)
        result=sess.run([out.name],{inp.name:one})[0]
        if result.shape != (1,NCLASS):
            raise RuntimeError(f"Row {i}: output {result.shape}, expected (1,57)")
        logits.append(result[0])
        if (i+1)%250==0 or i+1==len(X): print(f"Progress: {i+1}/{len(X)}")
    elapsed=time.perf_counter()-t0

    logits=np.asarray(logits,dtype=np.float32)
    pred=logits.argmax(1)
    acc=accuracy_score(y,pred)
    rep=classification_report(y,pred,labels=np.arange(NCLASS),target_names=labels,digits=4,zero_division=0)
    rd=classification_report(y,pred,labels=np.arange(NCLASS),target_names=labels,output_dict=True,zero_division=0)

    print("\n--- V2.1 ONNX LOCKED TEST RESULT ---")
    print(f"Accuracy   : {acc*100:.4f}%")
    print(f"Macro F1   : {rd['macro avg']['f1-score']*100:.4f}%")
    print(f"Weighted F1: {rd['weighted avg']['f1-score']*100:.4f}%")
    print("\nClassification report:\n"+rep)

    predfile=OUT/"locked_predictions_onnx_v2_1.csv"
    reportfile=OUT/"classification_report_onnx_v2_1.txt"
    cmfile=OUT/"confusion_matrix_onnx_v2_1.csv"
    summaryfile=OUT/"benchmark_summary_onnx_v2_1.json"
    pd.DataFrame({"text":texts,"true_intent":true_names,"predicted_intent":[labels[i] for i in pred],"confidence":logits.max(1)}).to_csv(predfile,index=False)
    reportfile.write_text(rep,encoding="utf-8")
    pd.DataFrame(confusion_matrix(y,pred,labels=np.arange(NCLASS)),index=labels,columns=labels).to_csv(cmfile)
    summary={
        "accuracy":float(acc),"macro_f1":float(rd["macro avg"]["f1-score"]),
        "weighted_f1":float(rd["weighted avg"]["f1-score"]),
        "rows":len(X),"total_time_sec":elapsed,
        "rows_per_sec":len(X)/elapsed,"ms_per_row":elapsed*1000/len(X),
        "input_shape":[1,24],"output_shape":[1,57],"locked_test_used":True,
        "training_performed":False
    }
    summaryfile.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\n--- INFERENCE SPEED ---")
    print(f"Total rows : {len(X)}")
    print(f"Total time : {elapsed:.4f} sec")
    print(f"Rows/sec   : {len(X)/elapsed:.2f}")
    print(f"ms/row     : {elapsed*1000/len(X):.4f}")
    print("\nSTATUS: V2.1 ONNX LOCKED 57-INTENT BENCHMARK COMPLETE")

if __name__=="__main__": main()
