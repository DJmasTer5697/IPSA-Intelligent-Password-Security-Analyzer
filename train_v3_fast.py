
import os, json, argparse, re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from joblib import dump

def make_tier(n):
    a=int(n*0.10); b=a+int(n*0.40)
    y=np.full(n,2,dtype=np.int8); y[:a]=0; y[a:b]=1
    return y

def metrics(y,p):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),
            "precision_macro":float(pr),"recall_macro":float(rc),"f1_macro":float(f1)}

def engineered(passwords):
    s=pd.Series(passwords, dtype="string").fillna("")
    L=s.str.len().astype(float)
    upper=s.str.count(r"[A-Z]").astype(float)
    lower=s.str.count(r"[a-z]").astype(float)
    digits=s.str.count(r"\d").astype(float)
    special=(L-upper-lower-digits).clip(lower=0)
    unique=s.map(lambda x: len(set(x))).astype(float)
    diversity=(unique/L.replace(0,np.nan)).fillna(0)
    # Shannon entropy of the observed character distribution (not theoretical pool entropy).
    def shannon(x):
        if not x: return 0.0
        c=pd.Series(list(x)).value_counts(normalize=True)
        return float(-(c*np.log2(c)).sum())
    entropy=s.map(shannon).astype(float)
    repetition=(1-unique/L.replace(0,np.nan)).fillna(0)
    repeated=s.map(lambda x: 1.0 if re.search(r"(.)\1", x) else 0.0).astype(float)
    sequential=s.str.contains(r"(?:abc|bcd|cde|def|123|234|345|456|567|678|789)",case=False,regex=True).astype(float)
    keyboard=s.str.lower().str.contains(r"(?:qwerty|asdf|zxcv|qaz|wsx|edc)",regex=True).astype(float)
    dictionary=s.str.lower().str.contains(r"(?:password|admin|welcome|qwerty|letmein|login|dragon|monkey|football|iloveyou)",regex=True).astype(float)
    transitions=s.map(lambda x: sum((x[i].isdigit()!=x[i-1].isdigit()) or (x[i].islower()!=x[i-1].islower()) or (x[i].isupper()!=x[i-1].isupper()) for i in range(1,len(x)))).astype(float)
    return np.column_stack([upper,lower,digits,special,unique,diversity,entropy,repetition,repeated,sequential,keyboard,dictionary,transitions])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="passwords_rockyou.csv")
    ap.add_argument("--n_per_class",type=int,default=100000, help="sample per exposure tier; default 100k")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if not os.path.exists(args.csv): raise SystemExit("Missing passwords_rockyou.csv")
    df=pd.read_csv(args.csv)
    df["password"]=df["password"].fillna("").astype(str).str.strip()
    df["rank"]=pd.to_numeric(df["rank"],errors="coerce")
    df=df.dropna(subset=["rank"]); df=df[df.password.str.len()>0]
    df=df.drop_duplicates("password").sort_values("rank").reset_index(drop=True)
    y_all=make_tier(len(df))
    rng=np.random.default_rng(args.seed)
    idx=[]
    for cls in [0,1,2]:
        pool=np.flatnonzero(y_all==cls)
        n=min(args.n_per_class,len(pool))
        idx.extend(rng.choice(pool,size=n,replace=False))
    idx=np.array(idx); rng.shuffle(idx)
    df=df.iloc[idx].reset_index(drop=True); y=y_all[idx]
    print(f"Full cleaned corpus: {len(y_all)}")
    print(f"Balanced experimental sample: {len(df)} ({args.n_per_class} per tier requested)")
    Xtr,Xte,ytr,yte=train_test_split(df.password,y,test_size=.20,stratify=y,random_state=args.seed)
    print(f"Train: {len(Xtr)} | Test: {len(Xte)}")

    # Character TF-IDF
    tf=TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=80000,sublinear_tf=True,dtype=np.float32)
    print("1/4 Character TF-IDF...")
    A=tf.fit_transform(Xtr); B=tf.transform(Xte)
    char=LogisticRegression(max_iter=200,C=2,class_weight="balanced",solver="lbfgs",random_state=args.seed)
    char.fit(A,ytr); p_char=char.predict(B); m_char=metrics(yte,p_char)

    # Engineered features excluding raw length
    print("2/4 IPSA engineered features (excluding length)...")
    Ftr=engineered(Xtr); Fte=engineered(Xte)
    cols=["uppercase","lowercase","digits","special","unique_chars","character_diversity","entropy","repetition_ratio","repeated_pattern","sequential_pattern","keyboard_pattern","dictionary_pattern","structural_transitions"]
    eng=make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=1,class_weight="balanced",solver="lbfgs",random_state=args.seed))
    eng.fit(Ftr,ytr); p_eng=eng.predict(Fte); m_eng=metrics(yte,p_eng)

    # Combined
    print("3/4 Combined character TF-IDF + IPSA features...")
    scaler=StandardScaler()
    Fs=scaler.fit_transform(Ftr); Ft=scaler.transform(Fte)
    comb=LogisticRegression(max_iter=250,C=2,class_weight="balanced",solver="lbfgs",random_state=args.seed)
    comb.fit(hstack([A,csr_matrix(Fs)]),ytr)
    p_comb=comb.predict(hstack([B,csr_matrix(Ft)])); m_comb=metrics(yte,p_comb)

    # Length baseline
    print("4/4 Length-only baseline...")
    Ltr=np.array([len(x) for x in Xtr],dtype=float).reshape(-1,1)
    Lte=np.array([len(x) for x in Xte],dtype=float).reshape(-1,1)
    lm=make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=1,class_weight="balanced",solver="lbfgs",random_state=args.seed))
    lm.fit(Ltr,ytr); p_len=lm.predict(Lte); m_len=metrics(yte,p_len)

    os.makedirs("model_v3_fast",exist_ok=True)
    result={"full_cleaned_rows":int(len(y_all)),"sample_rows":int(len(df)),"train_rows":int(len(Xtr)),"test_rows":int(len(Xte)),
            "sampling":"balanced random sample by frequency-rank tier","n_per_class_requested":args.n_per_class,
            "target":"empirical guessability/exposure tier derived from frequency rank",
            "tier_definition":{"0":"top 10% most frequent (high exposure)","1":"next 40% (medium exposure)","2":"bottom 50% (lower exposure)"},
            "length_only":m_len,"engineered_no_length":m_eng,"character_tfidf":m_char,"combined":m_comb}
    with open("model_v3_fast/metrics.json","w") as f: json.dump(result,f,indent=2)
    with open("model_v3_fast/classification_reports.txt","w") as f:
        for name,p in [("Length-only",p_len),("Engineered no length",p_eng),("Character TF-IDF",p_char),("Combined",p_comb)]:
            f.write(f"\n=== {name} ===\n")
            f.write(classification_report(yte,p,target_names=["High-exposure","Medium-exposure","Lower-exposure"],digits=4))
    np.savetxt("model_v3_fast/confusion_matrix_combined.csv",confusion_matrix(yte,p_comb),fmt="%d",delimiter=",")
    dump({"tfidf":tf,"model":comb,"scaler":scaler,"feature_columns":cols},"model_v3_fast/ipsa_v3_fast_combined.joblib")
    print("\n=== IPSA V3 FAST RESULTS ===")
    print(json.dumps({"length_only":m_len,"engineered_no_length":m_eng,"character_tfidf":m_char,"combined":m_comb},indent=2))
    print("\nSaved results in model_v3_fast/")

if __name__=="__main__": main()
