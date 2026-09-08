
import csv, io, os, urllib.request
URL="https://huggingface.co/datasets/ZoneTwelve/rockyou-in-size/resolve/main/rockyou.txt"
OUT="passwords_rockyou.csv"
print("Downloading public RockYou-derived corpus...")
try:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data=r.read()
except Exception as e:
    raise SystemExit("Download failed. Error: "+str(e))
text=data.decode("utf-8","ignore").splitlines()
seen=set(); rows=[]
for rank,p in enumerate(text,1):
    p=p.strip()
    if not p or p in seen: continue
    seen.add(p); rows.append((p,rank))
with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["password","rank"]); w.writerows(rows)
print(f"Saved {OUT}: {os.path.getsize(OUT)/1024/1024:.2f} MB; unique passwords: {len(rows)}")
