import requests

URL = "https://raw.githubusercontent.com/KalyanM45/Checking-Password-Strength-using-Machine-Learning/main/Password%20Strength.csv"
OUT = "passwords.csv"

r = requests.get(URL, timeout=60)
r.raise_for_status()
open(OUT, "wb").write(r.content)
print(f"Saved {OUT}: {len(r.content)/1e6:.2f} MB")
