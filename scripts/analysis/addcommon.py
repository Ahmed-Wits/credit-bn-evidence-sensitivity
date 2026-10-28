"""Shared helpers for the analysis additions (J1 to J8). Reads saved outputs only; nothing is retrained or refitted.
Bootstrap convention for every new interval: 2,000 resamples, seed 42, 95% percentile intervals, test rows resampled with predictions
and thresholds fixed, paired across models (same resample for every model)."""
import os, sys, json, hashlib
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np, pandas as pd

R = os.environ["CREDIT_BN_ROOT"]
WR = R + "/Submissions/4. Article/WRITEUP"
ADD = WR + "/analysis_additions"
PARTS = ADD + "/code/parts"
LEDP = WR + "/ledgers/numbers_ledger_v3.csv"
COLS = list(pd.read_csv(LEDP, dtype=str, nrows=0).columns)
LED = pd.read_csv(LEDP, dtype=str, keep_default_na=False).set_index("ledger_id")
MINUS = "−"
B_BOOT, SEED = 2000, 42

def sha12(rel):
    p = rel if rel.startswith("/") else R + "/" + rel
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]

def sgn(x, d, pts=False):
    s = f"{abs(x):.{d}f}"; sign = "+" if x > 0 or (x == 0) else MINUS
    return sign + s + (" pts" if pts else "")

class Part:
    def __init__(self, name):
        self.name = name; self.rows = []; self.checks = []; self.md = []
    def check(self, what, ok, detail=""):
        ok = bool(ok); self.checks.append(dict(item=what, ok=ok, detail=detail))
        print(("PASS " if ok else "FAIL ") + what + (" | " + detail if detail else ""))
        return ok
    def gate(self, what, ok, detail=""):
        """Reproduction gate: if it fails, stop the whole job (nothing new is produced)."""
        if not self.check("GATE: " + what, ok, detail):
            print("GATE FAILED: stopping before any new number is produced"); sys.exit(2)
    def gate_near(self, what, a, b, tol, detail=""):
        self.gate(what, abs(float(a) - float(b)) <= tol, "%s vs %s (tol %g) %s" % (a, b, tol, detail))
    def near(self, what, a, b, tol, detail=""):
        return self.check(what, abs(float(a) - float(b)) <= tol, "%s vs %s (tol %g) %s" % (a, b, tol, detail))
    def row(self, stage, quantity, value, display, unit, source_file, source_locator, derived="Y", notes="", display_rule="txt", ci_lo="", ci_hi="", ci_method="",
            n="", model="", variable="", mechanism="", level="", comparator="", imputer="", tested="not-tested", locked="N", caveat="", n_seeds_bn_less=""):
        d = {c: "" for c in COLS}
        d.update(stage=stage, quantity=quantity, value=repr(float(value)) if isinstance(value, (float, np.floating)) else str(value), display=display, unit=unit,
                 ci_lo=("" if ci_lo == "" else repr(float(ci_lo))), ci_hi=("" if ci_hi == "" else repr(float(ci_hi))), ci_method=ci_method, n=str(n), model=model, variable=variable,
                 mechanism=mechanism, level=level, comparator=comparator, imputer=imputer, tested=tested, source_file=source_file, source_locator=source_locator,
                 source_sha12=sha12(source_file), derived=derived, locked=locked, caveat=caveat, notes=notes, display_rule=display_rule, n_seeds_bn_less=str(n_seeds_bn_less))
        self.rows.append(d)
    def write(self, md_text):
        json.dump(dict(rows=self.rows, checks=self.checks), open(f"{PARTS}/{self.name}.json", "w"), indent=1)
        open(f"{PARTS}/{self.name}.md", "w").write(md_text)
        print("%s: %d rows, %d checks (%d fail)" % (self.name, len(self.rows), len(self.checks), sum(not c["ok"] for c in self.checks)))

def checks_md(p):
    L = ["| Check | Result | Detail |", "|---|---|---|"]
    for c in p.checks: L.append("| %s | %s | %s |" % (c["item"], "PASS" if c["ok"] else "FAIL", str(c["detail"]).replace("|", "/")))
    return "\n".join(L)

def boot_weights(n, B=B_BOOT, seed=SEED):
    """Row-resample counts, as in the Phase 8 test-row bootstrap (default_rng(42), multinomial); same weights for every model."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)

def pct_ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
