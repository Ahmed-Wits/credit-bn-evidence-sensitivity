"""python j17_fit.py NAME : fit one BN variant at seed 42 (EM fixed 100 iterations after acceptance) and save j17/NAME_fit.{pkl,json}, NAME_sens_rows.pkl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
V = {"after_full": ("after", []), "after_V1": ("after", ["90+ Days Late"]), "after_V2": ("after", ["30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"]), "application": ("application", [])}
name = sys.argv[1]; st, drop = V[name]
fit_variant(name, st, drop, ADD + "/j17", log=lambda s: print(s, flush=True), n_jobs=int(os.environ.get("RF_JOBS", "2")))
