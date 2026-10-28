# Reproducibility notes
- pgmpy is pinned at 1.1.2 (see requirements.txt). HillClimbSearch is deprecated and is removed in pgmpy 1.3.0 (replacement: pgmpy.causal_discovery.HillClimbSearch), so do not upgrade pgmpy.
- PYTHONHASHSEED must be 0 before Python starts. Use the kernel bn_sel (PYTHONHASHSEED=0); notebooks assert it in the first cell.
- Structure learning: shared/src/structure.py (learn_structure), validated 2026-10-10 on the locked 11-node frame: reproduces the 22 locked edges exactly (best seed 53, BIC -1,026,115.7308).
