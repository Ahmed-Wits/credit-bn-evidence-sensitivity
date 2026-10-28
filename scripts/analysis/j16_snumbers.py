"""J16: rows for Table S-Numbers (Section 5.1.2), in the style of the existing rows (Quantity, Value, Ledger ID, Claim ID)."""
from addcommon import *
V6 = pd.read_csv(ADD + "/ledger_additions_v6.csv", dtype=str, keep_default_na=False); J = V6[V6.source_file.str.contains("j16_blanket_only_lr")]
q = {r.quantity: r for r in J.itertuples()}
def get(prefix): m = [r for r in J.itertuples() if r.quantity.startswith(prefix)]; assert len(m) == 1, prefix; return m[0]
a = get("Linear LR on Age and Debt Ratio only"); b = get("Linear LR on the four blanket variables only, After acceptance: test AUC"); c = get("Linear LR on the four blanket variables only, After acceptance: paired"); d = get("Linear LR on all ten variables minus"); e = get("Linear LR on all ten variables, After acceptance")
rows = [("Section 5.1.2 (RQ2/RQ3 discussion)", "Linear LR, Age and Debt Ratio only, AUC difference, BN minus this LR", a.display, a.ledger_id, "CL-M11 (application analogue, gap G11)", "new (J16)"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "Linear LR, four blanket variables only, test AUC (after acceptance)", b.display, b.ledger_id, "CL-M11", "new (J16)"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "AUC difference, BN minus linear LR on the four blanket variables", c.display, c.ledger_id, "CL-M11", "new (J16)"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "AUC difference, linear LR on all ten variables minus linear LR on the four blanket variables", d.display, d.ledger_id, "CL-M11", "new (J16)"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "AUC difference, BN minus linear LR on all ten variables", e.display, e.ledger_id, "CL-M11", "new (J16), context"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "Test AUC, LR (binned, locked settings) on the four blanket variables", LED.loc["N-MON-037", "display"], "N-MON-037", "CL-M11", "existing row, not yet in Table S-Numbers (earlier run, saga C = 100)"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "Paired AUC difference, BN minus LR on the four blanket variables (earlier run)", LED.loc["N-MON-044", "display"], "N-MON-044", "CL-M11", "existing row, not yet in Table S-Numbers"),
        ("Section 5.1.2 (RQ2/RQ3 discussion)", "Paired AUC difference, LR with all variables minus LR on the four blanket variables (earlier run)", LED.loc["N-MON-043", "display"], "N-MON-043", "CL-M11", "existing row, not yet in Table S-Numbers")]
pd.DataFrame(rows, columns=["Section", "Quantity", "Value", "Ledger ID", "Claim ID", "Status"]).to_csv(ADD + "/j16_snumbers_rows.csv", index=False)
print(pd.DataFrame(rows, columns=["Section", "Quantity", "Value", "Ledger ID", "Claim ID", "Status"])[["Quantity", "Value", "Ledger ID"]].to_string())
