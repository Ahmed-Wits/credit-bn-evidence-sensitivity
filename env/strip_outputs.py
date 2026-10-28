import sys, json
# git clean filter: strips outputs and execution counts from a notebook as it is stored.
# Does not modify the file on disk. If the input is not valid notebook JSON it is passed through unchanged.
raw = sys.stdin.read()
try:
    nb = json.loads(raw)
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    sys.stdout.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
except Exception:
    sys.stdout.write(raw)
