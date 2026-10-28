"""safe_save: guarded saving for the SELECTION (sel_) and MONITORING RQ3 (mon_) pipelines.

Rules (all enforced):
  1. File name must start with a known prefix: 'sel_' or 'mon_' (or the prefix passed explicitly).
  2. Resolved path (case-insensitive, symlinks followed) must NOT be inside full_variable/ or archive/.
  3. 'sel_' files must be inside selection/ or Data/processed/. 'mon_' files must be inside monitoring_rq3/ ONLY.
  4. An existing path is refused unless overwrite=True.
Writes go to a temp file in the same folder, then os.replace (atomic).
"""
import os, json, pickle, hashlib

ROOT = os.environ["CREDIT_BN_ROOT"]
FORBIDDEN = ["full_variable", "archive"]
ALLOWED = ["selection", "data/processed"]          # allowed folders for the default 'sel_' prefix (kept for compatibility)
PREFIX = "sel_"
RULES = {"sel_": ["selection", "data/processed"], "mon_": ["monitoring_rq3"]}


class SaveRefused(Exception):
    pass


def _norm(p):
    return os.path.normcase(os.path.realpath(p)).lower()


def _inside(path, folder, root):
    f = _norm(os.path.join(root, folder))
    return path == f or path.startswith(f + os.sep)


def check_path(path, overwrite=False, root=ROOT, prefix=None):
    """Return the resolved path, or raise SaveRefused."""
    rp = _norm(path)
    name = os.path.basename(rp)
    if prefix is None:
        pf = next((k for k in RULES if name.startswith(k)), None)
        if pf is None:
            raise SaveRefused(f"file name must start with one of {list(RULES)}: {os.path.basename(path)}")
    else:
        pf = prefix.lower()
        if not name.startswith(pf):
            raise SaveRefused(f"file name must start with '{prefix}': {os.path.basename(path)}")
        if pf not in RULES:
            raise SaveRefused(f"unknown prefix '{prefix}' (known: {list(RULES)})")
    for f in FORBIDDEN:
        if _inside(rp, f, root):
            raise SaveRefused(f"path is inside protected folder '{f}/': {path}")
    allowed = RULES[pf]
    if not any(_inside(rp, a, root) for a in allowed):
        raise SaveRefused(f"'{pf}' files are only allowed inside {allowed}: {path}")
    if os.path.lexists(path) and not overwrite:
        raise SaveRefused(f"path already exists (use overwrite=True to replace): {path}")
    return os.path.realpath(path)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_save(obj, path, kind=None, overwrite=False, root=ROOT, prefix=None, **kw):
    """Save obj to path. kind: pkl | json | csv | txt | png (inferred from extension if None).
    Returns (resolved_path, sha256)."""
    real = check_path(path, overwrite=overwrite, root=root, prefix=prefix)
    kind = kind or os.path.splitext(path)[1].lstrip(".").lower()
    os.makedirs(os.path.dirname(real), exist_ok=True)
    tmp = real + ".tmp_safe_save"
    try:
        if kind in ("pkl", "pickle"):
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=4)
        elif kind == "json":
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        elif kind == "csv":
            obj.to_csv(tmp, **kw)
        elif kind in ("txt", "md"):
            with open(tmp, "w") as f:
                f.write(obj)
        elif kind == "png":
            obj.savefig(tmp, format="png", **({"dpi": 200, "bbox_inches": "tight"} | kw))
        else:
            raise SaveRefused(f"unsupported kind: {kind}")
        if os.path.lexists(real) and not overwrite:   # re-check just before the swap
            raise SaveRefused(f"path appeared during save: {path}")
        os.replace(tmp, real)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return real, _sha256(real)
