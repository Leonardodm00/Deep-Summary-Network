import json, os, sys, collections
root = sys.argv[1] if len(sys.argv) > 1 else "out"
exp  = sys.argv[2] if len(sys.argv) > 2 else "mea_joint_full"
for L in range(4):
    d = os.path.join(root, "%s_lane%d" % (exp, L))
    p = os.path.join(d, "trials.jsonl")
    if not os.path.exists(p):
        print("lane%d: NO trials.jsonl at %s" % (L, p)); continue
    recs = [json.loads(x) for x in open(p) if x.strip()]
    n = len(recs)
    failed = sum(1 for r in recs if r.get("failed"))
    ok = [r for r in recs if not r.get("failed")]
    se_missing = sum(1 for r in ok if not r.get("selected_epochs"))
    keys = collections.Counter()
    for r in recs[:5]:
        keys.update(r.keys())
    print("lane%d: n=%d failed=%d ok=%d  selected_epochs EMPTY in %d of %d ok"
          % (L, n, failed, len(ok), se_missing, len(ok)))
    if ok:
        r = ok[0]
        print("        schema_version=%s selection_primary=%s epsilon=%s"
              % (r.get("schema_version"), r.get("selection_primary"), r.get("epsilon")))
        print("        selected_epochs=%r n_seeds=%s n_seeds_ok=%s"
              % (r.get("selected_epochs"), r.get("n_seeds"), r.get("n_seeds_ok")))
        print("        objective=%s mean=%s ari_mean=%s sil_mean=%s"
              % (r.get("objective"), r.get("mean"), r.get("ari_mean"), r.get("sil_mean")))
        print("        has point_raw=%s  n_axes=%s  cell=%s"
              % ("point_raw" in r, len(r.get("point_raw", {})), r.get("cell")))
    if failed:
        f = [r for r in recs if r.get("failed")][0]
        print("        first FAILED record: objective=%s reason=%r"
              % (f.get("objective"), f.get("error") or f.get("reason")))