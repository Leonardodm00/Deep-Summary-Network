# Real MEA cohort -- config, specs, and the 4-lane joint search

**Date:** 6 August 2026
**Applies on top of:** commit `460649f` (the K3 culture-grouping merge)
**Contents:** 1 modified file, 5 new files, all under `Main/`

Turns a tree of `ptrain_*` well folders into a running joint search, with the
K3 culture grouping wired through so the 9 subregions of a well stay one
culture.

---

## What is delivered

| File | Kind | What it does |
|---|---|---|
| `Main/config.py` | MODIFIED | adds the `CohortConfig` dataclass and `ExperimentConfig.cohort` |
| `Main/make_mea_specs.py` | new | walks the cohort, writes `data.npz_specs` with a `culture` per well |
| `Main/hpc/Config/config_mea_joint_full.json` | new | the base config: cohort block + full search |
| `Main/hpc/run_mea_joint_search.pbs` | new | one lane; adds a specs pre-flight guard |
| `Main/hpc/make_mea_lanes.sh` | new | expands the base config into 4 lanes |
| `Main/Smoke_Tests/smoke_test_mea_specs.py` | new | 6 checks over cohort + generator |

`CohortConfig` defaults to empty, so every existing config is unaffected and
loads with **zero warnings** -- which matters, because `preflight_config.py`
treats "no warnings" as a pass signal.

---

## The three stages

    <root>/ptrain_A1/  ---[1. extraction, ALREADY DONE]--->
        <extract_root>/.../ptrain_A1/trace_subregion_00..08.npz
                       ---[2. make_mea_specs.py]--->
        npz_specs_mea.json
                       ---[3. run_mea_joint_search.pbs x4]---> search

Stage 1 is expensive (~5.8 GB, minutes per well) and is NOT part of a lane.
Stage 2 is seconds on the login node. Stage 3 is the queue job.

---

## 1. Fill in the cohort block

Edit `Main/hpc/Config/config_mea_joint_full.json`. Every `REPLACE/ME` path is a
placeholder:

```json
"cohort": {
  "class_roots": {
    "0": ["/abs/path/ctrl_plate_1", "/abs/path/ctrl_plate_2", "/abs/path/ctrl_plate_3"],
    "1": ["/abs/path/path_plate_1", "/abs/path/path_plate_2", "/abs/path/path_plate_3"]
  },
  "class_names": ["control", "pathological"],
  "well_glob": "ptrain_*",
  "extract_root": "/abs/path/extracted",
  "extract_layout": "{class_name}/{root_name}/{well}",
  "culture_template": "{root_name}__{well}"
}
```

`class_roots` holds the RAW parent folders -- the ones containing
`ptrain_A1 ... ptrain_B3`. `extract_root` + `extract_layout` say where the
extraction outputs went. **If your extraction outputs are laid out differently,
change `extract_layout`, not the code.** It expands `{class_index}`,
`{class_name}`, `{root_name}` and `{well}` per well.

Paths with spaces work but must be quoted in every shell command.

## 2. Generate the specs

```bash
cd ~/Deep-Summary-Network/Main
export PYTHONPATH="$(pwd)"

# look first, write nothing
python3 make_mea_specs.py --config hpc/Config/config_mea_joint_full.json --dry-run

# then write
python3 make_mea_specs.py --config hpc/Config/config_mea_joint_full.json
```

The dry run prints the inventory, the split apportionment per class, `U_eff`,
the batch size `M`, and the implied window length in samples. Read it before
writing: it is the only place the cohort is visible as numbers.

Add `--strict` to make a missing extraction output a hard failure (exit 3)
rather than a warning. Worth using once you believe extraction is complete.

**Extraction mode is detected, not assumed.** `trace_subregion_NN.npz` means
`per_region_single` -> N_SUB records per well sharing one culture, and
`data.n_channels` must be 1. A lone `traces.npz` means `multichannel` -> one
record per well, `data.n_channels` = 9. A cohort mixing the two is refused: the
backbone stem cannot be both widths.

## 3. Build the cache once, then launch 4 lanes

```bash
cd ~/Deep-Summary-Network/Main
bash hpc/make_mea_lanes.sh 4          # writes config_mea_joint_full_lane0..3

cd ~/Deep-Summary-Network
# build the cache ONCE -- concurrent lanes would race on first write
python3 Main/run_optimization.py \
  --config Main/hpc/Config/config_mea_joint_full_lane0.json --dry-run

for L in 0 1 2 3; do
  qsub -v CFG=hpc/Config/config_mea_joint_full_lane${L}.json,RESUME=auto \
       -N dsn_mea_L${L} -o dsn_mea_lane${L}.out \
       Main/hpc/run_mea_joint_search.pbs
done
```

`RESUME=auto` means the same qsub line starts and continues a lane. After a
walltime kill, resubmit unchanged.

```bash
qstat -u $USER
wc -l out/mea_joint_full_lane*/trials.jsonl      # progress per lane
cat out/mea_joint_full_lane0/search_state.json   # running best
```

---

## Geometry at 3 roots per class

3 roots x 6 wells = 18 cultures per class, 36 total; x 9 subregions = **324
trace records**. At `split_fractions` [0.6, 0.2, 0.2] that apportions to
**11 / 4 / 3 cultures per class**, so:

- `U_avail` = 11 training cultures per class
- `cultures_per_class_per_batch` = 9 fits without clamping (9 <= 11)
- `M = C * U_eff * q * (1 + N_s) = 2 * 9 * 1 * 3 = 54` rows per batch
- 8 cross-culture positives per anchor
- window `T = 180 s * 50 Hz = 9000` samples; at `L_c = 60,000` that is 6
  non-overlapping windows per trace, 54 per culture, 1188 training windows

Verified end to end on a fixture of exactly this shape: 324 records -> 36
cultures -> 22 / 8 / 6 culture split, **no culture spanning two splits**.

---

## Decisions worth revisiting

**`train_stride_s = 180`, i.e. non-overlapping.** Chosen because independent
windows are the honest reading of "I need statistics"; overlapping windows
inflate N with correlated samples. Set it to 90 for 12 windows/trace (2376
training windows) at the cost of 50% overlap. One number.

**`windows_per_culture_per_batch = 1`**, matching the l3c run, giving M = 54
against l3c's 81. Raising it to 2 gives M = 108 and uses the 9x larger
per-culture window pool. Deliberately left at 1 so only one thing changed.

**Augmentation parameters are INHERITED** from the synthetic/latent runs, not
chosen for real MEA. Same caveat as O5 in REAL_DATA_FINDINGS: the smoothing
bandwidth was tuned for a whole-culture rate, and whether it suits a subregion
rate pooling E = 9 electrodes is untested.

**`index_base = 0`** remains the open assumption from O1 -- bases 0 and 1 both
fit the 48x48 grid and no filename test separates them.

---

## Verify before the first qsub

```bash
cd ~/Deep-Summary-Network/Main
export PYTHONPATH="$(pwd)"

python3 -m py_compile $(find . -name '*.py' -not -path '*/__pycache__/*') \
  && echo "compile OK"
python3 -c "import make_mea_specs, run_optimization" && echo "imports OK"
python3 Smoke_Tests/smoke_test_mea_specs.py
python3 Smoke_Tests/smoke_test_culture_grouping.py

# CRLF check -- \r is inside ASCII, so a byte scan does NOT catch it
python3 -c "print(open('hpc/run_mea_joint_search.pbs','rb').read().count(b'\r'))"
# must print 0; if not:  sed -i 's/\r$//' hpc/run_mea_joint_search.pbs
```

Expected on the full suite: 36 of 37 pass. `smoke_test_loss_type_wiring.py`
fails 2 of 15 checks ("gate momentum 0 = cumulative", "census is
JSON-serialisable"). That failure is **PRE-EXISTING on main and unrelated** --
reproduced on the unpatched tree, byte-identical before and after.

Also add to `.gitattributes` while you are here, since the repo has 21 `.pbs`
files and the explicit rules cover only `.sh`:

```
*.pbs    text eol=lf
*.slurm  text eol=lf
```
