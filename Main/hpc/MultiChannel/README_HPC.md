# Running the channel-subset extractor on the HPC (PBS / davinci-1)

## Files
    channel_subset_extraction.py        core extractor (import this)
    channel_subset_viz.py               headless PNG renderers + CLI
    run_channel_subset_extraction.py    batch CLI driver (saves traces.npz + PNGs)
    generate_burst_data.py              REQUIRED dependency (IFR primitive, imported lazily)
    run_extractor.pbs                   single-recording PBS job
    run_extractor_array.pbs             array job over many recordings
    smoke_test_channel_subsets_*.py     tests (run once after transfer)

Keep them in the SAME directory (the driver imports the other two by name).

## Environment
Only numpy, scipy, matplotlib are needed (NO torch -- this is extraction, not
training). Example:
    conda activate brian_env
    python3 -c "import numpy, scipy, matplotlib; print('ok')"
If matplotlib is missing:  pip install matplotlib

## Sanity check after transfer (do this once)
    python3 -m py_compile *.py
    for s in stage1 stage2 stage3 stage4 mode3 stage5; do \
        python3 smoke_test_channel_subsets_$s.py; done
    python3 smoke_test_channel_subset_viz.py
All should print "ALL ... CHECKS PASSED" and exit 0.

## One recording, from the command line
    python3 run_channel_subset_extraction.py /path/to/folder \
        --out-dir out/ --mode multichannel --fs-raw 10110.09 --base 0
Modes: multichannel (C,K) | per_region_single (C single-channel samples) |
whole_culture (1,K). Output: out/traces.npz (+ PNGs unless --no-plots).

## Confirm the index base / orientation (IMPORTANT, still unconfirmed)
Render the electrode map on a REAL multi-electrode folder and look at it:
    python3 channel_subset_viz.py /path/to/folder --mode multichannel \
        --base 0 --out-dir out/
Open out/subregion_map.png. If the active region sits in the wrong place or an
"out of grid" GeometryError is raised, re-run with --base 1.

## Submit as a PBS job
    qsub run_extractor.pbs                 # edit FOLDER/OUT/MODE inside first
Array over many recordings (list folders in RECORDINGS.txt, one per line):
    qsub run_extractor_array.pbs           # edit #PBS -J range to match line count

## Using the extractor from your own Python
    from channel_subset_extraction import extract_channel_subsets
    traces, fs_ifr = extract_channel_subsets(folder, mode="multichannel",
        n_subsets=9, electrodes_per_subset=9, mfr_threshold=0.1,
        fs_raw=10110.09, index_base=0)
    # traces is a LIST; len(traces) = samples this recording contributes
    #   multichannel      -> [ (C,K) ]        (one C-channel sample)
    #   per_region_single -> [ (K,) ] * C     (C single-channel samples)
    #   whole_culture     -> [ (K,) ]         (one single-channel sample)

## ASCII / encoding note (davinci, MobaXterm, scp from Windows)
All .py here are pure ASCII on purpose. If you edit them on a Windows tool and
hit a SyntaxError about a byte like 0x97, the file picked up a non-ASCII char;
re-save as UTF-8/ASCII with Unix line endings.
