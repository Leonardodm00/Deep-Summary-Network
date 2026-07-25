cd "/davinci-1/home/ldellamea/Deep Summary Network/Deep_v2/Main"
conda activate meacnn_cpu
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 inspect_latent_benchmark.py --n-per-class 4 4 4 4 --tau 0.07 \
    --duration-s 900 --out-dir latent_inspect_4c

python3 run_optimization.py --config hpc/config_latent_4class_hard.json --dry-run --verbose

time python3 run_optimization.py --config hpc/config_latent_4class_timing.json \
    --out-dir /tmp/timing --cache-dir /tmp/timing_cache --verbose

qsub hpc/dsn_latent_4class_hard.pbs
qsub hpc/dsn_latent_4class_easypos.pbs
