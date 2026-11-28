# Run VMC with RMSProp for Ising

Install the environment per the [shared instructions](../../shared/README.md#install-once-from-the-courserepository-root), then from `Final_proj`:

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ising-matplotlib
python RMSProp/Ising/training/run.py --n 4 --hidden 3 --batch-size 16 --steps 3 --log-every 1
```

The entry point fixes `--optimizer rmsprop` and rejects another name. Each run
writes to a new directory under `RMSProp/Ising/results/runs/`; `--output` must name a
directory that does not exist. Output files are as for
[VMC](../../VMC/Ising/training/PARAMETERS.md#output-and-evaluation). No optimizer state is
saved, so runs cannot be resumed.
