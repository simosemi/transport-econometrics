# RPNB on HPCC

Install once in the project directory:

```bash
bash hpcc/install_rpnb_env.sh
```

Submit a model run:

```bash
sbatch hpcc/run_rpnb.sbatch /path/to/crashes.csv /path/to/model.yaml /path/to/rpnb_runs 10
```

The job writes a timestamped run directory containing the model spec, logs,
coefficient tables, predictions, marginal effects, and workbook/HTML exports.

Resume after a walltime interruption:

```bash
sbatch hpcc/run_rpnb.sbatch --resume /path/to/rpnb_runs/rpnb_YYYYMMDD_HHMMSS_microseconds
```

The resumed job loads `model_spec.yaml`, `run_metadata.yaml`, and
`checkpoints/checkpoint_latest.npz` from the previous run directory.

Resource guidance depends mostly on observation count, number of random
parameters, number of simulation draws, and panel size.

| Observations | Draws | Random parameters | Suggested CPUs | Suggested memory |
|---:|---:|---:|---:|---:|
| 10,000 | 200 | 1 | 1-2 | 8G |
| 100,000 | 500 | 2 | 2-4 | 16G |
| 500,000 | 500 | 2-4 | 4-8 | 32G |
| 1,000,000 | 500 | 4+ | 8+ | 64G+ |

For large jobs, set `chunk_size` in the YAML spec to limit memory used per
likelihood block. Use `workers` carefully because each worker needs memory for
its chunk payload.

Recommended checkpoint setting:

```yaml
estimation:
  checkpoint_interval: 10
```

Each checkpoint stores the current optimizer iteration, parameter vector,
log-likelihood, and optimizer metadata. Use `checkpoint_interval: 0` to disable
checkpoint writing.
