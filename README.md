# PEARL Federated Learning Experiments

This repository packages the PEARL notebook into a reproducible Python project for remote Federated Learning experiments. The original notebook and raw TeX paper remain in the root as source material; the runnable experiment code now lives under `src/pearl`.

## Project layout

```text
configs/              YAML experiment configurations
src/pearl/            Python package for data, models, selection, training, and plots
src/pearl/cli/        Command-line entry points for remote runs
tests/                Lightweight tests for config and result utilities
data/                 Downloaded datasets, ignored by git
results/              Generated CSVs and figures, ignored by git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a CUDA server, install the PyTorch wheel that matches the server driver first, then run `pip install -e ".[dev]"`.

## Run a smoke experiment

After installing the project, use the console command:

```bash
pearl-run --config configs/smoke.yaml
```

Without installing the project, use the Python wrapper:

```bash
python scripts/run.py --config configs/smoke.yaml
```

This uses small subsets, 5 clients, 2 rounds, and three methods so you can verify the environment before launching the paper-scale job.

## Run the paper-scale configuration

```bash
pearl-run --config configs/paper_er150.yaml --resume
```

Equivalent wrapper form:

```bash
python scripts/run.py --config configs/paper_er150.yaml --resume
```

The `--resume` flag skips method/seed CSV files that already exist in the output directory. This is useful on remote servers where a job may be interrupted.

You can override any flat YAML parameter from the command line:

```bash
pearl-run --config configs/paper_er150.yaml --set rounds=10 --set num_clients=20
```

## Summarize and plot existing results

```bash
pearl-summarize results/paper_er150/results_all.csv --output results/paper_er150/summary_final.csv
pearl-plot results/paper_er150/results_all.csv --output-dir results/paper_er150/figures
```

Equivalent wrapper form:

```bash
python scripts/summarize.py results/paper_er150/results_all.csv --output results/paper_er150/summary_final.csv
python scripts/plot.py results/paper_er150/results_all.csv --output-dir results/paper_er150/figures
```

The runner also writes these files automatically after completing a full run.

## CIFAR-10 experiment suites

The CIFAR-10 extension keeps the paper's 50-client, 150-round, three-seed,
Dirichlet `alpha=0.3` setup. It uses a wider version of the same
encoder-decoder model (`model_width=32`, `latent_dim=128`) and evaluates global
metrics every five rounds to control runtime without changing training.

Run the three budget-matched graph configs, full-neighbour D-PSGD references,
and then the server references:

```bash
python scripts/run_cifar10_suite.py --suite core
```

Run only the dynamic ER resilience experiments:

```bash
python scripts/run_cifar10_suite.py --suite dynamic
```

Run every selected CIFAR-10 config in order:

```bash
python scripts/run_cifar10_suite.py --suite all
```

Resume is enabled by default. Use `--dry-run` to inspect the commands,
`--no-resume` to force reruns, and `--continue-on-error` to continue after a
failed config. The console reports suite, config, method/seed, and round
progress with elapsed times.

Before leaving a long run unattended, execute the complete preflight only:

```bash
python scripts/run_cifar10_suite.py --suite all --preflight-only
```

The normal suite performs this preflight automatically. It validates every
YAML and output path, checks disk and device availability, opens or downloads
CIFAR-10, and runs one synthetic round through all 18 implemented methods.
Use `--skip-dataset-check` or `--skip-smoke` only when intentionally bypassing
those checks.

For a detached Linux run:

```bash
tmux new -s pearl-cifar10
python -u scripts/run_cifar10_suite.py --suite all
```

Each invocation creates `results/cifar10/_suite_runs/<timestamp>/` containing
`suite.log`, one log per config, a smoke-test log, and an atomically updated
`status.json`. Resume accepts only CSVs containing the expected method, seed,
and final round; incomplete files are rerun. Per-run and combined CSV writes
are atomic, and `results_partial.csv` is refreshed after each method/seed.

### Comparison families

The direct decentralised configs under `configs/cifar10/` use the same graph
and at most one model-bearing peer per active client and round:

- `local_only`, `random_peer`, and `static_peer`
- `dpsgd_one_peer`: one-neighbour D-PSGD-style full-model mixing
- `model_similarity`: last-layer similarity selection, DFLStar-style signal
- `prototype_quality_exploration`: similarity, self-quality, and frequency signal
- `anchor_quality`: the PEARL-AQ validation/anchor selector
- `pearl_full`: the proposed selector and representation exchange

The full-neighbour D-PSGD configs are stored separately and labelled
`decentralized_reference`, because every client receives all active-neighbour
models and therefore exceeds the one-peer budget.

The separate `server_references150.yaml` config contains FedAvg, FedProx,
FedPer, FedRep, and Ditto. These ignore the graph and are reference anchors
rather than budget-matched competitors. FedPer jointly trains local heads and
the shared representation; FedRep alternates head and representation phases;
Ditto trains a FedAvg global model alongside proximal personalised models.

The similarity and validation selectors instantiate related-work signals in
the common PEARL runner. They are controlled signal baselines, not claims of
exact reproduction of the full external systems.

### Dynamic ER scope

The minimum dynamic suite tests node activity `a=0.8,0.6,0.4` and descriptor
refresh periods `r=5,10,20`. The core ER config supplies the shared `a=1.0`
and `r=1` reference, avoiding duplicate runs. Cold-start joins, exact external
peer-selection reproductions, CHOCO-SGD, and pFedMe are intentionally left for
a later expansion.

Results are separated by comparison family and experiment condition:

```text
results/cifar10/decentralized/<topology>_alpha03/
results/cifar10/decentralized_references/dpsgd_full/<topology>_alpha03/
results/cifar10/server/references_alpha03/
results/cifar10/dynamic/dropout/er_a<level>/
results/cifar10/dynamic/staleness/er_r<period>/
```

## Main outputs

Each run writes one CSV per method and seed under `results/<experiment>/runs/`, then combines them into:

- `results_all.csv`: all recorded round-level metrics
- `summary_final.csv`: final-round mean and standard deviation across seeds
- `figures/`: paper-ready plots for accuracy, macro-F1, worst-client accuracy, communication, selection entropy, and negative transfer

Dynamic result CSVs also include active fraction, mean active degree, the
fraction of active clients without an active peer, descriptor refresh period,
and descriptor age. Rows are explicitly labelled `server_reference`,
`decentralized_reference`, or `budget_matched_decentralized`.
