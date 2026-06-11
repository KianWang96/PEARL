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

## Main outputs

Each run writes one CSV per method and seed under `results/<experiment>/runs/`, then combines them into:

- `results_all.csv`: all recorded round-level metrics
- `summary_final.csv`: final-round mean and standard deviation across seeds
- `figures/`: paper-ready plots for accuracy, macro-F1, worst-client accuracy, communication, selection entropy, and negative transfer
