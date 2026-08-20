
# MAPF Anytime

Code for dissertation experiments on combining fast and deliberative MAPF
solvers.

## Setup

Python 3.10 or 3.11 is required. MAPF-LNS also requires Boost and Eigen3.

```bash
# macOS
xcode-select --install
brew install python@3.11 boost eigen

# Ubuntu/Debian
sudo apt-get install python3-venv build-essential \
  libboost-program-options-dev libboost-system-dev \
  libboost-filesystem-dev libeigen3-dev
```

Create the environment and build LaCAM, MAPF-LNS and the static feature
extractor:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[build,notebooks,pogema]'
scripts/build_native.sh .venv/bin/python
```

## Dataset generation

Dataset generation is independent of the instance source. Its input is a text
file containing one instance-manifest path per line; blank lines and lines
starting with `#` are ignored. Paths may be absolute or relative to the list.

For every instance, the pipeline performs:

1. static feature extraction;
2. one non-anytime LaCAM run and storage of its initial solution;
3. repeated MAPF-LNS runs starting from that exact solution;
4. aggregation into raw and model-ready CSV files.

The defaults are a 40-second LaCAM limit with seed `0`, followed by four
100-second LNS repetitions (seeds `1` through `4`) for each neighbourhood in
`2,4,8,16,32`. The PP replanning limit is 10 seconds. LNS early stopping is
disabled. Instances not solved by LaCAM remain censored in the dataset and
their LNS runs are marked as skipped.

### Local execution

Use a small end-to-end run to verify the installation:

```bash
python scripts/generate_dataset.py run \
  --instances datasets/movingai/instances.txt \
  --run-name smoke \
  --limit 5 \
  --lacam-timeout 5 \
  --lns-timeout 5 \
  --neighborhood-sizes 4 \
  --lns-repetitions 1 \
  --replan-time-limit 1
```

Run the complete default protocol locally with:

```bash
python scripts/generate_dataset.py run \
  --instances datasets/movingai/instances.txt \
  --run-name movingai-replan10
```

The stages can also be invoked separately. Each command belonging to a run
must receive exactly the same options:

```bash
options=(
  --instances datasets/movingai/instances.txt
  --run-name movingai-replan10
)

python scripts/generate_dataset.py lacam-worker "${options[@]}"
python scripts/generate_dataset.py lns-worker "${options[@]}"
python scripts/generate_dataset.py aggregate "${options[@]}"
```

Stopping after `lacam-worker` leaves the static features, LaCAM checkpoints and
initial solutions available for a later LNS stage. CSV aggregation deliberately
requires all configured LNS checkpoints to be complete.

### Slurm execution

Submit the complete dependency chain from the repository root. It uses 500
array workers by default:

```bash
scripts/slurm/submit_dataset.sh \
  datasets/movingai/instances.txt movingai-replan10
```

An optional integer after the run name changes the worker count. All remaining
arguments are forwarded to the generator. For example:

```bash
DATASET_PARTITION=short \
scripts/slurm/submit_dataset.sh \
  datasets/pogema/dataset_instances.txt pogema-replan10 500 \
  --lacam-timeout 40 \
  --lacam-seed 1 \
  --lns-timeout 100 \
  --lns-repetitions 4 \
  --neighborhood-sizes 4,8,16,32 \
  --replan-time-limit 10
```

Solver arrays use `medium` by default with a 48-hour task limit. To use the
`short` partition, which automatically selects its 12-hour limit:

```bash
DATASET_PARTITION=short \
scripts/slurm/submit_dataset.sh \
  datasets/movingai/instances.txt movingai-replan10 500
```

`DATASET_TIME_LIMIT` can override the selected task limit when required:

```bash
DATASET_PARTITION=short DATASET_TIME_LIMIT=10:00:00 \
scripts/slurm/submit_dataset.sh \
  datasets/movingai/instances.txt movingai-replan10 500
```

The final aggregation job always runs on `short`. Solver jobs use homogeneous
Cascade Lake CPUs. LaCAM and LNS are each followed by retry arrays requesting
8, 20, 48 and finally 96 GiB. Every stage skips completed checkpoints, so an
identical submission also resumes an interrupted run rather than repeating
finished work.

### Progress and aggregation

Inspect a default run with:

```bash
python scripts/generate_dataset.py status \
  --instances datasets/movingai/instances.txt \
  --run-name movingai-replan10
```

For a customised run, repeat its custom options in the `status` or `aggregate`
command. This is intentional: `config.json` prevents checkpoints produced by
different protocols or instance lists from being mixed.

Aggregation is normally submitted automatically. It can be repeated locally
after all checkpoints are complete:

```bash
python scripts/generate_dataset.py aggregate \
  --instances datasets/movingai/instances.txt \
  --run-name movingai-replan10
```

Aggregation validates and reads the existing checkpoints in one pass using
eight bounded I/O threads. This does not change checkpoint or CSV formats. Use
`--aggregate-workers` to tune the reader concurrency if necessary.

Each run is stored under `datasets/runs/<name>/`:

```text
config.json
solutions/
checkpoints/static/
checkpoints/lacam/
checkpoints/lns/
dataset_raw.csv
dataset.csv
```

`dataset_raw.csv` contains one row per instance, neighbourhood and LNS
repetition. `dataset.csv` averages repetitions for each instance and
neighbourhood while retaining the complete mean SOC and SoD curves.

To compare different LaCAM seeds, use a different run name for each seed. For
example, four LaCAM seeds with two LNS repetitions each preserve a total of
eight LNS repetitions per instance and neighbourhood:

```bash
for seed in 0 1 2 3; do
  scripts/slurm/submit_dataset.sh \
    datasets/movingai/instances.txt "movingai-lacam-seed${seed}" 500 \
    --lacam-seed "$seed" \
    --lns-repetitions 2
done
```

## Policy experiments

An experiment JSON selects a training dataset, an evaluation instance list,
repetitions, policies and policy-specific runs. The tracked example runs both
learned policies and all three nonlearnable baselines on MovingAI.

```bash
python scripts/run_experiment.py prepare \
  policy_experiments/example_movingai_experiment.json \
  policy_experiments/example_movingai_experiment
```

Preparation excludes every evaluation instance from model training before
performing the train/test split. It writes `plan.json` and prints the required
Slurm array range. Submit that range from the repository root:

```bash
sbatch --array=0-449 \
  scripts/slurm/experiment.slurm \
  policy_experiments/example_movingai_experiment
```

The exact upper index depends on the configuration; use the value printed by
`prepare`. A single task or all tasks can also be run directly:

```bash
python scripts/run_experiment.py run policy_experiments/example_movingai_experiment --task 0
python scripts/run_experiment.py run policy_experiments/example_movingai_experiment
```

Aggregate completed tasks with:

```bash
python scripts/run_experiment.py aggregate policy_experiments/example_movingai_experiment
```

This produces `results.csv`; individual JSON files retain policy decisions and
solver traces.

## S2 solver benchmark

This benchmark compares two continuations of the same initial LaCAM solution:
continuing anytime LaCAM or switching to MAPF-LNS for the remaining time. It
uses the normal `.venv`.

Run a small seeded local sample:

```bash
python scripts/benchmark_s2.py prepare benchmark_runs/s2-smoke \
  --sample-size 31 --time-limit 100 --replan-time-limit 10
python scripts/benchmark_s2.py run benchmark_runs/s2-smoke
python scripts/benchmark_s2.py aggregate benchmark_runs/s2-smoke
```

Run the default 500-instance benchmark as a 500-worker Slurm array on the
short partition and homogeneous Cascade Lake CPUs:

```bash
python scripts/benchmark_s2.py prepare benchmark_runs/s2-replan10 \
  --time-limit 100 --replan-time-limit 10 \
  --neighborhood-sizes 4 8 16 32
python scripts/benchmark_s2.py submit benchmark_runs/s2-replan10 \
  --workers 500
python scripts/benchmark_s2.py status benchmark_runs/s2-replan10
```

Submission automatically schedules aggregation after the array. Results are
written to `benchmark_runs/s2-replan10/results.jsonl`; the analysis and plots
are in `notebooks/benchmarks/s2_benchmark.ipynb`. Each worker runs LaCAM once
per instance and reuses that exact initial solution for every requested LNS
neighbourhood.

## Licence and solver provenance

Original code is released under the MIT licence. Bundled and optional solvers
retain their upstream licences. In particular, LaCAM3 is MIT-licensed and
MAPF-LNS uses the USC Research License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for revisions,
modifications, licences and citations.
