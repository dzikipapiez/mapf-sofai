# MAPF Anytime

Code for dissertation on SOFAI-inspired metacognition for Anytime MAPF.

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

## Reproduction data

The datasets, benchmark output, system results and generated instances are
distributed in the [`data-v1` GitHub release](https://github.com/dzikipapiez/mapf-sofai/releases/tag/data-v1). Download its assets into a temporary directory and
install `zstd` (`brew install zstd` on macOS or `sudo apt-get install zstd` on
Ubuntu/Debian).

From the repository root, place the files as follows:

```bash
ASSETS=/path/to/downloaded/data-v1

zstd -d -f "$ASSETS/movingai_final.csv.zst" -o datasets/movingai_final.csv
zstd -d -f "$ASSETS/pogema_final.csv.zst" -o datasets/pogema_final.csv
zstd -d -f "$ASSETS/s2_results.jsonl.zst" -o notebooks/benchmarks/s2_results.jsonl

zstd -dc "$ASSETS/experiment_instances.tar.zst" | tar -xf -

mkdir -p results
for name in \
  system_results_movingai_movingai \
  system_results_movingai_pogema \
  system_results_pogema_movingai \
  system_results_pogema_pogema
do
  zstd -d -f "$ASSETS/$name.csv.zst" -o "results/$name.csv"
done
```

The instance archive already contains the `datasets/` prefix, so it must be
extracted from the repository root. It contains the 4,950 MovingAI and 4,959
POGEMA manifests used here, their instance lists, and every referenced map and
scenario. The four system-result files are the complete stitched learned and
nonlearnable results, named as `training-dataset_evaluation-dataset`. The
system-analysis notebook selects one of these files through its `RESULTS`
list. `SHA256SUMS` in the release can be used to verify every download.

## Repository structure

```text
.
│
├── datasets/                  Instances from MovingAI and POGEMA-Maze datasets, + lists of instances used for experiments.
│   │
│   ├── movingai/
│   │
│   └── pogema/
│
├── notebooks/
│   │
│   ├── benchmarks/            Comparisons of alternative solvers referenced in the report.
│   │
│   ├── demonstrations/        Small empirical studies illustrating solver behaviour.
│   │
│   ├── models/                S1 termination and S2 quality-model training and evaluation.
│   │
│   └── system_analysis/ 	   Analysis and plots for the system experiments.
│
├── metacognitive_experiments/ Directory for specifications of system experiments. Contains an example experiment.
│
├── scripts/                   Entry points for datasets, benchmarks and experiments.
│   │
│   └── slurm/                 Slurm wrappers for select scripts.
│
└── src/mapf_anytime/          Installable Python package and bundled native solver integrations.
    │
    ├── datasets/              MovingAI and POGEMA instance-generation utilities.
    │
    ├── metacognitive/         Learned and nonlearnable metacognitive modules.
    │
    └── solvers/               Python interfaces, LaCAM, MAPF-LNS, feature extraction utilities.
```

Large generated instances, checkpoints, experiment results, build products and
CSV datasets are intentionally excluded from Git.

## System experiments

An experiment specification selects the training dataset, evaluation instances,
repetitions, metacognitive modules and budgets. The included example evaluates
both learned modules and all three nonlearnable baselines on MovingAI.

Run preparation, every task and aggregation locally as one pipeline:

```bash
CONFIG=metacognitive_experiments/example_movingai_experiment.json
RUN=metacognitive_experiments/example_movingai_experiment
python scripts/run_experiment.py prepare "$CONFIG" "$RUN" && \
  python scripts/run_experiment.py run "$RUN" && \
  python scripts/run_experiment.py aggregate "$RUN"
```

For Slurm, preparation prints the required array range. The included example
produces 450 tasks (`0-449`). Submit the array asynchronously, then aggregate
once it has completed:

```bash
CONFIG=metacognitive_experiments/example_movingai_experiment.json
RUN=metacognitive_experiments/example_movingai_experiment
python scripts/run_experiment.py prepare "$CONFIG" "$RUN"
sbatch --array=0-449 scripts/slurm/experiment.slurm "$RUN"
python scripts/run_experiment.py aggregate "$RUN"
```

The most important specification fields are:

| Field                         | Meaning                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset`                   | Dataset on which predictive models are trained.                                                                                                  |
| `instance_list`             | Instances on which the systems are evaluated.                                                                                                    |
| `repetitions`               | Number of times the experiment is repeated.                                                                                                      |
| `metacognitive_modules`     | Module specifications, each containing a`name`, optional training settings in `prepare`, and one or more parameter dictionaries in `runs`. |
| `runs[].budget`             | Shared time budget for the complete sequence.                                                                                                    |
| `runs[].early_stop_seconds` | Optional MAPF-LNS stagnation limit.                                                                                                              |

## Notebooks - models

These notebooks contain evaluations & training of the predictive models used for the system.

## Notebooks - demonstrations

These self-contained notebooks contain generations of some figures used in the report.

## Notebooks - system analysis

This notebook generates figures analysing the performance of systems. (given the results from system experiments)

## Notebooks - benchmakrs

These notebooks run/visualise results of the benchmarks used:

* mapf_gpt_s1_colab.ipynb runs the mapf gpt benchmark on google colab (we were unable to run it otherwise due to lack of access to GPUs)
* s2_benchmark.ipynb visualises the results of MAPF_LNS and Anytime LaCAM3 as S2 solvers.

## Benchmarks

### S2 solver benchmark

This benchmark compares continuing anytime LaCAM with switching from the same
initial LaCAM solution to MAPF-LNS. From the repository root, run the complete
500-instance MovingAI benchmark locally with:

```bash
RUN=benchmark_runs/s2
python scripts/benchmark_s2.py prepare "$RUN" \
  --time-limit 100 --replan-time-limit 10 \
  --neighborhood-sizes 4 8 16 32
python scripts/benchmark_s2.py run "$RUN"
python scripts/benchmark_s2.py aggregate "$RUN"
```

On Slurm, replace the last two commands with the following asynchronous
submission; it schedules aggregation after the solver array finishes:

```bash
python scripts/benchmark_s2.py submit "$RUN" --workers 500
```

The aggregated output is `benchmark_runs/s2/results.jsonl`. Set `RESULTS_FILE`
in `notebooks/benchmarks/s2_benchmark.ipynb` to this path to reproduce the
analysis and figures.

### LaCAM PIBT candidates

This benchmark compares LaCAM using one PIBT candidate with ten candidates
evaluated sequentially. It samples the same 50 MovingAI instances for both
variants and writes the paired raw results to
`notebooks/benchmarks/lacam_k1_vs_k10.csv`:

```bash
python scripts/compare_lacam_pibt.py \
  --instances datasets/movingai/instances.txt \
  --count 50 --timeout 40 --seed 0 --sample-seed 0 --workers 10
```

## Dataset generation

The input is a text file containing one instance path per line. The lists used
for the experiments are `datasets/movingai/instances.txt` and
`datasets/pogema/instances.txt`.

Run the entire pipeline locally with one command:

```bash
python scripts/generate_dataset.py run --instances INSTANCE_LIST --run-name RUN_NAME [OPTIONS]
```

The experiments used the Slurm wrapper adapted to the Oxford ARC cluster:

```bash
DATASET_PARTITION=medium DATASET_TIME_LIMIT=2-00:00:00 LACAM_CPUS_PER_TASK=1 scripts/slurm/submit_dataset.sh INSTANCE_LIST RUN_NAME WORKERS [OPTIONS]
```

`RUN_NAME` names the output directory and `WORKERS` is the number of Slurm
array workers (default `500`). Repeating an identical command resumes an
interrupted run.

The options accepted by both commands are:

| Option                        |                    Default | Meaning                                                   |
| ----------------------------- | -------------------------: | --------------------------------------------------------- |
| `--run-dir PATH`            | `datasets/runs/RUN_NAME` | Override the output directory.                            |
| `--lacam-timeout SEC`       |                     `40` | Per-instance LaCAM time limit.                            |
| `--lacam-seed N`            |                      `0` | LaCAM random seed.                                        |
| `--lacam-pibt-num N`        |                      `1` | Number of PIBT candidates considered by LaCAM.            |
| `--lns-timeout SEC`         |                    `100` | Time limit for each MAPF-LNS run.                         |
| `--neighborhood-sizes LIST` |              `4,8,16,32` | Comma-separated LNS neighbourhood sizes.                  |
| `--lns-repetitions N`       |                      `4` | Runs per instance and neighbourhood, using seeds`1..N`. |
| `--replan-time-limit SEC`   |                     `10` | Prioritized-planning replanning limit within MAPF-LNS.    |
| `--lns-max-iterations N`    |                `1000000` | Maximum LNS iterations per run.                           |
| `--aggregate-workers N`     |                      `8` | Threads used to read checkpoints during aggregation.      |

The Slurm-only environment variables are:

| Variable                |  Setting above | Meaning                                 |
| ----------------------- | -------------: | --------------------------------------- |
| `DATASET_PARTITION`   |     `medium` | Partition used by solver arrays.        |
| `DATASET_TIME_LIMIT`  | `2-00:00:00` | Per-array-task wall-time limit.         |
| `LACAM_CPUS_PER_TASK` |          `1` | CPUs assigned to each LaCAM array task. |

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
repetition. `dataset.csv` averages LNS repetitions for each instance.

## Licence and solver provenance

Original code is released under the MIT licence. Bundled and optional solvers
retain their upstream licences. In particular, LaCAM3 is MIT-licensed and
MAPF-LNS uses the USC Research License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for revisions,
modifications, licences and citations.
