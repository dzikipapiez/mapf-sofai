# Dataset schema

Generation is deliberately ordered:

1. Static features are computed without running a solver.
2. LaCAM produces a solution, runtime, and solution-derived delay features.
3. MAPF-LNS produces the per-second curve. The iteration probe is read from
   this same trace; it is not another solver run.

## Static model features

`num_agents`, `num_obstacles`, `agent_density`, `obstacle_density`,
`avg_shortest_path_distance`, `min_shortest_path_distance`,
`max_shortest_path_distance`, `cells_at_sp_ratio`, `num_total_cells`,
`num_free_cells`, `lower_bound_soc`, `sp_collision_count`,
`sp_vertex_collision_count`, `sp_edge_collision_count`.

The C++ call also returns individual shortest-path distances. They are used
immediately to derive LaCAM delays and are not duplicated as a CSV column.

## LaCAM model features

`lacam_initial_sod`, `lacam_fraction_delayed_agents`, `lacam_avg_delay`,
`lacam_delay_90th_percentile`.

`lacam_initial_soc`, `lacam_runtime_seconds`, and `lacam_solved` are retained
as outcomes/context, but are not aliases for those four features.

## LNS runtime

In `dataset_raw.csv`, `lns_wall_seconds` is the complete wrapper wall time for
one repetition. In `dataset.csv`, it is the mean across repetitions. It
includes input-path creation, the native process, trace and path parsing, and
temporary-directory cleanup.

## Repetitions and curves

`dataset_raw.csv` has one row per instance, neighbourhood size and repetition.
It retains the repetition number, LNS seed, status, wall time, probe values,
and that repetition's `lns_best_soc_by_second` and
`lns_best_sod_by_second` curves.

`dataset.csv` has one row per instance and neighbourhood size. It retains the
pointwise mean curves, mean wall/probe values, `lns_repetitions`, and
`lns_successful_repetitions`. This is the dataset consumed by the layered
policy. The dataset retains unsolved LaCAM rows, although layered-policy
training filters them out to reproduce the original experiment split.

## LNS probe features

For the default five iterations:

- `lns_sod_reduction_after_5_iterations`
- `lns_sod_reduction_per_agent_after_5_iterations`
- `lns_internal_runtime_after_5_iterations`
- `lns_5iter_probe_incomplete`

Changing `--probe-iterations` changes the number in these names.

## Targets

The averaged CSV adds only canonical target names:

- `target_sod_improvement`
- `target_sodn_improvement`
- `target_sodlb_improvement`
- `target_sodfraction_improvement`
- `target_elbow_time_seconds`

It does not persist the old aliases (`lns_final_sod`,
`final_lns_improvement_per_agent`, `lacam_sodn`, and similar), cap columns, or
log-transformed copies. Those are exact transformations and should be created
by the model that needs them.

Both CSVs retain `lns_best_soc_by_second` and `lns_best_sod_by_second`; only
`dataset.csv` averages them across repetitions.
