from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from xgboost import XGBClassifier, XGBRegressor

from ..features import FeatureSet, LACAM_FEATURES, STATIC_FEATURES
from ..problem import MapfProblem
from ..solution import MapfSolution
from .base import S1Request, S2Request, SequenceContext


LNS_TIMES = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0)
NEIGHBOURHOOD_SIZES = (4, 8, 16, 32)
LNS_TIME_FEATURE = "lns_horizon_seconds"
LNS_BASE_FEATURES = (*STATIC_FEATURES, *LACAM_FEATURES, "lns_neighborhood_size")
LNS_FEATURES = (*LNS_BASE_FEATURES, LNS_TIME_FEATURE)
LACAM_HORIZON = 40
SURVIVAL_FOLDS = 5
SURVIVAL_LIMIT_PERCENTILE = 0.8
LNS_PARAMETERS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=3.0,
)
SURVIVAL_PARAMETERS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=5.0,
)


@dataclass(frozen=True)
class TrainingConfig:
    split_seed: int = 42
    jobs: int = 1
    survival_limit_percentile: float = SURVIVAL_LIMIT_PERCENTILE


class LayeredVariableNeighbourhoodMetacognitiveModuleV1:
    """Variable-neighbour metacognitive module with a conservative survival ensemble."""

    charge_feature_time = True
    early_stop_seconds: float | None = None

    def __init__(
        self,
        survival: XGBClassifier | list[XGBClassifier],
        lns_models: XGBRegressor | list[XGBRegressor],
        lacam_cdf: np.ndarray,
        average_lns_curve: np.ndarray,
        mean_lacam_runtime: float,
        horizon: int,
        neighborhood_sizes: tuple[int, ...] = NEIGHBOURHOOD_SIZES,
        seed: int = 1,
        survival_limit_percentile: float = SURVIVAL_LIMIT_PERCENTILE,
        survival_runtime_multiplier: float = 1.0,
    ):
        self.survival = survival if isinstance(survival, list) else [survival]
        self.lns_models = lns_models if isinstance(lns_models, list) else [lns_models]
        self.lacam_cdf = lacam_cdf
        self.average_lns_curve = average_lns_curve
        self.mean_lacam_runtime = mean_lacam_runtime
        self.horizon = horizon
        self.neighborhood_sizes = neighborhood_sizes
        self.seed = seed
        self.survival_limit_percentile = survival_limit_percentile
        self.survival_runtime_multiplier = survival_runtime_multiplier
        self.remaining_seconds = 0.0

    def start_sequence(self, budget: float, instances: int) -> None:
        self.remaining_seconds = max(0.0, float(budget))

    def observe(self, stage: str, seconds: float) -> None:
        self.remaining_seconds = max(0.0, self.remaining_seconds - seconds)

    @classmethod
    def load(
        cls, directory: str | Path, seed: int = 1
    ) -> "LayeredVariableNeighbourhoodMetacognitiveModuleV1":
        directory = Path(directory)
        parameters = json.loads(
            (directory / "parameters.json").read_text(encoding="utf-8")
        )
        paths = sorted((directory / "models").glob("lacam_*.json"))
        if not paths:
            paths = [directory / "models" / "lacam.json"]
        expected = int(parameters.get("survival_models", 1))
        if len(paths) != expected:
            raise FileNotFoundError(
                f"Expected {expected} LaCAM models, found {len(paths)}"
            )
        survival = []
        for path in paths:
            model = XGBClassifier()
            model.load_model(path)
            survival.append(model)
        neighborhoods = tuple(parameters["neighborhood_sizes"])
        if parameters.get("quality_model_type") != "pooled_time_neighborhood_feature":
            raise ValueError(
                "Metacognitive-module artifacts use the retired per-action quality models; "
                "prepare the metacognitive module again to train pooled time-neighbourhood models"
            )
        if parameters.get("quality_model_features") != list(LNS_FEATURES):
            raise ValueError("Pooled LNS model features do not match the metacognitive module")
        if parameters.get("quality_model_horizons") != list(LNS_TIMES):
            raise ValueError("Pooled LNS model horizons do not match the metacognitive module")
        paths = sorted(
            (directory / "models").glob("lns_time_neighborhood_[0-4].json")
        )
        expected = int(parameters["quality_models"])
        if len(paths) != expected:
            raise FileNotFoundError(
                f"Expected {expected} pooled LNS quality models, found {len(paths)}"
            )
        models = []
        for path in paths:
            model = XGBRegressor()
            model.load_model(path)
            models.append(model)
        return cls(
            survival,
            models,
            pd.read_csv(directory / "population_lacam_cdf.csv")[
                "termination_probability"
            ].to_numpy(),
            pd.read_csv(directory / "average_lns_improvement.csv")[
                "average_improvement"
            ].to_numpy(),
            parameters["mean_lacam_runtime_seconds"],
            parameters["lacam_horizon_seconds"],
            neighborhoods,
            seed,
            parameters.get("survival_limit_percentile", SURVIVAL_LIMIT_PERCENTILE),
        )

    def choose_s1(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
    ) -> S1Request:
        maximum = min(self.horizon, int(self.remaining_seconds))
        if maximum == 0:
            return S1Request(0.0, info={"termination_probability": 0.0})
        frame = pd.DataFrame(
            [
                {**features.static, "time_step": second}
                for second in range(1, maximum + 1)
            ]
        )
        currents, member_scores, model_limits = [], [], []
        for model in self.survival:
            hazards = np.clip(model.predict_proba(frame)[:, 1], 1e-9, 1 - 1e-9)
            current = np.r_[0.0, 1.0 - np.cumprod(1.0 - hazards)]
            scores = _allocation(
                dict(enumerate(current)),
                self.lacam_cdf,
                self.remaining_seconds,
                context.instances_left - 1,
                range(maximum + 1),
            )
            currents.append(current)
            member_scores.append(scores)
            model_limits.append(max(scores, key=lambda t: (scores[t], -t)))
        scores = {
            second: float(np.mean([values[second] for values in member_scores]))
            for second in range(maximum + 1)
        }
        model_limit = float(np.quantile(model_limits, self.survival_limit_percentile))
        unmultiplied_limit = max(
            float(model_limit), self.remaining_seconds / context.instances_left
        )
        limit = min(
            float(LACAM_HORIZON),
            self.survival_runtime_multiplier * unmultiplied_limit,
        )
        return S1Request(
            float(limit),
            info={
                "expected_solved": float(
                    np.interp(
                        model_limit, np.arange(len(scores)), list(scores.values())
                    )
                ),
                "termination_probability": float(
                    np.mean(
                        [
                            np.interp(limit, np.arange(len(current)), current)
                            for current in currents
                        ]
                    )
                ),
                "model_limit": float(model_limit),
                "model_limits": [float(value) for value in model_limits],
                "model_limit_percentile": self.survival_limit_percentile,
                "survival_runtime_multiplier": self.survival_runtime_multiplier,
                "unmultiplied_limit": unmultiplied_limit,
                "fair_share": self.remaining_seconds / context.instances_left,
                "scores": scores,
            },
        )

    def choose_s2(
        self,
        context: SequenceContext,
        problem: MapfProblem,
        features: FeatureSet,
        solution: MapfSolution,
    ) -> S2Request | None:
        base = {**features.static, **features.lacam}
        candidates = pd.DataFrame(
            [
                {**base, "lns_neighborhood_size": neighborhood}
                for neighborhood in self.neighborhood_sizes
            ],
            columns=LNS_BASE_FEATURES,
        )
        frame = _lns_design(candidates)
        member_predictions = np.stack(
            [
                np.expm1(model.predict(frame)).reshape(
                    len(self.neighborhood_sizes), len(LNS_TIMES)
                )
                for model in self.lns_models
            ]
        )
        predicted_curves = member_predictions.mean(axis=0)
        cap = features.lacam["lacam_initial_sod"] / features.static["lower_bound_soc"]
        future = context.instances_left - 1
        spendable = max(0.0, self.remaining_seconds - future * self.mean_lacam_runtime)
        skip = _allocation(
            {0.0: 0.0},
            self.average_lns_curve,
            spendable,
            future,
            (0.0,),
        )
        scores = {(None, 0.0): skip[0.0]}
        predictions = {}
        for neighborhood_index, neighborhood in enumerate(self.neighborhood_sizes):
            curve = np.maximum.accumulate(
                np.clip(
                    predicted_curves[neighborhood_index],
                    0.0,
                    cap,
                )
            )
            predictions[neighborhood] = curve
            limits = tuple(seconds for seconds in LNS_TIMES if seconds <= spendable)
            scores.update(
                {
                    (neighborhood, seconds): score
                    for seconds, score in _allocation(
                        dict(zip(LNS_TIMES, curve)),
                        self.average_lns_curve,
                        spendable,
                        future,
                        limits,
                    ).items()
                }
            )
        neighborhood, limit = max(
            scores,
            key=lambda action: (
                scores[action],
                -action[1],
                -(action[0] or 0),
            ),
        )
        if neighborhood is None:
            return None
        return S2Request(
            limit,
            neighborhood,
            self.seed,
            early_stop_seconds=self.early_stop_seconds,
            info={
                "predicted_improvement": float(
                    predictions[neighborhood][LNS_TIMES.index(limit)]
                ),
                "scores": {
                    (
                        "skip"
                        if candidate_neighborhood is None
                        else f"n{candidate_neighborhood}:{seconds:g}"
                    ): score
                    for (candidate_neighborhood, seconds), score in scores.items()
                },
            },
        )


def prepare(
    dataset: str | Path,
    output: str | Path,
    config: TrainingConfig = TrainingConfig(),
    exclude_instances: set[str] | None = None,
) -> None:
    """Train five pooled time-and-neighbourhood quality models."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    models.mkdir(exist_ok=True)
    if not 0.5 <= config.survival_limit_percentile <= 1.0:
        raise ValueError("survival_limit_percentile must be between 0.5 and 1")

    dataset = Path(dataset)
    lns = _lns_rows(dataset)
    runtimes = _runtime_rows(dataset)
    complete = lns.groupby(
        ["instance_id", "lacam_seed"]
    ).lns_neighborhood_size.nunique()
    complete_keys = set(complete[complete == len(NEIGHBOURHOOD_SIZES)].index)
    excluded = set(exclude_instances or ())
    runtimes = runtimes[~runtimes.instance_id.astype(str).isin(excluded)].reset_index(
        drop=True
    )
    runtime_keys = set(
        runtimes[["instance_id", "lacam_seed"]].itertuples(index=False, name=None)
    )
    eligible_keys = complete_keys & runtime_keys
    lns = lns[
        [
            key in eligible_keys
            for key in lns[["instance_id", "lacam_seed"]].itertuples(
                index=False, name=None
            )
        ]
    ]

    domain = _domain(runtimes)
    strata = pd.qcut(runtimes.observed_step, 10, labels=False, duplicates="drop")
    indices = np.arange(len(runtimes))
    folds = list(_folds(runtimes, indices, strata, config.split_seed))
    fold_of = np.empty(len(runtimes), dtype=int)
    for fold, (_, holdout) in enumerate(folds, start=1):
        fold_of[holdout] = fold
    split = pd.DataFrame(
        {
            "instance_id": runtimes.instance_id,
            "lacam_seed": runtimes.lacam_seed,
            "fold": fold_of,
        }
    )
    split.to_csv(output / "instance_split.csv", index=False)

    actual = {
        (neighborhood, seconds): []
        for neighborhood in NEIGHBOURHOOD_SIZES
        for seconds in LNS_TIMES
    }
    predictions = {key: [] for key in actual}
    for fold, (fit, holdout) in enumerate(folds):
        fit_ids = set(runtimes.iloc[fit].instance_id.astype(str))
        holdout_ids = set(runtimes.iloc[holdout].instance_id.astype(str))
        fit_rows = lns[lns.instance_id.astype(str).isin(fit_ids)]
        holdout_rows = lns[lns.instance_id.astype(str).isin(holdout_ids)]
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=config.split_seed + fold,
            n_jobs=config.jobs,
            tree_method="hist",
            **LNS_PARAMETERS,
        )
        model.fit(_lns_design(fit_rows), np.log1p(_lns_targets(fit_rows)))
        model.save_model(models / f"lns_time_neighborhood_{fold}.json")
        fold_predictions = np.maximum(
            0.0,
            np.expm1(model.predict(_lns_design(holdout_rows))),
        ).reshape(len(holdout_rows), len(LNS_TIMES))
        for neighborhood in NEIGHBOURHOOD_SIZES:
            selected = holdout_rows.lns_neighborhood_size.eq(neighborhood).to_numpy()
            for column, seconds in enumerate(LNS_TIMES):
                key = neighborhood, seconds
                actual[key].append(
                    holdout_rows.loc[selected, f"target_{seconds:g}"].to_numpy()
                )
                predictions[key].append(fold_predictions[selected, column])

    lns_metrics = {}
    for neighborhood in NEIGHBOURHOOD_SIZES:
        neighborhood_metrics = {}
        for seconds in LNS_TIMES:
            key = neighborhood, seconds
            observed = np.concatenate(actual[key])
            predicted = np.concatenate(predictions[key])
            neighborhood_metrics[f"{seconds:g}"] = {
                "mae": float(mean_absolute_error(observed, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            }
        lns_metrics[str(neighborhood)] = neighborhood_metrics

    columns = [*STATIC_FEATURES, "time_step"]
    survival_targets, probabilities = [], []
    for fold, (fit, holdout) in enumerate(folds):
        train_rows = _person_period(runtimes.iloc[fit])
        validation_rows = _person_period(runtimes.iloc[holdout])
        survival = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **SURVIVAL_PARAMETERS,
            random_state=config.split_seed + fold,
            n_jobs=config.jobs,
            tree_method="hist",
            early_stopping_rounds=75,
        )
        survival.fit(
            train_rows[columns],
            train_rows.target,
            eval_set=[(validation_rows[columns], validation_rows.target)],
            verbose=False,
        )
        survival.save_model(models / f"lacam_{fold}.json")
        survival_targets.append(validation_rows.target.to_numpy())
        probabilities.append(survival.predict_proba(validation_rows[columns])[:, 1])
    survival_target = np.concatenate(survival_targets)
    probability = np.concatenate(probabilities)

    horizon = LACAM_HORIZON
    cdf = _empirical_cdf(runtimes, horizon)
    pd.DataFrame(
        {"seconds": np.arange(horizon + 1), "termination_probability": cdf}
    ).to_csv(output / "population_lacam_cdf.csv", index=False)
    average_lns = _average_best_curve(lns)
    pd.DataFrame(
        {"seconds": np.arange(len(average_lns)), "average_improvement": average_lns}
    ).to_csv(output / "average_lns_improvement.csv", index=False)
    solved = runtimes.event_observed
    _json(
        output / "parameters.json",
        {
            "lacam_horizon_seconds": horizon,
            "mean_lacam_runtime_seconds": float(
                runtimes.loc[solved, "lacam_runtime_seconds"].mean()
            ),
            "neighborhood_sizes": list(NEIGHBOURHOOD_SIZES),
            "objective": "expected number of solved instances",
            "training_domain": domain,
            "split_protocol": "stratified",
            "lns_hyperparameters": LNS_PARAMETERS,
            "survival_models": SURVIVAL_FOLDS,
            "quality_model_type": "pooled_time_neighborhood_feature",
            "quality_model_features": list(LNS_FEATURES),
            "quality_model_horizons": list(LNS_TIMES),
            "quality_models": SURVIVAL_FOLDS,
            "survival_limit_percentile": config.survival_limit_percentile,
        },
    )
    _json(
        output / "metrics.json",
        {
            "survival_row_metrics": {
                "roc_auc": float(roc_auc_score(survival_target, probability)),
                "average_precision": float(
                    average_precision_score(survival_target, probability)
                ),
                "log_loss": float(log_loss(survival_target, probability)),
            },
            "lns_oof_metrics": lns_metrics,
            "fold_instances": split.groupby("fold").instance_id.nunique().to_dict(),
        },
    )


def _lns_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "lacam_seed" not in frame:
        frame["lacam_seed"] = 0
    if "repetition" in frame:
        raise ValueError(
            "Layered metacognitive module training requires averaged dataset.csv, "
            "not repetition-level dataset_raw.csv"
        )
    required_columns = {
        "lns_repetitions",
        "lns_successful_repetitions",
        "lns_best_sod_by_second",
        *LNS_BASE_FEATURES,
    }
    missing = sorted(required_columns - set(frame))
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    frame = frame[frame.lns_neighborhood_size.isin(NEIGHBOURHOOD_SIZES)].copy()
    parsed = frame.lns_best_sod_by_second.map(json.loads)
    valid = parsed.map(len) > max(LNS_TIMES)
    frame, parsed = frame.loc[valid].copy(), parsed.loc[valid]
    curves, improvements = [], []
    for row, values in zip(frame.itertuples(), parsed):
        curve = np.asarray(values, dtype=float)
        curve = np.minimum.accumulate(curve)
        improvement = (
            np.maximum(0.0, row.lacam_initial_sod - curve) / row.lower_bound_soc
        )
        curves.append(improvement)
        improvements.append(
            [
                float(np.interp(seconds, np.arange(len(improvement)), improvement))
                for seconds in LNS_TIMES
            ]
        )
    frame["improvement_curve"] = curves
    for index, seconds in enumerate(LNS_TIMES):
        frame[f"target_{seconds:g}"] = [values[index] for values in improvements]
    required = [
        *LNS_BASE_FEATURES,
        *(f"target_{seconds:g}" for seconds in LNS_TIMES),
    ]
    frame[required] = frame[required].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=required)


def _lns_design(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.loc[
        frame.index.repeat(len(LNS_TIMES)), list(LNS_BASE_FEATURES)
    ].reset_index(drop=True)
    rows[LNS_TIME_FEATURE] = np.tile(LNS_TIMES, len(frame))
    return rows[list(LNS_FEATURES)]


def _lns_targets(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"target_{seconds:g}" for seconds in LNS_TIMES]
    return frame[columns].to_numpy().reshape(-1)


def _runtime_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "lacam_seed" not in frame:
        frame["lacam_seed"] = 0
    columns = [
        "source",
        "instance_id",
        "map_name",
        "lacam_seed",
        "lns_neighborhood_size",
        "lacam_solved",
        "lacam_runtime_seconds",
        *STATIC_FEATURES,
    ]
    frame = frame[columns]
    numeric = ["lacam_seed", "lacam_solved", "lacam_runtime_seconds"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric)
    frame["lacam_seed"] = frame.lacam_seed.astype(int)
    frame = (
        frame.sort_values(["instance_id", "lacam_seed", "lns_neighborhood_size"])
        .drop_duplicates(["instance_id", "lacam_seed"])
        .drop(columns="lns_neighborhood_size")
        .reset_index(drop=True)
    )
    frame["event_observed"] = frame.lacam_solved.eq(1) & frame.lacam_runtime_seconds.le(
        LACAM_HORIZON
    )
    frame["event_step"] = (
        np.ceil(frame.lacam_runtime_seconds)
        .astype(int)
        .clip(lower=1, upper=LACAM_HORIZON)
    )
    frame["observed_step"] = np.where(
        frame.event_observed,
        frame.event_step,
        LACAM_HORIZON,
    )
    return frame


def _domain(frame: pd.DataFrame) -> str:
    sources = frame.source.astype(str).str.lower()
    if sources.str.contains("pogema").all():
        return "pogema"
    if sources.str.contains("movingai").all():
        return "movingai"
    raise ValueError("Training data must contain only MovingAI or only Pogema rows")


def _folds(frame, indices, strata, seed):
    return StratifiedGroupKFold(SURVIVAL_FOLDS, shuffle=True, random_state=seed).split(
        indices,
        strata.iloc[indices],
        groups=frame.iloc[indices].instance_id,
    )


def _person_period(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.observed_step.astype(int).to_numpy()
    rows = frame.loc[frame.index.repeat(counts), list(STATIC_FEATURES)].reset_index(
        drop=True
    )
    rows["time_step"] = np.concatenate([np.arange(1, count + 1) for count in counts])
    target = np.zeros(len(rows), dtype=np.int8)
    endpoints = np.cumsum(counts) - 1
    target[endpoints[frame.event_observed.to_numpy()]] = 1
    rows["target"] = target
    return rows


def _empirical_cdf(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    survival, values = 1.0, [0.0]
    for second in range(1, horizon + 1):
        at_risk = np.count_nonzero(frame.observed_step >= second)
        events = np.count_nonzero(frame.event_observed & (frame.event_step == second))
        if at_risk:
            survival *= 1 - events / at_risk
        values.append(1 - survival)
    return np.asarray(values)


def _average_best_curve(frame: pd.DataFrame, horizon: int = 101) -> np.ndarray:
    best_curves = []
    group_columns = ["instance_id"]
    if "lacam_seed" in frame:
        group_columns.append("lacam_seed")
    for _, rows in frame.groupby(group_columns):
        curves = [
            row.improvement_curve[:horizon]
            for row in rows.itertuples()
            if len(row.improvement_curve) >= horizon
        ]
        if len(curves) == len(NEIGHBOURHOOD_SIZES):
            best_curves.append(np.max(curves, axis=0))
    if not best_curves:
        raise ValueError(f"No complete 0..{horizon - 1}-second LNS curves")
    return np.maximum.accumulate(np.mean(best_curves, axis=0))


def _allocation(current: dict[float, float], average, budget, future, limits):
    scores = {}
    for limit in limits:
        future_value = 0.0
        if future:
            per_instance = max(0.0, budget - limit) / future
            future_value = future * float(
                np.interp(per_instance, np.arange(len(average)), average)
            )
        scores[limit] = float(current[limit] + future_value)
    return scores


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
