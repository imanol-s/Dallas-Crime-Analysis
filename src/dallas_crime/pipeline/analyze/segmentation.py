"""ZIP segmentation: deterministic k-means, silhouette, ARI, and cluster stability."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from dallas_crime.pipeline.analyze.core import (
    CLUSTER_MIN_SIZE_THRESHOLD,
    CLUSTER_PRACTICAL_SILHOUETTE_THRESHOLD,
    CLUSTER_STABILITY_ARI_THRESHOLD,
    SEGMENTATION_FEATURE_GROUPS,
    SEGMENTATION_NAMES,
    SEGMENTATION_PREPROCESSING_MODES,
)


def _standardize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = frame[columns].apply(pd.to_numeric, errors="coerce")
    std = standardized.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return (standardized - standardized.mean()) / std


def _prepare_segmentation_working_frame(
    model_df: pd.DataFrame,
    *,
    features: list[str],
    preprocessing_mode: str,
) -> pd.DataFrame:
    working = model_df[["zip", *features]].copy()
    for feature in features:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working = working.dropna(subset=features).reset_index(drop=True)
    if working.empty or preprocessing_mode == "raw":
        return working

    quantile_map = {
        "winsor_5_95": (0.05, 0.95),
        "winsor_10_90": (0.10, 0.90),
    }
    if preprocessing_mode not in quantile_map:
        raise ValueError(f"unsupported segmentation preprocessing mode: {preprocessing_mode}")
    lower_quantile, upper_quantile = quantile_map[preprocessing_mode]
    for feature in features:
        lower_bound = working[feature].quantile(lower_quantile)
        upper_bound = working[feature].quantile(upper_quantile)
        working[feature] = working[feature].clip(lower=lower_bound, upper=upper_bound)
    return working


def _iter_segmentation_feature_sets(available_features: list[str]) -> list[list[str]]:
    feature_sets: list[list[str]] = []
    minimum_size = 2 if len(available_features) >= 2 else len(available_features)
    for size in range(len(available_features), minimum_size - 1, -1):
        for subset in itertools.combinations(available_features, size):
            feature_sets.append(list(subset))
    return feature_sets


def _deterministic_kmeans(values: np.ndarray, *, k: int, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
    row_order = np.argsort(values.sum(axis=1), kind="mergesort")
    anchors = np.linspace(0, len(row_order) - 1, num=k, dtype=int)
    centroids = values[row_order[anchors]].copy()
    labels = np.zeros(len(values), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centroids[cluster_id] = values[mask].mean(axis=0)

    return labels, centroids


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(values) <= len(unique_labels):
        return None

    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    scores: list[float] = []
    for index, label in enumerate(labels):
        same_cluster = labels == label
        same_cluster[index] = False
        intra_cluster = float(distances[index, same_cluster].mean()) if same_cluster.any() else 0.0
        nearest_cluster = min(
            float(distances[index, labels == other_label].mean())
            for other_label in unique_labels
            if other_label != label
        )
        denominator = max(intra_cluster, nearest_cluster)
        if denominator == 0:
            scores.append(0.0)
        else:
            scores.append((nearest_cluster - intra_cluster) / denominator)
    return float(np.mean(scores))


def _cluster_name(domain: str, cluster_order: int, *, cluster_count: int) -> str:
    names = SEGMENTATION_NAMES.get(domain)
    if names:
        if cluster_count == 2 and len(names) >= 3:
            names = (names[0], names[-1])
        if cluster_order < len(names):
            return names[cluster_order]
    return f"{domain}_{cluster_order + 1}"


def _fit_segmentation_labels(
    working: pd.DataFrame,
    *,
    domain: str,
    features: list[str],
    k: int,
) -> tuple[np.ndarray, float | None]:
    standardized = _standardize_frame(working, features)
    values = standardized.to_numpy(dtype=float)
    k = max(2, min(k, len(working)))
    labels, centroids = _deterministic_kmeans(values, k=k)
    centroid_order = np.argsort(centroids.mean(axis=1), kind="mergesort")
    remap = {int(old): int(new) for new, old in enumerate(centroid_order)}
    mapped_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    return mapped_labels, _silhouette_score(values, mapped_labels)


def _evaluate_segmentation_solution(
    working: pd.DataFrame,
    *,
    domain: str,
    features: list[str],
    k: int,
) -> dict[str, object]:
    labels, silhouette = _fit_segmentation_labels(
        working,
        domain=domain,
        features=features,
        k=k,
    )
    ari_values: list[float] = []
    if len(features) > 1:
        for feature_to_drop in features:
            reduced_features = [feature for feature in features if feature != feature_to_drop]
            reduced_labels, _ = _fit_segmentation_labels(
                working[["zip", *reduced_features]].copy(),
                domain=domain,
                features=reduced_features,
                k=k,
            )
            ari = _adjusted_rand_index(labels, reduced_labels)
            if ari is not None:
                ari_values.append(float(ari))

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    min_cluster_size = int(cluster_sizes.min()) if not cluster_sizes.empty else 0
    min_cluster_size_threshold = max(CLUSTER_MIN_SIZE_THRESHOLD, int(np.ceil(len(working) * 0.10)))
    max_cluster_share = float(cluster_sizes.max() / len(working)) if len(working) > 0 else np.nan
    mean_ari = float(np.mean(ari_values)) if ari_values else np.nan
    silhouette_pass = pd.notna(silhouette) and silhouette >= CLUSTER_PRACTICAL_SILHOUETTE_THRESHOLD
    stability_pass = pd.notna(mean_ari) and mean_ari >= CLUSTER_STABILITY_ARI_THRESHOLD
    size_pass = min_cluster_size >= min_cluster_size_threshold
    return {
        "labels": labels,
        "silhouette": silhouette,
        "cluster_sizes": cluster_sizes,
        "cluster_count": int(cluster_sizes.shape[0]),
        "min_cluster_size": min_cluster_size,
        "min_cluster_size_threshold": min_cluster_size_threshold,
        "max_cluster_share": max_cluster_share,
        "leave_one_feature_out_mean_ari": mean_ari,
        "silhouette_pass": int(silhouette_pass),
        "stability_pass": int(stability_pass),
        "size_pass": int(size_pass),
        "practical_utility_pass": int(silhouette_pass and stability_pass and size_pass),
        "selected_k": int(k),
    }


def _select_segmentation_solution(
    model_df: pd.DataFrame,
    *,
    domain: str,
    available_features: list[str],
) -> dict[str, object]:
    candidate_solutions: list[dict[str, object]] = []
    mode_priority = {
        mode: len(SEGMENTATION_PREPROCESSING_MODES) - index
        for index, mode in enumerate(SEGMENTATION_PREPROCESSING_MODES)
    }
    for features in _iter_segmentation_feature_sets(available_features):
        for preprocessing_mode in SEGMENTATION_PREPROCESSING_MODES:
            working = _prepare_segmentation_working_frame(
                model_df,
                features=features,
                preprocessing_mode=preprocessing_mode,
            )
            for k in (2, 3):
                if len(working) < k:
                    continue
                solution = _evaluate_segmentation_solution(
                    working,
                    domain=domain,
                    features=features,
                    k=k,
                )
                solution["working"] = working.copy()
                solution["selected_features"] = tuple(features)
                solution["preprocessing_mode"] = preprocessing_mode
                solution["feature_count"] = len(features)
                solution["mode_priority"] = mode_priority.get(preprocessing_mode, 0)
                candidate_solutions.append(solution)
    if not candidate_solutions:
        return {}
    candidate_solutions.sort(
        key=lambda solution: (
            int(solution["practical_utility_pass"]),
            int(solution["size_pass"]),
            int(solution["stability_pass"]),
            int(solution["silhouette_pass"]),
            int(solution["min_cluster_size"]),
            -float(solution["max_cluster_share"]),
            float(solution["leave_one_feature_out_mean_ari"])
            if pd.notna(solution["leave_one_feature_out_mean_ari"])
            else float("-inf"),
            float(solution["silhouette"])
            if pd.notna(solution["silhouette"])
            else float("-inf"),
            int(solution["feature_count"]),
            int(solution["mode_priority"]),
            -int(solution["selected_k"]),
        ),
        reverse=True,
    )
    return candidate_solutions[0]


def _comb2(value: int) -> float:
    return float(value * (value - 1) / 2)


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float | None:
    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        return None
    contingency = pd.crosstab(pd.Series(labels_a), pd.Series(labels_b))
    total = int(contingency.to_numpy().sum())
    total_pairs = _comb2(total)
    if total_pairs == 0:
        return None
    index = float(sum(_comb2(int(value)) for value in contingency.to_numpy().ravel()))
    row_pairs = float(sum(_comb2(int(value)) for value in contingency.sum(axis=1)))
    col_pairs = float(sum(_comb2(int(value)) for value in contingency.sum(axis=0)))
    expected = (row_pairs * col_pairs) / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (row_pairs + col_pairs)
    denominator = max_index - expected
    if denominator == 0:
        return 1.0
    return float((index - expected) / denominator)


def _build_segmentation_artifacts(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = model_df[["zip"]].copy() if "zip" in model_df.columns else pd.DataFrame()
    profile_rows: list[dict[str, object]] = []

    for domain, feature_group in SEGMENTATION_FEATURE_GROUPS.items():
        available = [column for column in feature_group if column in model_df.columns]
        cluster_column = f"{domain}_cluster"
        if cluster_column not in assignments.columns:
            assignments[cluster_column] = pd.Series(pd.NA, index=assignments.index, dtype="string")
        if not available:
            continue

        solution = _select_segmentation_solution(
            model_df,
            domain=domain,
            available_features=available,
        )
        if not solution:
            continue
        working = solution["working"]
        if len(working) < 3:
            continue
        mapped_labels = np.asarray(solution["labels"], dtype=int)
        silhouette = solution["silhouette"]
        cluster_count = int(solution["cluster_count"])
        selected_features = list(solution["selected_features"])

        domain_assignments = working[["zip"]].copy()
        domain_assignments[cluster_column] = pd.Series(
            [
                _cluster_name(domain, int(label), cluster_count=cluster_count)
                for label in mapped_labels
            ],
            dtype="string",
        )
        assignments = assignments.drop(columns=[cluster_column]).merge(
            domain_assignments,
            on="zip",
            how="left",
        )

        for cluster_id in sorted(np.unique(mapped_labels)):
            cluster_name_str = _cluster_name(domain, int(cluster_id), cluster_count=cluster_count)
            cluster_frame = working.loc[mapped_labels == cluster_id].copy()
            for feature in selected_features:
                profile_rows.append(
                    {
                        "domain": domain,
                        "cluster_label": cluster_name_str,
                        "zip_count": int(len(cluster_frame)),
                        "silhouette_score": silhouette,
                        "feature": feature,
                        "mean_value": float(cluster_frame[feature].mean()),
                    }
                )

    if assignments.empty:
        assignments = pd.DataFrame(
            columns=["zip", "crime_cluster", "socioeconomic_cluster", "market_cluster"]
        )
    assignments = assignments.sort_values("zip", ignore_index=True)

    profiles = pd.DataFrame.from_records(
        profile_rows,
        columns=["domain", "cluster_label", "zip_count", "silhouette_score", "feature", "mean_value"],
    )
    if not profiles.empty:
        profiles = profiles.sort_values(
            ["domain", "cluster_label", "feature"],
            ignore_index=True,
        )
    return assignments, profiles


def _build_cluster_stability_artifacts(model_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "zip_count",
        "feature_count",
        "selected_feature_set",
        "preprocessing_mode",
        "selected_k",
        "cluster_count",
        "min_cluster_size",
        "min_cluster_size_threshold",
        "max_cluster_share",
        "silhouette_score",
        "leave_one_feature_out_mean_ari",
        "silhouette_pass",
        "stability_pass",
        "size_pass",
        "practical_utility_pass",
    ]
    rows: list[dict[str, object]] = []

    for domain, feature_group in SEGMENTATION_FEATURE_GROUPS.items():
        available = [column for column in feature_group if column in model_df.columns]
        if not available:
            continue

        solution = _select_segmentation_solution(
            model_df,
            domain=domain,
            available_features=available,
        )
        if not solution:
            continue
        working = solution["working"]
        if len(working) < 3:
            continue
        rows.append(
            {
                "domain": domain,
                "zip_count": int(len(working)),
                "feature_count": int(solution["feature_count"]),
                "selected_feature_set": ", ".join(solution["selected_features"]),
                "preprocessing_mode": str(solution["preprocessing_mode"]),
                "selected_k": int(solution["selected_k"]),
                "cluster_count": int(solution["cluster_count"]),
                "min_cluster_size": int(solution["min_cluster_size"]),
                "min_cluster_size_threshold": int(solution["min_cluster_size_threshold"]),
                "max_cluster_share": float(solution["max_cluster_share"]),
                "silhouette_score": solution["silhouette"],
                "leave_one_feature_out_mean_ari": solution["leave_one_feature_out_mean_ari"],
                "silhouette_pass": int(solution["silhouette_pass"]),
                "stability_pass": int(solution["stability_pass"]),
                "size_pass": int(solution["size_pass"]),
                "practical_utility_pass": int(solution["practical_utility_pass"]),
            }
        )

    diagnostics = pd.DataFrame.from_records(rows, columns=columns)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("domain", kind="mergesort", ignore_index=True)
    return diagnostics
