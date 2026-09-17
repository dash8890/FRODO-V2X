from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Config:
    """Configuration for the standalone finite-field privacy tests."""

    prime: int = 65_537
    episodes: int = 350
    steps_per_episode: int = 100
    coalition_size: int = 4
    private_coefficients: int = 3
    operand_rows: int = 3
    common_dimension: int = 12
    kernel_columns: int = 2
    mi_bins: int = 20
    mi_bin_sensitivity: Tuple[int, ...] = (20, 30, 50)
    pseudocount: float = 1e-3
    bootstrap_replicates: int = 2_000
    mi_permutation_replicates: int = 1_000
    attack_permutation_replicates: int = 10_000
    attack_train_fraction: float = 0.50
    seed: int = 20_260_818
    output_directory: Path = Path("privacy_test_outputs_aligned")


CONFIG = Config()
SCHEMES = ("GMDS", "SPC", "FRODO")
COLORS = {
    "GMDS": "#4C78A8",
    "SPC": "#F28E2B",
    "FRODO": "#2E8B57",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",      # Changed from "serif"
        "font.sans-serif": ["Arial"],     # Added explicitly
        "font.size": 60,
        "axes.labelsize": 60,
        "axes.titlesize": 60,
        "legend.fontsize": 40,
        "xtick.labelsize": 60,
        "ytick.labelsize": 60,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


# ============================================================
# 1. EXACT FINITE-FIELD LINEAR ALGEBRA
# ============================================================


def rref_mod(
    matrix: np.ndarray,
    prime: int,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Return reduced row-echelon form and pivot columns over GF(prime)."""

    reduced = np.asarray(matrix, dtype=np.int64).copy() % prime
    rows, columns = reduced.shape
    pivots = []
    pivot_row = 0

    for column in range(columns):
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if candidates.size == 0:
            continue

        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]

        inverse = pow(int(reduced[pivot_row, column]), prime - 2, prime)
        reduced[pivot_row] = (reduced[pivot_row] * inverse) % prime

        for row in range(rows):
            if row == pivot_row or reduced[row, column] == 0:
                continue
            factor = int(reduced[row, column])
            reduced[row] = (
                reduced[row] - factor * reduced[pivot_row]
            ) % prime

        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break

    return reduced, tuple(pivots)


def rank_mod(matrix: np.ndarray, prime: int) -> int:
    """Return matrix rank over GF(prime)."""

    return len(rref_mod(matrix, prime)[1])


def nullspace_basis(matrix: np.ndarray, prime: int) -> np.ndarray:
    """Return a row basis N satisfying matrix @ N.T = 0 over GF(prime)."""

    reduced, pivots = rref_mod(matrix, prime)
    columns = reduced.shape[1]
    free_columns = [column for column in range(columns) if column not in pivots]
    basis = []

    for free_column in free_columns:
        vector = np.zeros(columns, dtype=np.int64)
        vector[free_column] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            vector[pivot_column] = (
                -reduced[pivot_row, free_column]
            ) % prime
        basis.append(vector)

    if not basis:
        return np.empty((0, columns), dtype=np.int64)
    return np.stack(basis, axis=0)


def inverse_mod(matrix: np.ndarray, prime: int) -> np.ndarray:
    """Invert a square matrix over GF(prime)."""

    matrix = np.asarray(matrix, dtype=np.int64) % prime
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("matrix must be square")

    augmented = np.concatenate(
        [matrix.copy(), np.eye(rows, dtype=np.int64)],
        axis=1,
    )

    for pivot in range(rows):
        candidates = np.flatnonzero(augmented[pivot:, pivot])
        if candidates.size == 0:
            raise ValueError("matrix is singular over the selected field")

        selected = pivot + int(candidates[0])
        if selected != pivot:
            augmented[[pivot, selected]] = augmented[[selected, pivot]]

        inverse = pow(int(augmented[pivot, pivot]), prime - 2, prime)
        augmented[pivot] = (augmented[pivot] * inverse) % prime

        for row in range(rows):
            if row == pivot or augmented[row, pivot] == 0:
                continue
            factor = int(augmented[row, pivot])
            augmented[row] = (
                augmented[row] - factor * augmented[pivot]
            ) % prime

    return augmented[:, rows:] % prime


def random_full_column_rank_matrix(
    rng: np.random.Generator,
    rows: int,
    columns: int,
    prime: int,
) -> np.ndarray:
    """Draw a full-column-rank matrix over GF(prime)."""

    if rows < columns:
        raise ValueError("rows must be at least columns")

    while True:
        matrix = rng.integers(
            0,
            prime,
            size=(rows, columns),
            dtype=np.int64,
        )
        if rank_mod(matrix, prime) == columns:
            return matrix


def sample_from_row_space(
    rng: np.random.Generator,
    rows: int,
    basis: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Sample independent row-wise uniform combinations of a row basis."""

    if basis.shape[0] == 0:
        return np.zeros((rows, basis.shape[1]), dtype=np.int64)

    coefficients = rng.integers(
        0,
        prime,
        size=(rows, basis.shape[0]),
        dtype=np.int64,
    )
    return (coefficients @ basis) % prime


def centered_field(values: np.ndarray, prime: int) -> np.ndarray:
    """Map GF(prime) representatives to approximately [-0.5, 0.5)."""

    values = np.asarray(values, dtype=np.int64) % prime
    signed = np.where(values > prime // 2, values - prime, values)
    return signed.astype(np.float64) / float(prime)


# ============================================================
# 2. SERVER-SPECIFIC POLYNOMIAL ENCODING AND MASKING
# ============================================================


def evaluation_matrix(config: Config) -> np.ndarray:
    """Return the coalition evaluation matrix for the private polynomial."""

    points = np.arange(1, config.coalition_size + 1, dtype=np.int64)
    exponents = np.arange(config.private_coefficients, dtype=np.int64)
    return pow_matrix(points, exponents, config.prime)


def pow_matrix(
    points: np.ndarray,
    exponents: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Calculate point-by-exponent powers over GF(prime)."""

    result = np.empty((points.size, exponents.size), dtype=np.int64)
    for row, point in enumerate(points):
        for column, exponent in enumerate(exponents):
            result[row, column] = pow(int(point), int(exponent), prime)
    return result


def encode_private_coefficients(
    coefficients: np.ndarray,
    evaluations: np.ndarray,
    prime: int,
) -> Tuple[np.ndarray, ...]:
    """
    Evaluate a private matrix-valued polynomial at every coalition point.

    coefficients has shape
        (number of coefficients, operand rows, common dimension).
    """

    encoded = np.einsum(
        "nc,crd->nrd",
        evaluations,
        coefficients,
        dtype=np.int64,
    ) % prime
    return tuple(encoded[index] for index in range(encoded.shape[0]))


def build_joint_product_operator(
    evaluations: np.ndarray,
    kernels: Tuple[np.ndarray, ...],
    prime: int,
) -> np.ndarray:
    """
    Build the linear map from one row of private coefficients to all
    coalition-assigned products.

    A vector in its nullspace changes the underlying private input while
    leaving every server-specific encoded product unchanged.
    """

    coefficient_count = evaluations.shape[1]
    common_dimension = kernels[0].shape[0]
    kernel_columns = kernels[0].shape[1]
    operator = np.zeros(
        (
            len(kernels) * kernel_columns,
            coefficient_count * common_dimension,
        ),
        dtype=np.int64,
    )

    for server, kernel in enumerate(kernels):
        row_slice = slice(
            server * kernel_columns,
            (server + 1) * kernel_columns,
        )
        for coefficient in range(coefficient_count):
            column_slice = slice(
                coefficient * common_dimension,
                (coefficient + 1) * common_dimension,
            )
            operator[row_slice, column_slice] = (
                evaluations[server, coefficient] * kernel.T
            ) % prime

    return operator % prime


def build_episode_context(
    rng: np.random.Generator,
    config: Config,
) -> Dict[str, object]:
    """Build public kernels, evaluation maps, and relevant nullspaces."""

    evaluations = evaluation_matrix(config)
    kernels = tuple(
        random_full_column_rank_matrix(
            rng,
            config.common_dimension,
            config.kernel_columns,
            config.prime,
        )
        for _ in range(config.coalition_size)
    )

    server_nullspaces = tuple(
        nullspace_basis(kernel.T, config.prime) for kernel in kernels
    )
    concatenated_kernel = np.concatenate(kernels, axis=1)
    common_mask_space = nullspace_basis(
        concatenated_kernel.T,
        config.prime,
    )
    if common_mask_space.shape[0] == 0:
        raise RuntimeError("the common masking nullspace is empty")

    joint_operator = build_joint_product_operator(
        evaluations,
        kernels,
        config.prime,
    )
    joint_private_nullspace = nullspace_basis(joint_operator, config.prime)
    if joint_private_nullspace.shape[0] == 0:
        raise RuntimeError("the joint encode-and-product nullspace is empty")

    interpolation_rows = evaluations[: config.private_coefficients]
    interpolation_inverse = inverse_mod(interpolation_rows, config.prime)

    return {
        "evaluations": evaluations,
        "kernels": kernels,
        "server_nullspaces": server_nullspaces,
        "common_mask_space": common_mask_space,
        "joint_private_nullspace": joint_private_nullspace,
        "interpolation_inverse": interpolation_inverse,
    }


def build_fixed_masks(
    rng: np.random.Generator,
    context: Dict[str, object],
    config: Config,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Draw masks retained throughout one episode for fixed-mask regimes."""

    server_nullspaces = context["server_nullspaces"]
    sfpd_masks = tuple(
        sample_from_row_space(
            rng,
            config.operand_rows,
            nullspace,
            config.prime,
        )
        for nullspace in server_nullspaces
    )

    global_mask = sample_from_row_space(
        rng,
        config.operand_rows,
        context["common_mask_space"],
        config.prime,
    )
    smds_masks = tuple(global_mask.copy() for _ in server_nullspaces)

    return {"SPC": sfpd_masks, "GMDS": smds_masks}


def protect_encoded_operands(
    scheme: str,
    encoded: Tuple[np.ndarray, ...],
    rng: np.random.Generator,
    context: Dict[str, object],
    fixed_masks: Dict[str, Tuple[np.ndarray, ...]],
    config: Config,
) -> Tuple[np.ndarray, ...]:
    """Protect every server-specific encoded operand."""

    protected = []
    for server, operand in enumerate(encoded):
        if scheme == "FRODO":
            mask = sample_from_row_space(
                rng,
                config.operand_rows,
                context["server_nullspaces"][server],
                config.prime,
            )
        else:
            mask = fixed_masks[scheme][server]
        protected.append((operand + mask) % config.prime)
    return tuple(protected)


def reconstruct_coefficients(
    encoded_values: Tuple[np.ndarray, ...],
    interpolation_inverse: np.ndarray,
    coefficient_count: int,
    prime: int,
) -> np.ndarray:
    """Interpolate private coefficient matrices from coalition evaluations."""

    selected = np.stack(encoded_values[:coefficient_count], axis=0)
    return np.einsum(
        "cn,nrd->crd",
        interpolation_inverse,
        selected,
        dtype=np.int64,
    ) % prime


# ============================================================
# 3. TEST 1: TEMPORAL RECONSTRUCTION-LINKAGE MI
# ============================================================


def generate_temporal_data(
    config: Config,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate consecutive private changes and changes reconstructed from
    coalition transcripts.

    Fixed masks cancel under differencing. Fresh FRODO masks do not.
    """

    rng = np.random.default_rng(config.seed)
    private_summary = np.empty(
        (config.episodes, config.steps_per_episode - 1),
        dtype=np.int64,
    )
    observed_summary = {
        scheme: np.empty_like(private_summary) for scheme in SCHEMES
    }

    projection = rng.integers(
        1,
        config.prime,
        size=(
            config.private_coefficients,
            config.operand_rows,
            config.common_dimension,
        ),
        dtype=np.int64,
    )

    for episode in range(config.episodes):
        context = build_episode_context(rng, config)
        fixed_masks = build_fixed_masks(rng, context, config)
        previous_private = None
        previous_transcript = {scheme: None for scheme in SCHEMES}

        for step in range(config.steps_per_episode):
            private = rng.integers(
                0,
                config.prime,
                size=(
                    config.private_coefficients,
                    config.operand_rows,
                    config.common_dimension,
                ),
                dtype=np.int64,
            )
            encoded = encode_private_coefficients(
                private,
                context["evaluations"],
                config.prime,
            )

            transcripts = {
                scheme: protect_encoded_operands(
                    scheme,
                    encoded,
                    rng,
                    context,
                    fixed_masks,
                    config,
                )
                for scheme in SCHEMES
            }

            if step > 0:
                private_change = (private - previous_private) % config.prime
                private_summary[episode, step - 1] = int(
                    np.sum(private_change * projection, dtype=np.int64)
                    % config.prime
                )

                for scheme in SCHEMES:
                    transcript_change = tuple(
                        (current - previous) % config.prime
                        for current, previous in zip(
                            transcripts[scheme],
                            previous_transcript[scheme],
                        )
                    )
                    reconstructed = reconstruct_coefficients(
                        transcript_change,
                        context["interpolation_inverse"],
                        config.private_coefficients,
                        config.prime,
                    )
                    observed_summary[scheme][episode, step - 1] = int(
                        np.sum(reconstructed * projection, dtype=np.int64)
                        % config.prime
                    )

            previous_private = private
            previous_transcript = transcripts

    return {
        scheme: {
            "private": private_summary.copy(),
            "observation": observed_summary[scheme],
        }
        for scheme in SCHEMES
    }


def bin_indices(values: np.ndarray, bins: int, prime: int) -> np.ndarray:
    """Map finite-field representatives to fixed equal-width bin indices."""

    indices = (np.asarray(values, dtype=np.int64) * bins) // prime
    return np.minimum(indices, bins - 1).astype(np.int64)


def joint_counts_by_episode(
    private: np.ndarray,
    observation: np.ndarray,
    bins: int,
    prime: int,
) -> np.ndarray:
    """Build one joint count matrix per episode."""

    x_bins = bin_indices(private, bins, prime)
    y_bins = bin_indices(observation, bins, prime)
    counts = np.empty((private.shape[0], bins, bins), dtype=np.float64)
    for episode in range(private.shape[0]):
        flat = x_bins[episode] * bins + y_bins[episode]
        counts[episode] = np.bincount(
            flat,
            minlength=bins * bins,
        ).reshape(bins, bins)
    return counts


def mutual_information_from_counts(
    counts: np.ndarray,
    pseudocount: float,
) -> float:
    """Calculate plug-in discrete mutual information in nats."""

    joint = np.asarray(counts, dtype=np.float64) + pseudocount
    joint /= joint.sum()
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    return float(np.sum(joint * np.log(joint / (marginal_x @ marginal_y))))


def bootstrap_mi(
    episode_counts: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Episode-block-bootstrap confidence interval for aggregate MI."""

    episodes = episode_counts.shape[0]
    point = mutual_information_from_counts(
        episode_counts.sum(axis=0),
        config.pseudocount,
    )
    flattened = episode_counts.reshape(episodes, -1)
    samples = np.empty(config.bootstrap_replicates, dtype=np.float64)

    for replicate in range(config.bootstrap_replicates):
        indices = rng.integers(0, episodes, size=episodes)
        resampled = flattened[indices].sum(axis=0).reshape(
            episode_counts.shape[1:]
        )
        samples[replicate] = mutual_information_from_counts(
            resampled,
            config.pseudocount,
        )

    standard_error = float(samples.std(ddof=1))
    lower = max(0.0, point - 1.96 * standard_error)
    upper = point + 1.96 * standard_error
    return point, float(lower), float(upper)


def temporal_permutation_null(
    private: np.ndarray,
    observation: np.ndarray,
    bins: int,
    config: Config,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Episode-preserving permutation null for histogram MI.

    Observation order is independently permuted within every episode.
    """

    x_bins = bin_indices(private, bins, config.prime)
    y_bins = bin_indices(observation, bins, config.prime)
    observed_counts = joint_counts_by_episode(
        private,
        observation,
        bins,
        config.prime,
    ).sum(axis=0)
    observed_mi = mutual_information_from_counts(
        observed_counts,
        config.pseudocount,
    )

    null_values = np.empty(config.mi_permutation_replicates, dtype=np.float64)
    for replicate in range(config.mi_permutation_replicates):
        aggregate = np.zeros((bins, bins), dtype=np.float64)
        for episode in range(private.shape[0]):
            permuted = rng.permutation(y_bins[episode])
            flat = x_bins[episode] * bins + permuted
            aggregate += np.bincount(
                flat,
                minlength=bins * bins,
            ).reshape(bins, bins)
        null_values[replicate] = mutual_information_from_counts(
            aggregate,
            config.pseudocount,
        )

    p_value = float(
        (np.count_nonzero(null_values >= observed_mi) + 1)
        / (null_values.size + 1)
    )
    return float(null_values.mean()), p_value


def run_temporal_test(
    data: Dict[str, Dict[str, np.ndarray]],
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run temporal-linkage MI, bootstrap, permutation, and bin sensitivity."""

    bootstrap_rng = np.random.default_rng(config.seed + 10)
    permutation_rng = np.random.default_rng(config.seed + 11)
    rows = []
    sensitivity_rows = []

    for scheme in SCHEMES:
        private = data[scheme]["private"]
        observation = data[scheme]["observation"]
        counts = joint_counts_by_episode(
            private,
            observation,
            config.mi_bins,
            config.prime,
        )
        estimate, lower, upper = bootstrap_mi(
            counts,
            config,
            bootstrap_rng,
        )
        null_mean, p_value = temporal_permutation_null(
            private,
            observation,
            config.mi_bins,
            config,
            permutation_rng,
        )
        rows.append(
            {
                "Scheme": scheme,
                "Temporal MI (nats)": estimate,
                "95% CI lower": lower,
                "95% CI upper": upper,
                "Permutation-null MI": null_mean,
                "Excess MI": max(0.0, estimate - null_mean),
                "Temporal p-value": p_value,
            }
        )

        for bins in config.mi_bin_sensitivity:
            sensitivity_counts = joint_counts_by_episode(
                private,
                observation,
                bins,
                config.prime,
            )
            sensitivity_rows.append(
                {
                    "Scheme": scheme,
                    "Bins": bins,
                    "MI (nats)": mutual_information_from_counts(
                        sensitivity_counts.sum(axis=0),
                        config.pseudocount,
                    ),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(sensitivity_rows)


# ============================================================
# 4. TEST 2: SAME-PRODUCT DISTINGUISHABILITY
# ============================================================


def make_same_product_private_pair(
    rng: np.random.Generator,
    context: Dict[str, object],
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct distinct underlying private inputs with equal SV products."""

    first = rng.integers(
        0,
        config.prime,
        size=(
            config.private_coefficients,
            config.operand_rows,
            config.common_dimension,
        ),
        dtype=np.int64,
    )

    while True:
        row_differences = sample_from_row_space(
            rng,
            config.operand_rows,
            context["joint_private_nullspace"],
            config.prime,
        )
        difference = row_differences.reshape(
            config.operand_rows,
            config.private_coefficients,
            config.common_dimension,
        ).transpose(1, 0, 2)
        if np.any(difference):
            break

    second = (first + difference) % config.prime
    return first, second


def verify_same_products(
    first: np.ndarray,
    second: np.ndarray,
    context: Dict[str, object],
    config: Config,
) -> int:
    """Return the maximum modular mismatch over all coalition products."""

    encoded_first = encode_private_coefficients(
        first,
        context["evaluations"],
        config.prime,
    )
    encoded_second = encode_private_coefficients(
        second,
        context["evaluations"],
        config.prime,
    )
    maximum = 0
    for server, kernel in enumerate(context["kernels"]):
        product_first = (encoded_first[server] @ kernel) % config.prime
        product_second = (encoded_second[server] @ kernel) % config.prime
        mismatch = (product_first - product_second) % config.prime
        maximum = max(maximum, int(np.max(mismatch)))
    return maximum


def stratified_train_test_indices(
    labels: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce a stratified train/test split within one episode."""

    train_indices = []
    test_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        split = int(round(train_fraction * indices.size))
        split = min(max(split, 1), indices.size - 1)
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    return np.asarray(train_indices), np.asarray(test_indices)


def nearest_centroid_attack(
    features: np.ndarray,
    labels: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Run a train/test two-sample distinguishability attack."""

    train, test = stratified_train_test_indices(
        labels,
        config.attack_train_fraction,
        rng,
    )
    centroid_zero = features[train][labels[train] == 0].mean(axis=0)
    centroid_one = features[train][labels[train] == 1].mean(axis=0)
    test_features = features[test]
    test_labels = labels[test]
    distance_zero = np.sum((test_features - centroid_zero) ** 2, axis=1)
    distance_one = np.sum((test_features - centroid_one) ** 2, axis=1)
    score = distance_zero - distance_one
    prediction = (score > 0).astype(np.int64)
    accuracy = float(np.mean(prediction == test_labels))
    auc = float(roc_auc_score(test_labels, score))
    return accuracy, auc


def generate_attack_metrics(
    config: Config,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    """Generate episode-level attack metrics and exact equality checks."""

    rng = np.random.default_rng(config.seed + 20)
    split_rng = np.random.default_rng(config.seed + 21)
    metrics = {
        scheme: {
            "accuracy": np.empty(config.episodes, dtype=np.float64),
            "auc": np.empty(config.episodes, dtype=np.float64),
        }
        for scheme in SCHEMES
    }
    maximum_product_mismatch = 0

    for episode in range(config.episodes):
        context = build_episode_context(rng, config)
        fixed_masks = build_fixed_masks(rng, context, config)
        first, second = make_same_product_private_pair(rng, context, config)
        maximum_product_mismatch = max(
            maximum_product_mismatch,
            verify_same_products(first, second, context, config),
        )

        encoded_candidates = {
            0: encode_private_coefficients(
                first,
                context["evaluations"],
                config.prime,
            ),
            1: encode_private_coefficients(
                second,
                context["evaluations"],
                config.prime,
            ),
        }
        labels = np.tile(
            np.array([0, 1], dtype=np.int64),
            config.steps_per_episode // 2,
        )
        if labels.size < config.steps_per_episode:
            labels = np.append(labels, 0)
        rng.shuffle(labels)

        for scheme in SCHEMES:
            rows = []
            for label in labels:
                transcript = protect_encoded_operands(
                    scheme,
                    encoded_candidates[int(label)],
                    rng,
                    context,
                    fixed_masks,
                    config,
                )
                rows.append(
                    np.concatenate(
                        [
                            centered_field(value, config.prime).ravel()
                            for value in transcript
                        ]
                    )
                )
            features = np.stack(rows, axis=0)
            accuracy, auc = nearest_centroid_attack(
                features,
                labels,
                config,
                split_rng,
            )
            metrics[scheme]["accuracy"][episode] = accuracy
            metrics[scheme]["auc"][episode] = auc

    return metrics, maximum_product_mismatch


def bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Episode-block-bootstrap confidence interval for a mean."""

    indices = rng.integers(
        0,
        values.size,
        size=(replicates, values.size),
        dtype=np.int64,
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lower), float(upper)


def sign_flip_pvalue(
    values: np.ndarray,
    null_value: float,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """One-sided episode-level randomization p-value for above-chance attack."""

    centered = values - null_value
    observed = centered.mean()
    exceedances = 0
    completed = 0
    batch_size = 1_000

    while completed < replicates:
        current = min(batch_size, replicates - completed)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(current, values.size),
            replace=True,
        )
        permuted = (signs * centered).mean(axis=1)
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += current

    return float((exceedances + 1) / (replicates + 1))


def summarize_attack(
    metrics: Dict[str, Dict[str, np.ndarray]],
    config: Config,
) -> pd.DataFrame:
    """Summarize classifier accuracy, AUROC, intervals, and chance test."""

    bootstrap_rng = np.random.default_rng(config.seed + 22)
    permutation_rng = np.random.default_rng(config.seed + 23)
    rows = []

    for scheme in SCHEMES:
        accuracy, accuracy_lower, accuracy_upper = bootstrap_mean_ci(
            metrics[scheme]["accuracy"],
            config.bootstrap_replicates,
            bootstrap_rng,
        )
        auc, auc_lower, auc_upper = bootstrap_mean_ci(
            metrics[scheme]["auc"],
            config.bootstrap_replicates,
            bootstrap_rng,
        )
        p_value = sign_flip_pvalue(
            metrics[scheme]["accuracy"],
            0.5,
            config.attack_permutation_replicates,
            permutation_rng,
        )
        rows.append(
            {
                "Scheme": scheme,
                "Attack accuracy": accuracy,
                "Accuracy 95% CI lower": accuracy_lower,
                "Accuracy 95% CI upper": accuracy_upper,
                "AUROC": auc,
                "AUROC 95% CI lower": auc_lower,
                "AUROC 95% CI upper": auc_upper,
                "Attack p-value": p_value,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 5. PAPER-READY REPORTING AND FIGURES
# ============================================================


def format_interval(estimate: float, lower: float, upper: float) -> str:
    """Format a point estimate and confidence interval."""

    return f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"


def build_paper_table(
    temporal_report: pd.DataFrame,
    attack_report: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise paper-ready results table."""

    combined = temporal_report.merge(attack_report, on="Scheme", how="inner")
    return pd.DataFrame(
        {
            "Scheme": combined["Scheme"],
            "Temporal MI, nats (95% CI)": [
                format_interval(
                    row["Temporal MI (nats)"],
                    row["95% CI lower"],
                    row["95% CI upper"],
                )
                for _, row in combined.iterrows()
            ],
            "Null MI": [
                f"{value:.4f}" for value in combined["Permutation-null MI"]
            ],
            "Temporal p-value": [
                f"{value:.4g}" for value in combined["Temporal p-value"]
            ],
            "Attack accuracy (95% CI)": [
                format_interval(
                    row["Attack accuracy"],
                    row["Accuracy 95% CI lower"],
                    row["Accuracy 95% CI upper"],
                )
                for _, row in combined.iterrows()
            ],
            "AUROC (95% CI)": [
                format_interval(
                    row["AUROC"],
                    row["AUROC 95% CI lower"],
                    row["AUROC 95% CI upper"],
                )
                for _, row in combined.iterrows()
            ],
            "Attack p-value": [
                f"{value:.4g}" for value in combined["Attack p-value"]
            ],
        }
    )


def print_report(
    temporal_report: pd.DataFrame,
    sensitivity: pd.DataFrame,
    attack_report: pd.DataFrame,
    maximum_product_mismatch: int,
    config: Config,
) -> pd.DataFrame:
    """Print human-readable and LaTeX-ready results."""

    paper_table = build_paper_table(temporal_report, attack_report)
    print("\nALIGNED FINITE-FIELD PRIVACY TESTS")
    print(
        f"GF({config.prime}); {config.episodes} episodes x "
        f"{config.steps_per_episode} steps; "
        f"{config.episodes * (config.steps_per_episode - 1):,} "
        "within-episode consecutive pairs"
    )
    print(
        f"Coalition size={config.coalition_size}; private polynomial "
        f"coefficients={config.private_coefficients}; "
        f"bootstrap replicates={config.bootstrap_replicates:,}"
    )
    print(
        "Maximum modular same-product mismatch: "
        f"{maximum_product_mismatch} (required: 0)"
    )
    print("\nREPORTABLE RESULTS")
    print(paper_table.to_string(index=False))

    print("\nMI BIN-COUNT SENSITIVITY")
    sensitivity_pivot = sensitivity.pivot(
        index="Scheme",
        columns="Bins",
        values="MI (nats)",
    ).reindex(SCHEMES)
    print(sensitivity_pivot.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nLATEX TABLE BODY")
    print(
        "Scheme & Temporal MI (95\\% CI) & Null MI & Temporal $p$ & "
        "Attack accuracy (95\\% CI) & AUROC (95\\% CI) & Attack $p$ \\\\"
    )
    for _, row in paper_table.iterrows():
        print(
            f"{row['Scheme']} & "
            f"{row['Temporal MI, nats (95% CI)']} & "
            f"{row['Null MI']} & "
            f"{row['Temporal p-value']} & "
            f"{row['Attack accuracy (95% CI)']} & "
            f"{row['AUROC (95% CI)']} & "
            f"{row['Attack p-value']} \\\\"
        )
    return paper_table


def plot_temporal_mi(report: pd.DataFrame, config: Config) -> None:
    """Display and save the temporal-linkage MI figure."""

    ordered = report.set_index("Scheme").loc[list(SCHEMES)].reset_index()
    values = ordered["Temporal MI (nats)"].to_numpy()
    errors = np.maximum(
        0.0,
        np.vstack(
            [
                values - ordered["95% CI lower"].to_numpy(),
                ordered["95% CI upper"].to_numpy() - values,
            ]
        ),
    )
    positions = np.arange(len(SCHEMES))
    
    # FIX: Completely different color palette for the three schemes
    temporal_palette = {
        "GMDS": "#C0C0C0",  # Muted Gray
        "SPC": "#008080",   # Muted Teal
        "FRODO": "#B07AA1", # Soft Purple
    }

    figure, axis = plt.subplots(figsize=(20, 18))
    axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=6,
        width=0.62,
        color=[temporal_palette[scheme] for scheme in SCHEMES],
        edgecolor="black",
        linewidth=1.1,
    )
    axis.scatter(
        positions,
        ordered["Permutation-null MI"],
        marker="D",
        s=400,
        color="black",
        label="Permutation null",
        zorder=3,
    )
    axis.set_xticks(positions, SCHEMES)
    axis.set_ylabel("Projected temporal MI (nats)")
    axis.set_title("Consecutive-task temporal linkage")
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    # ADD THESE TWO LINES HERE
    y_bottom, y_top = axis.get_ylim()
    axis.set_ylim(-0.5*y_bottom, y_top * 1.05) 

    # YOUR UPDATED LEGEND LINE
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=40,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
    )

    figure.tight_layout()
    figure.savefig(config.output_directory / "temporal_linkage_mi.pdf")
    figure.savefig(config.output_directory / "temporal_linkage_mi.png")
    plt.show()


def plot_attack(report: pd.DataFrame, config: Config) -> None:
    """Display and save same-product attack accuracy and AUROC."""

    ordered = report.set_index("Scheme").loc[list(SCHEMES)].reset_index()
    positions = np.arange(len(SCHEMES))
    width = 0.35
    accuracy = ordered["Attack accuracy"].to_numpy()
    auc = ordered["AUROC"].to_numpy()
    accuracy_error = np.vstack(
        [
            accuracy - ordered["Accuracy 95% CI lower"].to_numpy(),
            ordered["Accuracy 95% CI upper"].to_numpy() - accuracy,
        ]
    )
    auc_error = np.vstack(
        [
            auc - ordered["AUROC 95% CI lower"].to_numpy(),
            ordered["AUROC 95% CI upper"].to_numpy() - auc,
        ]
    )

    figure, axis = plt.subplots(figsize=(20, 18))
    
    # FIX: Assign a single distinct color for the Accuracy metric
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        yerr=accuracy_error,
        capsize=5,
        color="#4C78A8",  # Standard blue for Accuracy
        edgecolor="black",
        label="Accuracy",
    )
    
    # FIX: Assign a single distinct color for the AUROC metric
    axis.bar(
        positions + width / 2,
        auc,
        width,
        yerr=auc_error,
        capsize=5,
        color="#F28E2B",  # Standard orange for AUROC
        edgecolor="black",
        hatch="//",
        alpha=0.8,
        label="AUROC",
    )
    
    axis.axhline(0.5, color="black", linestyle="--", linewidth=4.0, label="Chance perf.")
    axis.set_xticks(positions, SCHEMES)
    axis.set_ylim(0.4, 1.03)
    axis.set_ylabel("Distinguishability")
    axis.set_title("Same-product private-input attack")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=40,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
    )
    figure.tight_layout()
    figure.savefig(config.output_directory / "same_product_attack.pdf")
    figure.savefig(config.output_directory / "same_product_attack.png")
    plt.show()


def save_tables(
    paper_table: pd.DataFrame,
    temporal_report: pd.DataFrame,
    sensitivity: pd.DataFrame,
    attack_report: pd.DataFrame,
    config: Config,
) -> None:
    """Save all reportable and diagnostic tables."""

    paper_table.to_csv(config.output_directory / "paper_table.csv", index=False)
    temporal_report.to_csv(
        config.output_directory / "temporal_mi_results.csv",
        index=False,
    )
    sensitivity.to_csv(
        config.output_directory / "temporal_mi_bin_sensitivity.csv",
        index=False,
    )
    attack_report.to_csv(
        config.output_directory / "same_product_attack_results.csv",
        index=False,
    )


def main(config: Config = CONFIG) -> pd.DataFrame:
    """Run both aligned tests, print results, save files, and show figures."""

    config.output_directory.mkdir(parents=True, exist_ok=True)

    print("Generating temporal-linkage data...", flush=True)
    temporal_data = generate_temporal_data(config)
    print("Running temporal linkage tests...", flush=True)
    temporal_report, sensitivity = run_temporal_test(temporal_data, config)

    print("Generating same-product distinguishability data...", flush=True)
    attack_metrics, maximum_product_mismatch = generate_attack_metrics(config)
    if maximum_product_mismatch != 0:
        raise AssertionError(
            "same-product construction failed: modular products differ"
        )
    print("Summarizing attack metrics...", flush=True)
    attack_report = summarize_attack(attack_metrics, config)

    paper_table = print_report(
        temporal_report,
        sensitivity,
        attack_report,
        maximum_product_mismatch,
        config,
    )
    save_tables(
        paper_table,
        temporal_report,
        sensitivity,
        attack_report,
        config,
    )
    plot_temporal_mi(temporal_report, config)
    plot_attack(attack_report, config)
    return paper_table


if __name__ == "__main__":
    main()
