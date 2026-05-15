"""Multimodal PSO peak detection on the smoothed 3D RGB histogram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import M3DHPConfig
from .distance import non_euclidean_distance_matrix


@dataclass(frozen=True)
class PSOResult:
    """Personal-best positions and fitness values from PSO."""

    positions: np.ndarray
    fitness: np.ndarray


def histogram_fitness(histogram: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Look up histogram fitness at floating particle positions."""
    bins = histogram.shape[0]
    idx = np.rint(positions).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    return histogram[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)


def initialize_particles(
    histogram: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seed particles near occupied histogram bins, with small jitter.

    The optimization target is still the smoothed 3D histogram from the paper;
    this initialization simply avoids wasting most particles in empty RGB space.
    """
    bins = histogram.shape[0]
    flat = histogram.reshape(-1)
    occupied = np.flatnonzero(flat > 0)

    if len(occupied) == 0:
        positions = rng.uniform(0, bins - 1, size=(population_size, 3))
    else:
        weights = flat[occupied].astype(np.float64)
        weights = weights / weights.sum()
        sampled = rng.choice(occupied, size=population_size, replace=True, p=weights)
        coords = np.column_stack(np.unravel_index(sampled, histogram.shape)).astype(np.float32)
        positions = coords + rng.uniform(-0.5, 0.5, size=coords.shape)
        positions = np.clip(positions, 0, bins - 1)

    velocities = rng.uniform(-bins / 8.0, bins / 8.0, size=(population_size, 3))
    return positions.astype(np.float32), velocities.astype(np.float32)


def multimodal_pso_peaks(
    smoothed_histogram: np.ndarray,
    config: M3DHPConfig,
) -> PSOResult:
    """Detect local/global histogram peaks using multimodal PSO.

    Each particle is an RGB histogram coordinate. Fitness is the smoothed
    histogram value at that coordinate. The attractor for each particle is the
    personal best with the largest electrostatic interaction force, matching
    the multimodal PSO idea described in the paper.
    """
    rng = np.random.default_rng(config.random_seed)
    bins = smoothed_histogram.shape[0]
    positions, velocities = initialize_particles(
        smoothed_histogram,
        config.population_size,
        rng,
    )

    pbest_positions = positions.copy()
    pbest_fitness = histogram_fitness(smoothed_histogram, pbest_positions)

    for iteration in range(config.iterations):
        progress = iteration / max(config.iterations - 1, 1)
        inertia = config.w_max - (config.w_max - config.w_min) * progress

        # Electrostatic interaction: alpha * f_i * f_j / ned(i, j).
        distances = non_euclidean_distance_matrix(
            pbest_positions,
            pbest_positions,
            normalizer=bins - 1,
        )
        np.fill_diagonal(distances, np.inf)
        forces = (
            config.alpha
            * pbest_fitness[:, None]
            * pbest_fitness[None, :]
            / np.maximum(distances, 1e-12)
        )
        np.fill_diagonal(forces, -np.inf)
        attractor_index = np.argmax(forces, axis=1)
        attractors = pbest_positions[attractor_index]

        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (pbest_positions - positions)
            + config.c2 * r2 * (attractors - positions)
        )
        positions = np.clip(positions + velocities, 0, bins - 1)

        current_fitness = histogram_fitness(smoothed_histogram, positions)
        improved = current_fitness > pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = current_fitness[improved]

        # Local search from the paper: use nearest personal best to refine.
        distances = non_euclidean_distance_matrix(
            pbest_positions,
            pbest_positions,
            normalizer=bins - 1,
        )
        np.fill_diagonal(distances, np.inf)
        nearest_index = np.argmin(distances, axis=1)
        nearest = pbest_positions[nearest_index]
        nearest_fitness = pbest_fitness[nearest_index]

        direction = np.where(
            (nearest_fitness >= pbest_fitness)[:, None],
            nearest - pbest_positions,
            pbest_positions - nearest,
        )
        trial = pbest_positions + config.c1 * rng.random(size=positions.shape) * direction
        trial = np.clip(trial, 0, bins - 1)
        trial_fitness = histogram_fitness(smoothed_histogram, trial)
        improved = trial_fitness > pbest_fitness
        pbest_positions[improved] = trial[improved]
        pbest_fitness[improved] = trial_fitness[improved]

    return PSOResult(positions=pbest_positions, fitness=pbest_fitness)


def filter_dominant_peaks(
    pso_result: PSOResult,
    config: M3DHPConfig,
) -> np.ndarray:
    """Remove weaker PSO peaks that are too close to stronger peaks."""
    if len(pso_result.positions) == 0:
        raise ValueError("PSO returned no peak candidates")

    order = np.argsort(-pso_result.fitness)
    sorted_positions = pso_result.positions[order]
    sorted_fitness = pso_result.fitness[order]
    max_fitness = float(sorted_fitness[0])
    min_fitness = max_fitness * config.min_peak_relative_fitness

    dominant = []
    for candidate, fitness in zip(sorted_positions, sorted_fitness):
        if fitness < min_fitness:
            continue
        if all(np.linalg.norm(candidate - peak) >= config.peak_distance_limit for peak in dominant):
            dominant.append(candidate)
        if len(dominant) >= config.max_clusters:
            break

    if not dominant:
        dominant.append(sorted_positions[0])

    return np.asarray(dominant, dtype=np.float32)
