"""Permutation hypothesis tests for Ohio overdose vineyards.

The null model keeps the Ohio county adjacency surface fixed and randomly
reassigns county time series to county labels. Each trial builds a vineyard
from the scrambled county labels and records one or more vineyard statistics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "pop_normalized_county_drug_death.csv"
DEFAULT_ADJ_PATH = PROJECT_ROOT / "data" / "ohio_neighbors.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "null_distributions" / "ohio_vineyard_null_distribution.csv"


@dataclass(frozen=True)
class StatisticSpec:
    """Metadata for a vineyard test statistic."""

    name: str
    description: str
    compute: Callable[["VineyardSummary", int], float]
    larger_is_extreme: bool = True


@dataclass(frozen=True)
class VineyardSummary:
    """Cached quantities used by statistic functions."""

    finite_persistences: tuple[float, ...]
    all_positive_persistences: tuple[float, ...]

    @classmethod
    def from_vineyard(cls, vineyard: object) -> "VineyardSummary":
        finite = []
        all_positive = []
        for vine in vineyard.vines:
            persistence = float(vine.get_dist_from_diag())
            if persistence > 0:
                all_positive.append(persistence)
                if np.isfinite(persistence):
                    finite.append(persistence)
        finite.sort(reverse=True)
        all_positive.sort(reverse=True)
        return cls(
            finite_persistences=tuple(finite),
            all_positive_persistences=tuple(all_positive),
        )


def _top(values: tuple[float, ...], k: int) -> tuple[float, ...]:
    return values[: max(1, k)]


def top_persistence(summary: VineyardSummary, k: int) -> float:
    del k
    return summary.finite_persistences[0] if summary.finite_persistences else 0.0


def top_k_mean(summary: VineyardSummary, k: int) -> float:
    values = _top(summary.finite_persistences, k)
    return float(np.mean(values)) if values else 0.0


def top_k_sum(summary: VineyardSummary, k: int) -> float:
    return float(np.sum(_top(summary.finite_persistences, k)))


def total_persistence(summary: VineyardSummary, k: int) -> float:
    del k
    return float(np.sum(summary.finite_persistences))


def top_persistence_share(summary: VineyardSummary, k: int) -> float:
    del k
    total = total_persistence(summary, k=1)
    return top_persistence(summary, k=1) / total if total > 0 else 0.0


def finite_vine_count(summary: VineyardSummary, k: int) -> float:
    del k
    return float(len(summary.finite_persistences))


def persistence_entropy(summary: VineyardSummary, k: int) -> float:
    del k
    total = total_persistence(summary, k=1)
    if total <= 0:
        return 0.0
    probabilities = np.array(summary.finite_persistences, dtype=float) / total
    return float(-np.sum(probabilities * np.log(probabilities)))


STATISTICS: dict[str, StatisticSpec] = {
    "top_persistence": StatisticSpec(
        name="top_persistence",
        description="Persistence of the single most persistent finite vine.",
        compute=top_persistence,
    ),
    "top_k_mean": StatisticSpec(
        name="top_k_mean",
        description="Mean persistence among the top k finite vines.",
        compute=top_k_mean,
    ),
    "top_k_sum": StatisticSpec(
        name="top_k_sum",
        description="Total persistence among the top k finite vines.",
        compute=top_k_sum,
    ),
    "total_persistence": StatisticSpec(
        name="total_persistence",
        description="Total persistence across all finite nontrivial vines.",
        compute=total_persistence,
    ),
    "top_persistence_share": StatisticSpec(
        name="top_persistence_share",
        description="Share of total finite persistence carried by the top vine.",
        compute=top_persistence_share,
    ),
    "finite_vine_count": StatisticSpec(
        name="finite_vine_count",
        description="Number of finite nontrivial vines.",
        compute=finite_vine_count,
    ),
    "persistence_entropy": StatisticSpec(
        name="persistence_entropy",
        description="Entropy of the finite persistence distribution.",
        compute=persistence_entropy,
    ),
}

STAT_ALIASES = {
    "top": "top_persistence",
    "top-persistence": "top_persistence",
    "top_k_persistence": "top_k_mean",
    "top-k-persistence": "top_k_mean",
    "top-k-mean": "top_k_mean",
    "top-k-sum": "top_k_sum",
    "total": "total_persistence",
    "share": "top_persistence_share",
    "count": "finite_vine_count",
    "entropy": "persistence_entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the observed Ohio vineyard, scramble county labels repeatedly, "
            "and compare the observed statistic to the permutation null."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"County time-series CSV. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--adj-path",
        type=Path,
        default=DEFAULT_ADJ_PATH,
        help=f"Ohio adjacency file. Default: {DEFAULT_ADJ_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV path for the null distribution. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of scrambled-label null trials. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260526,
        help="Random seed for reproducible label scrambling. Default: 20260526",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Homology degree passed to toVineyard. Default: 1",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for top-k statistics. Default: 5",
    )
    parser.add_argument(
        "--stat",
        action="append",
        default=None,
        help=(
            "Statistic to test. Repeat or comma-separate for multiple statistics. "
            "Use 'all' to compute every built-in statistic. Default: top_persistence"
        ),
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Apply the same 12-month rolling mean option present in ohio_vineyards.py.",
    )
    parser.add_argument(
        "--no-cumulative",
        action="store_true",
        help="Use monthly values instead of the default cumulative transform.",
    )
    parser.add_argument(
        "--quiet-vineyard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress noisy output from polygonal_surface/vineyard internals. Default: true",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N trials. Use 0 to disable. Default: 10",
    )
    parser.add_argument(
        "--list-stats",
        action="store_true",
        help="List available statistics and exit.",
    )
    return parser.parse_args()


def resolve_stats(raw_stats: list[str] | None) -> list[StatisticSpec]:
    names = []
    if raw_stats:
        for item in raw_stats:
            names.extend(part.strip() for part in item.split(",") if part.strip())

    if not names:
        names = ["top_persistence"]

    if any(name.lower() == "all" for name in names):
        return list(STATISTICS.values())

    resolved = []
    for name in names:
        normalized = name.lower().replace(" ", "_")
        normalized = STAT_ALIASES.get(normalized, normalized)
        if normalized not in STATISTICS:
            choices = ", ".join(sorted(STATISTICS))
            raise SystemExit(f"Unknown statistic '{name}'. Choose one of: {choices}, or all.")
        resolved.append(STATISTICS[normalized])
    return list(dict.fromkeys(resolved))


def list_stats() -> None:
    print("Available statistics:")
    for spec in STATISTICS.values():
        print(f"  {spec.name}: {spec.description}")


def load_county_values(csv_path: Path, smoothing: bool, cumulative: bool) -> dict[str, list[float]]:
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "county"})
    df = df.loc[~df["county"].astype(str).str.fullmatch("Total", case=False, na=False)].copy()
    df = df.set_index("county")
    df.columns = np.arange(df.shape[1])
    df = df.apply(pd.to_numeric, errors="raise")

    if smoothing:
        df = df.T.rolling(12, min_periods=1).mean().T
    if cumulative:
        df = df.cumsum(axis=1)

    return {county: df.loc[county].tolist() for county in df.index}


def validate_counties(region_values: dict[str, list[float]], ps: object) -> list[str]:
    data_counties = set(region_values)
    surface_counties = {county for county in ps.R if county != "exterior"}
    missing_from_data = sorted(surface_counties - data_counties)
    missing_from_surface = sorted(data_counties - surface_counties)
    if missing_from_data or missing_from_surface:
        details = []
        if missing_from_data:
            details.append(f"missing from data: {missing_from_data}")
        if missing_from_surface:
            details.append(f"missing from adjacency surface: {missing_from_surface}")
        raise ValueError("County names do not match between data and adjacency file (" + "; ".join(details) + ").")
    return sorted(surface_counties)


def copy_region_values(region_values: dict[str, list[float]], counties: list[str]) -> dict[str, list[float]]:
    return {county: list(region_values[county]) for county in counties}


def scramble_region_values(
    region_values: dict[str, list[float]],
    counties: list[str],
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    shuffled_sources = rng.permutation(counties)
    return {
        county: list(region_values[source_county])
        for county, source_county in zip(counties, shuffled_sources)
    }


def with_optional_silence(quiet: bool):
    if quiet:
        return contextlib.redirect_stdout(io.StringIO())
    return contextlib.nullcontext()


def build_vineyard(ps: PS, region_values: dict[str, list[float]], degree: int, quiet: bool) -> object:
    with with_optional_silence(quiet):
        return ps.toVineyard(region_values, dim=degree)


def compute_statistics(vineyard: object, specs: list[StatisticSpec], k: int) -> dict[str, float]:
    summary = VineyardSummary.from_vineyard(vineyard)
    return {spec.name: spec.compute(summary, k) for spec in specs}


def empirical_p_value(null_values: pd.Series, observed: float, larger_is_extreme: bool) -> float:
    if larger_is_extreme:
        extreme_count = int((null_values >= observed).sum())
    else:
        extreme_count = int((null_values <= observed).sum())
    return (extreme_count + 1) / (len(null_values) + 1)


def summarize_results(
    observed: dict[str, float],
    null_df: pd.DataFrame,
    specs: list[StatisticSpec],
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        values = null_df[spec.name]
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "statistic": spec.name,
                "observed": observed[spec.name],
                "null_mean": mean,
                "null_sd": sd,
                "null_q025": float(values.quantile(0.025)),
                "null_q500": float(values.quantile(0.5)),
                "null_q975": float(values.quantile(0.975)),
                "empirical_p_value": empirical_p_value(values, observed[spec.name], spec.larger_is_extreme),
                "z_score": (observed[spec.name] - mean) / sd if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def print_run_header(args: argparse.Namespace, specs: list[StatisticSpec], county_count: int) -> None:
    stat_names = ", ".join(spec.name for spec in specs)
    transform = "cumulative" if not args.no_cumulative else "monthly"
    if args.smoothing:
        transform = f"12-month-smoothed {transform}"
    print(f"Data: {args.data_path}")
    print(f"Adjacency: {args.adj_path}")
    print(f"Counties: {county_count}")
    print(f"Degree: {args.degree}")
    print(f"Transform: {transform}")
    print(f"Statistics: {stat_names}")
    print(f"Trials: {args.trials}")
    print(f"Seed: {args.seed}")


def main() -> None:
    args = parse_args()

    if args.list_stats:
        list_stats()
        return

    from polygonal_surface import PolygonalSurface as PS

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.k < 1:
        raise SystemExit("--k must be at least 1.")

    specs = resolve_stats(args.stat)
    cumulative = not args.no_cumulative

    with with_optional_silence(args.quiet_vineyard):
        ps = PS.read_adj(args.adj_path)
    region_values = load_county_values(args.data_path, smoothing=args.smoothing, cumulative=cumulative)
    counties = validate_counties(region_values, ps)

    print_run_header(args, specs, county_count=len(counties))

    observed_vineyard = build_vineyard(
        ps=ps,
        region_values=copy_region_values(region_values, counties),
        degree=args.degree,
        quiet=args.quiet_vineyard,
    )
    observed = compute_statistics(observed_vineyard, specs, args.k)
    print("\nObserved statistics:")
    for name, value in observed.items():
        print(f"  {name}: {value:.12g}")

    rng = np.random.default_rng(args.seed)
    rows = []
    start = time.perf_counter()
    for trial in range(1, args.trials + 1):
        trial_values = scramble_region_values(region_values, counties, rng)
        vineyard = build_vineyard(
            ps=ps,
            region_values=trial_values,
            degree=args.degree,
            quiet=args.quiet_vineyard,
        )
        row = {"trial": trial}
        row.update(compute_statistics(vineyard, specs, args.k))
        rows.append(row)

        if args.progress_every and trial % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"Completed {trial}/{args.trials} trials in {elapsed:.1f}s")

    null_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    null_df.to_csv(args.output, index=False)

    summary = summarize_results(observed, null_df, specs)

    print("\nPermutation-test summary (larger statistic = more extreme):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved null distribution: {args.output}")
    print("\nObserved statistic JSON:")
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
