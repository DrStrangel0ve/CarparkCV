"""Correlate ultrasonic dips with CV vehicle detections.

This utility loads the processed ultrasonic dip features (RAK CSV) and the
computer-vision detection results, estimates a linear time mapping between the
sensor clocks, and reports which ultrasonic dips line up with CV vehicle
peaks. People and bicycles are ignored when scoring the alignment because the
ultrasonic sensor only reacts reliably to vehicles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class MappingConfig:
    rak_csv: Path = Path("RAK_DATA_F2025_Test2.csv")
    cv_csv: Path = Path("detection_results.csv")
    min_dip_depth: float = 180.0  # Only keep deep dips (likely vehicles)
    cv_peak_prominence: float = 0.8  # Prominence relative to vehicle max
    cv_peak_min_distance: int = 15  # Number of frames between peaks
    max_alignment_error: float = 2.5  # seconds tolerance for a match
    cv_start_time: float | None = None  # Known CV start timestamp (seconds)
    rak_start_time: float | None = None  # Known RAK start timestamp (seconds)
    scan_window: float = 10.0  # seconds around base offset to explore
    scan_step: float = 0.5  # resolution of offset sweep (seconds)
    rak_raw_path: Path = Path("RAK_DATA_F2025_Test2.TXT")
    rak_trim_baseline: float = 220.0
    rak_trim_tolerance: float = 5.0
    rak_trim_duration: float = 5.0  # seconds
    cv_enriched_output: Path = Path("detection_results_with_rak_distance.csv")


@dataclass
class AlignmentResult:
    scale: float
    offset: float
    pairings: pd.DataFrame
    sweep_summary: pd.DataFrame


def load_ultrasonic_events(rak_path: Path, min_depth: float) -> pd.DataFrame:
    df = pd.read_csv(rak_path)
    filtered = df[df["max_depth"] >= min_depth].copy()
    filtered.sort_values("start_time", inplace=True)
    return filtered.reset_index(drop=True)


def read_rak_raw_series(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    values = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                time_str, value_str = line.split(",")
                times.append(float(time_str))
                values.append(float(value_str))
            except ValueError:
                continue

    if not times:
        raise ValueError(f"No valid readings found in {raw_path}")

    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def trim_initial_stable_period_series(
    times: np.ndarray,
    values: np.ndarray,
    baseline_value: float,
    tolerance: float,
    stable_duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, values

    threshold = baseline_value - tolerance
    stable_duration_units = stable_duration_seconds * 1000.0  # times are in ms
    start_idx = None

    for idx, (t, v) in enumerate(zip(times, values)):
        if v >= threshold:
            if start_idx is None:
                start_idx = idx
            if t - times[start_idx] >= stable_duration_units:
                return times[start_idx:], values[start_idx:]
        else:
            start_idx = None

    return times, values


def load_trimmed_rak_series(config: MappingConfig) -> tuple[np.ndarray, np.ndarray]:
    raw_times, raw_values = read_rak_raw_series(config.rak_raw_path)
    trimmed_times, trimmed_values = trim_initial_stable_period_series(
        raw_times,
        raw_values,
        baseline_value=config.rak_trim_baseline,
        tolerance=config.rak_trim_tolerance,
        stable_duration_seconds=config.rak_trim_duration,
    )

    times_seconds = trimmed_times / 1000.0
    return times_seconds, trimmed_values

def _simple_peak_indices(values: np.ndarray, threshold: float, min_separation: int) -> np.ndarray:
    """Return indices of local maxima above a threshold without SciPy."""
    arr = np.asarray(values, dtype=float)
    count = arr.size
    if count < 3:
        return np.empty(0, dtype=int)

    separation = max(int(min_separation), 1)
    indices: list[int] = []

    for idx in range(1, count - 1):
        current = arr[idx]
        if current < threshold:
            continue
        if current <= arr[idx - 1] or current <= arr[idx + 1]:
            continue

        if indices and idx - indices[-1] < separation:
            if current > arr[indices[-1]]:
                indices[-1] = idx
            continue

        indices.append(idx)

    return np.asarray(indices, dtype=int)


def extract_cv_vehicle_peaks(cv_path: Path, prominence_ratio: float, distance_frames: int) -> pd.DataFrame:
    cv = pd.read_csv(cv_path)
    vehicle_series = cv["vehicles_in_frame"].to_numpy()

    if vehicle_series.max() <= 0:
        raise ValueError("CV data contains no vehicle detections (all zeros)")

    prominence = vehicle_series.max() * prominence_ratio
    peak_indices = _simple_peak_indices(vehicle_series, threshold=prominence, min_separation=distance_frames)

    peaks_df = cv.iloc[peak_indices].copy()
    peaks_df = peaks_df[["time_seconds", "vehicles_in_frame", "frame_number"]]
    peaks_df.rename(columns={"time_seconds": "cv_time", "vehicles_in_frame": "vehicles"}, inplace=True)
    peaks_df.sort_values("cv_time", inplace=True)
    peaks_df.reset_index(drop=True, inplace=True)

    return peaks_df


def _paired_regression(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        raise ValueError("Need at least two events in each series for calibration")

    # Use least squares to solve y = scale * x + offset
    A = np.vstack([x_arr, np.ones_like(x_arr)]).T
    scale, offset = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    return float(scale), float(offset)


def estimate_time_mapping(rak_times: np.ndarray, cv_times: np.ndarray) -> Tuple[float, float]:
    # Pick comparable subsets based on order statistics
    n = min(len(rak_times), len(cv_times))
    if n < 2:
        raise ValueError("Not enough overlapping events to estimate mapping")

    # Use the top and bottom quartiles to stabilise regression
    quantile_idx = np.linspace(0.1, 0.9, num=min(n, 8))
    rak_sample = np.quantile(rak_times, quantile_idx)
    cv_sample = np.quantile(cv_times, quantile_idx)

    return _paired_regression(rak_sample, cv_sample)


def map_dips_to_cv(
    rak_events: pd.DataFrame,
    cv_peaks: pd.DataFrame,
    scale: float,
    offset: float,
    tolerance: float,
) -> pd.DataFrame:
    mapped_times = rak_events["start_time"].to_numpy() * scale + offset
    rak_events = rak_events.assign(mapped_time=mapped_times)

    cv_times = cv_peaks["cv_time"].to_numpy()

    matches = []
    cv_idx = 0
    for idx, row in rak_events.iterrows():
        mapped = row["mapped_time"]

        # Advance cv index to closest event
        while cv_idx + 1 < len(cv_times) and abs(cv_times[cv_idx + 1] - mapped) < abs(cv_times[cv_idx] - mapped):
            cv_idx += 1

        cv_time = cv_times[cv_idx]
        time_error = cv_time - mapped
        matched = abs(time_error) <= tolerance

        matches.append(
            {
                "rak_idx": idx,
                "rak_time": row["start_time"],
                "rak_depth": row["max_depth"],
                "mapped_time": mapped,
                "cv_time": cv_time,
                "cv_vehicles": cv_peaks.iloc[cv_idx]["vehicles"],
                "time_error": time_error,
                "is_match": matched,
            }
        )

    return pd.DataFrame(matches)


def score_alignment(
    rak_events: pd.DataFrame,
    cv_peaks: pd.DataFrame,
    scale: float,
    offset: float,
    tolerance: float,
) -> tuple[int, float, float, pd.DataFrame]:
    matches = map_dips_to_cv(
        rak_events,
        cv_peaks,
        scale=scale,
        offset=offset,
        tolerance=tolerance,
    )

    match_mask = matches["is_match"]
    match_count = int(match_mask.sum())

    if match_count:
        abs_errors = matches.loc[match_mask, "time_error"].abs()
        mean_abs_error = float(abs_errors.mean())
        median_abs_error = float(abs_errors.median())
    else:
        mean_abs_error = float("inf")
        median_abs_error = float("inf")

    return match_count, mean_abs_error, median_abs_error, matches


def sweep_offsets(
    rak_events: pd.DataFrame,
    cv_peaks: pd.DataFrame,
    scale: float,
    base_offset: float,
    window: float,
    step: float,
    tolerance: float,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    offsets = np.arange(base_offset - window, base_offset + window + step / 2.0, step)

    summary_records = []
    best_offset = base_offset
    best_matches = pd.DataFrame()
    best_match_count = -1
    best_mean_abs_error = float("inf")

    for offset in offsets:
        match_count, mean_abs_error, median_abs_error, matches = score_alignment(
            rak_events,
            cv_peaks,
            scale=scale,
            offset=offset,
            tolerance=tolerance,
        )

        summary_records.append(
            {
                "offset": offset,
                "match_count": match_count,
                "mean_abs_error": mean_abs_error,
                "median_abs_error": median_abs_error,
            }
        )

        if match_count > best_match_count or (
            match_count == best_match_count and mean_abs_error < best_mean_abs_error
        ):
            best_offset = offset
            best_matches = matches
            best_match_count = match_count
            best_mean_abs_error = mean_abs_error

    summary_df = pd.DataFrame(summary_records)
    return best_offset, best_matches, summary_df


def correlate_ultrasonic_with_cv(
    config: MappingConfig = MappingConfig(),
    base_offset: float | None = None,
) -> AlignmentResult:
    rak_events = load_ultrasonic_events(config.rak_csv, config.min_dip_depth)
    cv_peaks = extract_cv_vehicle_peaks(
        config.cv_csv,
        prominence_ratio=config.cv_peak_prominence,
        distance_frames=config.cv_peak_min_distance,
    )

    if base_offset is not None:
        scale = 1.0
        sweep_center = base_offset
    elif config.cv_start_time is not None and config.rak_start_time is not None:
        scale = 1.0
        sweep_center = config.cv_start_time - config.rak_start_time
    else:
        scale, sweep_center = estimate_time_mapping(
            rak_events["start_time"].to_numpy(),
            cv_peaks["cv_time"].to_numpy(),
        )

    best_offset, matches, sweep = sweep_offsets(
        rak_events,
        cv_peaks,
        scale=scale,
        base_offset=sweep_center,
        window=config.scan_window,
        step=config.scan_step,
        tolerance=config.max_alignment_error,
    )

    return AlignmentResult(scale=scale, offset=best_offset, pairings=matches, sweep_summary=sweep)


def interactive_offset_alignment(config: MappingConfig, initial_offset: float = 0.0) -> float:
    """Launch an overlay plot to help the user tune the CV-RAK offset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for the interactive alignment viewer."
        ) from exc

    rak_times_sec, rak_values = load_trimmed_rak_series(config)
    if rak_times_sec.size == 0:
        raise RuntimeError("No ultrasonic samples available to plot.")

    cv_df = pd.read_csv(config.cv_csv)
    required_columns = {"time_seconds", "vehicles_in_frame"}
    missing = required_columns.difference(cv_df.columns)
    if missing:
        raise RuntimeError(
            f"CV CSV is missing the required columns: {', '.join(sorted(missing))}"
        )

    cv_times = cv_df["time_seconds"].to_numpy(dtype=float)
    vehicle_counts = cv_df["vehicles_in_frame"].to_numpy(dtype=float)
    if cv_times.size == 0:
        raise RuntimeError("CV detections file contains no rows to plot.")

    plt.ion()
    fig, ax_ultra = plt.subplots(figsize=(12, 6))
    ax_cv = ax_ultra.twinx()

    ax_ultra.set_xlabel("RAK timeline (seconds)")
    ax_ultra.set_ylabel("Ultrasonic distance (raw units)", color="tab:blue")
    ax_cv.set_ylabel("Vehicles per frame", color="tab:orange")

    (rak_line,) = ax_ultra.plot(
        rak_times_sec,
        rak_values,
        color="tab:blue",
        label="Ultrasonic distance",
        alpha=0.8,
    )
    (cv_line,) = ax_cv.plot(
        cv_times - initial_offset,
        vehicle_counts,
        color="tab:orange",
        label="CV vehicles",
        alpha=0.7,
    )

    fig.legend(handles=[rak_line, cv_line], loc="upper right")
    offset = float(initial_offset)

    def _update_plot(current_offset: float) -> None:
        shifted = cv_times - current_offset
        cv_line.set_xdata(shifted)
        x_min = float(min(np.min(rak_times_sec), np.min(shifted)))
        x_max = float(max(np.max(rak_times_sec), np.max(shifted)))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        ax_ultra.set_xlim(x_min, x_max)
        ax_ultra.set_title(
            f"Ultrasonic vs CV overlay (offset = {current_offset:.3f} s)"
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.05)

    _update_plot(offset)
    plt.show(block=False)
    print()
    print("Interactive overlay running. Enter a numeric offset in seconds to replot,")
    print("or press Enter (or type 'done') once the alignment looks correct.")

    while True:
        user_input = input(
            "Offset seconds (blank to accept, 'q' to keep current): "
        ).strip()
        if not user_input:
            break
        lowered = user_input.lower()
        if lowered in {"done", "accept"}:
            break
        if lowered in {"q", "quit", "exit"}:
            print("Keeping current offset without further changes.")
            break
        try:
            offset = float(user_input)
        except ValueError:
            print("Please enter a numeric offset (e.g. 12.5 or -3.0).")
            continue
        _update_plot(offset)

    plt.ioff()
    return offset


def main() -> None:
    config = MappingConfig()
    base_offset = None

    while True:
        user_input = input(
            "Enter initial CV - RAK time offset in seconds (CV minus RAK): "
        ).strip()
        if not user_input:
            print("No value entered, defaulting to 0.0 seconds")
            base_offset = 0.0
            break
        try:
            base_offset = float(user_input)
            break
        except ValueError:
            print("Please enter a numeric value (e.g. 12.5 or -3.0)")

    plot_prompt = input(
        "Open a Matplotlib overlay to fine-tune the offset? [y/N]: "
    ).strip()
    if plot_prompt.lower() in {"y", "yes"}:
        try:
            base_offset = interactive_offset_alignment(
                config,
                initial_offset=base_offset or 0.0,
            )
        except Exception as exc:  # pragma: no cover - user interaction path
            print(f"Unable to launch interactive overlay: {exc}")

    result = correlate_ultrasonic_with_cv(config, base_offset=base_offset)

    matches = result.pairings
    total = len(matches)
    confirmed = int(matches["is_match"].sum())
    print("=" * 72)
    print("ULTRASONIC <-> CV ALIGNMENT SUMMARY")
    print("=" * 72)
    print(f"Scale factor (cv seconds / rak unit): {result.scale:.6f}")
    print(f"Best offset (cv seconds when rak time = 0): {result.offset:.3f}s")
    print(
        f"Offset sweep: centre={base_offset:.3f}s, window=+/-{config.scan_window}s, "
        f"step={config.scan_step}s"
    )
    print(f"Deep dips analysed: {total}")
    match_rate = confirmed / total if total else 0.0
    print(f"Matches within +/-{config.max_alignment_error} s: {confirmed} ({match_rate:.1%})")
    print()

    sweep_sorted = result.sweep_summary.sort_values(
        by=["match_count", "mean_abs_error"],
        ascending=[False, True],
    )
    top_rows = sweep_sorted.head(10).copy()
    if not top_rows.empty:
        print("Offset sweep results (top candidates):")
        formatted = top_rows.copy()
        formatted["offset"] = formatted["offset"].map(lambda v: f"{v:.3f}")

        def _fmt_metric(val: float) -> str:
            return f"{val:.3f}" if np.isfinite(val) else "inf"

        formatted["mean_abs_error"] = formatted["mean_abs_error"].map(_fmt_metric)
        formatted["median_abs_error"] = formatted["median_abs_error"].map(_fmt_metric)
        print(
            formatted[["offset", "match_count", "mean_abs_error", "median_abs_error"]]
            .to_string(index=False)
        )
        print()

    # Build CV-enriched dataset with interpolated RAK distance
    cv_full = pd.read_csv(config.cv_csv)
    rak_times_sec, rak_values = load_trimmed_rak_series(config)

    if rak_times_sec.size > 1:
        mapped_times = rak_times_sec * result.scale + result.offset
        order = np.argsort(mapped_times)
        mapped_times_sorted = mapped_times[order]
        rak_values_sorted = rak_values[order]

        unique_times, unique_indices = np.unique(mapped_times_sorted, return_index=True)
        unique_values = rak_values_sorted[unique_indices]

        cv_times = cv_full["time_seconds"].to_numpy()
        interpolated = np.interp(
            cv_times,
            unique_times,
            unique_values,
            left=np.nan,
            right=np.nan,
        )
    else:
        interpolated = np.full(len(cv_full), np.nan)

    cv_enriched = cv_full.copy()
    cv_enriched["rak_distance"] = interpolated

    cv_enriched.to_csv(config.cv_enriched_output, index=False)

    if confirmed:
        print("Top confirmed matches:")
        print(matches[matches["is_match"]].nsmallest(10, columns="time_error")[
            ["rak_time", "mapped_time", "cv_time", "time_error", "rak_depth", "cv_vehicles"]
        ])

    unmatched = matches[~matches["is_match"]]
    if not unmatched.empty:
        print()
        print("Deep dips with no nearby CV vehicle peak:")
        print(unmatched.nlargest(10, columns="rak_depth")[
            ["rak_time", "mapped_time", "cv_time", "time_error", "rak_depth"]
        ])

    # Optionally export
    matches.to_csv("rak_cv_correlated_events.csv", index=False)
    result.sweep_summary.to_csv("rak_cv_offset_sweep.csv", index=False)
    print()
    print("Detailed pairing saved to rak_cv_correlated_events.csv")
    print("Offset sweep summary saved to rak_cv_offset_sweep.csv")
    print(f"CV data with RAK distance saved to: {config.cv_enriched_output}")


if __name__ == "__main__":
    main()
