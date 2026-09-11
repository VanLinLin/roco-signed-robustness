"""Primary public-WAFT signed axes on the official sampled flow points.

This is a CPU-only, inference-free analysis.  It opens the committed submission
and official sampling map read-only, then writes only runs/signed_axes_v1/.
"""

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/ssd8/van/contest/RoCo")
FLOW_PATH = ROOT / "submissions/v1/flow_robustness.hdf5"
INDEX_PATH = ROOT / "tools/subsampling/robust_sampling_map_v2/robust.json"
REPLICATION_PATH = ROOT / "runs/official_delta_v1/summary_left_right/SUMMARY.json"
OUT_DIR = ROOT / "runs/signed_axes_v1"
JSON_PATH = OUT_DIR / "SIGNED_AXES.json"
TABLE_PATH = OUT_DIR / "TABLE.md"
NBOOT = 10_000
SEED = 20_260_911
REFERENCE_DIRECTIONAL_GAIN = {
    "frost": 0.1421,
    "spatter": 0.2749,
    "zoom_blur": 1.0347,
    "snow": 0.9489,
    "rain": 0.9676,
}


def segment_sums(values, starts):
    """Sum a pointwise vector over contiguous robust.json record boundaries."""
    return np.add.reduceat(values, starts).astype(np.float64, copy=False)


def scene_sums(record_values, record_scene, n_scenes):
    """Pool record totals into scenes without averaging records."""
    return np.bincount(record_scene, weights=record_values,
                       minlength=n_scenes).astype(np.float64, copy=False)


def finite_flow(dataset, condition):
    """Read one HDF5 flow group and reject nonfinite exported predictions."""
    flow = dataset[:].astype(np.float64)
    if flow.ndim != 2 or flow.shape[1] != 2:
        raise ValueError(f"{condition}: expected Nx2 flow, got {flow.shape}")
    if not np.isfinite(flow).all():
        raise ValueError(f"{condition}: nonfinite flow value")
    return flow


def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main():
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    entries = index["entries"]
    starts = np.asarray([entry["start"] for entry in entries], dtype=np.int64)
    stops = np.asarray([entry["stop"] for entry in entries], dtype=np.int64)
    n_points = int(index["shape"][0])
    if not len(entries) or starts[0] != 0 or stops[-1] != n_points:
        raise ValueError("robust.json entries do not span the declared point array")
    if not np.array_equal(stops[:-1], starts[1:]):
        raise ValueError("robust.json record boundaries are not contiguous")

    scenes = sorted({entry["path"].split("/")[0] for entry in entries})
    scene_index = {scene: i for i, scene in enumerate(scenes)}
    record_scene = np.asarray(
        [scene_index[entry["path"].split("/")[0]] for entry in entries],
        dtype=np.int64,
    )
    n_scenes = len(scenes)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_scenes, size=(NBOOT, n_scenes))

    print(f"{len(entries)} records, {n_scenes} scenes, {n_points} points", flush=True)
    by_condition = {}
    with h5py.File(FLOW_PATH, "r") as h5:
        if "clean" not in h5:
            raise ValueError("missing clean/flow in submission")
        clean = finite_flow(h5["clean/flow"], "clean")
        if clean.shape[0] != n_points:
            raise ValueError(f"clean point count {clean.shape[0]} != {n_points}")

        clean_mag = np.linalg.norm(clean, axis=1)
        clean_record_sum = segment_sums(clean_mag, starts)
        clean_sum = float(clean_record_sum.sum())
        clean_sq_point = np.einsum("ij,ij->i", clean, clean)
        clean_sq_record = segment_sums(clean_sq_point, starts)
        clean_sq_scene = scene_sums(clean_sq_record, record_scene, n_scenes)
        clean_sq_sum = float(clean_sq_scene.sum())
        moving = clean_mag > 1.0
        moving_count = int(moving.sum())
        if clean_sum <= 0 or clean_sq_sum <= 0 or moving_count == 0:
            raise ValueError("clean flow has an empty magnitude denominator")
        del clean_sq_point, clean_sq_record

        conditions = sorted(key for key in h5.keys() if key != "clean")
        if len(conditions) != 20:
            raise ValueError(f"expected 20 corruptions, found {len(conditions)}")

        for condition in conditions:
            corrupt = finite_flow(h5[f"{condition}/flow"], condition)
            if corrupt.shape != clean.shape:
                raise ValueError(
                    f"{condition}: flow shape {corrupt.shape} != clean {clean.shape}"
                )
            corrupt_mag = np.linalg.norm(corrupt, axis=1)
            corrupt_record_sum = segment_sums(corrupt_mag, starts)
            record_ratios = np.divide(
                corrupt_record_sum,
                clean_record_sum,
                out=np.full_like(corrupt_record_sum, np.nan),
                where=clean_record_sum > 0,
            )
            valid_record_ratios = record_ratios[np.isfinite(record_ratios)]
            if len(valid_record_ratios) != len(entries):
                raise ValueError(f"{condition}: zero-magnitude record denominator")

            dot_point = np.einsum("ij,ij->i", clean, corrupt)
            dot_record = segment_sums(dot_point, starts)
            dot_scene = scene_sums(dot_record, record_scene, n_scenes)
            dot_sum = float(dot_scene.sum())
            directional_gain = dot_sum / clean_sq_sum
            bootstrap_gain = (
                dot_scene[draws].sum(axis=1)
                / clean_sq_scene[draws].sum(axis=1)
            )
            ci_lo, ci_hi = np.percentile(bootstrap_gain, [2.5, 97.5])

            corrupt_sq_sum = float(np.einsum("ij,ij->", corrupt, corrupt))
            residual_sq_sum = (
                corrupt_sq_sum
                - 2.0 * directional_gain * dot_sum
                + directional_gain * directional_gain * clean_sq_sum
            )
            # The expression is algebraically nonnegative; guard only roundoff.
            residual_sq_sum = max(0.0, residual_sq_sum)
            orthogonal_fraction = residual_sq_sum / corrupt_sq_sum
            collapsed_count = int((moving & (corrupt_mag < 0.5 * clean_mag)).sum())
            inflated_count = int((moving & (corrupt_mag > 2.0 * clean_mag)).sum())

            result = {
                "ratio_of_sums": float(corrupt_record_sum.sum() / clean_sum),
                "mean_per_record": float(np.mean(valid_record_ratios)),
                "median_per_record": float(np.median(valid_record_ratios)),
                "median_per_point": float(
                    np.median(corrupt_mag[moving] / clean_mag[moving])
                ),
                "directional_gain": float(directional_gain),
                "directional_gain_ci95": [float(ci_lo), float(ci_hi)],
                "directional_gain_per_scene": {
                    scene: float(dot_scene[i] / clean_sq_scene[i])
                    for i, scene in enumerate(scenes)
                },
                "orthogonal_fraction": float(orthogonal_fraction),
                "collapse_fraction": float(collapsed_count / moving_count),
                "inflation_fraction": float(inflated_count / moving_count),
                "counts": {
                    "points": n_points,
                    "records": len(entries),
                    "moving_clean_gt_1px": moving_count,
                    "collapsed_lt_half_clean": collapsed_count,
                    "inflated_gt_twice_clean": inflated_count,
                },
                "pooled_sums": {
                    "clean_magnitude": clean_sum,
                    "corrupt_magnitude": float(corrupt_record_sum.sum()),
                    "clean_squared": clean_sq_sum,
                    "clean_corrupt_dot": dot_sum,
                    "corrupt_squared": corrupt_sq_sum,
                    "orthogonal_residual_squared": residual_sq_sum,
                },
            }
            by_condition[condition] = result
            print(
                f"{condition:20s} ratio={result['ratio_of_sums']:.4f} "
                f"gain={directional_gain:.4f} "
                f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]",
                flush=True,
            )
            del corrupt, corrupt_mag, corrupt_record_sum, record_ratios
            del valid_record_ratios, dot_point, dot_record, dot_scene, bootstrap_gain

    replication = json.loads(REPLICATION_PATH.read_text(encoding="utf-8"))
    crosscheck = {}
    for condition, stated_value in REFERENCE_DIRECTIONAL_GAIN.items():
        file_value = float(replication["conditions"][condition]["directional_gain"])
        if abs(file_value - stated_value) > 5e-5:
            raise ValueError(
                f"replication reference mismatch for {condition}: "
                f"task={stated_value}, file={file_value}"
            )
        primary_value = by_condition[condition]["directional_gain"]
        difference = abs(primary_value - file_value)
        crosscheck[condition] = {
            "primary": primary_value,
            "replication": file_value,
            "absolute_difference": difference,
            "exceeds_0.1": difference > 0.1,
        }

    output = {
        "schema_version": 1,
        "source_hdf5": str(FLOW_PATH),
        "sampling_index": str(INDEX_PATH),
        "scope": "primary public WAFT; official sampled points; CPU-only analysis",
        "n_corruptions": len(by_condition),
        "n_points_per_condition": n_points,
        "n_records_per_condition": len(entries),
        "scenes": scenes,
        "bootstrap": {
            "unit": "whole scene",
            "draws": NBOOT,
            "seed": SEED,
            "percentiles": [2.5, 97.5],
        },
        "definitions": {
            "ratio_of_sums": "sum(||f_corrupt||) / sum(||f_clean||)",
            "mean_per_record": "mean over record-level ratios of magnitude sums",
            "median_per_record": "median over record-level ratios of magnitude sums",
            "median_per_point": "median(||f_corrupt|| / ||f_clean||) where ||f_clean|| > 1 px",
            "directional_gain": "sum(<f_clean,f_corrupt>) / sum(||f_clean||^2)",
            "orthogonal_fraction": "sum(||f_corrupt-g*f_clean||^2) / sum(||f_corrupt||^2), using pooled g",
            "collapse_fraction": "fraction where ||f_clean|| > 1 px and ||f_corrupt|| < 0.5*||f_clean||",
            "inflation_fraction": "fraction where ||f_clean|| > 1 px and ||f_corrupt|| > 2*||f_clean||",
        },
        "by_condition": by_condition,
        "replication_crosscheck": {
            "source": str(REPLICATION_PATH),
            "threshold": 0.1,
            "conditions": crosscheck,
        },
    }

    ordered = sorted(by_condition.items(), key=lambda item: item[1]["ratio_of_sums"])
    lines = [
        "# Primary public-WAFT signed axes",
        "",
        "20 個 corruption，依 `ratio_of_sums` 由小到大排序。CI 是 10 個 scene "
        "整塊 bootstrap 10,000 次，seed `20260911`。",
        "",
        "| condition | ratio_of_sums | median_per_record | directional_gain [CI] | orthogonal_fraction | collapse_frac | inflation_frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, result in ordered:
        lo, hi = result["directional_gain_ci95"]
        lines.append(
            f"| {condition} | {result['ratio_of_sums']:.6f} | "
            f"{result['median_per_record']:.6f} | "
            f"{result['directional_gain']:.6f} [{lo:.6f}, {hi:.6f}] | "
            f"{result['orthogonal_fraction']:.6f} | "
            f"{result['collapse_fraction']:.6f} | "
            f"{result['inflation_fraction']:.6f} |"
        )

    large_differences = [
        (condition, values)
        for condition, values in crosscheck.items()
        if values["exceeds_0.1"]
    ]
    lines += ["", "## Replication checkpoint 交叉檢查", ""]
    if large_differences:
        lines.append("下列 directional gain 與 replication checkpoint 的差距超過 0.1：")
        lines.append("")
        for condition, values in large_differences:
            lines.append(
                f"- `{condition}`: primary {values['primary']:.6f}, replication "
                f"{values['replication']:.6f}, absolute difference "
                f"{values['absolute_difference']:.6f}."
            )
    else:
        lines.append(
            "指定的 frost、spatter、zoom_blur、snow、rain 均未出現超過 0.1 的差距。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(JSON_PATH, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    write_atomic(TABLE_PATH, "\n".join(lines) + "\n")
    print(f"wrote {JSON_PATH}", flush=True)
    print(f"wrote {TABLE_PATH}", flush=True)


if __name__ == "__main__":
    main()
