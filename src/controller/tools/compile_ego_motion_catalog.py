#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Compile a ClipGT ego-motion-only catalog for controller benchmarks."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTROLLER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_ROOT))

from benchmark.ego_motion_catalog import (  # noqa: E402
    EgoMotionCatalog,
    make_ego_motion_record,
)


class EgoMotionTransform:
    """Decode only egomotion from a ClipGT raw sample."""

    def __init__(
        self,
        record_duration_seconds: float,
        time_step: float,
    ) -> None:
        self.record_duration_seconds = record_duration_seconds
        self.time_step = time_step

    def __call__(self, raw_sample: Any) -> dict[str, Any]:
        from alpamayo.data.decoders.ego.support import egopose

        ego = egopose.load_clipgt_egopose(
            {
                "egomotion_estimate.parquet": getattr(
                    raw_sample, "egomotion_estimate.parquet"
                )
            }
        )
        if ego is None:
            return {"ego_available": False}
        timestamps_us = np.asarray(ego["tparam"], dtype=float) * 1e6 + int(ego["tmin"])
        xyz = np.asarray(ego["xyzs"], dtype=float)
        quats = np.asarray(ego["quats"], dtype=float)
        yaw = np.unwrap(
            np.asarray([_yaw_from_scalar_first_quat(quat) for quat in quats])
        )

        duration_us = int(round(self.record_duration_seconds * 1e6))
        requested_start_us = int(raw_sample.t0s[-1])
        latest_start_us = int(timestamps_us[-1]) - duration_us
        if latest_start_us < int(timestamps_us[0]):
            return {"ego_available": False}
        start_us = min(max(requested_start_us, int(timestamps_us[0])), latest_start_us)
        segment_timestamps_us = np.arange(
            start_us,
            start_us + duration_us + 1,
            int(round(self.time_step * 1e6)),
            dtype=np.int64,
        )

        return {
            "ego_available": True,
            "clip_key": raw_sample.clip_key,
            "clip_url": raw_sample.clip_url,
            "sample_index": int(raw_sample.sample_index),
            "sample_index_in_clip": int(raw_sample.sample_index_in_clip),
            "segment_start_us": int(start_us),
            "segment_end_us": int(segment_timestamps_us[-1]),
            "ego_timestamps_us": (
                segment_timestamps_us - segment_timestamps_us[0]
            ).tolist(),
            "ego_x": np.interp(
                segment_timestamps_us, timestamps_us, xyz[:, 0]
            ).tolist(),
            "ego_y": np.interp(
                segment_timestamps_us, timestamps_us, xyz[:, 1]
            ).tolist(),
            "ego_yaw": np.interp(segment_timestamps_us, timestamps_us, yaw).tolist(),
        }


def compile_catalog(args: argparse.Namespace) -> EgoMotionCatalog:
    """Read ClipGT samples and compile benchmark ego-motion records."""
    from alpamayo.data.dataset.formats.clipgt import ClipgtDataset, ClipgtDatasetConfig

    dataset = ClipgtDataset(
        config=ClipgtDatasetConfig(
            metadb_path=args.metadb_path,
            url_prefix=args.url_prefix,
            camera_indices=[args.camera_index],
            n_frames_per_sample=1,
            n_samples_per_clip=args.samples_per_clip,
            start_safe_margin_seconds=args.start_safe_margin_seconds,
            end_safe_margin_seconds=args.end_safe_margin_seconds,
            image_sequence_max_span_seconds=0.1,
            video_cache_type="dummy",
            t0_subsample_method="uniform",
            last_timestamp_sample_method="uniform",
            max_retries=args.max_retries,
        ),
        transform=EgoMotionTransform(
            record_duration_seconds=args.record_duration_seconds,
            time_step=args.time_step,
        ),
    )

    records = []
    sample_count = min(args.max_samples or len(dataset), len(dataset))
    for sample_index in range(0, sample_count, args.stride):
        sample = dataset[sample_index]
        if not sample.get("ego_available", True):
            continue
        record = _sample_to_record(sample)
        records.append(record)
        if len(records) % args.log_every == 0:
            print(f"Compiled {len(records)} ego-motion records...")

    if not records:
        raise ValueError(
            "No ego-motion records were compiled from the requested dataset range."
        )
    return EgoMotionCatalog(records=records)


def _sample_to_record(sample: dict[str, Any]):
    name = f"{sample['clip_key']}_{sample['sample_index_in_clip']}"
    description = (
        f"ClipGT ego motion {sample['clip_key']} "
        f"sample {sample['sample_index_in_clip']}"
    )
    return make_ego_motion_record(
        name=name,
        description=description,
        source_id=str(sample["clip_url"]),
        timestamps_us=[
            int(timestamp_us) for timestamp_us in sample["ego_timestamps_us"]
        ],
        x=[float(x) for x in sample["ego_x"]],
        y=[float(y) for y in sample["ego_y"]],
        yaw=[float(yaw) for yaw in sample["ego_yaw"]],
        metadata={
            "clip_key": sample["clip_key"],
            "sample_index": sample["sample_index"],
            "sample_index_in_clip": sample["sample_index_in_clip"],
            "segment_start_us": sample["segment_start_us"],
            "segment_end_us": sample["segment_end_us"],
        },
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _yaw_from_scalar_first_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadb-path", required=True, help="ClipGT metadb parquet path"
    )
    parser.add_argument(
        "--url-prefix",
        default=None,
        help="Optional prefix for relative ClipGT URLs",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output ego-motion catalog JSON"
    )
    parser.add_argument(
        "--selected-output", help="Optional output JSON for the selected catalog"
    )
    parser.add_argument(
        "--k", type=int, help="Select this many records with k-disk coverage"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum dataset samples to read",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Read every Nth dataset sample"
    )
    parser.add_argument(
        "--samples-per-clip", type=int, default=5, help="Uniform samples per clip"
    )
    parser.add_argument(
        "--record-duration-seconds",
        type=float,
        default=25.0,
        help="Total ego trajectory duration per catalog record",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.1,
        help="Ego trajectory timestep in seconds",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera slot used for sample timing",
    )
    parser.add_argument("--start-safe-margin-seconds", type=float, default=1.6)
    parser.add_argument("--end-safe-margin-seconds", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selected_output and args.k is None:
        raise ValueError("--selected-output requires --k.")
    if args.k is not None and args.selected_output is None:
        raise ValueError(
            "--k requires --selected-output to avoid overwriting --output."
        )
    catalog = compile_catalog(args)
    output_path = Path(args.output)
    catalog.save(output_path)
    print(f"Wrote {len(catalog.records)} ego-motion records to {output_path}")
    if args.k is not None:
        selected_catalog = catalog.with_selected_cover(args.k)
        selected_output = Path(args.selected_output)
        selected_catalog.save(selected_output)
        print(
            f"Selected {len(selected_catalog.selected_names)} records; "
            f"covering radius={selected_catalog.selection_metadata['covering_radius']:.3f}"
        )
        print(f"Wrote selected ego-motion catalog to {selected_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
