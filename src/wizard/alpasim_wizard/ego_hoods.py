# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Re-render carline hood images into each scene's front-wide camera.

A carline bank's hood image is drawn for one camera, but each vehicle's camera
sits slightly differently, so the shared image misses the recorded hood edge.
Transform specs list, per carline bank, a source image and the f-theta camera
it is drawn for. For every selected scene of a listed carline, the source image
is re-rendered (pure rotation plus intrinsics, translation ignored) into that
scene's front-wide camera from its USDZ calibration, and written as a bank next
to the carline's other cameras. Clips of unlisted carlines use their carline
bank unchanged. Banks are named by the content of all their files and cached
in the banks directory the renderer mounts, so each is generated once and shared
between runs, and a changed carline bank never reuses a stale copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

MASK_SIZE = (1920, 1080)
GENERATE_WORKERS = 8
# Part of every transformed bank's name. Bump it when a change to this module changes
# the rendered images, so that cached banks are regenerated instead of reused.
RENDER_VERSION = 1


@dataclass(frozen=True)
class FthetaCamera:
    """F-theta camera at MASK_SIZE; R maps camera (RDF) rays to the rig (FLU) frame."""

    cx: float
    cy: float
    pixel_to_angle: tuple[float, ...]
    angle_to_pixel: tuple[float, ...]
    R: tuple[tuple[float, ...], ...]
    A: tuple[tuple[float, ...], ...]  # affine from ideal to actual pixel offsets

    @staticmethod
    def from_dict(d: dict) -> FthetaCamera:
        return FthetaCamera(
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            pixel_to_angle=tuple(float(v) for v in d["pixel_to_angle"]),
            angle_to_pixel=tuple(float(v) for v in d["angle_to_pixel"]),
            R=tuple(tuple(float(v) for v in row) for row in d["R"]),
            A=tuple(tuple(float(v) for v in row) for row in d["A"]),
        )

    @staticmethod
    def from_usdz(usdz_path: str, camera: str) -> FthetaCamera:
        """The camera's calibration from the USDZ's rig_trajectories.json, rescaled to MASK_SIZE."""
        with zipfile.ZipFile(usdz_path) as zf:
            data = json.loads(zf.read("rig_trajectories.json"))
        cal = next(
            v
            for k, v in data["camera_calibrations"].items()
            if k.split("@")[0] == camera
        )
        if cal["camera_model"]["type"] != "ftheta":
            raise ValueError(
                f"{usdz_path}: {camera} is {cal['camera_model']['type']}, not ftheta"
            )
        params = cal["camera_model"]["parameters"]
        c, d, e = params.get("linear_cde") or (1.0, 0.0, 0.0)
        width, height = params["resolution"]
        # Resizing the image scales actual pixel offsets, so the scale folds into the
        # affine term and the principal point moves with half-pixel centres.
        sx, sy = MASK_SIZE[0] / width, MASK_SIZE[1] / height
        cx, cy = params["principal_point"]
        return FthetaCamera.from_dict(
            {
                "cx": (cx + 0.5) * sx - 0.5,
                "cy": (cy + 0.5) * sy - 0.5,
                "pixel_to_angle": params["pixeldist_to_angle_poly"],
                "angle_to_pixel": params["angle_to_pixeldist_poly"],
                "R": np.asarray(cal["T_sensor_rig"], float)[:3, :3].tolist(),
                "A": (np.diag([sx, sy]) @ np.array([[c, d], [e, 1.0]])).tolist(),
            }
        )

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(np.asarray(self.A))
        du, dv = u - self.cx, v - self.cy
        dx = inv[0, 0] * du + inv[0, 1] * dv
        dy = inv[1, 0] * du + inv[1, 1] * dv
        r = np.hypot(dx, dy)
        theta = np.polynomial.polynomial.polyval(r, self.pixel_to_angle)
        scale = np.divide(np.sin(theta), r, out=np.zeros_like(r), where=r > 0)
        return np.stack([dx * scale, dy * scale, np.cos(theta)], axis=-1)

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        rho = np.hypot(x, y)
        r = np.polynomial.polynomial.polyval(np.arctan2(rho, z), self.angle_to_pixel)
        scale = np.divide(r, rho, out=np.zeros_like(rho), where=rho > 0)
        dx, dy = x * scale, y * scale
        a = self.A
        return (
            self.cx + a[0][0] * dx + a[0][1] * dy,
            self.cy + a[1][0] * dx + a[1][1] * dy,
        )


def warp(image: np.ndarray, src: FthetaCamera, dst: FthetaCamera) -> np.ndarray:
    """Re-render ``image`` (seen by ``src``) as ``dst`` sees it: bilinear, edges replicated."""
    h, w = image.shape[:2]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rig = dst.rays(u, v) @ np.asarray(dst.R).T
    map_x, map_y = src.project(rig @ np.asarray(src.R))
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    x0 = np.minimum(np.floor(map_x).astype(int), w - 2)
    y0 = np.minimum(np.floor(map_y).astype(int), h - 2)
    fx = (map_x - x0)[..., None]
    fy = (map_y - y0)[..., None]
    img = image.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def load_image(path: str) -> np.ndarray:
    """RGBA image at MASK_SIZE (some banks are stored at twice that size)."""
    img = Image.open(path).convert("RGBA")
    if img.size != MASK_SIZE:
        img = img.resize(MASK_SIZE, Image.Resampling.BOX)
    return np.asarray(img)


@dataclass(frozen=True)
class _Job:
    bank_dir: str
    source_image: str
    source_camera: FthetaCamera
    target_camera: FthetaCamera
    camera: str
    carline_bank: str


def _bank_digest(bank_dir: str) -> str:
    """Hash of a bank's file names and contents."""
    key = hashlib.sha256()
    for name in sorted(os.listdir(bank_dir)):
        key.update(name.encode() + b"\0")
        key.update(hashlib.sha256(Path(bank_dir, name).read_bytes()).digest())
    return key.hexdigest()


def _share(bank_dir: str) -> None:
    """Make a bank group-writable: the cache is shared by the team."""
    for name in os.listdir(bank_dir):
        os.chmod(os.path.join(bank_dir, name), 0o664)
    os.chmod(bank_dir, 0o2775)


def _write_bank(job: _Job) -> None:
    """Write one transformed bank: the re-rendered camera plus the carline bank's other files."""
    if os.path.isdir(job.bank_dir):
        return
    parent = os.path.dirname(job.bank_dir)
    tmp = tempfile.mkdtemp(prefix=".tmp-", dir=parent)
    try:
        for name in os.listdir(job.carline_bank):
            if name != f"{job.camera}.png":
                shutil.copyfile(
                    os.path.join(job.carline_bank, name), os.path.join(tmp, name)
                )
        image = warp(load_image(job.source_image), job.source_camera, job.target_camera)
        Image.fromarray(image).save(os.path.join(tmp, f"{job.camera}.png"))
        _share(tmp)
        try:
            os.rename(tmp, job.bank_dir)
        except OSError:
            if not os.path.isdir(
                job.bank_dir
            ):  # another process won the race otherwise
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _copy_bank(src: str, dst: str) -> None:
    """Copy an unchanged carline bank into the output directory once."""
    if os.path.isdir(dst):
        return
    tmp = tempfile.mkdtemp(prefix=".tmp-", dir=os.path.dirname(dst))
    try:
        shutil.copytree(src, tmp, dirs_exist_ok=True)
        _share(tmp)
        try:
            os.rename(tmp, dst)
        except OSError:
            if not os.path.isdir(dst):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_specs(
    spec_paths: list[str],
) -> tuple[str, dict[str, tuple[str, FthetaCamera]], dict[str, str]]:
    """The specs' camera, their sources (bank -> image, camera) and stock bank directories."""
    cameras = set()
    sources: dict[str, tuple[str, FthetaCamera]] = {}
    stock: dict[str, str] = {}
    for spec_path in map(Path, spec_paths):
        spec = yaml.safe_load(spec_path.read_text())
        cameras.add(spec["camera_logical_id"])
        banks = (spec_path.parent / spec["banks"]).resolve()
        for bank in sorted(p for p in banks.iterdir() if p.is_dir()):
            if bank.name in stock:
                raise ValueError(
                    f"Ego hood bank {bank.name} is in both {stock[bank.name]} and {bank}"
                )
            stock[bank.name] = str(bank)
        for bank, entry in spec["sources"].items():
            if bank in sources:
                raise ValueError(
                    f"Ego hood bank {bank} has a source in more than one spec"
                )
            sources[bank] = (
                str((spec_path.parent / entry["image"]).resolve()),
                FthetaCamera.from_dict(entry["camera"]),
            )
    if len(cameras) != 1:
        raise ValueError(f"Ego hood specs name different cameras: {sorted(cameras)}")
    return cameras.pop(), sources, stock


def generate_ego_hood_banks(
    spec_paths: list[str],
    carlines: str,
    banks_dir: str,
    usdz_paths: dict[str, str],
    overrides_out: str,
) -> dict[str, dict]:
    """Make banks for the clips in ``usdz_paths`` (clip ID -> USDZ) and write their overrides.

    ``spec_paths`` are transform specs whose sources and stock bank directories are
    combined. ``carlines`` maps clip IDs to carline banks (an ego mask overrides file);
    clips missing from it are left out and keep the default bank. Banks are written to
    ``banks_dir``. Returns the overrides written to ``overrides_out``.
    """
    camera, sources, stock = _load_specs(spec_paths)
    base = yaml.safe_load(Path(carlines).read_text())
    # The cache is shared: directories this run creates are group-writable so that
    # other users' runs can add banks.
    missing = []
    path = os.path.abspath(banks_dir)
    while not os.path.isdir(path):
        missing.append(path)
        path = os.path.dirname(path)
    for path in reversed(missing):
        os.makedirs(path, exist_ok=True)
        if os.stat(path).st_uid == os.getuid():
            os.chmod(path, 0o2775)

    unlisted = sorted(set(usdz_paths) - set(base))
    if unlisted:
        logger.warning(
            "Ego hood banks: %d of %d selected clips are not in %s and render without "
            "an ego hood: %s",
            len(unlisted),
            len(usdz_paths),
            carlines,
            ", ".join(unlisted[:10]) + (", ..." if len(unlisted) > 10 else ""),
        )

    overrides: dict[str, dict] = {}
    jobs: list[_Job] = []
    digests: dict[str, str] = {}
    plain: dict[str, str] = {}  # carline -> its copy's bank name
    for clip_id, usdz in sorted(usdz_paths.items()):
        entry = base.get(clip_id)
        if entry is None:
            continue
        carline = entry["rig_config_id"]
        carline_bank = stock[carline]
        if carline not in digests:
            digests[carline] = _bank_digest(carline_bank)
        if carline not in sources:
            plain[carline] = f"{carline}_{digests[carline][:12]}"
            overrides[clip_id] = {**entry, "rig_config_id": plain[carline]}
            continue
        image, source_camera = sources[carline]
        target_camera = FthetaCamera.from_usdz(usdz, camera)
        key = hashlib.sha256(f"v{RENDER_VERSION}".encode())
        key.update(Path(image).read_bytes())
        key.update(repr((source_camera, target_camera)).encode())
        key.update(digests[carline].encode())
        bank = f"transformed_{carline}_{clip_id.removeprefix('clipgt-')}_{key.hexdigest()[:12]}"
        jobs.append(
            _Job(
                bank_dir=os.path.join(banks_dir, bank),
                source_image=image,
                source_camera=source_camera,
                target_camera=target_camera,
                camera=camera,
                carline_bank=carline_bank,
            )
        )
        overrides[clip_id] = {**entry, "rig_config_id": bank}

    if plain:
        logger.warning(
            "Ego hood banks: %d of %d selected clips use unchanged textures for "
            "carline banks without a calibrated front-wide source in %s: %s. "
            "The hood may not align with the scene camera; inspect rendered frames "
            "before using evaluation results.",
            len(overrides) - len(jobs),
            len(usdz_paths),
            ", ".join(spec_paths),
            ", ".join(sorted(plain)),
        )

    for carline, bank in sorted(plain.items()):
        _copy_bank(stock[carline], os.path.join(banks_dir, bank))
    todo = [job for job in jobs if not os.path.isdir(job.bank_dir)]
    logger.info(
        "Ego hood banks: %d transformed (%d to generate), %d carline banks unchanged, in %s",
        len(jobs),
        len(todo),
        len(plain),
        banks_dir,
    )
    if todo:
        with ProcessPoolExecutor(min(GENERATE_WORKERS, len(todo))) as pool:
            list(pool.map(_write_bank, todo))

    # Some renderer versions load every bank in the mounted directory at startup.
    used = {entry["rig_config_id"] for entry in overrides.values()}
    cached = [name for name in os.listdir(banks_dir) if not name.startswith(".")]
    if len(cached) > 2 * len(used):
        logger.warning(
            "Ego hood banks: %s holds %d banks, of which this run uses %d. Renderers "
            "that load every mounted bank need host memory for all of them; remove "
            "banks that no current run uses if renderers run out of memory.",
            banks_dir,
            len(cached),
            len(used),
        )

    os.makedirs(os.path.dirname(overrides_out), exist_ok=True)
    with open(overrides_out, "w") as fh:
        yaml.safe_dump(overrides, fh, sort_keys=True)
    return overrides
