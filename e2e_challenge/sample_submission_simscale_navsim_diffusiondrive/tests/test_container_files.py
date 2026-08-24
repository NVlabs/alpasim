# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Release-boundary tests for the standalone DiffusionDrive submission."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import runpy
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = SUBMISSION_ROOT / "navsim_diffusiondrive_challenge"
MODEL_PACKAGE = RUNTIME_PACKAGE / "simscale_diffusiondrive"

FORBIDDEN_IMPORTS = (
    "navsim_transfuser_challenge",
    "navsim_gtrs_dense_challenge",
    "reference",
    "navsim.agents.diffusiondrive",
)

ASSET_NAME = "diffusiondrive_sim_navhard.ckpt"
EXPECTED_SIZE = "243596717"
EXPECTED_SHA256 = "8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f"
VERIFY_IMAGE_SCRIPT = SUBMISSION_ROOT / "scripts/verify_image.py"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def _load_image_verifier():
    assert VERIFY_IMAGE_SCRIPT.is_file(), "missing testable image verifier"
    spec = importlib.util.spec_from_file_location(
        "diffusiondrive_verify_image", VERIFY_IMAGE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_imports_stay_inside_submission_boundary() -> None:
    violations: list[str] = []
    for path in sorted(RUNTIME_PACKAGE.rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_IMPORTS):
                relative_path = path.relative_to(SUBMISSION_ROOT)
                violations.append(f"{relative_path}: {module}")

    assert not violations, "Forbidden runtime imports:\n" + "\n".join(violations)


def test_package_initializers_exist() -> None:
    missing = [
        path.relative_to(SUBMISSION_ROOT)
        for path in (RUNTIME_PACKAGE / "__init__.py", MODEL_PACKAGE / "__init__.py")
        if not path.is_file()
    ]
    assert not missing, f"Missing package initializers: {missing}"


def test_dockerfile_is_an_independent_offline_diffusiondrive_image() -> None:
    text = (SUBMISSION_ROOT / "Dockerfile").read_text()
    assert text.startswith(
        "ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime\n"
        "FROM ${BASE_IMAGE}"
    )
    for required in (
        "FROM ${BASE_IMAGE} AS grpc_builder",
        "ARG DIFFUSIONDRIVE_SOURCE_DIR",
        "pip install -r /tmp/requirements.txt",
        "COPY src/grpc /tmp/alpasim_grpc",
        "grpcio-tools==1.62.2",
        (
            "COPY ${DIFFUSIONDRIVE_SOURCE_DIR}/navsim_diffusiondrive_challenge/"
            "alpasim_grpc_build_only.toml /tmp/alpasim_grpc/pyproject.toml"
        ),
        "compile_protos",
        "pip install --no-deps --no-build-isolation /tmp/alpasim_grpc",
        "USER root",
        f"DIFFUSIONDRIVE_CHECKPOINT_PATH=/app/assets/diffusiondrive/{ASSET_NAME}",
        "DIFFUSIONDRIVE_DEVICE=cuda",
        "DIFFUSIONDRIVE_MAX_BATCH_SIZE=1",
        "DIFFUSIONDRIVE_BATCH_WINDOW_MS=2",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        f"ARG DIFFUSIONDRIVE_ASSET_FILE={ASSET_NAME}",
        "${DIFFUSIONDRIVE_SOURCE_DIR}/navsim_diffusiondrive_challenge",
        "/app/navsim_diffusiondrive_challenge",
        "USER 10001:10001",
        '["python", "-m", "navsim_diffusiondrive_challenge.driver"]',
    ):
        assert required in text
    for forbidden in (
        "DIFFUSIONDRIVE_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "alpasim-e2e-simscale-ltf",
        "apt-get",
        "git clone",
        "curl ",
        "wget ",
        "--mount",
        "navsim_transfuser_challenge",
        "navsim_gtrs_dense_challenge",
        "reference/SimScale",
        "pytest",
        "pytest-asyncio",
        "hatchling",
    ):
        assert forbidden not in text


def test_grpc_runtime_metadata_excludes_build_and_test_dependencies() -> None:
    metadata = tomllib.loads(
        (RUNTIME_PACKAGE / "alpasim_grpc_build_only.toml").read_text()
    )
    assert metadata["build-system"] == {
        "requires": ["setuptools==80.9.0"],
        "build-backend": "setuptools.build_meta",
    }
    assert metadata["project"]["name"] == "alpasim_grpc"
    assert metadata["project"]["version"] == "0.54.0"
    assert metadata["project"]["dependencies"] == [
        "dataclasses-json==0.6.7",
        "grpcio==1.74.0",
        "numpy==1.26.4",
        "protobuf==4.25.8",
    ]


def test_dockerignore_is_a_strict_diffusiondrive_allowlist() -> None:
    expected = [
        "**",
        "!src/",
        "!src/grpc/",
        "!src/grpc/**",
        "!e2e_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/requirements.txt",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/assets/",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/assets/diffusiondrive/",
        "e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/assets/diffusiondrive/*",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/assets/diffusiondrive/.diffusiondrive-build-*",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/navsim_diffusiondrive_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/navsim_diffusiondrive_challenge/**",
        "**/__pycache__/",
        "**/*.pyc",
    ]
    for ignore_name in (".dockerignore", "Dockerfile.dockerignore"):
        assert (SUBMISSION_ROOT / ignore_name).read_text().splitlines() == expected
    assert not any(ASSET_NAME in line for line in expected)


def test_shell_scripts_are_executable_offline_and_hardened() -> None:
    prepare = SUBMISSION_ROOT / "scripts/prepare_assets.sh"
    build = SUBMISSION_ROOT / "scripts/build_image.sh"
    launcher = SUBMISSION_ROOT / "run_local_container.sh"
    assert all(os.access(path, os.X_OK) for path in (prepare, build, launcher))

    for text in (prepare.read_text(), build.read_text()):
        assert EXPECTED_SIZE in text
        assert EXPECTED_SHA256 in text
        assert "sha256sum" in text
        assert "curl" not in text
        assert "wget" not in text
    build_text = build.read_text()
    assert "docker image inspect" in build_text
    assert "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" in build_text
    assert "alpasim-e2e-simscale-ltf:latest" not in build_text
    assert "--pull" not in build_text

    launcher_text = launcher.read_text()
    for required in (
        "--rm",
        "--init",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "--pids-limit 1024",
        "--memory 32g",
        "--cpus 8",
        "--tmpfs /tmp:",
        "--tmpfs /run:",
        "127.0.0.1:",
        "DIFFUSIONDRIVE_MAX_BATCH_SIZE",
        "ALPASIM_CONTESTANT_REPLICA_INDEX",
        "ALPASIM_CONTESTANT_REPLICAS",
    ):
        assert required in launcher_text
    for forbidden in (
        "DIFFUSIONDRIVE_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "--volume",
        "/var/run/docker.sock",
    ):
        assert forbidden not in launcher_text


def test_requirements_are_pinned_for_independent_build() -> None:
    requirements = (SUBMISSION_ROOT / "requirements.txt").read_text().splitlines()
    assert requirements == [
        "dataclasses-json==0.6.7",
        "grpcio==1.74.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.9.0.80",
        "pillow==11.1.0",
        "protobuf==4.25.8",
        "setuptools==80.9.0",
        "timm==1.0.27",
        "torchvision==0.21.0",
    ]


def test_prepare_assets_supports_verified_atomic_override(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    destination = tmp_path / "assets"
    env = os.environ | {
        "DIFFUSIONDRIVE_ASSET_DIR": str(destination),
        "DIFFUSIONDRIVE_EXPECTED_SIZE": "4",
        "DIFFUSIONDRIVE_EXPECTED_SHA256": hashlib.sha256(b"test").hexdigest(),
    }
    subprocess.run(
        ["bash", str(SUBMISSION_ROOT / "scripts/prepare_assets.sh"), str(source)],
        check=True,
        env=env,
    )
    asset = destination / ASSET_NAME
    assert asset.read_bytes() == b"test"
    assert asset.stat().st_mode & 0o777 == 0o444
    assert list(destination.glob(".diffusiondrive-prepare-*")) == []


@pytest.mark.parametrize(
    ("source_exists", "size", "sha", "message"),
    [
        (False, "4", hashlib.sha256(b"test").hexdigest(), "missing checkpoint"),
        (True, "5", hashlib.sha256(b"test").hexdigest(), "checkpoint size"),
        (True, "4", "0" * 64, "checkpoint sha256"),
    ],
)
def test_prepare_assets_rejects_invalid_source_without_replacing_asset(
    tmp_path: Path,
    source_exists: bool,
    size: str,
    sha: str,
    message: str,
) -> None:
    source = tmp_path / "model.ckpt"
    if source_exists:
        source.write_bytes(b"test")
    destination = tmp_path / "assets"
    destination.mkdir()
    existing = destination / ASSET_NAME
    existing.write_bytes(b"existing")
    env = os.environ | {
        "DIFFUSIONDRIVE_ASSET_DIR": str(destination),
        "DIFFUSIONDRIVE_EXPECTED_SIZE": size,
        "DIFFUSIONDRIVE_EXPECTED_SHA256": sha,
    }
    result = subprocess.run(
        ["bash", str(SUBMISSION_ROOT / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert existing.read_bytes() == b"existing"
    assert list(destination.glob(".diffusiondrive-prepare-*")) == []


def _write_build_fake_docker(fake_bin: Path) -> None:
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
        'if [[ "${1:-}" == "image" && "${2:-}" == "inspect" '
        '&& "$*" == *"--format"* ]]; then\n'
        "  printf '%s\\n' \"${DOCKER_CONFIG_USER:-10001:10001}\"\n"
        "fi\n"
        'if [[ "${1:-}" == "build" ]]; then\n'
        '  snapshot_name=""\n'
        '  for arg in "$@"; do\n'
        '    case "${arg}" in\n'
        '      DIFFUSIONDRIVE_ASSET_FILE=*) snapshot_name="${arg#*=}" ;;\n'
        "    esac\n"
        "  done\n"
        "  stat -c 'snapshot-mode=%a' \"${SNAPSHOT_DIR}/${snapshot_name}\" "
        '>> "${DOCKER_CALLS}"\n'
        "fi\n"
    )
    docker.chmod(0o755)


def _build_script_env(tmp_path: Path) -> dict[str, str]:
    asset = tmp_path / "model.ckpt"
    asset.write_bytes(b"test")
    asset.chmod(0o600)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_build_fake_docker(fake_bin)
    return os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(tmp_path / "docker-calls.txt"),
        "SNAPSHOT_DIR": str(SUBMISSION_ROOT / "assets/diffusiondrive"),
        "DIFFUSIONDRIVE_ASSET_PATH": str(asset),
        "DIFFUSIONDRIVE_EXPECTED_SIZE": "4",
        "DIFFUSIONDRIVE_EXPECTED_SHA256": hashlib.sha256(b"test").hexdigest(),
        "IMAGE": "test-diffusiondrive:image",
    }


def test_build_script_stages_readable_asset_and_probes_image(tmp_path: Path) -> None:
    env = _build_script_env(tmp_path)
    calls = Path(env["DOCKER_CALLS"])
    subprocess.run(
        ["bash", str(SUBMISSION_ROOT / "scripts/build_image.sh")],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    recorded = calls.read_text()
    assert "image inspect alpasim-e2e-simscale-ltf:latest" not in recorded
    assert "build --tag test-diffusiondrive:image" in recorded
    assert "BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" in recorded
    assert (
        "DIFFUSIONDRIVE_SOURCE_DIR="
        "e2e_challenge/sample_submission_simscale_navsim_diffusiondrive" in recorded
    )
    assert str(SUBMISSION_ROOT.parents[1]) in recorded
    assert "--pull" not in recorded
    assert "snapshot-mode=444" in recorded
    assert (
        "image inspect --format {{.Config.User}} test-diffusiondrive:image" in recorded
    )
    run_calls = [line for line in recorded.splitlines() if line.startswith("run ")]
    assert run_calls[0].startswith("run --rm --user 0:0 --entrypoint python")
    assert run_calls[1].startswith("run --rm --entrypoint python")
    assert "--user" not in run_calls[1]
    assert f"/app/assets/diffusiondrive/{ASSET_NAME}" in recorded
    assert 'checkpoint.open("rb")' in recorded
    assert "hashlib.sha256()" in recorded
    assert 'os.environ["DIFFUSIONDRIVE_PROBE_EXPECTED_SIZE"]' in recorded
    assert 'import_module("navsim_diffusiondrive_challenge.driver")' in recorded
    assert 'Path("/app/assets/ltf")' in recorded
    assert 'Path("/app/navsim_transfuser_challenge")' in recorded
    assert 'Path("/app/assets/gtrs_dense")' in recorded
    assert 'Path("/app/navsim_gtrs_dense_challenge")' in recorded
    assert 'name.startswith("LTF_")' in recorded
    assert 'name.startswith("GTRS_")' in recorded
    assert "find_spec(" in recorded
    assert '"navsim_transfuser_challenge"' in recorded
    assert '"navsim_gtrs_dense_challenge"' in recorded
    assert "os.getuid()" in recorded
    assert "os.getgid()" in recorded
    assert (
        list(
            (SUBMISSION_ROOT / "assets/diffusiondrive").glob(".diffusiondrive-build-*")
        )
        == []
    )


def test_build_script_rejects_wrong_configured_user(tmp_path: Path) -> None:
    env = _build_script_env(tmp_path) | {"DOCKER_CONFIG_USER": "root"}
    result = subprocess.run(
        ["bash", str(SUBMISSION_ROOT / "scripts/build_image.sh")],
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "image Config.User root, expected 10001:10001" in result.stderr
    assert (
        list(
            (SUBMISSION_ROOT / "assets/diffusiondrive").glob(".diffusiondrive-build-*")
        )
        == []
    )


def test_image_verifier_accepts_an_isolated_non_root_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_image_verifier()
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"test")
    imported: list[str] = []
    monkeypatch.setattr(verifier.os, "getuid", lambda: 10001)
    monkeypatch.setattr(verifier.os, "getgid", lambda: 10001)
    monkeypatch.setattr(verifier.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        verifier.importlib, "import_module", lambda name: imported.append(name)
    )

    verifier.verify_filesystem(tmp_path)
    verifier.verify_runtime(
        checkpoint,
        4,
        hashlib.sha256(b"test").hexdigest(),
        {},
    )

    assert imported == ["navsim_diffusiondrive_challenge.driver"]


@pytest.mark.parametrize(
    "relative_path",
    (
        "app/assets/ltf",
        "app/navsim_transfuser_challenge",
        "app/assets/gtrs_dense",
        "app/navsim_gtrs_dense_challenge",
    ),
)
def test_image_verifier_rejects_inherited_model_paths(
    tmp_path: Path, relative_path: str
) -> None:
    verifier = _load_image_verifier()
    forbidden_path = tmp_path / relative_path
    forbidden_path.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unexpected inherited path"):
        verifier.verify_filesystem(tmp_path)


@pytest.mark.parametrize("name", ("LTF_DEVICE", "GTRS_CHECKPOINT_PATH"))
def test_image_verifier_rejects_inherited_environment(name: str) -> None:
    verifier = _load_image_verifier()

    with pytest.raises(RuntimeError, match="unexpected inherited environment"):
        verifier.verify_environment({name: "unexpected"})


@pytest.mark.parametrize(
    "module_name",
    ("navsim_transfuser_challenge", "navsim_gtrs_dense_challenge"),
)
def test_image_verifier_rejects_inherited_modules(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_image_verifier()
    monkeypatch.setattr(
        verifier.importlib.util,
        "find_spec",
        lambda name: object() if name == module_name else None,
    )

    with pytest.raises(RuntimeError, match="unexpected inherited module"):
        verifier.verify_modules()


def test_image_verifier_root_mode_checks_filesystem_and_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_image_verifier()
    calls: list[str] = []
    monkeypatch.setattr(
        verifier, "verify_filesystem", lambda: calls.append("filesystem")
    )
    monkeypatch.setattr(verifier, "verify_modules", lambda: calls.append("modules"))

    verifier.main(["filesystem"])

    assert calls == ["filesystem", "modules"]


@pytest.mark.parametrize(("uid", "gid"), ((0, 10001), (10001, 0)))
def test_image_verifier_rejects_wrong_runtime_identity(
    uid: int, gid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_image_verifier()
    monkeypatch.setattr(verifier.os, "getuid", lambda: uid)
    monkeypatch.setattr(verifier.os, "getgid", lambda: gid)

    with pytest.raises(RuntimeError, match="expected uid:gid 10001:10001"):
        verifier.verify_identity()


def test_checkpoint_is_ignored_and_asset_contract_is_documented() -> None:
    assert (
        SUBMISSION_ROOT / "assets/diffusiondrive/.gitignore"
    ).read_text().splitlines() == ["*", "!.gitignore"]
    text = (SUBMISSION_ROOT / "assets/README.md").read_text()
    for required in (
        "DiffusionDrive NAVHARD",
        "OpenDriveLab/SimScale",
        "SimScale_ckpts/DiffusionDrive/" + ASSET_NAME,
        "hf download",
        "--repo-type dataset",
        ASSET_NAME,
        "243,596,717 bytes",
        EXPECTED_SHA256,
        "scripts/prepare_assets.sh",
    ):
        assert required in text
    assert "/high_perf_store4/" not in text


def test_benchmark_result_reports_runtime_and_output_contract() -> None:
    namespace = runpy.run_path(str(SUBMISSION_ROOT / "scripts/benchmark_model.py"))
    result = namespace["_benchmark_result"](
        batch_size=1,
        first_forward_ms=12.5,
        samples=[1.0, 2.0, 10.0],
        peak_vram_bytes=1234,
        device_name="NVIDIA Test GPU",
        output_shape=(1, 8, 3),
    )
    assert result == {
        "batch_size": 1,
        "cuda_version": torch.version.cuda,
        "device_name": "NVIDIA Test GPU",
        "first_forward_ms": 12.5,
        "iterations": 3,
        "numpy_version": np.__version__,
        "output_shape": [1, 8, 3],
        "p50_ms": 2.0,
        "p95_ms": pytest.approx(9.2),
        "peak_vram_bytes": 1234,
        "torch_version": torch.__version__,
    }


def test_readme_documents_reproducible_workflow_and_acceptance() -> None:
    text = (SUBMISSION_ROOT / "README.md").read_text()
    for required in (
        "DiffusionDrive NAVHARD",
        "OpenDriveLab/SimScale",
        "SimScale_ckpts/DiffusionDrive/" + ASSET_NAME,
        ASSET_NAME,
        "243,596,717 bytes",
        EXPECTED_SHA256,
        "scripts/prepare_assets.sh",
        "scripts/build_image.sh",
        "run_local_container.sh",
        "scripts/probe_container.py",
        "alpasim_wizard",
        "diffusiondrive_inference > 0",
        "inference_error = 0",
        "all-fallback",
        "host mounts",
        "reference/SimScale",
        "same-timestamp RPC `DynamicState`",
        "Pose history is used only when",
        "dynamic_state_fallback",
        "4d352ba",
    ):
        assert required in text
    for forbidden in (
        "DIFFUSIONDRIVE_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
    ):
        assert forbidden not in text


def test_readme_documents_independent_image_build() -> None:
    text = (SUBMISSION_ROOT / "README.md").read_text()
    build_section = text.split("## Build", 1)[1].split("## Local Smoke And Probe", 1)[0]
    for required in (
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        "requirements.txt",
        "src/grpc",
        "independent",
        "scripts/build_image.sh",
    ):
        assert required in build_section
    for forbidden in (
        "alpasim-e2e-simscale-ltf:latest",
        "sha256:77d2066ccb3d79038438f59da2cc024202da2b509a0c0c9368bd5249c682a9e2",
        "6,593,675,543 bytes",
        "sha256:b4ce29c824322532c7992c7ef8fa95461f1128483591a6ac8aed24332f723795",
        "6,837,387,745 bytes",
    ):
        assert forbidden not in text
    for forbidden in ("243.7 MB", "The legacy builder sent"):
        assert forbidden not in build_section


def test_readme_uses_portable_paths_and_device_selection() -> None:
    text = (SUBMISSION_ROOT / "README.md").read_text()

    for required in (
        "OpenDriveLab/SimScale",
        "SimScale_ckpts/DiffusionDrive/" + ASSET_NAME,
        "hf download",
        "--repo-type dataset",
        "/path/to/alpasim-nuplan-track",
    ):
        assert required in text
    assert "/high_perf_store4/" not in text
    assert "aggregate run UUID" not in text
    assert re.search(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text) is None
    assert re.search(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", text) is None
