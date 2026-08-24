# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import os
import re
import runpy
import subprocess
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION = "sample_submission_simscale_navsim_gtrs_dense"
EXPECTED_IMAGE = "alpasim-e2e-simscale-gtrs-dense:latest"
EXPECTED_CONTAINER_PREFIX = "alpasim-e2e-simscale-gtrs-dense"
EXPECTED_IMAGE_DEFAULT = 'IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"'
EXPECTED_CONTAINER_PREFIX_DEFAULT = (
    f'PREFIX="${{ALPASIM_DRIVER_NAME_PREFIX:-{EXPECTED_CONTAINER_PREFIX}}}"'
)
ASSET_NAME = "gtrs_dense_resnet_sim_reward_navhard.ckpt"
EXPECTED_SIZE = "269095388"
EXPECTED_SHA256 = "8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409"
EXPERT_ASSET_NAME = "gtrs_dense_resnet_sim_expert_navhard.ckpt"
EXPERT_SHA256 = "2496b82f5f256d7de09fca656c7634967b8660eb12e5c10386a587283629a7ff"
VOV_REWARD_ASSET_NAME = "gtrs_dense_vov_sim_reward_navhard.ckpt"
VOV_REWARD_SHA256 = "7567d269bd8d0757cf906c30612bf1ad167ac7310e8af0ead74dc7798fe54c99"
VOV_EXPERT_ASSET_NAME = "gtrs_dense_vov_sim_expert_navhard.ckpt"
VOV_EXPERT_SHA256 = "badcf3e7c3e2ecc1d7ecb9fc744c78420c368f96e47b89d1681ade7833cd5e57"
VOCAB_NAME = "navhard_8192.npy"
VOCAB_SIZE = 3_932_288
VOCAB_SHA256 = "cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35"
RELEASE_VOCAB_NAME = "navsim_16384.npy"
RELEASE_VOCAB_SIZE = 7_864_448
RELEASE_VOCAB_SHA256 = (
    "e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d"
)
PUBLIC_BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
SOURCE_DIR = f"e2e_challenge/{SAMPLE.name}"
REPO_ROOT = SAMPLE.parents[1]
RELEASE_CONTRACT_FILES = (
    SAMPLE / "Dockerfile",
    SAMPLE / "scripts/prepare_assets.sh",
    SAMPLE / "scripts/build_image.sh",
    SAMPLE / "assets/README.md",
    SAMPLE / "README.md",
    SAMPLE / "navsim_gtrs_dense_challenge/simscale_gtrs_dense/NOTICE.md",
    *sorted((SAMPLE / "scripts").glob("*.py")),
    *sorted((SAMPLE / "navsim_gtrs_dense_challenge").rglob("*.py")),
)


def test_submission_has_fixed_resnet_reward_identity() -> None:
    assert SAMPLE.name == EXPECTED_SUBMISSION


def test_build_image_has_exact_resnet_reward_image_default() -> None:
    lines = [
        line.strip()
        for line in (SAMPLE / "scripts/build_image.sh").read_text().splitlines()
    ]

    assert f'DEFAULT_IMAGE="{EXPECTED_IMAGE}"' in lines
    assert 'IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"' in lines


def test_canonical_asset_scripts_do_not_reference_expert_submission() -> None:
    for script in ("prepare_assets.sh", "build_image.sh"):
        text = (SAMPLE / f"scripts/{script}").read_text()
        assert "sample_submission_simscale_navsim_gtrs_dense_resnet_expert" not in text


def test_launcher_has_exact_resnet_reward_defaults() -> None:
    lines = (SAMPLE / "run_local_container.sh").read_text().splitlines()

    assert EXPECTED_IMAGE_DEFAULT in lines
    assert EXPECTED_CONTAINER_PREFIX_DEFAULT in lines


def test_dockerfile_builds_gtrs_from_public_pytorch_runtime() -> None:
    text = (SAMPLE / "Dockerfile").read_text()

    assert text.startswith(
        "ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime\n"
        "FROM ${BASE_IMAGE}"
    )
    for required in (
        "ARG INSTALL_DEPENDENCIES=1",
        "ARG GTRS_SOURCE_DIR",
        "pip install -r /tmp/requirements.txt",
        "COPY src/grpc /tmp/alpasim_grpc",
        "pip install /tmp/alpasim_grpc",
        "USER root",
        "GTRS_CHECKPOINT_PATH=/app/assets/gtrs_dense/${GTRS_CHECKPOINT_NAME}",
        "GTRS_VOCAB_PATH=/app/assets/gtrs_dense/navsim_16384.npy",
        "GTRS_DEVICE=cuda",
        "GTRS_MAX_BATCH_SIZE=1",
        "GTRS_BATCH_WINDOW_MS=2",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "${GTRS_SOURCE_DIR}/assets/gtrs_dense/${GTRS_ASSET_FILE}",
        "/app/assets/gtrs_dense/${GTRS_CHECKPOINT_NAME}",
        "${GTRS_SOURCE_DIR}/assets/gtrs_dense/navhard_8192.npy",
        "/app/assets/gtrs_dense/navhard_8192.npy",
        "${GTRS_SOURCE_DIR}/assets/gtrs_dense/navsim_16384.npy",
        "/app/assets/gtrs_dense/navsim_16384.npy",
        "/app/navsim_gtrs_dense_challenge",
        "USER 10001:10001",
        'CMD ["python", "-m", "navsim_gtrs_dense_challenge.driver"]',
    ):
        assert required in text

    for forbidden in (
        "GTRS_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "alpasim-e2e-simscale-ltf",
        "apt-get",
        "git clone",
        "curl ",
        "wget ",
        "--mount",
    ):
        assert forbidden not in text


def test_dockerfile_enables_speed_enhancement_by_default() -> None:
    text = (SAMPLE / "Dockerfile").read_text()

    assert "GTRS_SPEED_ENHANCEMENT=1" in text


def test_requirements_are_pinned_for_gtrs_runtime() -> None:
    assert (SAMPLE / "requirements.txt").read_text().splitlines() == [
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


def test_dockerignore_is_a_strict_gtrs_allowlist() -> None:
    lines = (SAMPLE / "Dockerfile.dockerignore").read_text().splitlines()

    assert lines == [
        "**",
        "!src/",
        "!src/grpc/",
        "!src/grpc/**",
        "!e2e_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/requirements.txt",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/",
        "e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/*",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/navhard_8192.npy",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/navsim_16384.npy",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/.gtrs-build-*",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/navsim_gtrs_dense_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/navsim_gtrs_dense_challenge/**",
        "**/__pycache__/",
        "**/*.pyc",
    ]
    assert (SAMPLE / ".dockerignore").read_text().splitlines() == lines
    assert not any(ASSET_NAME in line for line in lines)
    assert not any(
        forbidden in line
        for line in lines
        for forbidden in ("reference", "/tests", "/scripts", "README")
    )


@pytest.mark.parametrize("ignore_name", [".dockerignore", "Dockerfile.dockerignore"])
def test_dockerignore_finally_excludes_python_bytecode(ignore_name: str) -> None:
    lines = (SAMPLE / ignore_name).read_text().splitlines()
    source_allowlist = (
        "!e2e_challenge/"
        "sample_submission_simscale_navsim_gtrs_dense/"
        "navsim_gtrs_dense_challenge/**"
    )
    cache_exclusions = ["**/__pycache__/", "**/*.pyc"]

    assert source_allowlist in lines
    assert lines[-2:] == cache_exclusions
    assert lines.index(source_allowlist) < lines.index(cache_exclusions[0])


def test_shell_scripts_are_executable_and_harden_the_container() -> None:
    paths = [
        SAMPLE / "scripts/prepare_assets.sh",
        SAMPLE / "scripts/build_image.sh",
        SAMPLE / "run_local_container.sh",
    ]
    assert all(os.access(path, os.X_OK) for path in paths)

    prepare = paths[0].read_text()
    build = paths[1].read_text()
    for text in (prepare, build):
        assert EXPECTED_SIZE in text
        assert EXPECTED_SHA256 in text
        assert EXPERT_SHA256 in text
        assert VOV_REWARD_SHA256 in text
        assert VOV_EXPERT_SHA256 in text
        assert "sha256sum" in text
        assert "curl" not in text
        assert "wget" not in text
        assert f'EXPECTED_SIZE="{EXPECTED_SIZE}"' in text
        assert f'EXPECTED_SHA256="{EXPECTED_SHA256}"' in text
        assert "GTRS_EXPECTED_SIZE" not in text
        assert "GTRS_EXPECTED_SHA256" not in text
    assert "docker image inspect" in build
    assert PUBLIC_BASE_IMAGE in build
    assert "alpasim-e2e-simscale-ltf:latest" not in build
    assert "--pull" not in build

    launcher = paths[2].read_text()
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
        "GTRS_SCORER_MODE=${SCORER_MODE}",
        "GTRS_SPEED_PROXY=${SPEED_PROXY}",
        '"longitudinal_0p5s"',
        "GTRS_CURVATURE_WEIGHT=${CURVATURE_WEIGHT}",
        "GTRS_HEADING_CHANGE_WEIGHT=${HEADING_CHANGE_WEIGHT}",
        "GTRS_TRAJECTORY_TIME_SCALE=${TRAJECTORY_TIME_SCALE}",
        "GTRS_MAX_BATCH_SIZE",
        "ALPASIM_CONTESTANT_REPLICA_INDEX",
        "ALPASIM_CONTESTANT_REPLICAS",
    ):
        assert required in launcher
    for forbidden in (
        "GTRS_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "--volume",
        "/var/run/docker.sock",
    ):
        assert forbidden not in launcher


def _write_fake_wc(fake_bin: Path, expected_size: str = EXPECTED_SIZE) -> None:
    wc = fake_bin / "wc"
    wc.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' '{expected_size}'\n")
    wc.chmod(0o755)


def _write_fixed_identity_tools(
    fake_bin: Path,
    expected_sha: str = EXPECTED_SHA256,
    expected_size: str = EXPECTED_SIZE,
) -> None:
    _write_fake_wc(fake_bin, expected_size)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_text(
        f"#!/usr/bin/env bash\nprintf '%s  %s\\n' '{expected_sha}' \"${{1:-asset}}\"\n"
    )
    sha256sum.chmod(0o755)


def test_prepare_assets_success_with_mocked_identity_tools(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    destination = tmp_path / "assets"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fixed_identity_tools(fake_bin)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GTRS_ASSET_DIR": str(destination),
    }

    subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        check=True,
        env=env,
    )

    assert (destination / ASSET_NAME).read_bytes() == b"test"


def test_prepare_assets_uses_canonical_default_directory(tmp_path: Path) -> None:
    challenge = tmp_path / "e2e_challenge"
    sample = challenge / SAMPLE.name
    script = sample / "scripts/prepare_assets.sh"
    script.parent.mkdir(parents=True)
    script.write_text((SAMPLE / "scripts/prepare_assets.sh").read_text())

    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fixed_identity_tools(fake_bin, expected_sha=EXPERT_SHA256)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GTRS_METHOD": "expert",
    }

    subprocess.run(["bash", str(script), str(source)], check=True, env=env)

    destination = sample / "assets/gtrs_dense" / EXPERT_ASSET_NAME
    assert destination.read_bytes() == b"test"


def test_prepare_assets_rejects_identity_override(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    env = os.environ | {
        "GTRS_ASSET_DIR": str(tmp_path / "assets"),
        "GTRS_EXPECTED_SIZE": "4",
        "GTRS_EXPECTED_SHA256": hashlib.sha256(b"test").hexdigest(),
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert "checkpoint size" in result.stderr


@pytest.mark.parametrize(
    ("source_exists", "message"),
    [(False, "missing checkpoint"), (True, "checkpoint size")],
)
def test_prepare_assets_rejects_missing_or_wrong_size(
    tmp_path: Path,
    source_exists: bool,
    message: str,
) -> None:
    source = tmp_path / "model.ckpt"
    if source_exists:
        source.write_bytes(b"test")
    env = os.environ | {"GTRS_ASSET_DIR": str(tmp_path / "assets")}

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_prepare_assets_rejects_wrong_sha(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_wc(fake_bin)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GTRS_ASSET_DIR": str(tmp_path / "assets"),
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert "checkpoint sha256" in result.stderr


def test_prepare_failure_preserves_existing_asset_and_cleans_temp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"invalid")
    destination = tmp_path / "assets"
    destination.mkdir()
    checkpoint = destination / ASSET_NAME
    checkpoint.write_bytes(b"existing")
    env = os.environ | {"GTRS_ASSET_DIR": str(destination)}

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert checkpoint.read_bytes() == b"existing"
    assert list(destination.glob(".gtrs-prepare-*")) == []


def test_build_script_rejects_identity_override(tmp_path: Path) -> None:
    asset = tmp_path / "model.ckpt"
    asset.write_bytes(b"test")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\nexit 0\n")
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GTRS_ASSET_PATH": str(asset),
        "GTRS_EXPECTED_SIZE": "4",
        "GTRS_EXPECTED_SHA256": hashlib.sha256(b"test").hexdigest(),
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "checkpoint size" in result.stderr


def test_build_script_rejects_unknown_method_before_docker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "GTRS_METHOD": "distill",
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "ERROR: GTRS_METHOD=distill is invalid; expected reward or expert"
    )
    assert not calls.exists()


def test_build_script_rejects_unknown_backbone_before_docker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "GTRS_BACKBONE": "vit",
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "ERROR: GTRS_BACKBONE=vit is invalid; expected resnet or vov"
    )
    assert not calls.exists()


def test_build_script_rejects_invalid_dependency_mode_before_docker(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "INSTALL_DEPENDENCIES": "sometimes",
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "ERROR: INSTALL_DEPENDENCIES=sometimes is invalid; expected 0 or 1"
    )
    assert not calls.exists()


def _run_launcher_with_fake_docker(
    tmp_path: Path,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    for variable in (
        "GTRS_SPEED_ENHANCEMENT",
        "GTRS_SCORER_MODE",
        "GTRS_EP_EXPONENT",
        "GTRS_SPEED_TOP_K",
        "GTRS_SPEED_WEIGHT",
        "GTRS_BACKBONE",
        "GTRS_METHOD",
        "IMAGE",
    ):
        env.pop(variable, None)
    env |= {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "ALPASIM_DOCKER_GPUS": "none",
        **(overrides or {}),
    }
    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
    )
    return result, calls


@pytest.mark.parametrize(
    ("backbone", "method", "expected_image"),
    [
        ("resnet", "reward", EXPECTED_IMAGE),
        (
            "resnet",
            "expert",
            "alpasim-e2e-simscale-gtrs-dense-resnet-expert:latest",
        ),
        ("vov", "reward", "alpasim-e2e-simscale-gtrs-dense-vov-reward:latest"),
        ("vov", "expert", "alpasim-e2e-simscale-gtrs-dense-vov-expert:latest"),
    ],
)
def test_launcher_selects_fixed_variant_image(
    tmp_path: Path,
    backbone: str,
    method: str,
    expected_image: str,
) -> None:
    result, calls = _run_launcher_with_fake_docker(
        tmp_path,
        {"GTRS_BACKBONE": backbone, "GTRS_METHOD": method},
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert docker_run.endswith(expected_image)
    assert "-e GTRS_BACKBONE=" not in docker_run
    assert "-e GTRS_METHOD=" not in docker_run


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("GTRS_BACKBONE", "vit", "expected resnet or vov"),
        ("GTRS_METHOD", "distill", "expected reward or expert"),
    ],
)
def test_launcher_rejects_invalid_variant_before_docker(
    tmp_path: Path,
    variable: str,
    value: str,
    message: str,
) -> None:
    result, calls = _run_launcher_with_fake_docker(tmp_path, {variable: value})

    assert result.returncode == 2
    assert message in result.stderr
    assert not calls.exists()


def test_launcher_speed_enhancement_is_enabled_by_default(tmp_path: Path) -> None:
    result, calls = _run_launcher_with_fake_docker(tmp_path)

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert "-e GTRS_SPEED_ENHANCEMENT=1" in docker_run
    assert "-e GTRS_SCORER_MODE=nc_dac_ep" in docker_run
    assert "-e GTRS_EP_EXPONENT=3" in docker_run
    assert "-e GTRS_SPEED_TOP_K=64" in docker_run
    assert "-e GTRS_SPEED_WEIGHT=3" in docker_run


def test_launcher_speed_enhancement_can_be_disabled(tmp_path: Path) -> None:
    result, calls = _run_launcher_with_fake_docker(
        tmp_path,
        {"GTRS_SPEED_ENHANCEMENT": "0"},
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert "-e GTRS_SPEED_ENHANCEMENT=0" in docker_run
    assert "-e GTRS_SCORER_MODE=nc_dac_ep" in docker_run
    assert "-e GTRS_EP_EXPONENT=1" in docker_run
    assert "-e GTRS_SPEED_TOP_K=0" in docker_run
    assert "-e GTRS_SPEED_WEIGHT=0" in docker_run


def test_launcher_speed_enhancement_allows_advanced_overrides(
    tmp_path: Path,
) -> None:
    result, calls = _run_launcher_with_fake_docker(
        tmp_path,
        {
            "GTRS_EP_EXPONENT": "10",
            "GTRS_SPEED_TOP_K": "32",
            "GTRS_SPEED_WEIGHT": "0.1",
        },
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert "-e GTRS_SPEED_ENHANCEMENT=1" in docker_run
    assert "-e GTRS_SCORER_MODE=nc_dac_ep" in docker_run
    assert "-e GTRS_EP_EXPONENT=10" in docker_run
    assert "-e GTRS_SPEED_TOP_K=32" in docker_run
    assert "-e GTRS_SPEED_WEIGHT=0.1" in docker_run


def test_launcher_speed_enhancement_rejects_invalid_switch_before_docker(
    tmp_path: Path,
) -> None:
    result, calls = _run_launcher_with_fake_docker(
        tmp_path,
        {"GTRS_SPEED_ENHANCEMENT": "true"},
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "ERROR: GTRS_SPEED_ENHANCEMENT must be 0 or 1"
    assert not calls.exists()


@pytest.mark.parametrize("scale", ["nan", "0.9", "1.3"])
def test_launcher_rejects_invalid_trajectory_time_scale_before_docker(
    tmp_path: Path,
    scale: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "GTRS_TRAJECTORY_TIME_SCALE": scale,
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert "GTRS_TRAJECTORY_TIME_SCALE" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("launcher", "expected_image", "expected_scorer"),
    [
        (
            SAMPLE / "run_local_container.sh",
            "alpasim-e2e-simscale-gtrs-dense:latest",
            "nc_dac_ep",
        ),
    ],
)
def test_launcher_preserves_selected_image_asset_defaults_when_unset(
    tmp_path: Path,
    launcher: Path,
    expected_image: str,
    expected_scorer: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.pop("GTRS_CHECKPOINT_PATH", None)
    env.pop("GTRS_VOCAB_PATH", None)
    env |= {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "ALPASIM_DOCKER_GPUS": "none",
    }

    result = subprocess.run(
        ["bash", str(launcher)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert "GTRS_CHECKPOINT_PATH" not in docker_run
    assert "GTRS_VOCAB_PATH" not in docker_run
    assert f"-e GTRS_SCORER_MODE={expected_scorer}" in docker_run
    assert "-e GTRS_TRAJECTORY_TIME_SCALE=1.0" in docker_run
    assert docker_run.endswith(expected_image)


@pytest.mark.parametrize(
    "overrides",
    [
        {"GTRS_CHECKPOINT_PATH": "/experiment/checkpoint.ckpt"},
        {"GTRS_VOCAB_PATH": "/experiment/navhard_8192.npy"},
        {
            "GTRS_CHECKPOINT_PATH": "/experiment/checkpoint.ckpt",
            "GTRS_VOCAB_PATH": "/experiment/navhard_8192.npy",
        },
    ],
)
def test_launcher_forwards_only_explicit_asset_overrides(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.pop("GTRS_CHECKPOINT_PATH", None)
    env.pop("GTRS_VOCAB_PATH", None)
    env |= {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "ALPASIM_DOCKER_GPUS": "none",
        **overrides,
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    for variable, value in overrides.items():
        assert f"-e {variable}={value}" in docker_run
    for unset_variable in {
        "GTRS_CHECKPOINT_PATH",
        "GTRS_VOCAB_PATH",
    } - overrides.keys():
        assert unset_variable not in docker_run


def test_launcher_forwards_safety_gate_anchor_configuration(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "ALPASIM_DOCKER_GPUS": "none",
        "GTRS_SCORER_MODE": "safety_gate_ep",
        "GTRS_EP_EXPONENT": "3",
        "GTRS_SPEED_TOP_K": "64",
        "GTRS_SPEED_WEIGHT": "3",
        "GTRS_SPEED_PROXY": "longitudinal",
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    docker_run = calls.read_text().splitlines()[-1]
    assert "-e GTRS_SCORER_MODE=safety_gate_ep" in docker_run
    assert "-e GTRS_EP_EXPONENT=3" in docker_run
    assert "-e GTRS_SPEED_TOP_K=64" in docker_run
    assert "-e GTRS_SPEED_WEIGHT=3" in docker_run
    assert "-e GTRS_SPEED_PROXY=longitudinal" in docker_run


@pytest.mark.parametrize(
    (
        "backbone",
        "method",
        "asset_name",
        "expected_sha",
        "image",
        "service_version",
    ),
    [
        (
            "resnet",
            "reward",
            ASSET_NAME,
            EXPECTED_SHA256,
            EXPECTED_IMAGE,
            "simscale-gtrs-dense-e2e",
        ),
        (
            "resnet",
            "expert",
            EXPERT_ASSET_NAME,
            EXPERT_SHA256,
            "alpasim-e2e-simscale-gtrs-dense-resnet-expert:latest",
            "simscale-gtrs-dense-resnet-expert-e2e",
        ),
        (
            "vov",
            "reward",
            VOV_REWARD_ASSET_NAME,
            VOV_REWARD_SHA256,
            "alpasim-e2e-simscale-gtrs-dense-vov-reward:latest",
            "simscale-gtrs-dense-vov-reward-e2e",
        ),
        (
            "vov",
            "expert",
            VOV_EXPERT_ASSET_NAME,
            VOV_EXPERT_SHA256,
            "alpasim-e2e-simscale-gtrs-dense-vov-expert:latest",
            "simscale-gtrs-dense-vov-expert-e2e",
        ),
    ],
)
def test_build_script_stages_fixed_variant_and_probes_image(
    tmp_path: Path,
    backbone: str,
    method: str,
    asset_name: str,
    expected_sha: str,
    image: str,
    service_version: str,
) -> None:
    asset = tmp_path / "model.ckpt"
    asset.write_bytes(b"test")
    asset.chmod(0o600)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
        'if [[ "${1:-}" == "build" ]]; then\n'
        '  snapshot_name=""\n'
        '  for arg in "$@"; do\n'
        '    case "${arg}" in\n'
        '      GTRS_ASSET_FILE=*) snapshot_name="${arg#*=}" ;;\n'
        "    esac\n"
        "  done\n"
        "  stat -c 'snapshot-mode=%a' \"${SNAPSHOT_DIR}/${snapshot_name}\" "
        '>> "${DOCKER_CALLS}"\n'
        "fi\n"
        "exit 0\n"
    )
    docker.chmod(0o755)
    _write_fixed_identity_tools(
        fake_bin,
        expected_sha,
        "332348155" if backbone == "vov" else EXPECTED_SIZE,
    )
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "SNAPSHOT_DIR": str(SAMPLE / "assets/gtrs_dense"),
        "GTRS_ASSET_PATH": str(asset),
        "GTRS_BACKBONE": backbone,
        "GTRS_METHOD": method,
    }

    subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        check=True,
        env=env,
        cwd=tmp_path,
    )

    recorded = calls.read_text()
    assert f"build --tag {image}" in recorded
    assert f"BASE_IMAGE={PUBLIC_BASE_IMAGE}" in recorded
    assert "INSTALL_DEPENDENCIES=1" in recorded
    assert f"GTRS_SOURCE_DIR={SOURCE_DIR}" in recorded
    assert f"GTRS_CHECKPOINT_NAME={asset_name}" in recorded
    assert f"GTRS_BACKBONE={backbone}" in recorded
    assert f"GTRS_SERVICE_VERSION={service_version}" in recorded
    assert "--pull" not in recorded
    assert "snapshot-mode=444" in recorded
    build_call = next(
        line for line in recorded.splitlines() if line.startswith("build ")
    )
    assert build_call.endswith(str(REPO_ROOT))
    assert "run --rm --user 10001:10001 --entrypoint python" in recorded
    assert f"/app/assets/gtrs_dense/{asset_name}" in recorded
    assert "/app/assets/gtrs_dense/navhard_8192.npy" in recorded
    assert "/app/assets/gtrs_dense/navsim_16384.npy" in recorded
    assert 'checkpoint.open("rb")' in recorded
    assert "stream.read(1024 * 1024)" in recorded
    assert "hashlib.sha256()" in recorded
    assert "expected_size = int(sys.argv[2])" in recorded
    assert "expected_sha = sys.argv[3]" in recorded
    assert "expected_version = sys.argv[4]" in recorded
    assert expected_sha in recorded
    assert service_version in recorded
    assert "GTRS_PROBE_EXPECTED_SIZE" not in recorded
    assert "GTRS_PROBE_EXPECTED_SHA256" not in recorded
    assert "size != expected_size" in recorded
    assert "actual_sha != expected_sha" in recorded
    assert VOCAB_SHA256 in recorded
    assert RELEASE_VOCAB_SHA256 in recorded
    assert "np.load(vocabulary_path, allow_pickle=False)" in recorded
    assert "(8192, 40, 3)" in recorded
    assert "(16384, 40, 3)" in recorded
    assert "np.float32" in recorded
    assert "import navsim_gtrs_dense_challenge.driver" in recorded
    assert list((SAMPLE / "assets/gtrs_dense").glob(".gtrs-build-*")) == []


def test_build_script_uses_canonical_expert_asset_by_default(tmp_path: Path) -> None:
    sample = tmp_path / "e2e_challenge" / SAMPLE.name
    script = sample / "scripts/build_image.sh"
    script.parent.mkdir(parents=True)
    script.write_text((SAMPLE / "scripts/build_image.sh").read_text())
    asset_dir = sample / "assets/gtrs_dense"
    asset_dir.mkdir(parents=True)
    (asset_dir / EXPERT_ASSET_NAME).write_bytes(b"test")
    for vocabulary in (VOCAB_NAME, RELEASE_VOCAB_NAME):
        (asset_dir / vocabulary).touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "docker-calls.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "${DOCKER_CALLS}"\n'
    )
    docker.chmod(0o755)
    python = fake_bin / "python"
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
    python.chmod(0o755)
    _write_fixed_identity_tools(fake_bin, EXPERT_SHA256)
    env = os.environ.copy()
    env.pop("GTRS_ASSET_PATH", None)
    env |= {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "GTRS_METHOD": "expert",
    }

    subprocess.run(["bash", str(script)], check=True, env=env, cwd=tmp_path)

    recorded = calls.read_text()
    assert f"GTRS_CHECKPOINT_NAME={EXPERT_ASSET_NAME}" in recorded
    assert list(asset_dir.glob(".gtrs-build-*")) == []


def test_checkpoint_is_ignored_by_submission_asset_rules() -> None:
    lines = (SAMPLE / "assets/gtrs_dense/.gitignore").read_text().splitlines()

    assert lines == ["*", "!.gitignore", "!navhard_8192.npy", "!navsim_16384.npy"]


def test_official_navhard_vocabulary_asset_identity() -> None:
    path = SAMPLE / "assets/gtrs_dense" / VOCAB_NAME
    assert path.stat().st_size == VOCAB_SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == VOCAB_SHA256


def test_release_vocabulary_asset_identity() -> None:
    path = SAMPLE / "assets/gtrs_dense" / RELEASE_VOCAB_NAME
    assert path.stat().st_size == RELEASE_VOCAB_SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RELEASE_VOCAB_SHA256


def test_asset_readme_documents_canonical_resnet_asset_contract() -> None:
    text = (SAMPLE / "assets/README.md").read_text()

    for required in (
        "ResNet reward NAVHARD",
        "OpenDriveLab/SimScale",
        "SimScale_ckpts/GTRS_Dense/" + ASSET_NAME,
        "SimScale_ckpts/GTRS_Dense/" + EXPERT_ASSET_NAME,
        "hf download",
        "--repo-type dataset",
        "exp/_ckpts/weights/GTRS_Dense/" + ASSET_NAME,
        ASSET_NAME,
        "ResNet expert NAVHARD",
        "exp/_ckpts/weights/GTRS_Dense/" + EXPERT_ASSET_NAME,
        EXPERT_ASSET_NAME,
        "269,095,388 bytes",
        EXPECTED_SHA256,
        EXPERT_SHA256,
        "GTRS_METHOD=expert bash scripts/prepare_assets.sh",
        "GTRS_METHOD=expert bash scripts/build_image.sh",
        "`GTRS_ASSET_DIR` and",
        "`GTRS_ASSET_PATH` overrides",
        "image's checkpoint identity is fixed at build time",
        "runtime does not select or switch checkpoints",
        "navsim_16384.npy",
        RELEASE_VOCAB_SHA256,
        "pose",
    ):
        assert required in text
    assert text.count("exactly 269,095,388 bytes") == 2


def _assert_only_known_checkpoint_identities(asset_dir: Path) -> None:
    checkpoint_names = {
        path.name for path in asset_dir.iterdir() if path.suffix == ".ckpt"
    }

    assert checkpoint_names <= {
        ASSET_NAME,
        EXPERT_ASSET_NAME,
        VOV_REWARD_ASSET_NAME,
        VOV_EXPERT_ASSET_NAME,
    }


def test_asset_directory_rejects_other_checkpoint_identities() -> None:
    _assert_only_known_checkpoint_identities(SAMPLE / "assets/gtrs_dense")


def test_asset_directory_allows_expert_checkpoint_identity(tmp_path: Path) -> None:
    (tmp_path / EXPERT_ASSET_NAME).touch()

    _assert_only_known_checkpoint_identities(tmp_path)


def test_asset_directory_rejects_unknown_checkpoint_identity(tmp_path: Path) -> None:
    (tmp_path / "unrelated.ckpt").touch()

    with pytest.raises(AssertionError):
        _assert_only_known_checkpoint_identities(tmp_path)


def test_current_release_surface_contains_vov_implementation() -> None:
    release_text = "\n".join(path.read_text() for path in RELEASE_CONTRACT_FILES)

    for required in (
        VOV_REWARD_ASSET_NAME,
        VOV_EXPERT_ASSET_NAME,
        "V-99-eSE",
        "GTRS_BACKBONE",
    ):
        assert required in release_text


def test_readme_documents_resnet_workflow_and_closed_loop_acceptance() -> None:
    text = (SAMPLE / "README.md").read_text()

    for required in (
        "ResNet34 backbone",
        "reward checkpoint",
        "navsim_16384.npy",
        "8,192 NAVHARD vocabulary",
        "same-timestamp",
        "4d352ba",
        "FP32",
        ASSET_NAME,
        "269,095,388 bytes",
        EXPECTED_SHA256,
        EXPECTED_IMAGE,
        PUBLIC_BASE_IMAGE,
        "INSTALL_DEPENDENCIES=0",
        "gtrs_inference > 0",
        "inference_error = 0",
        "NuPlan/MTGS Smoke",
    ):
        assert required in text

    for forbidden in (
        "TBD",
        "TODO",
        "Build evidence is pending",
        "Probe evidence is pending",
        "smoke evidence is pending",
        "The build uses only the local",
    ):
        assert forbidden not in text


def test_readme_documents_speed_enhancement_profile() -> None:
    text = (SAMPLE / "README.md").read_text()

    for required in (
        "GTRS_SPEED_ENHANCEMENT=1",
        "GTRS_SPEED_ENHANCEMENT=0",
        "NC * DAC * EP",
        "EP=3",
        "K=64",
        "lambda=3",
        "GTRS_SCORER_MODE",
        "GTRS_EP_EXPONENT",
        "GTRS_SPEED_TOP_K",
        "GTRS_SPEED_WEIGHT",
    ):
        assert required in text


def test_readme_uses_portable_paths_and_device_selection() -> None:
    text = (SAMPLE / "README.md").read_text()

    for required in (
        "OpenDriveLab/SimScale",
        "SimScale_ckpts/GTRS_Dense/" + ASSET_NAME,
        "hf download",
        "--repo-type dataset",
        "/path/to/alpasim-nuplan-track",
    ):
        assert required in text
    assert "/high_perf_store4/" not in text
    assert "aggregate run UUID" not in text
    assert re.search(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text) is None
    assert re.search(r"\b[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\b", text) is None
    assert re.search(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", text) is None


def test_readme_default_setup_is_reward_with_release_vocabulary() -> None:
    text = (SAMPLE / "README.md").read_text()
    checkpoint_section = text.split("## Prepare The Checkpoint", maxsplit=1)[1].split(
        "## Build", maxsplit=1
    )[0]

    assert "reward checkpoint" in checkpoint_section
    assert "SimScale_ckpts/GTRS_Dense/" + ASSET_NAME in checkpoint_section
    assert EXPERT_ASSET_NAME not in checkpoint_section
    assert "navsim_16384.npy" in text
    assert "default" in text


def test_notice_documents_backbone_provenance_and_offline_encoders() -> None:
    text = (
        SAMPLE / "navsim_gtrs_dense_challenge/simscale_gtrs_dense/NOTICE.md"
    ).read_text()

    for required in (
        "ResNet34 and latent LiDAR Transfuser graph",
        "transfuser_backbone.py",
        "09cf91ba32a42443f14c44e9ac8630831ca537064b5201958736c075f153b89a",
        "utils/attn.py",
        "pretrained=False",
        "load all release weights strictly",
        "VoV V-99-eSE",
        "backbones/vov.py",
        "8c43d93743ba977c987c2abdc4bb0ab76394371e4e7695febc94b547b7bd2b2f",
    ):
        assert required in text


def test_benchmark_defaults_to_resnet_pose_16384_release() -> None:
    script = SAMPLE / "scripts/benchmark_model.py"
    text = script.read_text()

    assert "ResNet reward NAVHARD" in text
    assert "4096 image key/value tokens" in text
    assert "verified 16,384-candidate trajectory vocabulary" in text
    assert "FP32" in text

    namespace = runpy.run_path(str(script))
    assert namespace["DEFAULT_CHECKPOINT"] == SAMPLE / "assets/gtrs_dense" / ASSET_NAME
    assert namespace["DEFAULT_VOCABULARY"] == (
        SAMPLE / "assets/gtrs_dense/navsim_16384.npy"
    )
