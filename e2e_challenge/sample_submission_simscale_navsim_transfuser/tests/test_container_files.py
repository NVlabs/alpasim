# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[1]
REPO_ROOT = SAMPLE.parents[1]


def test_requirements_pin_the_inference_runtime_only() -> None:
    requirements = (SAMPLE / "requirements.txt").read_text().splitlines()

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
    lowered = "\n".join(requirements).lower()
    for forbidden in ("navsim", "nuplan", "lightning", "hydra"):
        assert forbidden not in lowered


def test_dockerfile_uses_pinned_cuda_base_and_embeds_checkpoint() -> None:
    text = (SAMPLE / "Dockerfile").read_text()

    assert text.startswith(
        "ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime\n"
        "FROM ${BASE_IMAGE}"
    )
    for required in (
        "HOME=/tmp/home",
        "TMPDIR=/tmp",
        "XDG_CACHE_HOME=/tmp/.cache",
        "TORCH_HOME=/tmp/torch",
        "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions",
        "HF_HOME=/tmp/huggingface",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "ALPASIM_DRIVER_LOG_DIR=/run/alpasim-driver",
        "TORCH_NUM_THREADS=1",
        "TORCH_NUM_INTEROP_THREADS=1",
        "COPY src/grpc /tmp/alpasim_grpc",
        "pip install /tmp/alpasim_grpc",
        "pip check",
        "alpasim_grpc.__version__ == (0, 54, 0)",
        "torch.__version__.startswith('2.6.0')",
        "torchvision.__version__.startswith('0.21.0')",
        "ARG LTF_ASSET_FILE=ltf_sim_navtest.ckpt",
        "COPY e2e_challenge/sample_submission_simscale_navsim_transfuser/assets/ltf/"
        "${LTF_ASSET_FILE} /app/assets/ltf/ltf_sim_navtest.ckpt",
        "LTF_CHECKPOINT_PATH=/app/assets/ltf/ltf_sim_navtest.ckpt",
        "groupadd --gid 10001 alpasim",
        "useradd --uid 10001 --gid 10001",
        "chmod -R a+rX /app",
        "EXPOSE 6789",
        "STOPSIGNAL SIGTERM",
        "USER 10001:10001",
        'CMD ["python", "-m", "navsim_transfuser_challenge.driver"]',
    ):
        assert required in text
    assert text.index("requirements.txt") < text.index("COPY src/grpc")
    assert text.index(
        "COPY e2e_challenge/sample_submission_simscale_navsim_transfuser/navsim"
    ) < (text.index("USER 10001:10001"))
    for forbidden in (
        "LTF_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "git clone",
        "curl ",
        "wget ",
        "huggingface-cli",
        "RUNTIME_STAGE",
        "INSTALL_DEPENDENCIES",
        "prebuilt",
    ):
        assert forbidden not in text


def test_dockerignore_is_a_strict_allowlist() -> None:
    lines = (SAMPLE / "Dockerfile.dockerignore").read_text().splitlines()

    assert lines == [
        "**",
        "!src/",
        "!src/grpc/",
        "!src/grpc/**",
        "!e2e_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/Dockerfile",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/requirements.txt",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/assets/",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/assets/ltf/",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/"
        "assets/ltf/ltf_sim_navtest.ckpt",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/"
        "assets/ltf/.ltf-build-*",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/"
        "navsim_transfuser_challenge/",
        "!e2e_challenge/sample_submission_simscale_navsim_transfuser/"
        "navsim_transfuser_challenge/**",
    ]
    assert not any(
        forbidden in line
        for line in lines
        for forbidden in ("reference", "/tests", "/scripts", "README")
    )


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
        assert "git-lfs.github.com/spec/v1" in text
        assert "EXPECTED_SIZE" in text
        assert "EXPECTED_SHA256" in text
        assert "sha256sum" in text

    launcher = paths[2].read_text()
    for required in (
        "--rm",
        "--init",
        "--name",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "--pids-limit 1024",
        "--memory 32g",
        "--cpus 8",
        "--tmpfs /tmp:",
        "--tmpfs /run:",
        "127.0.0.1:",
        "ALPASIM_CONTESTANT_REPLICA_INDEX",
        "ALPASIM_CONTESTANT_REPLICAS",
    ):
        assert required in launcher
    for forbidden in (
        "LTF_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
        "--volume",
        "/var/run/docker.sock",
    ):
        assert forbidden not in launcher


def test_launcher_signal_handlers_cleanup_and_exit() -> None:
    launcher = (SAMPLE / "run_local_container.sh").read_text()
    handler = launcher.split("handle_signal() {", 1)[1].split("\n}", 1)[0]

    assert "cleanup" in handler
    assert "trap - EXIT" in handler
    assert 'exit "${status}"' in handler
    assert "trap 'handle_signal 130' INT" in launcher
    assert "trap 'handle_signal 143' TERM" in launcher


def test_prepare_assets_supports_verified_override(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"test")
    sha = hashlib.sha256(b"test").hexdigest()
    destination = tmp_path / "assets"
    env = os.environ | {
        "LTF_ASSET_DIR": str(destination),
        "LTF_EXPECTED_SIZE": "4",
        "LTF_EXPECTED_SHA256": sha,
    }

    subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        check=True,
        env=env,
    )

    assert (destination / "ltf_sim_navtest.ckpt").read_bytes() == b"test"


def test_prepare_assets_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    source = tmp_path / "pointer.ckpt"
    source.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 224560669\n"
    )

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Git LFS pointer" in result.stderr


@pytest.mark.parametrize(
    ("source_exists", "expected_size", "expected_sha", "message"),
    [
        (False, "4", hashlib.sha256(b"test").hexdigest(), "missing checkpoint"),
        (True, "5", hashlib.sha256(b"test").hexdigest(), "checkpoint size"),
        (True, "4", "0" * 64, "checkpoint sha256"),
    ],
)
def test_prepare_assets_rejects_missing_size_and_hash_errors(
    tmp_path: Path,
    source_exists: bool,
    expected_size: str,
    expected_sha: str,
    message: str,
) -> None:
    source = tmp_path / "model.ckpt"
    if source_exists:
        source.write_bytes(b"test")
    env = os.environ | {
        "LTF_ASSET_DIR": str(tmp_path / "assets"),
        "LTF_EXPECTED_SIZE": expected_size,
        "LTF_EXPECTED_SHA256": expected_sha,
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_prepare_assets_failure_preserves_checkpoint_and_cleans_temp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"invalid")
    destination = tmp_path / "assets"
    destination.mkdir()
    checkpoint = destination / "ltf_sim_navtest.ckpt"
    checkpoint.write_bytes(b"existing")
    env = os.environ | {
        "LTF_ASSET_DIR": str(destination),
        "LTF_EXPECTED_SIZE": str(len(b"invalid")),
        "LTF_EXPECTED_SHA256": "0" * 64,
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "scripts/prepare_assets.sh"), str(source)],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert "checkpoint sha256" in result.stderr
    assert checkpoint.read_bytes() == b"existing"
    assert list(destination.glob(".ltf-prepare-*")) == []


def test_build_wrapper_verifies_asset_and_invokes_docker_from_repo_root(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "model.ckpt"
    asset.write_bytes(b"test")
    sha = hashlib.sha256(b"test").hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_file = tmp_path / "docker-args.txt"
    snapshot_copy = tmp_path / "snapshot.ckpt"
    snapshot_mode = tmp_path / "snapshot-mode.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$@" > "${DOCKER_ARGS_FILE}"\n'
        "snapshot_name=''\n"
        "while (($#)); do\n"
        "  if [[ \"$1\" == '--build-arg' ]]; then\n"
        "    shift\n"
        '    snapshot_name="${1#LTF_ASSET_FILE=}"\n'
        "  fi\n"
        "  shift\n"
        "done\n"
        'snapshot="${SAMPLE_ASSET_DIR}/${snapshot_name}"\n'
        'cp -- "${snapshot}" "${DOCKER_SNAPSHOT_COPY}"\n'
        'stat -c \'%A\' "${snapshot}" > "${DOCKER_SNAPSHOT_MODE}"\n'
    )
    docker.chmod(0o755)
    default_asset = SAMPLE / "assets/ltf/ltf_sim_navtest.ckpt"
    default_before = default_asset.stat() if default_asset.exists() else None
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_ARGS_FILE": str(args_file),
        "DOCKER_SNAPSHOT_COPY": str(snapshot_copy),
        "DOCKER_SNAPSHOT_MODE": str(snapshot_mode),
        "SAMPLE_ASSET_DIR": str(SAMPLE / "assets/ltf"),
        "IMAGE": "test-ltf:image",
        "LTF_ASSET_PATH": str(asset),
        "LTF_EXPECTED_SIZE": "4",
        "LTF_EXPECTED_SHA256": sha,
    }

    subprocess.run(
        ["bash", str(SAMPLE / "scripts/build_image.sh")],
        check=True,
        env=env,
        cwd=tmp_path,
    )

    args = args_file.read_text().splitlines()
    assert args[:6] == [
        "build",
        "--tag",
        "test-ltf:image",
        "--file",
        str(SAMPLE / "Dockerfile"),
        "--build-arg",
    ]
    assert args[6].startswith("LTF_ASSET_FILE=.ltf-build-")
    assert args[7:] == [
        str(REPO_ROOT),
    ]
    assert snapshot_copy.read_bytes() == b"test"
    assert "w" not in snapshot_mode.read_text()
    assert list((SAMPLE / "assets/ltf").glob(".ltf-build-*")) == []
    default_after = default_asset.stat() if default_asset.exists() else None
    assert default_after == default_before


def test_build_wrapper_has_no_local_prebuilt_runtime_mode() -> None:
    text = (SAMPLE / "scripts/build_image.sh").read_text()

    for forbidden in (
        "LTF_BASE_IMAGE",
        "LTF_INSTALL_DEPENDENCIES",
        "RUNTIME_STAGE",
        "--target prebuilt",
        "docker image inspect",
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    "overrides",
    [
        {"ALPASIM_DRIVER_REPLICAS": "abc"},
        {"ALPASIM_DRIVER_REPLICAS": "0"},
        {"ALPASIM_DRIVER_REPLICAS": "1025"},
        {"ALPASIM_DRIVER_BASE_PORT": "0"},
        {"ALPASIM_DRIVER_CONTAINER_PORT": "65536"},
        {
            "ALPASIM_DRIVER_BASE_PORT": "65535",
            "ALPASIM_DRIVER_REPLICAS": "2",
        },
        {"ALPASIM_DRIVER_DETACH": "2"},
    ],
)
def test_launcher_rejects_invalid_numeric_inputs_before_docker(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "docker-called"
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\n" 'touch "${DOCKER_CALLED}"\n' "exit 99\n")
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLED": str(marker),
        "ALPASIM_DRIVER_REPLICAS": "1",
        "ALPASIM_DRIVER_BASE_PORT": "6789",
        "ALPASIM_DRIVER_CONTAINER_PORT": "6789",
        "ALPASIM_DRIVER_DETACH": "0",
        "ALPASIM_DOCKER_GPUS": "none",
    }
    env.update(overrides)

    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("prefix", ["-leading-dash", "contains space", "bad/name"])
def test_launcher_rejects_invalid_name_prefix_before_docker(
    tmp_path: Path,
    prefix: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "docker-called"
    docker = fake_bin / "docker"
    docker.write_text("#!/usr/bin/env bash\n" 'touch "${DOCKER_CALLED}"\n' "exit 99\n")
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DOCKER_CALLED": str(marker),
        "ALPASIM_DRIVER_NAME_PREFIX": prefix,
        "ALPASIM_DRIVER_DETACH": "1",
        "ALPASIM_DOCKER_GPUS": "none",
    }

    result = subprocess.run(
        ["bash", str(SAMPLE / "run_local_container.sh")],
        text=True,
        capture_output=True,
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "ALPASIM_DRIVER_NAME_PREFIX" in result.stderr
    assert not marker.exists()


def test_checkpoint_is_ignored_by_submission_asset_rules() -> None:
    lines = (SAMPLE / "assets/ltf/.gitignore").read_text().splitlines()

    assert "*" in lines
    assert "!.gitignore" in lines


def test_asset_docs_link_official_checkpoint_and_prepare_it() -> None:
    texts = (
        (SAMPLE / "README.md").read_text(),
        (SAMPLE / "assets/README.md").read_text(),
    )

    for text in texts:
        for required in (
            "OpenDriveLab/SimScale",
            "SimScale_ckpts/LTF/ltf_sim_navtest.ckpt",
            "hf download",
            "--repo-type dataset",
            "scripts/prepare_assets.sh",
        ):
            assert required in text


def test_readme_documents_supported_workflow_and_acceptance() -> None:
    text = (SAMPLE / "README.md").read_text()

    for required in (
        "NuPlan/MTGS",
        "CAM_L0",
        "CAM_F0",
        "CAM_R0",
        "224,560,669",
        "9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69",
        "prepare_assets.sh",
        "build_image.sh",
        "run_local_container.sh",
        "ALPASIM_DRIVER_NAME_PREFIX",
        "read-only root filesystem",
        "Build",
        "Local Smoke And Probe",
        "NuPlan/MTGS Smoke",
        "ltf_inference > 0",
        "all-fallback run is not a",
        "same-timestamp RPC `DynamicState`",
        "Pose history is used only when",
        "dynamic_state_fallback",
        "4d352ba",
        "/path/to/alpasim-nuplan-track",
    ):
        assert required in text
    assert "/high_perf_store4/" not in text
    for forbidden in (
        "LTF_" + "DYNAMICS_" + "SOURCE",
        "DYNAMICS_" + "SOURCE",
    ):
        assert forbidden not in text
