#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Small competitor CLI for the AlpaSim E2E challenge API."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_API_BASE = "https://rgf863fslk.execute-api.us-east-1.amazonaws.com"
CONFIG_DIR = Path(os.environ.get("ALPASIM_CONFIG_DIR", "~/.alpasim")).expanduser()
CONFIG_PATH = CONFIG_DIR / "challenge.json"
MAX_CONTROLLER_GAINS_FILE_BYTES = 8 * 1024
CONTROLLER_GAIN_BOUNDS: dict[str, tuple[float, float]] = {
    "long_position_weight": (0.0, 10.0),
    "lat_position_weight": (0.0, 10.0),
    "heading_weight": (0.0, 10.0),
    "acceleration_weight": (0.0, 10.0),
    "rel_front_steering_angle_weight": (0.0, 10.0),
    "rel_acceleration_weight": (0.0, 10.0),
}
IDX_START_PENALTY_BOUNDS = (0, 19)


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(prog="alpasim-challenge")
    parser.add_argument(
        "--token",
        default=os.environ.get("ALPASIM_TOKEN"),
        help="Bearer token. Defaults to ALPASIM_TOKEN or saved config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("auth-url", help="Print the browser login URL.")

    configure = subparsers.add_parser("configure-token", help="Save a CLI token.")
    configure.add_argument("token", nargs="?", help="Token from POST /cli/token.")

    subparsers.add_parser("me", help="Show authenticated user and registration.")
    subparsers.add_parser("ecr-login", help="Run docker login for your team ECR repo.")
    terms = subparsers.add_parser(
        "terms", help="View, accept, or check competition terms."
    )
    terms_subparsers = terms.add_subparsers(dest="terms_command", required=True)
    terms_subparsers.add_parser("show", help="Print the current private terms.")
    terms_subparsers.add_parser("status", help="Show your team terms status.")
    terms_subparsers.add_parser(
        "accept", help="Review and accept the current terms as the signed-in user."
    )
    subparsers.add_parser(
        "limits", help="Show competition status and submission quota."
    )
    leaderboard = subparsers.add_parser(
        "leaderboard", help="Show the public leaderboard."
    )
    leaderboard.add_argument(
        "--track", choices=("pai", "nuplan"), help="Leaderboard track."
    )

    submit = subparsers.add_parser("submit", help="Submit an already-pushed image URI.")
    submit.add_argument("image_uri")
    submit.add_argument("--track", required=True, choices=("pai", "nuplan"))
    submit.add_argument(
        "--controller-gains",
        type=Path,
        help="Optional JSON file containing bounded nonlinear-controller gains.",
    )
    submit.add_argument("--metadata-source", default="cli", help=argparse.SUPPRESS)

    submissions = subparsers.add_parser(
        "submissions", help="List your team's recent submissions."
    )
    submissions.add_argument(
        "--track", choices=("pai", "nuplan"), help="Filter by track."
    )

    status = subparsers.add_parser("status", help="Get one submission status.")
    status.add_argument("submission_id")

    args = parser.parse_args()
    client = ChallengeClient(token=args.token or config.get("token"))

    if args.command == "auth-url":
        print(f"{DEFAULT_API_BASE}/auth/huggingface/login?cli=1")
        return 0
    if args.command == "configure-token":
        token = args.token or getpass.getpass("CLI token: ")
        save_config({"token": token})
        print(f"Saved token to {CONFIG_PATH}")
        return 0
    if args.command == "me":
        print_json(client.get("/me"))
        return 0
    if args.command == "ecr-login":
        require_terms_ready(client)
        ecr_login(client)
        return 0
    if args.command == "terms":
        if args.terms_command == "show":
            show_terms(client)
            return 0
        if args.terms_command == "status":
            print_json(client.get("/terms/status"))
            return 0
        if args.terms_command == "accept":
            accept_terms(client)
            return 0
    if args.command == "limits":
        print_limits(client)
        return 0
    if args.command == "leaderboard":
        print_json(client.get(path_with_query("/leaderboard", track=args.track)))
        return 0
    if args.command == "submit":
        # Check before the Docker manifest probe. A terms rejection is entirely
        # local/API-side and never creates a submission or consumes quota.
        require_terms_ready(client)
        ensure_image_exists(args.image_uri)
        payload = {
            "image_uri": args.image_uri,
            "track": args.track,
            "metadata": {"source": args.metadata_source},
        }
        if args.controller_gains is not None:
            payload["controller_gains"] = load_controller_gains(args.controller_gains)
        print_json(
            client.post(
                "/submissions",
                payload,
            )
        )
        return 0
    if args.command == "submissions":
        print_json(client.get(path_with_query("/submissions", track=args.track)))
        return 0
    if args.command == "status":
        print_json(client.get(f"/submissions/{args.submission_id}"))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


class ChallengeClient:
    def __init__(self, *, token: str | None):
        self.api_base = DEFAULT_API_BASE
        self.token = token

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(
            f"{self.api_base}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                error = json.loads(body)
            except json.JSONDecodeError:
                error = {"error": body or exc.reason}
            raise SystemExit(
                f"API request failed: HTTP {exc.code} {json.dumps(error)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"API request failed: {exc}") from exc


def ecr_login(client: ChallengeClient) -> None:
    auth = client.post("/ecr/login")
    registry = auth["registry"]
    username = auth["username"]
    password = auth["password"]

    process = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password.encode("utf-8"),
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(process.returncode)

    print(f"Logged in to {registry}")
    print(f"Team repository: {auth['image_uri_prefix']}")
    print()
    print("Next steps:")
    print("  docker tag <local-image>:<local-tag> \\")
    print(f"    {auth['image_uri_prefix']}:<version>")
    print()
    print(f"  docker push {auth['image_uri_prefix']}:<version>")
    print()
    print(
        "  uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit --track <pai|nuplan> \\"
    )
    print(f"    {auth['image_uri_prefix']}:<version>")
    print()
    print("Submissions are limited. Only submit images you want evaluated.")


def show_terms(client: ChallengeClient) -> dict[str, Any]:
    terms = client.get("/terms/current")
    print(f"Competition terms version: {terms['terms_version']}")
    print(f"SHA-256: {terms['terms_sha256']}")
    print()
    print(terms["terms_text"], end="" if terms["terms_text"].endswith("\n") else "\n")
    print()
    print_terms_status(terms)
    return terms


def accept_terms(client: ChallengeClient) -> None:
    terms = show_terms(client)
    if terms.get("actor_accepted"):
        print("You have already accepted this exact terms version.")
        if not terms.get("captain_accepted"):
            print("Your team captain must still accept it using their own account.")
        return

    expected = f"ACCEPT {terms['terms_version']}"
    confirmation = input(f"Type {expected} to accept: ").strip()
    if confirmation != expected:
        raise SystemExit("Terms were not accepted; no server-side change was made.")

    status = client.post(
        "/terms/accept",
        {
            "accepted": True,
            "terms_version": terms["terms_version"],
            "terms_sha256": terms["terms_sha256"],
        },
    )
    print("Acceptance recorded.")
    print_terms_status(status)


def require_terms_ready(client: ChallengeClient) -> None:
    status = client.get("/terms/status")
    if status.get("team_ready"):
        return
    version = status.get("terms_version") or "current"
    if not status.get("actor_accepted") and not status.get("captain_accepted"):
        reason = "Both you and your team captain must accept"
    elif not status.get("actor_accepted"):
        reason = "You must accept"
    else:
        reason = "Your team captain must accept"
    raise SystemExit(
        f"{reason} competition terms version {version} before uploading or "
        "submitting. Run `alpasim_challenge.py terms show` and "
        "`alpasim_challenge.py terms accept`."
    )


def print_terms_status(status: dict[str, Any]) -> None:
    print(f"Your acceptance is current: {bool(status.get('actor_accepted'))}")
    print(f"Captain acceptance is current: {bool(status.get('captain_accepted'))}")
    print(f"Team is eligible to upload and submit: {bool(status.get('team_ready'))}")


def ensure_image_exists(image_uri: str) -> None:
    """Confirm Docker can resolve an image manifest before submitting it."""
    if image_uri.rsplit("@", 1)[0].endswith(":latest"):
        raise SystemExit(
            "The latest tag is not allowed for submissions. "
            "Use a specific image tag or digest instead."
        )

    try:
        process = subprocess.run(
            ["docker", "manifest", "inspect", image_uri],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Docker is required to validate the submitted image. "
            "Install Docker, then run `ecr-login` and retry."
        ) from exc

    if process.returncode == 0:
        return

    detail = process.stderr.strip() or process.stdout.strip()
    message = (
        f"Could not find or access Docker image {image_uri!r}. "
        "Check the image URI and tag, ensure the image has been pushed, "
        "then run `ecr-login` and retry."
    )
    if detail:
        message = f"{message}\nDocker reported: {detail}"
    raise SystemExit(message)


def print_limits(client: ChallengeClient) -> None:
    me = client.get("/me")
    submissions = client.get("/submissions?limit=1")
    competition = me.get("competition") or {}
    registration = me.get("registration") or {}

    print(
        f"Competition: {competition.get('name') or competition.get('competition_id')}"
    )
    print(f"Status: {competition.get('status')}")
    print(f"Team: {registration.get('team_id') or submissions.get('team_id')}")
    print(f"Monthly submission limit: {submissions.get('monthly_submission_limit')}")
    print(f"Submitted this month: {submissions.get('current_month_submission_count')}")
    print(f"Remaining this month: {submissions.get('remaining_monthly_submissions')}")


def load_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(config, dict):
        return {}
    token = config.get("token")
    return {"token": str(token)} if token else {}


def path_with_query(path: str, **params: str | None) -> str:
    filtered = {key: value for key, value in params.items() if value}
    if not filtered:
        return path
    return f"{path}?{urllib.parse.urlencode(filtered)}"


def load_controller_gains(path: Path) -> dict[str, float | int]:
    """Read and validate the optional public nonlinear-MPC gain-set file."""
    try:
        if path.stat().st_size > MAX_CONTROLLER_GAINS_FILE_BYTES:
            raise ValueError(
                f"controller gains file exceeds {MAX_CONTROLLER_GAINS_FILE_BYTES} bytes"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read controller gains file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"controller gains file {path} is not valid JSON: {exc}"
        ) from exc
    return validate_controller_gains(value)


def validate_controller_gains(value: Any) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise SystemExit("controller gains must be a JSON object")
    if not value:
        raise SystemExit(
            "controller gains must not be empty; omit the option to use defaults"
        )

    allowed = set(CONTROLLER_GAIN_BOUNDS) | {"idx_start_penalty"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SystemExit(f"unsupported controller gain(s): {', '.join(unknown)}")

    normalized: dict[str, float | int] = {}
    for name, raw_value in value.items():
        if isinstance(raw_value, bool):
            raise SystemExit(f"controller gain {name} must be numeric, not boolean")
        if name == "idx_start_penalty":
            if not isinstance(raw_value, int):
                raise SystemExit("controller gain idx_start_penalty must be an integer")
            lower, upper = IDX_START_PENALTY_BOUNDS
            if not lower <= raw_value <= upper:
                raise SystemExit(
                    f"controller gain idx_start_penalty must be in [{lower}, {upper}]"
                )
            normalized[name] = raw_value
            continue

        if not isinstance(raw_value, (int, float)):
            raise SystemExit(f"controller gain {name} must be numeric")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise SystemExit(f"controller gain {name} must be finite")
        lower, upper = CONTROLLER_GAIN_BOUNDS[name]
        if not lower <= numeric_value <= upper:
            raise SystemExit(f"controller gain {name} must be in [{lower}, {upper}]")
        normalized[name] = numeric_value
    return normalized


def save_config(config: dict[str, str]) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)
        CONFIG_PATH.chmod(0o600)
    except OSError as exc:
        raise SystemExit(f"Could not write CLI config {CONFIG_PATH}: {exc}") from exc


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
