# Challenge CLI

A Python CLI is provided for the E2E challenge--use this CLI to authenticate, upload images, submit evaluations, and inspect
results. See the [challenge README](../README.md) for the contestant image
contract.

> Note: run the following commands from the repo root.

## Register

Before using the CLI, ensure you are registered. Open the
[Hugging Face Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026),
connect, fill in the form, and wait for approval.

Once approved, additional submitters (image push and submission permissions) whose user ids were
added to the submission form can connect at the Hugging Face Space and join the team.

## Authenticate

The following commands describe how to authenticate yourself. CLI tokens expire
after 12 hours, so you may need to re-authenticate periodically.

```bash
# to get the URL for authentication, go to the URL provided by:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py auth-url
# and copy the token

# run the cli with configure-token and paste the token when prompted
uv run e2e_challenge/competitor_cli/alpasim_challenge.py configure-token

# confirm authentication works by checking authentication status
uv run e2e_challenge/competitor_cli/alpasim_challenge.py me
```

## View/Accept the Competition Terms

The terms and conditions can be shown/provided after CLI authentication.
Review them and record your individual acceptance with:

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py terms show
uv run e2e_challenge/competitor_cli/alpasim_challenge.py terms accept
uv run e2e_challenge/competitor_cli/alpasim_challenge.py terms status
```

`terms accept` prints the exact version and SHA-256 digest, then requires you to
type `ACCEPT <version>`. Every user who uploads or requests an evaluation must
accept using their own Hugging Face-authenticated account. In addition, the
registered team captain must accept before any member of that team can upload
or submit. If the published version or digest changes, everyone must review and
accept the replacement.

## Submit Images and Request Evaluation

```bash
# Log in to ECR to push images:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
# Tag your local image with the provided ECR repository URI and a specific tag (not "latest")
# Note: Submitted images must be under 40 GiB. Use specific tags; `latest` is rejected.
docker tag <local-image>:<tag> \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>

# Then push the image to ECR:
docker push \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>

# ...and submit the image URI for evaluation:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit --track pai \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
```

Before sending the submission request, the CLI verifies that Docker can resolve
the exact image manifest and rejects the `latest` tag. If this check fails,
confirm the image URI and tag, push the image, and rerun `ecr-login` before
trying again.

`submit` requires an explicit track. Use `--track pai` for the Physical AI AV
track or `--track nuplan` for the nuPlan track.

### Optional Nonlinear-Controller Gains

The official evaluator uses a trusted nonlinear MPC controller. To tune its
existing gain set, pass a small JSON object with your submission:

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit --track pai \
  --controller-gains e2e_challenge/starter_kit/controller_gains.example.json \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
```

The optional file may contain any subset of these fields:

- `long_position_weight`, `lat_position_weight`, `heading_weight`: 0 to 10
- `acceleration_weight`: 0 to 10
- `rel_front_steering_angle_weight`, `rel_acceleration_weight`: 0 to 10
- `idx_start_penalty`: integer 0 to 19

Only these gains are accepted. The CLI validates the file before submission and
the evaluator validates it again; all other controller and simulator settings
remain fixed.

## Inspect

```bash
# Check competition info, limits, leaderboard, submissions, and submission status with:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py limits
uv run e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard --track pai
uv run e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard --track nuplan
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submissions --track pai
uv run e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
```
