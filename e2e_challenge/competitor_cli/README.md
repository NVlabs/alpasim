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
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
```


## Inspect

```bash
# Check competition info, limits, leaderboard, submissions, and submission status with:
uv run e2e_challenge/competitor_cli/alpasim_challenge.py limits
uv run e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submissions
uv run e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
```
