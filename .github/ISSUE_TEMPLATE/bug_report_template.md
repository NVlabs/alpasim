---
name: Bug report
about: Create a bug report to help us improve Alpamayo
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'yesfandiari'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Deployment: [Docker Compose (single machine) or Slurm / multi-node]
 - Renderer backend: [NuRec (default) or OmniDreams via FlashDreams]
 - Driving policy under test: [Alpamayo-R1, Alpamayo 1.5, VaVAM, Transfuser/LTFv6, or custom]
 - Scene / dataset used (see `data/scenes/`)
 - Affected gRPC service(s), and image/tag if applicable

**Environment details**
 - Hardware: GPU type(s), VRAM, number of GPUs / nodes
 - Operating System
 - CUDA / NVIDIA driver version (from `nvidia-smi`)
 - Docker / Docker Compose versions (or Slurm details for cluster runs)

**Additional context**
Add any other context about the problem here.
