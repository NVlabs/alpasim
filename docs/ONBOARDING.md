# Onboarding
Alpasim depends on access to the following:

* Hugging Face access
    * Used for downloading simulation artifacts
    * Data is [here](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release )
    * See info on data [here](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/blob/main/README.md#dataset-format)
    for more information on the contents of artifacts used to define scenes
    * You will need to create a free Hugging Face account if you do not already have one and create
    an access token with read access. See [access tokens](https://huggingface.co/settings/tokens).
    * Once you have the token, set it as an environment variable: `export HF_TOKEN=<token>`
* A version of `uv` installed (see [here](https://docs.astral.sh/uv/getting-started/installation/))
    * Example installation command for Ubuntu: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
* Docker installed (see [setup instructions](https://docs.docker.com/engine/install/ubuntu/))
* Docker compose installed (see [setup instructions](https://docs.docker.com/compose/install/linux/))
    * The wizard needs `docker`, `docker-compose-plugin`, and `docker-buildx-plugin`
    * Docker needs to be able to run without `sudo`. If you see a permission error when running
    `docker` commands, add yourself to the docker group: `sudo usermod -aG docker $USER`
* CUDA 12.6 or greater installed (see [here](https://developer.nvidia.com/cuda-downloads) for instructions)
* Install the NVIDIA Container Toolkit (see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

Once you have access to the above, please follow instructions in the [tutorial](/docs/TUTORIAL.md) to
get started running Alpasim.
