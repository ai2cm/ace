FROM python:3.12-slim

# Install git and curl since they are required for running with gantry.
RUN apt-get update && apt-get install -y --no-install-recommends git curl

# Install the Hugging Face CLI and ensure it is in the search path.
RUN curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Set environment variables for performant uploads based on recommendation from
# Paul Laskowski. HF_XET_HIGH_PERFORMANCE=1 enables optimal parallelization of
# uploads; HF_XET_SHARD_CACHE_SIZE_LIMIT=40GiB is larger than the default 16GB,
# which also facilitates performance. See documentation here for more details:
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_SHARD_CACHE_SIZE_LIMIT=40GiB

ENTRYPOINT ["/bin/bash"]
