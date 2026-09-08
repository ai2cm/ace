FROM python:3.12-slim

# Install git and curl since they are required for running with gantry.
RUN apt-get update && apt-get install -y --no-install-recommends git curl

# Install the Hugging Face CLI and ensure it is in the search path.
RUN curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Set environment variables for performant uploads per Paul Laskowski.
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_SHARD_CACHE_SIZE_LIMIT=40GiB

ENTRYPOINT ["/bin/bash"]
