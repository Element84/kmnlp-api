ARG PYTHON_VERSION=3.12
ARG BUILD_ENV_IS_CI=1

###############
# Base image #
###############

FROM python:${PYTHON_VERSION}-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.28 /uv /uvx /bin/
ARG BUILD_ENV_IS_CI

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /code

# Install git and ssh
RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-client shellcheck libatomic1 ffmpeg && \
    apt-get clean

# Create venv and use it
RUN uv venv
ENV PATH="/code/.venv/bin:$PATH"

# Add requirements file so we can install our dependencies
COPY pyproject.toml .
COPY uv.lock .

# Add GitLab's SSH host key only if running locally.
# CI will use HTTPS to fetch from GitLab.
RUN \
    if [ "${BUILD_ENV_IS_CI:-0}" != "1" ]; then \
      mkdir -p -m 0700 ~/.ssh && ssh-keyscan repo.element84.com > ~/.ssh/known_hosts; \
    fi

# Update git config to fetch llm-agent private repo over HTTPS using basic auth.
# the HTTPS basic auth password will be the CI_JOB_TOKEN provided in secrets.
# Run ONLY if build is not local (i.e. is CI)
RUN --mount=type=secret,id=CI_JOB_TOKEN \
    if [ "${BUILD_ENV_IS_CI:-0}" -eq 1 ]; then \
      git config --global \
        url."https://gitlab-ci-token:$(cat /run/secrets/CI_JOB_TOKEN)@repo.element84.com/".insteadOf \
        ssh://git@repo.element84.com/; \
    fi

# Ensure lock file is up to date
RUN uv lock --check
RUN --mount=type=ssh uv sync --all-extras
RUN natural-language-geocoding init
# RUN git config --global --unset-all url."https://gitlab-ci-token:*@repo.element84.com/".insteadOf

#####################
# Backend API image
#####################

FROM base AS api

COPY . /code

WORKDIR /code

# Build app
RUN python -m build

ENV PATH="/code/.venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
# --proxy-headers + --forwarded-allow-ips "*" trust X-Forwarded-Proto/-For from the
# TLS-terminating load balancer so request.url.scheme reflects the real https scheme.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

##############
# Dask image #
##############

FROM base AS dask

WORKDIR /code

CMD ["bash"]
