# The grader of an exported STM-Bench episode: `python -m stmbench.cli grade`, nothing else.
#
# Build (context = the repository root, so that the source and the pinned requirements are in it):
#   docker build -f deploy/osworld/grader/Dockerfile -t stmbench-grader:<tag> .
# Run (what deploy/osworld/stm_metric.py does; the export is read-only and there is no network):
#   docker run --rm --network none -v <export dir>:/export:ro stmbench-grader:<tag> grade /export --scenario <id> --seed <n>
#
# Build the image from the same commit the instrument in the virtual machine runs: a replay is
# only the recorded episode when the code and the numerics are the same (deploy/osworld/README.md).
# To fix the base image's libraries as well, build with
#   --build-arg PYTHON_IMAGE=python:3.13-slim@sha256:<digest>
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# The exact versions the instrument's environment uses too. Installed before the source, so that
# a code change does not reinstall them.
COPY deploy/osworld/grader/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --no-deps: the pins above are the dependencies; pip must not resolve newer ones for the package.
# What was copied for the build goes afterwards (the directory itself stays, it is the working
# directory of this layer), so that nothing but the installed copy of the package remains.
WORKDIR /build
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY stmsim ./stmsim
COPY stmbench ./stmbench
RUN pip install --no-deps . && rm -rf /build/*

RUN useradd --create-home --uid 10001 grader
USER grader
WORKDIR /home/grader
ENTRYPOINT ["python", "-m", "stmbench.cli"]
