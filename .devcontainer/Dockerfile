# Debian-based container with CMake, Python, and pipx support
FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-venv ninja-build git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install pipx && \
    pipx ensurepath

ENV PATH=/root/.local/bin:${PATH}
