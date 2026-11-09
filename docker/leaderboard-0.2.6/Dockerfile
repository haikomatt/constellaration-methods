# Pins to the leaderboard's frozen benchmark version (2026-10-08, constellaration-bench discussion 7;
# the Space's requirements.txt: "constellaration==0.2.6  # Frozen benchmark version"). Same recipe as
# docker/leaderboard-fidelity (0.2.1), version changed only. Built 2026-11-06 to re-check the
# own-lineage P3 designs and fronts under the version the leaderboard actually scores with.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install constellaration==0.2.6 pytest

WORKDIR /workspace
