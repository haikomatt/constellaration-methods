FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned to the exact versions every result under docs/results/ was produced
# with. constellaration==0.2.6 transitively pins vmecpp==0.4.11 and
# scikit-learn==1.5.2; the explicit scikit-learn pin is defensive so a future
# constellaration bump cannot silently drag it along (see pyproject.toml).
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

COPY . .

CMD ["pytest", "tests/", "-q"]
