# Stan probe image: cmdstanpy + stanc3.
# Tag: panproto-test-stan
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cmdstanpy==1.2.5 numpy==1.26.4

# A failed download or compiler build must fail the image build.
# The installer validates CmdStan by compiling its Bernoulli example.
RUN python -c "import sys; import cmdstanpy; sys.exit(0 if cmdstanpy.install_cmdstan(version='2.40.0', progress=False) else 1)"

ENTRYPOINT ["python"]
