# Pyro probe image: pyro-ppl + torch.
# Tag: panproto-test-pyro
#
# The CPU build of torch: a probe has no accelerator, and the default
# Linux build carries gigabytes of libraries for one.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

RUN pip install --no-cache-dir \
    pyro-ppl==1.9.1 \
    torch==2.5.1 \
    numpy==1.26.4

ENTRYPOINT ["python"]
