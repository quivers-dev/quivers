"""The image build propagates CmdStan's installation outcome to Docker."""

from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize("installed", (False, True))
def test_stan_image_build_requires_successful_installation(installed: bool) -> None:
    """Execute the Dockerfile command against both installer return values.

    Parameters
    ----------
    installed
        Whether the CmdStan installer reports success.
    """
    dockerfile = Path("tests/transpile/docker/stan/Dockerfile").read_text()
    command = next(
        line.removeprefix("RUN ")
        for line in dockerfile.splitlines()
        if line.startswith("RUN python -c ") and "install_cmdstan" in line
    )
    executable, flag, code = shlex.split(command)
    assert (executable, flag) == ("python", "-c")
    # The installer reports failure as a Boolean rather than raising it.
    # Run in a child process because the build command controls its exit status.
    bootstrap = (
        "import sys, types\n"
        "module = types.ModuleType('cmdstanpy')\n"
        f"module.install_cmdstan = lambda **kwargs: {installed!r}\n"
        "sys.modules['cmdstanpy'] = module\n"
        f"exec({code!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap], capture_output=True, text=True, check=False
    )
    assert result.returncode == (0 if installed else 1), result.stderr
