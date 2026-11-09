"""Workaround: on fresh cluster nodes hyperlux's JIT configure (cmake under subprocess PIPE) can stall
indefinitely; streaming the build output instead of capturing it avoids that. Import before hyperlux/hyperdrone."""
import subprocess
import sys

import hyperlux._jit._build as _build


def _run(arguments, env):
    subprocess.run([str(a) for a in arguments], env=env, check=True, stdout=sys.stderr, stderr=sys.stderr)


_build.run = _run
