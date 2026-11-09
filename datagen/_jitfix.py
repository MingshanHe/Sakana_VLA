"""Workarounds for hyperlux's JIT (cmake) hanging on the cluster. Import before hyperlux/hyperdrone.

1. `cmake --build` (conda CMake 4.4, libuv with io_uring) can hang forever after "[100%] Built target" with a defunct
   child: libuv's io_uring path. UV_USE_IO_URING=0 disables it (set here for this process and all children).
2. The configure step under subprocess PIPE can stall; streaming the build output instead of capturing it avoids that.
"""
import os
import subprocess
import sys

os.environ.setdefault("UV_USE_IO_URING", "0")

import hyperlux._jit._build as _build


def _run(arguments, env):
    subprocess.run([str(a) for a in arguments], env=env, check=True, stdout=sys.stderr, stderr=sys.stderr)


_build.run = _run
