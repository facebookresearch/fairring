#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import subprocess
from setuptools import setup
from torch.utils import cpp_extension


def get_my_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_git_commit_hash():
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=get_my_dir(), capture_output=True)
    return result.stdout.decode("ascii").strip()


def get_git_is_dirty():
    result = subprocess.run(["git", "diff", "--quiet"], cwd=get_my_dir())
    return result.returncode != 0


def get_version():
    injected_version = os.environ.get("FAIRRING_VERSION", None)
    if injected_version is not None:
        return injected_version
    today = datetime.date.today().strftime("%Y.%m.%d")
    git_hash = get_git_commit_hash()
    is_dirty = get_git_is_dirty()
    return f"{today}+git{git_hash}{'.dirty' if is_dirty else ''}"


setup(
    name="fairring",
    version=get_version(),
    ext_modules=[
        cpp_extension.CUDAExtension(
            "_fairring",
            [
                "fairring/all_reduce.cc",
                "fairring/bindings.cc",
                "fairring/device.cc",
                "fairring/process_group.cc",
            ],
            include_dirs=[get_my_dir()],
            libraries=["nccl_static"],
            extra_compile_args={
                "cxx": [
                    "-g",
                ],
            },
            # These two options are needed to make NCCL and TensorPipe private
            # and hidden dependencies. By default, even when linking statically
            # to them, the linker would mark their symbols as global and/or
            # dynamic in the resulting shared object, which would cause these
            # symbols to be "deduplicated" at runtime with symbols of the same
            # name, such as those found in the PyTorch libraries. If these two
            # versions of the symbols come from different versions of NCCL or
            # TensorPipe, mixing them up will cause a mess (e.g., memory
            # corruption). With these flags we make it so that the only global
            # symbol advertised by this extension is the Python entrypoint. It
            # means that in the final process there will two copies of NCCL and
            # TensorPipe loaded into memory, but that's probably fine.
            extra_link_args=["-Wl,--version-script=version.map"],
            export_symbols=["PyInit__fairring"],
            # To work around https://github.com/pytorch/pytorch/issues/65918.
            define_macros=[("USE_C10D_NCCL", None)],
        ),
    ],
    packages=[
        "fairring",
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    setup_requires=["setuptools", "torch"],
    # Must include https://github.com/pytorch/pytorch/pull/65914 and
    # https://github.com/pytorch/pytorch/pull/66493.
    install_requires=["torch>=1.11.0.dev20211112"],
)
