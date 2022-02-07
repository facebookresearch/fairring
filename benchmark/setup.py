#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import setup
from torch.utils import cpp_extension


def get_my_dir():
    return os.path.dirname(os.path.abspath(__file__))


setup(
    name="fairring_benchmark",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "benchmark_fairring",
            [
                "../fairring/device.cc",
                "../fairring/machine.cc",
                "../fairring/process_group.cc",
                "benchmark.cc",
            ],
            include_dirs=[get_my_dir() + "/.."],
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
            export_symbols=["PyInit_benchmark_fairring"],
            # To work around https://github.com/pytorch/pytorch/issues/65918.
            define_macros=[("USE_C10D_NCCL", None)],
        ),
    ],
    py_modules=[
        "launch_benchmark",
        "utils",
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    entry_points={
        "console_scripts": [
            "launch_benchmark=launch_benchmark:main",
        ],
    },
    setup_requires=["setuptools", "torch"],
    # Must include https://github.com/pytorch/pytorch/pull/65914 and
    # https://github.com/pytorch/pytorch/pull/66493.
    install_requires=["torch>=1.11.0.dev20211112"],
)
