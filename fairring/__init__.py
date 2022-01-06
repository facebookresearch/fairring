#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch.distributed

# Must come after torch
import _fairring  # isort: skip


def create_process_group_fairring(store, rank, size, timeout):
    try:
        max_memory_allocated_in_bytes = int(os.environ["FAIRRING_TOTAL_MEMORY"])
    except KeyError:
        sys.stderr.write(
            "The Fairring process group will allocate up to 50 MiB of staging memory "
            "on each GPU. This can be tuned by setting the FAIRRING_TOTAL_MEMORY "
            "environment variable (to an amount of bytes).\n",
        )
        max_memory_allocated_in_bytes = 50 * 1024 * 1024
    try:
        max_padding_allocated_in_bytes = int(os.environ["FAIRRING_PADDING_MEMORY"])
    except KeyError:
        max_padding_allocated_in_bytes = 1024 * 1024
    try:
        min_parallelism = int(os.environ["FAIRRING_PARALLELISM"])
    except KeyError:
        min_parallelism = 16
    return _fairring.ProcessGroup(
        store, rank, size, _fairring.ProcessGroup.Options(
            max_memory_allocated_in_bytes=max_memory_allocated_in_bytes,
            max_padding_allocated_in_bytes=max_padding_allocated_in_bytes,
            min_parallelism=min_parallelism,
            is_high_priority_stream=True,
            timeout=timeout,
        )
    )


torch.distributed.Backend.register_backend("fairring", create_process_group_fairring)
