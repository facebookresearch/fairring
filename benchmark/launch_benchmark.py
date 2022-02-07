#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import multiprocessing
import os
import sys
from typing import List, Optional

import torch
import torch.distributed
from utils import recv_from_connections_and_join_processes

# Must come after torch or else it will fail because it won't find libc10.so
import benchmark_fairring  # isort: skip


def run_one_device(
    init_method: str,
    machine_idx: int,
    device_idx: int,
    num_machines: int,
    num_devices_per_machine: int,
    num_buckets: int,
    bucket_size: int,
    num_epochs: int,
    num_network_threads: Optional[int],
    num_sockets_per_network_thread: Optional[int],
    use_nccl: bool,
    parallelism: Optional[int],
    # pid_file: str,
    conn: multiprocessing.connection.Connection,
) -> List[int]:
    torch._C._set_print_stack_traces_on_fatal_signal(True)

    rdv_iterator = torch.distributed.rendezvous(
        init_method,
        machine_idx * num_devices_per_machine + device_idx,
        num_machines * num_devices_per_machine,
    )
    store, _, _ = next(rdv_iterator)

    assert 0 <= machine_idx < num_machines
    assert 0 <= device_idx < num_devices_per_machine

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_idx}"
    if num_network_threads is not None:
        os.environ["NCCL_SOCKET_NTHREADS"] = f"{num_network_threads}"
    if num_sockets_per_network_thread is not None:
        os.environ["NCCL_NSOCKS_PERTHREAD"] = f"{num_sockets_per_network_thread}"

    # print(os.getpid())
    # time.sleep(15)

    # with open(f"{pid_file}_{device_idx}", "wt") as f:
    #     f.write(f"{os.getpid()}")
    # time.sleep(20)

    # old_stderr = os.dup(2)
    # with open(f"/dev/shm/fairring_{machine_idx}_{device_idx}", "wb") as f:
    #     os.dup2(f.raw.fileno(), 2)
    # try:
    #     foo()
    # finally:
    #     os.dup2(old_stderr, 2)
    #     os.close(old_stderr)
    #     shutil.copyfile(f"/dev/shm/fairring_{machine_idx}_{device_idx}", f"{trace_file}-{device_idx}")
    #     os.remove(f"/dev/shm/fairring_{machine_idx}_{device_idx}")

    # from torch.profiler import *
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # ) as prof:
    with contextlib.nullcontext():
        client = benchmark_fairring.Client(
            machine_idx=machine_idx,
            device_idx=device_idx,
            num_machines=num_machines,
            num_devices_per_machine=num_devices_per_machine,
            num_buckets=num_buckets,
            bucket_size=bucket_size,
            num_epochs=num_epochs,
            store=store,
            use_nccl=use_nccl,
            parallelism=parallelism,
        )
        conn.send(client.run())

    # prof.export_chrome_trace(f"{trace_file}_{machine_idx}_{device_idx}")


def run_one_machine(
    init_method: str,
    machine_idx: int,
    num_machines: int,
    num_devices_per_machine: int,
    num_buckets: int,
    bucket_size: int,
    num_epochs: int,
    num_network_threads: Optional[int],
    num_sockets_per_network_thread: Optional[int],
    use_nccl: bool,
    parallelism: Optional[int],
    # pid_file: str,
) -> torch.Tensor:
    receiving_conns = []
    sending_conns = []
    for _ in range(num_devices_per_machine):
        recv_end, send_end = multiprocessing.get_context("spawn").Pipe()
        receiving_conns.append(recv_end)
        sending_conns.append(send_end)
    clients = [
        multiprocessing.get_context("spawn").Process(
            target=run_one_device,
            name=f"machine{machine_idx}_device{device_idx}",
            args=(
                init_method,
                machine_idx,
                device_idx,
                num_machines,
                num_devices_per_machine,
                num_buckets,
                bucket_size,
                num_epochs,
                num_network_threads,
                num_sockets_per_network_thread,
                use_nccl,
                parallelism,
                # pid_file,
                sending_conns[device_idx],
            ),
        )
        for device_idx in range(num_devices_per_machine)
    ]
    for t in clients:
        t.start()
    for c in sending_conns:
        c.close()

    stats = recv_from_connections_and_join_processes(
        list(zip(clients, receiving_conns))
    )

    return torch.tensor(stats, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="all-reduce benchmark")
    parser.add_argument(
        "--init-method",
        type=str,
        default="env://",
        help="How to do rendezvous between machines (uses PyTorch, hence see its doc)",
    )
    parser.add_argument(
        "--machine-idx",
        type=int,
        required=True,
        help="The rank of the machine on which this script was invoked (0-based)",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        required=True,
        help="On how many machines this script is being invoked (each with its own rank)",
    )
    parser.add_argument(
        "--num-devices-per-machine",
        type=int,
        required=True,
        help="How many clients this script should launch (each will use one GPU)",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        required=True,
        help="How many buffers to do an allreduce over in each epoch",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        required=True,
        help="How big each buffer should be (expressed in number of float32 elements)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=True,
        help="How many times to run the benchmark",
    )
    parser.add_argument(
        "--num-network-threads",
        type=int,
        help="The value of the NCCL_SOCKET_NTHREADS env var (see NCCL's doc)",
    )
    parser.add_argument(
        "--num-sockets-per-network-thread",
        type=int,
        help="The value of the NCCL_NSOCKS_PERTHREAD env var (see NCCL's doc)",
    )
    parser.add_argument(
        "--use-nccl",
        action="store_true",
    )
    # parser.add_argument(
    #     "--pid-file",
    #     type=str,
    # )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("wb"),
        default=sys.stdout.buffer,
    )

    args = parser.parse_args()

    res = run_one_machine(
        init_method=args.init_method,
        machine_idx=args.machine_idx,
        num_machines=args.num_machines,
        num_devices_per_machine=args.num_devices_per_machine,
        num_buckets=args.num_buckets,
        bucket_size=args.bucket_size,
        num_epochs=args.num_epochs,
        num_network_threads=args.num_network_threads,
        num_sockets_per_network_thread=args.num_sockets_per_network_thread,
        use_nccl=args.use_nccl,
        parallelism=args.parallelism,
        # pid_file=args.pid_file,
    )

    torch.save(res, args.output)


if __name__ == "__main__":
    main()
