/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fairring/process_group.h>

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(_fairring, module) {
  intrusive_ptr_class_<fairring::ProcessGroupFairring, c10d::ProcessGroup>
      processGroupFairring(module, "ProcessGroup");

  intrusive_ptr_class_<
      fairring::ProcessGroupFairring::OptionsFairring,
      c10d::ProcessGroup::Options>(processGroupFairring, "Options")
      .def(
          py::init<
              int64_t,
              int64_t,
              int64_t,
              bool,
              std::chrono::milliseconds>(),
          py::arg("max_memory_allocated_in_bytes"),
          py::arg("max_padding_allocated_in_bytes"),
          py::arg("min_parallelism"),
          py::arg("is_high_priority_stream"),
          py::arg("timeout"));

  processGroupFairring.def(
      py::init<
          const c10::intrusive_ptr<c10d::Store>&,
          int,
          int,
          c10::intrusive_ptr<
              fairring::ProcessGroupFairring::OptionsFairring>>(),
      py::call_guard<py::gil_scoped_release>());
}
