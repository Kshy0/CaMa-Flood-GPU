// CaMa-Flood-GPU pybind11 bindings for CUDA launchers
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include <torch/extension.h>

// Forward declarations of launcher functions (defined in per-function .cu files)
void launch_outflow(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor,
    torch::Tensor, int);

void launch_inflow(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int);

void launch_flood_stage(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int);

void launch_adaptive_time_step(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    float, float, int);

void launch_bifurcation_outflow(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int);

void launch_bifurcation_inflow(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int);

void launch_reservoir_outflow(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int);

void launch_levee_stage(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int);

void launch_levee_bifurcation_outflow(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_outflow", &launch_outflow);
    m.def("launch_inflow", &launch_inflow);
    m.def("launch_flood_stage", &launch_flood_stage);
    m.def("launch_adaptive_time_step", &launch_adaptive_time_step);
    m.def("launch_bifurcation_outflow", &launch_bifurcation_outflow);
    m.def("launch_bifurcation_inflow", &launch_bifurcation_inflow);
    m.def("launch_reservoir_outflow", &launch_reservoir_outflow);
    m.def("launch_levee_stage", &launch_levee_stage);
    m.def("launch_levee_bifurcation_outflow", &launch_levee_bifurcation_outflow);
}
