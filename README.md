# 3D Incompressible Navier-Stokes Solver (WIP)

A prototype Finite Element Method (FEM) solver for the 3D incompressible Navier-Stokes equations, written in MATLAB with C++ MEX acceleration.

**Status: Experimental / Work-in-Progress**

## Overview
This code implements a stabilized finite element formulation to simulate laminar flows in 3D domains. It is designed to test efficient preconditioning strategies and time-stepping schemes.

## Key Methods
* **Discretization:** Q2-Q1 Taylor-Hood elements on hexahedral grids.
* **Time Stepping:** BDF2 (Backward Differentiation Formula, 2nd Order) with BDF1 startup.
* **Stabilization:** SUPG (Streamline-Upwind Petrov-Galerkin) for convection-dominated flows.
* **Linear Solver:** GMRES with a Pressure Convection-Diffusion (PCD) preconditioner.
* **Acceleration:**
    * Geometric Multigrid (GMG) for internal preconditioner solves.
    * Computationally intensive assembly routines offloaded to C++ MEX files.

## Structure
* `driver_ns_3d.m`: Main simulation loop and user parameters.
* `ns_assembly.m`: Wrappers for MEX-based matrix assembly.
* `pcd_manager.m`: Construction and update logic for the PCD preconditioner.
* `gmg_system.m`: Geometric Multigrid solvers (V-cycles) and hierarchy generation.
* `setup_domain.m`: Grid generation and boundary condition handling.

## Requirements
* MATLAB (R2023b or newer recommended)
* Compatible C++ Compiler (configured via `mex -setup`)
* *Note:* MEX binaries must be compiled before running the driver.

## Usage
1.  Ensure all MEX files are compiled and in the path.
2.  Adjust parameters (Reynolds number, grid size, `dt`) in `driver_ns_3d.m`.
3.  Run `driver_ns_3d`.
4.  Output files (`.vtu`) are written to the directory for visualization in Paraview.

## Disclaimer
This is research code provided "as-is" for educational or testing purposes. It is not fully optimized or documented for general use.
