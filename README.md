# reactiveLJ Paper Simulations

This repository contains a suite of molecular dynamics simulations used to evaluate the **reactiveLJ bond-order potential** on a **Kremer-Grest polymer melt**.

## Purpose

The workflows in this project are designed to:
- generate and run reactiveLJ melt simulations,
- analyze resulting trajectories and observables,
- compare behavior across parameter choices and validation tests.

## Repository Layout

- `simulation_package/melt/`: melt data generation, analysis, energy checks, and Tersoff comparison workflows.
- `simulation_package/single_chain/`: single-chain data generation and analysis workflows.
- `hoomd-blue/`: HOOMD-blue source tree, maintained as its own nested Git repository.
