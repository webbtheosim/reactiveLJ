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
- `simulation_package/mpcd/`: MPCD-solvated polymer solution data generation and analysis workflows.
- `hoomd-blue/`: HOOMD-blue source tree, maintained as its own nested Git repository.

## ReactiveLJ `epsilon=0` Behavior

In all three data-generation workflows
(`melt/data_generation/run_reactive_lj.py`,
`single_chain/data_generation/run_reactive_lj.py`,
`mpcd/data_generation/run_reactive_lj.py`):

- `reactive_epsilon > 0`: run the standard ReactiveLJ protocol.
- `reactive_epsilon <= 0`: automatically disable the ReactiveLJ force and run with pure WCA for sticky-sticky pairs.

This means sticky beads remain type-labeled for analysis, but their nonbonded
interactions are the same as backbone beads when `epsilon <= 0`.
