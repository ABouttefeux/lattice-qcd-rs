# ![Lattice QCD rs](logo.svg)

![](https://img.shields.io/badge/language-Rust-orange)
[![](https://img.shields.io/badge/doc-Read_Me-blue)](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/index.html)
![Build](https://img.shields.io/github/workflow/status/ABouttefeux/lattice-qcd-rs/Rust)
[![Documentation](https://img.shields.io/github/workflow/status/ABouttefeux/lattice-qcd-rs/Rust-Doc?label=doc)](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/index.html)
![](https://img.shields.io/criterion/ABouttefeux/lattice_qcd_rs)

Classical lattice QCD simulation and tools.

## Usage

Add `lattice_qcd_rs = { version = "0.1.0", git = "https://github.com/ABouttefeux/lattice_qcd_rs", branch = "develop" }` into your `cargo.toml`.

Is it on crates.io ? No, for the moment it is not. Maybe I will add it. But for the moment it is still in developpement.

## Why ?

This some code for my PhD thesis.
Mainly I use [arXiv:0707.2458](https://arxiv.org/abs/0707.2458), [arXiv:0902.28568](https://arxiv.org/abs/0707.2458) and [arXiv:2010.07316](https://arxiv.org/abs/2010.07316) as a basis.

## Goal

The goal is to provide an easy to use, fast and safe library to do classical lattice simulation.

### Module

#### SU(3)

Provides tool for managing SU(3) and its generator, as well as fast matrix exponentials for the use of Monte-Carlo simulation.

#### Thread

Some functionality to allow simple parallel computation.

#### Lattice

Representation of a finite lattice

#### Integrator

Numerical integrator for the classical simulation.

#### Field

Representation of fields on the lattice.
