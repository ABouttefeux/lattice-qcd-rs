
//! Configuration for simulation

use serde::{Serialize, Deserialize};

use lattice_qcd_rs::{
    Real,
};

/// Configuration for the lattice
#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct LatticeConfig {
    lattice_beta: Real,
    lattice_number_of_points: usize,
    lattice_size: Real,
}

impl LatticeConfig {
    
    pub fn new(lattice_size: Real, lattice_number_of_points: usize, lattice_beta: Real) -> Option<Self> {
        if lattice_size < 0_f64 || lattice_number_of_points < 2 {
            return None;
        }
        Some(Self {lattice_size, lattice_number_of_points, lattice_beta})
    }
    
    pub const fn lattice_beta(&self) -> Real {
        self.lattice_beta
    }
    
    pub const fn lattice_number_of_points(&self) -> usize {
        self.lattice_number_of_points
    }
    
    pub const fn lattice_size(&self) -> Real {
        self.lattice_size
    }
    
    
}

/// Configuration for the simulation
#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct SimConfig {
    number_of_thermalisation: usize,
    number_between_renorm: usize,
    number_of_averages: usize,
    number_of_steps_between_average: usize,
    mc_config: MonteCarloConfig,
}

impl SimConfig {
    
    pub const fn new(
        mc_config: MonteCarloConfig,
        number_of_thermalisation: usize,
        number_between_renorm: usize,
        number_of_averages: usize,
        number_of_steps_between_average: usize
    ) -> Option<Self> {
        
        if number_of_thermalisation % number_between_renorm != 0 ||
            number_of_steps_between_average % number_between_renorm != 0
        {
            None
        }
        else{
            Some(Self{
                mc_config,
                number_of_thermalisation,
                number_between_renorm,
                number_of_averages,
                number_of_steps_between_average,
            })
        }
    }
    
    pub fn new_raw(
        number_of_rand: usize,
        spread: Real,
        number_of_thermalisation: usize,
        number_between_renorm: usize,
        number_of_averages: usize,
        number_of_steps_between_average: usize
    ) -> Option<Self> {
        
        let mc = MonteCarloConfig::new(number_of_rand, spread)?;
        Self::new(
            mc,
            number_of_thermalisation,
            number_between_renorm,
            number_of_averages,
            number_of_steps_between_average
        )
    }
    
    pub const fn mc_config(&self) -> &MonteCarloConfig {
        &self.mc_config
    }
    
    pub const fn number_of_thermalisation(&self) -> usize {
        self.number_of_thermalisation
    }
    
    pub const fn number_between_renorm(&self) -> usize {
        self.number_between_renorm
    }
    
    pub const fn number_of_averages(&self) -> usize {
        self.number_of_averages
    }
    
    pub const fn number_of_steps_between_average(&self) -> usize {
        self.number_of_steps_between_average
    }
}

/// Configuration for the Monte Carlo algorithm
#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    number_of_rand: usize,
    spread: Real,
}

impl MonteCarloConfig {
    pub fn new(number_of_rand: usize, spread: Real) -> Option<Self> {
        if spread <= 0_f64 || spread >= 1_f64 {
            None
        }
        else{
            Some(Self{number_of_rand, spread})
        }
    }
    
    pub const fn number_of_rand(&self) -> usize {
        self.number_of_rand
    }
    
    pub const fn spread(&self) -> Real {
        self.spread
    }
}

/// Configuration for the lattice and simulation
#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct Config {
    lattice_config: LatticeConfig,
    sim_config: SimConfig,
}

impl Config {
    pub const fn new(lattice_config: LatticeConfig, sim_config: SimConfig) -> Self{
        Self {lattice_config, sim_config}
    }
    
    pub const fn lattice_config(&self) -> &LatticeConfig {
        &self.lattice_config
    }
    
    pub const fn sim_config(&self) -> &SimConfig {
        &self.sim_config
    }
}
