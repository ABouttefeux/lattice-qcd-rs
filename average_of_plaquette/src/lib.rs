
//! Binary application to compute the average of the plaquette

extern crate lattice_qcd_rs;
extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;
extern crate bincode;
extern crate serde;
extern crate serde_json;
extern crate csv;


pub mod config;
pub mod sim;
pub mod config_scan;
pub mod data_analysis;
#[cfg(test)]
mod test;
