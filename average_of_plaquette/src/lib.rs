
//! Binary application to compute the average of the plaquette

//#![warn(clippy::as_conversions)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
//#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::implicit_hasher)]
#![warn(clippy::implicit_saturating_sub)]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::macro_use_imports)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::non_ascii_literal)]
//#![warn(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::todo)]
#![warn(clippy::trivially_copy_pass_by_ref)]
//#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]
#![warn(clippy::unnecessary_wraps)]


//#![warn(clippy::missing_errors_doc)]
//#![warn(missing_docs)]

extern crate lattice_qcd_rs;
extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;
extern crate bincode;
extern crate serde;
extern crate serde_json;
extern crate csv;
extern crate rand_xoshiro;
extern crate console;
extern crate plotters;

pub mod config;
pub mod sim;
pub mod config_scan;
pub mod data_analysis;
pub mod rng;
pub mod observable;
#[cfg(test)]
mod test;
