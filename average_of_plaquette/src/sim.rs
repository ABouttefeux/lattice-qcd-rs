
use lattice_qcd_rs::{
    simulation::*,
    ComplexField,
    lattice::Direction,
};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use super::{
    config::*,
    data_analysis::*,
};
use rayon::prelude::*;

/// Return the [`indicatif::ProgressBar`] template
pub const fn get_pb_template() -> &'static str {
    &"{prefix:14} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>6}/{len:6} [ETA {eta_precise}] {msg}"
}

/// Generate a hot configuration with the given config
pub fn generate_state_default(cfg: &LatticeConfig, rng: &mut impl rand::Rng) -> LatticeStateDefault {
    LatticeStateDefault::new_deterministe(cfg.lattice_size(), cfg.lattice_beta(), cfg.lattice_number_of_points(), rng).expect("Invalide Configuration")
}

/// Generate a [`MetropolisHastingsDeltaDiagnostic`] from a config
pub fn get_mc_from_config<Rng>(cfg: &MonteCarloConfig, rng: Rng) -> MetropolisHastingsDeltaDiagnostic<Rng>
    where Rng: rand::Rng,
{
    MetropolisHastingsDeltaDiagnostic::new(cfg.number_of_rand(), cfg.spread(), rng).expect("Invalide Configuration")
}

pub fn run_simulation_with_progress_bar_average<Rng>(
    config: &SimConfig,
    inital_state : LatticeStateDefault,
    mp : &MultiProgress,
    rng: Rng,
) -> (AverageData, LatticeStateDefault, Rng)
    where Rng: rand::Rng,
{
    run_simulation_with_progress_bar(config, inital_state, mp, rng, &|simulation| {
        simulation.average_trace_plaquette().unwrap().real() / 3.0
     })
}

pub fn run_simulation_with_progress_bar_volume<Rng>(
    config: &SimConfig,
    inital_state : LatticeStateDefault,
    mp : &MultiProgress,
    rng: Rng,
) -> (AverageData, LatticeStateDefault, Rng)
    where Rng: rand::Rng,
{
    run_simulation_with_progress_bar(config, inital_state, mp, rng, &|simulation| {
        let sum = simulation.lattice().get_points().par_bridge().map(|point| {
            simulation.link_matrix().get_pij(&point, &Direction::positives()[0], &Direction::positives()[1], simulation.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64)
        }).sum::<Option<f64>>().unwrap();
        let number_of_plaquette = simulation.lattice().get_number_of_points() as f64;
        let c1 = 8_f64 / 3_f64;
        let c2 = 1.951315_f64;
        let c3 = 6.8612_f64;
        let c4 = 2.92942132_f64;
        simulation.beta().powi(4) * ((sum / number_of_plaquette) - c1 / simulation.beta() - c2 / simulation.beta().powi(2) - c3 / simulation.beta().powi(3) - c4 * simulation.beta().ln() / simulation.beta().powi(4))
     })
}

/// Run a simulation with a progress bar
fn run_simulation_with_progress_bar<Rng>(
    config: &SimConfig,
    inital_state : LatticeStateDefault,
    mp : &MultiProgress,
    rng: Rng,
    closure: &dyn Fn(&LatticeStateDefault) -> f64,
) -> (AverageData, LatticeStateDefault, Rng)
    where Rng: rand::Rng,
{
    
    let mut mc = get_mc_from_config(config.mc_config(), rng);
    let mut simulation = inital_state;
    
    let pb_th = mp.add(ProgressBar::new((config.number_of_averages() * config.number_of_steps_between_average() + config.number_of_thermalisation()) as u64));
    pb_th.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb_th.set_prefix("Thermalisation");
    
    for _ in 0..config.number_of_thermalisation() / config.number_between_renorm() {
        let value = closure(&simulation);
        pb_th.set_message(&format!("V {:.6}, P {:.2}", value, mc.prob_replace_last()));
        for _ in 0..config.number_between_renorm() {
            simulation = simulation.monte_carlo_step(&mut mc).unwrap();
        }
        pb_th.inc(config.number_between_renorm() as u64);
        simulation.normalize_link_matrices();
    }
    
    pb_th.set_prefix("simulation");
    pb_th.tick();
    
    let mut return_data = vec![];
    let mut value = closure(&simulation);
    
    for i in 0..config.number_of_averages() {
        for _ in 0..config.number_of_steps_between_average() / config.number_between_renorm() {
            pb_th.set_message(&format!("A {:.6}, P {:.2}", value, mc.prob_replace_last()));
            for _ in 0..config.number_between_renorm() {
                simulation = simulation.monte_carlo_step(&mut mc).unwrap();
            }
            pb_th.inc(config.number_between_renorm() as u64);
            simulation.normalize_link_matrices();
            value = closure(&simulation);
        }
        return_data.push(AverageDataPoint::new(
            value,
            config.number_of_thermalisation() + (i + 1) * config.number_of_steps_between_average()
        ));
    }
    pb_th.finish_and_clear();
    (AverageData::new(return_data), simulation, mc.rng_owned())
}
