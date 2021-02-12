
use lattice_qcd_rs::{
    simulation::*,
    ComplexField,
};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use super::{
    config::*,
    data_analysis::*,
};

pub fn get_pb_template() -> &'static str {
    &"{prefix:14} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>4}/{len:4} [ETA {eta_precise}] {msg}"
}

pub fn generate_state_default(cfg: &LatticeConfig, rng: &mut impl rand::Rng) -> LatticeStateDefault {
    LatticeStateDefault::new_deterministe(cfg.lattice_size(), cfg.lattice_beta(), cfg.lattice_number_of_points(), rng).unwrap()
}

pub fn get_mc_from_config(cfg: &MonteCarloConfig) -> MetropolisHastingsDeltaDiagnostic {
    MetropolisHastingsDeltaDiagnostic::new(cfg.number_of_rand(), cfg.spread()).unwrap()
}

pub fn run_simulation_with_progress_bar(
    config: &SimConfig,
    inital_state : LatticeStateDefault,
    mp : &MultiProgress,
    rng: &mut impl rand::Rng
) -> AverageData {
    let mut mc = MCWrapper::new(get_mc_from_config(config.mc_config()), rng);
    
    let mut simulation = inital_state;
    let pb_th = mp.add(ProgressBar::new((config.number_of_averages() * config.number_of_steps_between_average() + config.number_of_thermalisation()) as u64));
    pb_th.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb_th.set_prefix("Thermalisation");
    
    for _ in 0..config.number_of_thermalisation() / config.number_between_renorm() {
        let average = simulation.average_trace_plaquette().unwrap().real();
        for _ in 0..config.number_between_renorm() {
            simulation = simulation.monte_carlo_step(&mut mc).unwrap();
            pb_th.inc(1);
        }
        pb_th.set_message(&format!("A {:.6}, P {:.2}", average, mc.mcd().prob_replace_last()));
        simulation.normalize_link_matrices();
    }
    
    
    pb_th.set_prefix("simulation");
    pb_th.reset_eta();
    //pb_th.reset();
    //pb_th.set_length((config.number_of_averages() * config.number_of_steps_between_average()) as u64);
    pb_th.tick();
    
    let mut return_data = vec![];
    
    let mut average = simulation.average_trace_plaquette().unwrap().real();
    
    for i in 0..config.number_of_averages() {
        for _ in 0..config.number_of_steps_between_average() / config.number_between_renorm() {
            pb_th.set_message(&format!("A {:.6}, P {:.2}", average, mc.mcd().prob_replace_last()));
            for _ in 0..config.number_between_renorm() {
                simulation = simulation.monte_carlo_step(&mut mc).unwrap();
                pb_th.inc(1);
            }
            simulation.normalize_link_matrices();
            average = simulation.average_trace_plaquette().unwrap().real();
        }
        return_data.push(AverageDataPoint::new(
            average,
            config.number_of_thermalisation() + (i + 1) * config.number_of_steps_between_average()
        ));
    }
    pb_th.finish_and_clear();
    AverageData::new(return_data)
}
