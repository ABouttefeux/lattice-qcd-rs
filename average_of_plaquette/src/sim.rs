
use lattice_qcd_rs::{
    simulation::*,
    ComplexField,
    lattice::{Direction, DirectionList, LatticePoint},
    utils::Sign,
    statistics,
    Real,
    field::Su3Adjoint,
    integrator::SymplecticEulerRayon,
    error::{StateInitializationError, MultiIntegrationError},
};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use super::{
    config::*,
    data_analysis::*,
    observable,
    data_analysis,
};
use rayon::prelude::*;
use na::{
    SVector,
};

/// Return the [`indicatif::ProgressBar`] template
pub const fn get_pb_template() -> &'static str {
    &"{prefix:14} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>6}/{len:6} [ETA {eta_precise}] {msg}"
}

/// Generate a hot configuration with the given config
pub fn generate_state_default<Rng: rand::Rng, const D: usize>(cfg: &LatticeConfig, rng: &mut Rng) -> LatticeStateDefault<D> {
    LatticeStateDefault::new_deterministe(cfg.lattice_size(), cfg.lattice_beta(), cfg.lattice_number_of_points(), rng).expect("Invalide Configuration")
}

/// Generate a [`MetropolisHastingsDeltaDiagnostic`] from a config
pub fn get_mc_from_config<Rng>(cfg: &MonteCarloConfig, rng: Rng) -> MetropolisHastingsDeltaDiagnostic<Rng>
where
    Rng: rand::Rng,
{
    MetropolisHastingsDeltaDiagnostic::new(cfg.spread(), rng).expect("Invalide Configuration")
}

/// Generate a [`MetropolisHastingsSweep`] from a config
pub fn get_mc_from_config_sweep<Rng>(cfg: &MonteCarloConfig, rng: Rng) -> MetropolisHastingsSweep<Rng>
where
    Rng: rand::Rng,
{
    MetropolisHastingsSweep::new(cfg.number_of_rand(), cfg.spread(), rng).expect("Invalide Configuration")
}

pub fn run_simulation_with_progress_bar_average<Rng>(
    config: &SimConfig,
    inital_state : LatticeStateDefault<4>,
    mp : &MultiProgress,
    rng: Rng,
) -> (AverageData, LatticeStateDefault<4>, Rng)
where
    Rng: rand::Rng,
{
    run_simulation_with_progress_bar(config, inital_state, mp, rng, &|simulation| {
        simulation.average_trace_plaquette().unwrap().real() / 3.0
     })
}

pub fn run_simulation_with_progress_bar_volume<Rng>(
    config: &SimConfig,
    inital_state : LatticeStateDefault<3>,
    mp : &MultiProgress,
    rng: Rng,
) -> (AverageData, LatticeStateDefault<3>, Rng)
where
    Rng: rand::Rng,
{
    run_simulation_with_progress_bar(config, inital_state, mp, rng, &|simulation| {
        observable::volume_obs_mean(simulation)
     })
}

/// Run a simulation with a progress bar
fn run_simulation_with_progress_bar<Rng, const D: usize>(
    config: &SimConfig,
    inital_state : LatticeStateDefault<D>,
    mp : &MultiProgress,
    rng: Rng,
    closure: &dyn Fn(&LatticeStateDefault<D>) -> f64,
) -> (AverageData, LatticeStateDefault<D>, Rng)
where
    Rng: rand::Rng,
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

#[derive(Debug)]
pub enum ThermalisationSimumlationError<Error>{
    SignObsZero,
    SimulationError(Error),
}

impl<Error> From<Error> for ThermalisationSimumlationError<Error> {
    fn from(data: Error) -> Self {
        Self::SimulationError(data)
    }
}

#[allow(clippy::needless_range_loop)]
/// thermalize the state using a observable as a mesurement.
/// It plot a graph and write a csv file for the correlation
pub fn thermalize_state<MC, F, const D: usize>(
    //config: &SimConfig,
    inital_state : LatticeStateDefault<D>,
    mc: &mut MC,
    mp : &MultiProgress,
    observable: F,
    prefix: &str,
    sufix: &str,
) -> Result<(LatticeStateDefault<D>, Real), ThermalisationSimumlationError<MC::Error>>
where
    MC: MonteCarlo<LatticeStateDefault<D>, D>,
    F: Fn(&LatticePoint<D>, &LatticeStateDefault<D>) -> f64 + Sync
{
    const NUMBER_OF_MEASURE_COMPUTE: usize = 1_000;
    const NUMBER_OF_PASS: usize = 1;
    const NUMBER_OF_PASS_AUTO_CORR: usize = 1_000;
    
    let pb_th = mp.add(ProgressBar::new(NUMBER_OF_MEASURE_COMPUTE as u64 + 1));
    pb_th.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb_th.set_prefix(&format!("T - est - {} {}", inital_state.lattice().dim(), sufix));
    
    let mut state = inital_state.monte_carlo_step(mc)?;

    let points = state.lattice().get_points().collect::<Vec<LatticePoint<D>>>();
    let measurement_val_init = points.par_iter()
        .map(|el| observable(el, &state))
        .sum::<f64>();
    
    for _ in 0..NUMBER_OF_PASS {
        state = state.monte_carlo_step(mc)?;
    }
    pb_th.inc(1);
    
    let mut measurement_val = points.par_iter()
        .map(|el| observable(el, &state))
        .sum::<f64>();
    let sign = Sign::sign(measurement_val - measurement_val_init);
    if sign == Sign::Zero {
        return Err(ThermalisationSimumlationError::SignObsZero);
    }
    loop {
        let  mut vec = [0_f64; NUMBER_OF_MEASURE_COMPUTE];
        for j in 0..NUMBER_OF_MEASURE_COMPUTE {
            for _ in 0..NUMBER_OF_PASS {
                state = state.monte_carlo_step(mc)?;
            }
            state.normalize_link_matrices();
            let val = points.par_iter()
                .map(|el| observable(el, &state))
                .sum::<f64>();
            vec[j] = val - measurement_val;
            measurement_val = val;
            pb_th.inc(1);
        }
        
        let mean = statistics::mean(&vec);
        pb_th.set_message(&format!("{:.6}", mean));
        if Sign::sign(mean) != sign {
            break;
        }
        pb_th.set_length(pb_th.length() + NUMBER_OF_MEASURE_COMPUTE as u64);
    }
    let init_vec = points.par_iter()
        .map(|el| observable(el, &state))
        .collect::<Vec<f64>>();
    let mut t_exp = 0.5_f64;
    let init_auto_corr = statistics::variance(&init_vec).abs();
    let mut last_auto_corr_mean = 1_f64;
    
    let mut vec_corr_plot = Vec::with_capacity(NUMBER_OF_PASS_AUTO_CORR);
    
    let mut auto_corr_limiter = 0_f64;
    loop {
        if last_auto_corr_mean < auto_corr_limiter {
            break;
        }
        pb_th.set_length(pb_th.length() + NUMBER_OF_PASS_AUTO_CORR as u64);
        let  mut vec_corr = [0_f64; NUMBER_OF_PASS_AUTO_CORR];
        for j in 0..NUMBER_OF_PASS_AUTO_CORR{
            for _ in 0..NUMBER_OF_PASS {
                state = state.monte_carlo_step(mc)?;
            }
            state.normalize_link_matrices();
            let vec = points.par_iter()
                .map(|el| observable(el, &state))
                .collect::<Vec<f64>>();
            let last_auto_corr = statistics::covariance(&init_vec, &vec).unwrap().abs();
            t_exp += last_auto_corr / init_auto_corr;
            pb_th.inc(1);
            pb_th.set_message(&format!("{:.2}, {:.6}", t_exp, last_auto_corr / init_auto_corr));
            vec_corr[j] = last_auto_corr / init_auto_corr;
        }
        vec_corr_plot.extend_from_slice(&vec_corr);
        last_auto_corr_mean = statistics::mean(&vec_corr);
        auto_corr_limiter += 0.1_f64;
    }
    let _ = data_analysis::plot_data_auto_corr(&vec_corr_plot, &format!("{}plot_auto_corr_{}_{}.svg", prefix, state.lattice().dim(), sufix));
    let _ = write_vec_to_file_csv(&[vec_corr_plot], &format!("{}raw_auto_corr_{}_{}.csv", prefix, state.lattice().dim(), sufix));
    pb_th.finish_and_clear();
    Ok((state, t_exp))
}

type MesurementAndLattice<const D: usize> = (LatticeStateDefault<D>, Vec<Vec<Real>>);

pub fn simulation_gather_measurement<MC, F, const D: usize>(
    //config: &SimConfig,
    inital_state : LatticeStateDefault<D>,
    mc: &mut MC,
    mp : &MultiProgress,
    observable: F,
    number_of_discard: usize,
    number_of_measurement: usize,
) -> Result<MesurementAndLattice<D>, ThermalisationSimumlationError<MC::Error>>
where
    MC: MonteCarlo<LatticeStateDefault<D>, D>,
    F: Fn(&LatticePoint<D>, &LatticeStateDefault<D>) -> f64 + Sync
{
    const NUMBER_OF_PASS: usize = 1;
    let pb = mp.add(ProgressBar::new((number_of_measurement * number_of_discard) as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix(&format!("sim - {}", inital_state.lattice().dim()));
    
    let mut state = inital_state;
    let mut vec = Vec::with_capacity(number_of_measurement);
    let points = state.lattice().get_points().collect::<Vec<LatticePoint<D>>>();
    
    for _ in 0..number_of_measurement {
        for _ in 0..number_of_discard {
            for _ in 0..NUMBER_OF_PASS {
                state = state.monte_carlo_step(mc)?;
            }
            state.normalize_link_matrices();
            pb.inc(1);
        }
        let vec_data = points.par_iter()
            .map(|el| observable(el, &state))
            .collect::<Vec<f64>>();
        vec.push(vec_data);
    }
    pb.finish_and_clear();
    Ok((state, vec))
}

#[derive(Clone, Debug, PartialEq)]
pub enum ThermalizeError {
    HmcError(MultiIntegrationError<StateInitializationError>),
    StateInitializationError(StateInitializationError),
}

impl From<MultiIntegrationError<StateInitializationError>> for ThermalizeError {
    fn from(err: MultiIntegrationError<StateInitializationError>) -> Self{
        Self::HmcError(err)
    }
}

impl From<StateInitializationError> for ThermalizeError {
    fn from(err: StateInitializationError) -> Self{
        Self::StateInitializationError(err)
    }
}

pub type ResultThermalizeE<Rng, const D: usize> = (LatticeStateWithEFieldSyncDefault<LatticeStateDefault<D>, D>, Rng);

const INTEGRATOR: SymplecticEulerRayon = SymplecticEulerRayon::new();

#[allow(clippy::useless_format)]
pub fn thermalize_with_e_field<Rng, const D: usize>(
    inital_state : LatticeStateDefault<D>,
    mp : &MultiProgress,
    rng: Rng,
    dt: f64,
) -> Result<ResultThermalizeE<Rng, D>, ThermalizeError>
where
    Direction<D>: DirectionList,
    SVector<Su3Adjoint, D>: Sync + Send,
    Rng: rand::Rng,
{
    
    const STEPS: usize = 300;
    const CYCLE: usize = 40;
    
    let pb = mp.add(ProgressBar::new((CYCLE) as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix(&format!("th - e"));
    
    let mut hmc = HybridMonteCarloDiagnostic::new(dt, STEPS, INTEGRATOR, rng);
    
    let mut state = inital_state;
    for _ in 0..CYCLE {
        state = state.monte_carlo_step(&mut hmc)?;
        state.normalize_link_matrices();
        pb.set_message(&format!("{:.6}   ", hmc.prob_replace_last()));
        pb.inc(1);
    }
    
    let mut rng = hmc.rng_owned();
    let state_with_e = LatticeStateWithEFieldSyncDefault::new_random_e(state.lattice().clone(), state.beta(), state.link_matrix_owned(), &mut rng)?;
    let state_e = state_with_e.simulate_symplectic(&INTEGRATOR, dt)?;
    pb.inc(1);
    pb.finish_and_clear();
    //Ok((leap.simulate_leap_n(dt, &INTEGRATOR, STEPS)?, rng))
    Ok((state_e, rng))
}
