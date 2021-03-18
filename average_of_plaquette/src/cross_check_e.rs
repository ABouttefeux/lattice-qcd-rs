use average_of_plaquette::{
    sim::*,
    config_scan::*,
    data_analysis::*,
    rng::*,
    observable,
 };
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use lattice_qcd_rs::{
    field::{
        Su3Adjoint,
    },
    simulation::{
        //HeatBathSweep,
        //OverrelaxationSweepReverse,
        //HybrideMethode,
        LatticeStateDefault,
        LatticeState,
        HybridMonteCarloDiagnostic,
        SimulationError,
        LatticeHamiltonianSimulationStateSyncDefault,
        LatticeHamiltonianSimulationStateNew,
        //LatticeHamiltonianSimulationState,
        SimulationStateLeapFrog,
        SimulationStateSynchrone,
        SimulationStateLeap,
    },
    lattice::{Direction, DirectionList, LatticePoint},
    integrator::SymplecticEulerRayon,
    dim::U3,
    statistics,
};
use nalgebra::{
    DefaultAllocator,
    VectorN,
    base::allocator::Allocator,
    DimName,
};
use plotters::prelude::*;


fn main() {
    main_cross_with_e();
}

fn main_cross_with_e() {
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1_f64), //size
        ScanPossibility::Default(4), // dim
        ScanPossibility::Default(6_f64), // Beta
    ).unwrap();
    let mc_cfg = MonteCarloConfigScan::new(
        ScanPossibility::Default(1),
        ScanPossibility::Default(0.1),
    ).unwrap();
    let sim_cfg = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Default(10_000), //th steps (unused)
        ScanPossibility::Default(1), // renormn (unused)
        ScanPossibility::Default(1_000), // number_of_averages
        ScanPossibility::Default(200) //between av
    ).unwrap();
    let config = ConfigScan::new(cfg_l, sim_cfg).unwrap();
    let array_config = config.get_all_config();
    
    let multi_pb = std::sync::Arc::new(MultiProgress::new());
    let pb = multi_pb.add(ProgressBar::new(array_config.len() as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix("Sim Total");
    pb.enable_steady_tick(499);
    
    let multi_pb_2 = multi_pb.clone();
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1000));
        multi_pb_2.set_move_cursor(true);
        multi_pb_2.join_and_clear()
    });
    pb.tick();
    
    let _result_all_sim = array_config.par_iter().enumerate().map(|(index, cfg)| {
        let mut rng = get_rand_from_seed(0xd6_4b_ef_fd_9f_c8_b2_a4);
        for _ in 0..index{
            rng.jump();
        }
        let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
        let mut mc = get_mc_from_config_sweep(cfg.sim_config().mc_config(), rng);
        //let mut hb = HeatBathSweep::new(rng);
        //let mut or1 = OverrelaxationSweepReverse::new();
        //let mut or2 = OverrelaxationSweepReverse::new();
        //let mut hm = HybrideMethode::new_empty();
        //hm.push_methods(&mut hb);
        //hm.push_methods(&mut or1);
        //hm.push_methods(&mut or2);
        
        
        let (sim_th, _t_exp) = thermalize_state(sim_init, &mut mc, &multi_pb, &observable::volume_obs).unwrap();
        let (state, _rng) = thermalize_with_e_field(sim_th, &multi_pb, mc.rng_owned()).unwrap();
        let _ = save_data_any(&state, &format!("sim_bin_{}_th_e.bin", index));
        
        
        const NUMBER_OF_MEASUREMENT: usize = 300;
        let (state, measure) = measure(state, NUMBER_OF_MEASUREMENT, &multi_pb).unwrap();
        
        let _ = save_data_any(&state, &format!("sim_bin_{}_e.bin", index));
        let _ = write_vec_to_file_csv(&measure, &format!("raw_measures_{}.csv", index));
        let _ = plot_data(&measure, DT, &format!("e_corr_{}.csv", index));
        pb.inc(1);
        (*cfg, measure)
    }).collect::<Vec<_>>();
    
    pb.finish();
    let _ = h.join();
}

type LeapFrogState<D> = SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<D>, D>, D>;

const DT: f64 = 0.1_f64;
const INTEGRATOR: SymplecticEulerRayon = SymplecticEulerRayon::new();

#[allow(clippy::useless_format)]
fn thermalize_with_e_field<D, Rng>(
    inital_state : LatticeStateDefault<D>,
    mp : &MultiProgress,
    rng: Rng,
) -> Result<(LeapFrogState<D>, Rng), SimulationError>
    where D: DimName + Eq,
    DefaultAllocator: Allocator<usize, D> + Allocator<Su3Adjoint, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Rng: rand::Rng,
{
    
    const STEPS: usize = 300;
    const CYCLE: usize = 40;
    
    let pb = mp.add(ProgressBar::new((CYCLE) as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix(&format!("th - e"));
    
    let mut hmc = HybridMonteCarloDiagnostic::new(DT, STEPS, INTEGRATOR, rng);
    
    let mut state = inital_state;
    for _ in 0..CYCLE{
        state = state.monte_carlo_step(&mut hmc)?;
        pb.inc(1);
    }
    
    let mut rng = hmc.rng_owned();
    let state_with_e = LatticeHamiltonianSimulationStateSyncDefault::new_random_e(state.lattice().clone(), state.beta(), state.link_matrix_owned(), &mut rng)?;
    let leap = state_with_e.simulate_to_leapfrog(DT, &INTEGRATOR)?;
    pb.inc(1);
    pb.finish_and_clear();
    Ok((leap.simulate_leap_n(DT, &INTEGRATOR, STEPS)?, rng))
}

#[allow(clippy::useless_format)]
fn measure(state_initial: LeapFrogState<U3>, number_of_measurement: usize, mp: &MultiProgress) -> Result<(LeapFrogState<U3>, Vec<Vec<f64>>), SimulationError> {
    
    let pb = mp.add(ProgressBar::new((number_of_measurement) as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix(&format!("sim"));
    
    let mut state = state_initial;
    let points = state.lattice().get_points().collect::<Vec<LatticePoint<_>>>();
    let mut vec = Vec::with_capacity(number_of_measurement);
    
    for _ in 0..number_of_measurement {
        let state_new = state.simulate_leap(DT, &INTEGRATOR)?;
        let vec_data = points.par_iter()
            .map(|pt| {
                observable::e_correletor(&state, &state_new, pt)
            })
            .collect::<Vec<f64>>();
        vec.push(vec_data);
        state = state_new;
        pb.inc(1);
    }
    
    pb.finish_and_clear();
    Ok((state, vec))
}


fn plot_data(data: &[Vec<f64>], delta_t: f64, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    
    let data_mean = data.iter().map(|el| statistics::mean(el)).collect::<Vec<f64>>();
    
    let mut y_min = data_mean[0];
    let mut y_max = data_mean[0];
    for el in &data_mean {
        y_min = y_min.min(*el);
        y_max = y_max.max(*el);
    }
    
    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0_f64..data_mean.len() as f64 * delta_t,
            (y_min).min(0_f64)..y_max * 1.1_f64,
        )?;
    
    chart.configure_mesh()
        .y_desc("Correlation E")
        .x_desc("t")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.draw_series(data_mean.iter().enumerate().map(|(index, el)| {
        Circle::new((index as f64 * delta_t , *el), 2, BLACK.filled())
    }))?;
    Ok(())
}
