use average_of_plaquette::{
    sim::*,
    config_scan::*,
    data_analysis::*,
    rng::*,
    observable,
    plot_console::*,
 };
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use lattice_qcd_rs::{
    field::{
        Su3Adjoint,
    },
    simulation::*,
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
    Complex,
    ComplexField,
};
use plotters::prelude::*;
use rustfft::FftPlanner;

fn main() {
    //test_fft();
    main_cross_with_e();
}

fn test_fft() {
    let mut data = (0..16_384_0).map(|index| {
        Complex::from(((index as f64) / 500_f64).cos())
    }).collect::<Vec<_>>();
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(data.len());
    let _ = plot_data_fft(&data, DT, &"test_data.svg");
    
    fft.process(&mut data);
    let _ = plot_data_fft(&data[0..data.len()/2], DT, &"test_fft.svg");
}

fn main_cross_with_e() {
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1_f64), //size
        ScanPossibility::Default(4), // dim
        ScanPossibility::Default(24_f64), // Beta
    ).unwrap();
    let mc_cfg = MonteCarloConfigScan::new(
        ScanPossibility::Default(1),
        ScanPossibility::Default(0.1),
    ).unwrap();
    let sim_cfg = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Default(10_000), //th steps (unused)
        ScanPossibility::Default(1), // renormn (unused)
        ScanPossibility::Default(1_000), // number_of_averages (unused)
        ScanPossibility::Default(200) //between av (unused)
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
    
    const SEED: u64 = 0xd6_4b_ef_fd_9f_c8_b2_a4;
    let _result_all_sim = array_config.par_iter().enumerate().map(|(index, cfg)| {
        let mut rng = get_rand_from_seed(SEED);
        for _ in 0..index{
            rng.jump();
        }
        let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
        let mut mc = get_mc_from_config_sweep(cfg.sim_config().mc_config(), rng);
        /*
        let mut hb = HeatBathSweep::new(rng);
        let mut or1 = OverrelaxationSweepReverse::new();
        let mut or2 = OverrelaxationSweepReverse::new();
        let mut hm = HybrideMethode::new_empty();
        hm.push_methods(&mut hb);
        hm.push_methods(&mut or1);
        hm.push_methods(&mut or2);
        */
        
        let (sim_th, _t_exp) = thermalize_state(sim_init, &mut mc, &multi_pb, &observable::volume_obs).unwrap();
        let (state, _rng) = thermalize_with_e_field(sim_th, &multi_pb, mc.rng_owned()).unwrap();
        let _ = save_data_any(&state, &format!("sim_bin_{}_th_e.bin", index));
        
        const NUMBER_OF_MEASUREMENT: usize = 3_000_000;
        let (state, measure) = measure(state, NUMBER_OF_MEASUREMENT, &multi_pb).unwrap();
        
        let _ = save_data_any(&state, &format!("sim_bin_{}_e.bin", index));
        let _ = write_vec_to_file_csv(&measure, &format!("raw_measures_{}.csv", index));
        let _ = plot_data(&measure, DT, &format!("e_corr_{}.svg", index));
        
        let mut measure_fft = measure.iter().map(|el| statistics::mean(el).into()).collect::<Vec<Complex<f64>>>();
        
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(measure_fft.len());
        
        fft.process(&mut measure_fft);
        let _ = plot_data_fft(&measure_fft[..measure_fft.len() / 2], DT, &format!("e_corr_{}_fft.svg", index));
        pb.inc(1);
        (*cfg, measure)
    }).collect::<Vec<_>>();
    
    pb.finish();
    let _ = h.join();
}


//const DT: f64 = 0.000_01_f64; // to big
//const DT: f64 = 0.000_000_1_f64; // OK ?
//const DT: f64 = 0.000_000_000_1_f64; // too small
const DT: f64 = 0.000_001_f64; // OK ?


const INTEGRATOR: SymplecticEulerRayon = SymplecticEulerRayon::new();

#[allow(clippy::useless_format)]
fn thermalize_with_e_field<D, Rng>(
    inital_state : LatticeStateDefault<D>,
    mp : &MultiProgress,
    rng: Rng,
) -> Result<(LeapFrogStateDefault<D>, Rng), SimulationError>
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
    for _ in 0..CYCLE {
        state = state.monte_carlo_step(&mut hmc)?;
        state.normalize_link_matrices();
        pb.set_message(&format!("{:.6}   ", hmc.prob_replace_last()));
        pb.inc(1);
    }
    
    let mut rng = hmc.rng_owned();
    let state_with_e = LatticeHamiltonianSimulationStateSyncDefault::new_random_e(state.lattice().clone(), state.beta(), state.link_matrix_owned(), &mut rng)?;
    let leap = state_with_e.simulate_to_leapfrog(DT, &INTEGRATOR)?;
    pb.inc(1);
    pb.finish_and_clear();
    //Ok((leap.simulate_leap_n(DT, &INTEGRATOR, STEPS)?, rng))
    Ok((leap, rng))
}

#[allow(clippy::useless_format)]
fn measure(state_initial: LeapFrogStateDefault<U3>, number_of_measurement: usize, mp: &MultiProgress) -> Result<(LeapFrogStateDefault<U3>, Vec<Vec<f64>>), SimulationError> {
    
    let pb = mp.add(ProgressBar::new((number_of_measurement) as u64));
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        get_pb_template()
    ));
    pb.set_prefix(&format!("simulating"));
    
    let mut state = state_initial.clone();
    let points = state.lattice().get_points().collect::<Vec<LatticePoint<_>>>();
    let mut vec = Vec::with_capacity(number_of_measurement + 1);
    
    let vec_data = points.par_iter()
        .map(|pt| {
            observable::e_correletor(&state_initial, &state_initial, pt).unwrap()
        })
        .collect::<Vec<f64>>();
    vec.push(vec_data);
    
    let mut vec_plot = vec![];
    let mut y_min = 0_f64;
    let mut y_max = 0_f64;
    
    for i in 0..number_of_measurement {
        let mut state_new = state.simulate_leap(DT, &INTEGRATOR)?;
        if i % 200 == 0 {
            pb.set_message(&format!(
                "H {:.6} - G {:.6} ",
                state_new.get_hamiltonian_total(),
                state_new.e_field().get_gauss_sum_div(state_new.link_matrix(), state_new.lattice()).unwrap(),
            ));
            state_new.state_mut().lattice_state_mut().normalize_link_matrices();
            
            /*
            let new_e = state_new.e_field().project_to_gauss(state_new.link_matrix(), state_new.lattice())
                .ok_or(SimulationError::NotValide)?;
            // let statement because of mutable_borrow_reservation_conflict
            // (https://github.com/rust-lang/rust/issues/59159)
            state_new.set_e_field(new_e);
            */
            
        }
        let vec_data = points.par_iter()
            .map(|pt| {
                observable::e_correletor(&state_initial, &state_new, pt).unwrap()
            })
            .collect::<Vec<f64>>();
        vec.push(vec_data);
        
        const PLOT_COUNT: usize = 1_000;
        
        if i % PLOT_COUNT == 0 {
            // TODO clean, move to function
            let last_data = statistics::mean(vec.last().unwrap());
            vec_plot.push(last_data);
            if vec_plot.len() > 1 {
                y_min = y_min.min(last_data);
                y_max = y_max.max(last_data);
                if y_min < y_max {
                    let _ = draw_chart(
                        &TextDrawingBackend(vec![PixelState::Empty; 5000]).into_drawing_area(),
                        0_f64..((vec_plot.len() - 1) * PLOT_COUNT) as f64 * DT,
                        y_min..y_max,
                        vec_plot.iter().enumerate().map(|(index, el)| ((index * PLOT_COUNT) as f64 * DT, *el)),
                        "E Corr"
                    );
                    let _ = console::Term::stderr().move_cursor_up(30);
                }
            }
            else {
                y_min = last_data;
                y_max = last_data;
            }
        }
        
        state = state_new;
        pb.inc(1);
    }
    
    pb.finish_and_clear();
    let _ = console::Term::stderr().move_cursor_down(30);
    Ok((state, vec))
}

const STEP_BY: usize = 1_000;

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
            (y_min).min(0_f64)..y_max,
        )?;
    
    chart.configure_mesh()
        .y_desc("Correlation E")
        .x_desc("t")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    
    
    /*
    chart.draw_series(data_mean.iter().enumerate().step_by(STEP_BY).map(|(index, el)| {
        Circle::new((index as f64 * delta_t , *el), 2, BLACK.filled())
    }))?;
    */
    
    chart.draw_series(
        LineSeries::new(data_mean.iter().enumerate().step_by(STEP_BY).map(|(index, el)| {
            (index as f64 * delta_t , *el)}),
            BLACK.filled(),
    ))?;
    
    Ok(())
}

fn plot_data_fft(data: &[Complex<f64>], delta_t: f64, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut y_min = data[0].modulus() / (data.len() as f64).sqrt();
    let mut y_max = data[0].modulus() / (data.len() as f64).sqrt();
    for el in data {
        y_min = y_min.min(el.modulus() / (data.len() as f64).sqrt());
        y_max = y_max.max(el.modulus() / (data.len() as f64).sqrt());
    }
    
    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .right_y_label_area_size(60)
        .build_cartesian_2d(
            (delta_t..data.len() as f64 * delta_t).log_scale(),
            (y_min.max(1E-15)..y_max * 1.1_f64).log_scale(),
        )?
        .set_secondary_coord(
            (delta_t..data.len() as f64 * delta_t).log_scale(),
            -std::f64::consts::PI..std::f64::consts::PI
        );
    
    chart.configure_mesh()
        .y_desc("Modulus")
        .x_desc("w")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.configure_secondary_axes()
        .axis_desc_style(("sans-serif", 15))
       .y_desc("Argument")
       .draw()?;
    
    /*
    chart.draw_series(data.iter().enumerate().step_by(STEP_BY).map(|(index, el)| {
        Circle::new((index as f64 * delta_t , el.modulus() / (data.len() as f64).sqrt() ), 2, BLACK.filled())
    }))?;
    
    chart.draw_secondary_series(data.iter().enumerate().step_by(STEP_BY).map(|(index, el)| {
        Circle::new((index as f64 * delta_t , el.argument()), 2, RED.filled())
    }))?;
    */
    
    chart.draw_series(
        LineSeries::new(
            data.iter().enumerate().step_by(1).map(|(index, el)| {
                (index as f64 * delta_t, el.modulus().max(1E-15) / (data.len() as f64).sqrt())
            }),
            &BLACK,
        )
    )?;
    
    chart.draw_secondary_series(
        LineSeries::new(
            data.iter().enumerate().step_by(1).map(|(index, el)| {
                (index as f64 * delta_t, el.argument())
            }),
            &RED,
        )
    )?;
    
    Ok(())
}
