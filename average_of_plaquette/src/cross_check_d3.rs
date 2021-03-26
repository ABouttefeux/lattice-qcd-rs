use average_of_plaquette::{
    sim::*,
    config_scan::*,
    data_analysis::*,
    rng::*,
    observable,
 };
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

fn main() {
    main_cross_check_volume();
}

pub fn get_values(min: f64, max: f64, pos: usize, number_of_pts: usize) -> f64 {
    min + (pos as f64) / (number_of_pts as f64 - 1_f64) * (max - min)
}

pub fn get_vect_dim(number_of_data: usize, beta: f64, low_pt: f64, high_pt: f64) -> Vec<usize> {
    let mut vec_dim = vec![];
    for i in 0..number_of_data {
        //0.5, 2.5
        let n: usize = (beta / get_values(low_pt, high_pt, i, number_of_data)).round() as usize;
        if !vec_dim.iter().any(|el|* el == n) {
            vec_dim.push(n);
        }
    }
    vec_dim
}


const BETA: f64 = 24_f64;
const NUMBER_OF_DATA: usize = 16;
const LOW_PT: f64 = 0.25_f64;
const HIGH_PT: f64 = 2.5_f64;

fn main_cross_check_volume() {
    
    let beta = BETA;
    let number_of_data = NUMBER_OF_DATA;
    
    let vec_dim = get_vect_dim(number_of_data, beta, LOW_PT, HIGH_PT);
    
    
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64), //size
        ScanPossibility::Vector(vec_dim), // dim
        ScanPossibility::Default(beta), // Beta
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
    //println!("{:}", serde_json::to_string_pretty( &config).unwrap());
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
        for _ in 0..index {
            rng.jump();
        }
        let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
        let mut mc = get_mc_from_config_sweep(cfg.sim_config().mc_config(), rng);
        let (sim_th, t_exp) = thermalize_state(sim_init, &mut mc, &multi_pb, &observable::volume_obs).unwrap();
        //let (av, sim_final, _) = run_simulation_with_progress_bar_volume(cfg.sim_config(), sim_th, &multi_pb, rng);
        let _ = save_data_n(cfg, &sim_th, &"_th");
        
        let(sim_final, result) = simulation_gather_measurement(sim_th, &mut mc, &multi_pb, &observable::volume_obs, cfg.sim_config().number_of_steps_between_average(), cfg.sim_config().number_of_averages()).unwrap();
        let _ = write_vec_to_file_csv(&result, &format!("raw_measures_{}.csv", cfg.lattice_config().lattice_number_of_points() ));
        let _ = save_data_n(cfg, &sim_final, &"");
        pb.inc(1);
        (*cfg, t_exp)
    }).collect::<Vec<_>>();
    
    pb.finish();
    
    //let _ = write_data_to_file_csv_with_n(&result);
    //let _ = plot_data_volume(&result);
    let _ = h.join();
    
}
