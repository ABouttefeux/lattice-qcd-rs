use average_of_plaquette::{
    sim::*,
    config_scan::*,
    data_analysis::*,
    rng::*,
 };
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

fn main() {
    main_cross_check_volume();
}

fn get_values(min: f64, max: f64, pos: usize, number_of_pts: usize) -> f64 {
    min + (pos as f64) / (number_of_pts as f64 - 1_f64) * (max - min)
}

fn main_cross_check_volume() {
    
    let beta = 24_f64;
    let mut vec_dim = vec![];
    let number_of_data = 8;
    let mut therm_setp = vec![];
    
    for i in 0..number_of_data {
        let n: usize = (beta / get_values(0.5, 2.5, i, number_of_data)).round() as usize;
        vec_dim.push(n);
        therm_setp.push( n.pow(4) * 1_000);
    }
    
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Vector(vec_dim), // dim
        ScanPossibility::Default(beta), // Beta
    ).unwrap();
    let mc_cfg = MonteCarloConfigScan::new(
        ScanPossibility::Default(1),
        ScanPossibility::Default(0.1),
    ).unwrap();
    let sim_cfg = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Vector(therm_setp), //th setps
        ScanPossibility::Default(1_000), // renormn
        ScanPossibility::Default(250), // number_of_averages
        ScanPossibility::Default(1_000) //between av
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
    
    let result = array_config.par_iter().enumerate().map(|(index, cfg)| {
        let mut rng = get_rand_from_seed(0xd6_4b_ef_fd_9f_c8_b2_a4);
        for _ in 0..index{
            rng.jump();
        }
        let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
        let (av, sim_final, _) = run_simulation_with_progress_bar_volume(cfg.sim_config(), sim_init, &multi_pb, rng);
        let _ = save_data_n(cfg, &sim_final);
        pb.inc(1);
        (*cfg, av)
    }).collect();
    
    pb.finish();
    
    let _ = write_data_to_file_csv_with_n(&result);
    let _ = plot_data_volume(&result);
    let _ = h.join();
    
}
