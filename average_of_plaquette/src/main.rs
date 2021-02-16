use average_of_plaquette::{
    sim::*,
    config_scan::*,
    data_analysis::*,
    rng::*,
 };
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
/*use rand::{
    SeedableRng,
    rngs::StdRng
};*/
use rayon::prelude::*;


fn main() {
    let mut vec = vec![];
    let mut therm_setp = vec![];
    for i in 0..81 {
        let beta = i as f64 / 10_f64;
        vec.push(beta);
        let r = beta.floor() + 1_f64;
        if r < 2_f64 {
            therm_setp.push(1_000_000)
        }
        else if r < 4_f64 {
            therm_setp.push(4_000_000)
        }
        else if r < 5_f64 {
            therm_setp.push(10_000_000)
        }
        else if r < 6_f64 {
            therm_setp.push(25_000_000)
        }
        else {
            therm_setp.push(50_000_000)
        }
    }
    
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Default(5),
        ScanPossibility::Vector(vec),
    ).unwrap();
    let mc_cfg = MonteCarloConfigScan::new(
        ScanPossibility::Default(1),
        ScanPossibility::Default(0.5),
    ).unwrap();
    let sim_cfg = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Vector(therm_setp), //th setps
        ScanPossibility::Default(500), // renormn
        ScanPossibility::Default(250), // number_of_averages
        ScanPossibility::Default(5_000) //between av
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
    
    let result = array_config.par_iter().map(|cfg| {
        let mut rng = get_rand_from_seed(0xd64beffd9fc8b2a4);
        let sim_init = generate_state_default(cfg.lattice_config(), &mut rng);
        let (av, sim_final) = run_simulation_with_progress_bar(cfg.sim_config(), sim_init, &multi_pb, &mut rng);
        let _ = save_data(cfg, &sim_final);
        pb.inc(1);
        (*cfg, av)
    }).collect();
    
    pb.finish();
    
    let _ = write_data_to_file_csv(&result);
    let _ = plot_data(&result);
    let _ = h.join();
}
