use super::{
    config::*,
    config_scan::*,
};

#[test]
fn config_scan() {
    
    let cfg_l = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Default(12),
        ScanPossibility::Vector(vec![0.5, 1.0])
    ).unwrap();
    assert_eq!(cfg_l.len_option(), Some(2));
    let mc_cfg = MonteCarloConfigScan::new(
        ScanPossibility::Default(20),
        ScanPossibility::Default(0.00001)
    ).unwrap();
    assert_eq!(mc_cfg.len_option(), None);
    let sim_cfg = SimConfigScan::new(
        mc_cfg.clone(),
        ScanPossibility::Default(10_000),
        ScanPossibility::Default(100),
        ScanPossibility::Default(100),
        ScanPossibility::Default(1_000)
    ).unwrap();
    
    let config = ConfigScan::new(cfg_l.clone(), sim_cfg.clone()).unwrap();
    assert_eq!(config.len_option(), Some(2));
    assert_eq!(config.get_all_config().len(), 2);
    
    let cfg_l_2 = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Default(12),
        ScanPossibility::Default(0.5)
    ).unwrap();
    let config_2 = ConfigScan::new(cfg_l_2, sim_cfg).unwrap();
    assert_eq!(config_2.get_all_config().len(), 1);
    
    let cfg_l_3 = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Vector(vec![23]),
        ScanPossibility::Vector(vec![0.5, 1.0])
    );
    
    assert!(! cfg_l_3.is_some());
    
    let cfg_l_4 = LatticeConfigScan::new(
        ScanPossibility::Default(1000_f64),
        ScanPossibility::Vector(vec![1, 4]),
        ScanPossibility::Vector(vec![0.5, 1.0])
    );
    
    assert!(! cfg_l_4.is_some());
    
    let sim_cfg_2 = SimConfigScan::new(
        mc_cfg.clone(),
        ScanPossibility::Vector(vec![10_000, 100_000]),
        ScanPossibility::Default(100),
        ScanPossibility::Default(100),
        ScanPossibility::Default(1_000)
    ).unwrap();
    
    let config_3 = ConfigScan::new(cfg_l.clone(), sim_cfg_2);
    assert!(config_3.is_some());
    
    let sim_cfg_3 = SimConfigScan::new(
        mc_cfg,
        ScanPossibility::Vector(vec![10_000, 100_000, 10_000]),
        ScanPossibility::Default(100),
        ScanPossibility::Default(100),
        ScanPossibility::Default(1_000)
    ).unwrap();
    
    let config_4 = ConfigScan::new(cfg_l, sim_cfg_3);
    assert!(! config_4.is_some());

}
