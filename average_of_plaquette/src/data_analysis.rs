

use lattice_qcd_rs::{
    Real,
};
use std::vec::Vec;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::fs::File;
use super::config::Config;
use plotters::prelude::*;
use lattice_qcd_rs::simulation::LatticeStateDefault;
use std::io::prelude::*;


#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct AverageDataPoint {
    average: Real,
    sim_number: usize,
}

impl AverageDataPoint {
    pub const fn new(average: Real, sim_number: usize) -> Self {
        Self {average, sim_number}
    }
    
    pub const fn average(&self) -> Real {
        self.average
    }
    
    pub const fn sim_number(&self) -> usize {
        self.sim_number
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AverageData {
    data: Vec<AverageDataPoint>,
    final_average: Option<Real>,
}

impl AverageData{
    pub const fn new(data: Vec<AverageDataPoint>) -> Self {
        Self {data, final_average: None}
    }
    
    pub fn final_average(&mut self) -> Real {
        match self.final_average {
            Some(i) => i,
            None => self.compute_average()
        }
    }
    
    fn compute_average(&mut self) -> Real {
        let av = self.data.par_iter().map(|el| el.average()).sum::<Real>() / (self.data.len() as f64);
        self.final_average = Some(av);
        av
    }
    
    pub const fn data(&self) -> &Vec<AverageDataPoint> {
        &self.data
    }
}

pub fn write_data_to_file_csv(data: &Vec<(Config, AverageData)>) -> std::io::Result<()> {
    
    let file = File::create("result.csv")?;
    let mut wtr = csv::Writer::from_writer(file);
    for (cfg, av) in data {
        let mut data_block = vec![cfg.lattice_config().lattice_beta(), av.clone().final_average()];
        let mut val = av.data().iter().map(|el| el.average()).collect();
        data_block.append(&mut val);
        wtr.serialize(& data_block)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn plot_data(data: &[(Config, AverageData)]) -> Result<(), Box<dyn std::error::Error>> {
    
    let betas = data.iter().map(|(cfg, _)| cfg.lattice_config().lattice_beta()).collect::<Vec<f64>>();
    let avg = data.iter().map(|(_, avg)| avg.clone().final_average()).collect::<Vec<f64>>();
    
    plot_data_beta(&betas, &avg)
}


fn plot_data_beta(betas: &[f64], avg: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("plot_beta.svg", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0_f64..betas.last().unwrap() * 1.1_f64,
            0.9_f64 * avg.first().unwrap()..avg.last().unwrap() * 1.1_f64
        )?;
    
    chart.configure_mesh()
        .y_desc("Average of Plaquette")
        .x_desc("Beta")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.draw_series(betas.iter().zip(avg.iter()).map(|(beta, avg)| {
        Circle::new((*beta, *avg), 2, BLACK.filled())
    }))?;
    Ok(())
}

pub fn save_data(cfg: &Config, state: &LatticeStateDefault) -> std::io::Result<()> {
    let encoded: Vec<u8> = bincode::serialize(&state).unwrap();
    let mut file = File::create(format!("sim_b_{}.bin", cfg.lattice_config().lattice_beta()))?;
    file.write_all(&encoded)?;
    Ok(())
}
