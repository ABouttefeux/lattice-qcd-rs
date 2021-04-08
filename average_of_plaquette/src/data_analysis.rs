
//! Data analysis and saving to files

use lattice_qcd_rs::{
    Real,
    lattice::{Direction, DirectionList},
    dim::DimName,
};
use std::vec::Vec;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::fs::File;
use super::config::Config;
use plotters::prelude::*;
use lattice_qcd_rs::simulation::LatticeStateDefault;
use std::io::prelude::*;
use na::{
    DefaultAllocator,
    VectorN,
    base::allocator::Allocator,
};


/// Data point for an average
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

/// Vector of average
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AverageData {
    data: Vec<AverageDataPoint>,
    final_average: Option<Real>,
}

impl AverageData{
    pub const fn new(data: Vec<AverageDataPoint>) -> Self {
        Self {data, final_average: None}
    }
    
    /// Average over all average data point
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

/// Write result to csv "result.csv"
#[allow(clippy::ptr_arg)]
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

/// Write result to csv "result.csv"
#[allow(clippy::ptr_arg)]
pub fn write_data_to_file_csv_with_n(data: &Vec<(Config, AverageData)>) -> std::io::Result<()> {
    let file = File::create("result.csv")?;
    let mut wtr = csv::Writer::from_writer(file);
    for (cfg, av) in data {
        let mut data_block = vec![
            cfg.lattice_config().lattice_beta(),
            cfg.lattice_config().lattice_number_of_points() as f64,
            av.clone().final_average()
        ];
        let mut val = av.data().iter().map(|el| el.average()).collect();
        data_block.append(&mut val);
        wtr.serialize(& data_block)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn write_vec_to_file_csv<T : Serialize>(data: &[T], name: &str) -> std::io::Result<()> {
    let file = File::create(name)?;
    write_csv_to_file(data, file)
}

pub fn write_csv_to_file<T : Serialize>(data: &[T], file: File) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_writer(file);
    for data_el in data {
        wtr.serialize(data_el)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Plot data to "plot_beta.svg"
pub fn plot_data_average(data: &[(Config, AverageData)]) -> Result<(), Box<dyn std::error::Error>> {
    
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

pub fn plot_data_volume(data: &[(Config, AverageData)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("plot_beta_volume.svg", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let betas_n = data.iter().map(|(cfg, _)| {
        cfg.lattice_config().lattice_beta() / cfg.lattice_config().lattice_number_of_points() as f64
    }).collect::<Vec<f64>>();
    let avg = data.iter().map(|(_, avg)| avg.clone().final_average()).collect::<Vec<f64>>();
    
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0_f64..betas_n.last().unwrap() * 1.1_f64,
            0_f64..avg.last().unwrap() * 1.1_f64
        )?;
    
    chart.configure_mesh()
        .x_desc("Beta / N")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.draw_series(betas_n.iter().zip(avg.iter()).map(|(beta, avg)| {
        Circle::new((*beta, *avg), 2, BLACK.filled())
    }))?;
    Ok(())
}

pub fn plot_data_auto_corr(auto_corr: &[f64], name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0_f64..auto_corr.len() as f64,
            *auto_corr.iter().min_by(|i, j| i.partial_cmp(j).unwrap()).unwrap()..*auto_corr.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap()
        )?;
    
    chart.configure_mesh()
        .y_desc("Auto Correlation")
        .x_desc("steps")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.draw_series(auto_corr.iter().enumerate().map(|(x, y)| {
        Circle::new((x as f64, *y), 2, BLACK.filled())
    }))?;
    Ok(())
}


/// save configuration to `format!("sim_b_{}.bin", cfg.lattice_config().lattice_beta())`
pub fn save_data<D>(cfg: &Config, state: &LatticeStateDefault<D>) -> std::io::Result<()>
    where D: DimName + Serialize,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    let encoded: Vec<u8> = bincode::serialize(&state).unwrap();
    let mut file = File::create(format!("sim_b_{}.bin", cfg.lattice_config().lattice_beta()))?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn save_data_n<D>(cfg: &Config, state: &LatticeStateDefault<D>, sufix: &str) -> std::io::Result<()>
    where D: DimName + Serialize,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    let encoded: Vec<u8> = bincode::serialize(&state).unwrap();
    let mut file = File::create(format!("sim_b_{}_n_{}{}.bin", cfg.lattice_config().lattice_beta(), cfg.lattice_config().lattice_number_of_points(), sufix))?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn save_data_any<T>(state: &T, file_name: &str) -> std::io::Result<()>
    where T: Serialize,
{
    let encoded: Vec<u8> = bincode::serialize(&state).unwrap();
    let mut file = File::create(file_name)?;
    file.write_all(&encoded)?;
    Ok(())
}
