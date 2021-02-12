
use serde::{Serialize, Deserialize};

use lattice_qcd_rs::{
    Real,
};
use super::config::*;


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScanPossibility<Type> {
    Default(Type),
    Vector(Vec<Type>),
}

impl<T> ScanPossibility<T>{
    pub fn len(&self) -> usize {
        match self {
            ScanPossibility::Default(_) => 1,
            ScanPossibility::Vector(v) => v.len(),
        }
    }
    
    pub fn len_option(&self) -> Option<usize> {
        match self {
            ScanPossibility::Default(_) => None,
            ScanPossibility::Vector(v) => Some(v.len()),
        }
    }
    
    pub fn is_empty(&self) -> bool {
        match self {
            ScanPossibility::Default(_) => false,
            ScanPossibility::Vector(v) => v.is_empty(),
        }
    }
    
    pub fn get(&self, pos: usize) -> Option<&T> {
        match self {
            ScanPossibility::Default(t) => Some(&t),
            ScanPossibility::Vector(v) => v.get(pos),
        }
    }
    
    pub fn all<F>(&self, mut f: F) -> bool
        where F: FnMut(&T) -> bool,
    {
        match self {
            ScanPossibility::Default(t) => f(&t),
            ScanPossibility::Vector(v) => v.iter().all(f),
        }
    }
}

impl<T> From<T> for ScanPossibility<T> {
    fn from(t: T) -> Self{
        ScanPossibility::Default(t)
    }
}

impl<T> From<Vec<T>> for ScanPossibility<T> {
    fn from(t: Vec<T>) -> Self{
        ScanPossibility::Vector(t)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatticeConfigScan {
    lattice_beta: ScanPossibility<Real>,
    lattice_number_of_points: ScanPossibility<usize>,
    lattice_size: ScanPossibility<Real>,
}

impl LatticeConfigScan {
    
    pub fn new(lattice_size: ScanPossibility<Real>, lattice_number_of_points: ScanPossibility<usize>, lattice_beta: ScanPossibility<Real>) -> Option<Self> {
        let s = Self {lattice_size, lattice_number_of_points, lattice_beta};
        if s.is_valide(){
            Some(s)
        }
        else{
            None
        }
    }
    
    pub const fn lattice_beta(&self) -> &ScanPossibility<Real> {
        &self.lattice_beta
    }
    
    pub const fn lattice_number_of_points(&self) -> &ScanPossibility<usize> {
        &self.lattice_number_of_points
    }
    
    pub const fn lattice_size(&self) -> &ScanPossibility<Real> {
        &self.lattice_size
    }
    
    pub fn is_valide_length(&self) -> bool {
        let len = self.get_len_array();
        let iter = len.iter().filter_map(|el| *el);
        iter.clone().max() == iter.min()
    }
    
    pub fn is_data_valide(&self) -> bool {
        for i in 0..self.lattice_size.len(){
            if *self.lattice_size.get(i).unwrap() < 0_f64{
                return false;
            }
        }
        for i in 0..self.lattice_number_of_points.len(){
            if *self.lattice_number_of_points.get(i).unwrap() < 2 {
                return false;
            }
        }
        true
    }
    
    pub fn is_valide(&self) -> bool {
        self.is_valide_length() && self.is_data_valide()
    }
    
    fn get_len_array(&self) -> [Option<usize>; 3] {
        [
            self.lattice_beta.len_option(),
            self.lattice_number_of_points.len_option(),
            self.lattice_size.len_option()
        ]
    }
    
    pub fn len_option(&self) -> Option<usize> {
        let len = self.get_len_array();
        for i in len.iter() {
            if i.is_some() {
                return *i;
            }
        }
        None
    }
    
    pub fn get(&self, pos: usize) -> Option<LatticeConfig> {
        Some(LatticeConfig::new(
            *self.lattice_size.get(pos)?,
            *self.lattice_number_of_points.get(pos)?,
            *self.lattice_beta.get(pos)?,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimConfigScan {
    number_of_thermalisation: ScanPossibility<usize>,
    number_between_renorm: ScanPossibility<usize>,
    number_of_averages: ScanPossibility<usize>,
    number_of_steps_between_average: ScanPossibility<usize>,
    mc_config: MonteCarloConfigScan,
}

impl SimConfigScan {
    
    pub fn new(
        mc_config: MonteCarloConfigScan,
        number_of_thermalisation: ScanPossibility<usize>,
        number_between_renorm: ScanPossibility<usize>,
        number_of_averages: ScanPossibility<usize>,
        number_of_steps_between_average: ScanPossibility<usize>
    ) -> Option<Self> {
        let s = Self{
            mc_config,
            number_of_thermalisation,
            number_between_renorm,
            number_of_averages,
            number_of_steps_between_average,
        };
        if s.is_valide(){
            Some(s)
        }
        else{
            None
        }
    }
    
    pub fn new_raw(
        number_of_rand: ScanPossibility<usize>,
        spread: ScanPossibility<Real>,
        number_of_thermalisation: ScanPossibility<usize>,
        number_between_renorm: ScanPossibility<usize>,
        number_of_averages: ScanPossibility<usize>,
        number_of_steps_between_average: ScanPossibility<usize>
    ) -> Option<Self> {
        
        let mc = MonteCarloConfigScan::new(number_of_rand, spread)?;
        Self::new(
            mc,
            number_of_thermalisation,
            number_between_renorm,
            number_of_averages,
            number_of_steps_between_average
        )
    }
    
    pub const fn mc_config(&self) -> &MonteCarloConfigScan {
        &self.mc_config
    }
    
    pub const fn number_of_thermalisation(&self) -> &ScanPossibility<usize> {
        &self.number_of_thermalisation
    }
    
    pub const fn number_between_renorm(&self) -> &ScanPossibility<usize> {
        &self.number_between_renorm
    }
    
    pub const fn number_of_averages(&self) -> &ScanPossibility<usize> {
        &self.number_of_averages
    }
    
    pub const fn number_of_steps_between_average(&self) -> &ScanPossibility<usize> {
        &self.number_of_steps_between_average
    }
    
    pub fn is_valide_length(&self) -> bool {
        let len = self.get_len_array();
        let iter = len.iter().filter_map(|el| *el);
        self.mc_config.is_valide_length() && iter.clone().max() == iter.min()
    }
    
    pub fn is_data_valide(&self) -> bool {
        // number_of_thermalisation % number_between_renorm != 0 ||
        // number_of_steps_between_average % number_between_renorm != 0
        for i in 0..self.number_of_thermalisation.len(){
            if let Some(mo) = self.number_between_renorm.get(i){
                if self.number_of_thermalisation.get(i).unwrap() % mo != 0 {
                    return false
                }
            }
            else {
                return false
            }
        }
        for i in 0..self.number_of_steps_between_average.len(){
            if let Some(mo) = self.number_between_renorm.get(i){
                if self.number_of_steps_between_average.get(i).unwrap() % mo != 0 {
                    return false
                }
            }
            else {
                return false
            }
        }
        self.mc_config.is_data_valide()
    }
    
    pub fn is_valide(&self) -> bool {
        self.is_valide_length() && self.is_data_valide()
    }
    
    fn get_len_array(&self) -> [Option<usize>; 5] {
        [
            self.number_of_thermalisation.len_option(),
            self.number_between_renorm.len_option(),
            self.number_of_averages.len_option(),
            self.number_of_steps_between_average.len_option(),
            self.mc_config.len_option()
        ]
    }
    
    pub fn len_option(&self) -> Option<usize> {
        let len = self.get_len_array();
        for i in len.iter() {
            if i.is_some(){
                return *i;
            }
        }
        None
    }
    
    pub fn get(&self, pos: usize) -> Option<SimConfig> {
        Some(SimConfig::new(
            self.mc_config.get(pos)?,
            *self.number_of_thermalisation.get(pos)?,
            *self.number_between_renorm.get(pos)?,
            *self.number_of_averages.get(pos)?,
            *self.number_of_steps_between_average.get(pos)?,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloConfigScan {
    number_of_rand: ScanPossibility<usize>,
    spread: ScanPossibility<Real>,
}

impl MonteCarloConfigScan {
    pub fn new(number_of_rand: ScanPossibility<usize>, spread: ScanPossibility<Real>) -> Option<Self> {
        let s = Self{number_of_rand, spread};
        if s.is_valide(){
            Some(s)
        }
        else{
            None
        }
    }
    
    pub const fn number_of_rand(&self) -> &ScanPossibility<usize> {
        &self.number_of_rand
    }
    
    pub const fn spread(&self) -> &ScanPossibility<Real> {
        &self.spread
    }
    
    fn get_len_array(&self) -> [Option<usize>; 2] {
        [
            self.number_of_rand.len_option(),
            self.spread.len_option(),
        ]
    }
    
    pub fn len_option(&self) -> Option<usize> {
        let len = self.get_len_array();
        for i in len.iter() {
            if i.is_some(){
                return *i;
            }
        }
        None
    }
    
    pub fn is_valide_length(&self) -> bool {
        let len = self.get_len_array();
        let iter = len.iter().filter_map(|el| *el);
        iter.clone().max() == iter.min()
    }
    
    pub fn is_data_valide(&self) -> bool {
        for i in 0..self.number_of_rand.len(){
            if *self.number_of_rand.get(i).unwrap() == 0 {
                return false;
            }
        }
        for i in 0..self.spread.len(){
            let s = *self.spread.get(i).unwrap();
            if s <= 0_f64 || s >= 1_f64 {
                return false;
            }
        }
        true
    }
    
    pub fn is_valide(&self) -> bool {
        self.is_valide_length() && self.is_data_valide()
    }
    
    pub fn get(&self, pos: usize) -> Option<MonteCarloConfig> {
        Some(MonteCarloConfig::new(
            *self.number_of_rand.get(pos)?,
            *self.spread.get(pos)?,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigScan {
    lattice_config: LatticeConfigScan,
    sim_config: SimConfigScan,
}

impl ConfigScan {
    pub fn new(lattice_config: LatticeConfigScan, sim_config: SimConfigScan) -> Option<Self>{
        let s = Self {lattice_config, sim_config};
        if s.is_valide(){
            Some(s)
        }
        else{
            None
        }
    }
    
    pub const fn lattice_config(&self) -> &LatticeConfigScan {
        &self.lattice_config
    }
    
    pub const fn sim_config(&self) -> &SimConfigScan {
        &self.sim_config
    }
    
    fn get_len_array(&self) -> [Option<usize>; 2] {
        [
            self.lattice_config.len_option(),
            self.sim_config.len_option(),
        ]
    }
    
    pub fn len_option(&self) -> Option<usize> {
        let len = self.get_len_array();
        for i in len.iter() {
            if i.is_some(){
                return *i;
            }
        }
        None
    }
    
    pub fn is_valide_length(&self) -> bool {
        let len = self.get_len_array();
        let iter = len.iter().filter_map(|el| *el);
        self.lattice_config.is_valide_length() &&
            self.sim_config.is_valide_length() &&
            iter.clone().max() == iter.min()
    }
    
    pub fn is_data_valide(&self) -> bool {
        self.lattice_config.is_data_valide() &&
            self.sim_config.is_data_valide()
    }
    
    pub fn is_valide(&self) -> bool {
        self.is_valide_length() && self.is_data_valide()
    }
    
    pub fn get(&self, pos: usize) -> Option<Config> {
        Some(Config::new(
            self.lattice_config.get(pos)?,
            self.sim_config.get(pos)?,
        ))
    }
    
    pub fn get_all_config(&self) -> Vec<Config> {
        match self.len_option() {
            None => vec![self.get(0).unwrap()],
            Some(len) => {
                let mut vec = vec![];
                for i in 0..len {
                    vec.push(self.get(i).unwrap());
                }
                vec
            }
        }
    }
}
