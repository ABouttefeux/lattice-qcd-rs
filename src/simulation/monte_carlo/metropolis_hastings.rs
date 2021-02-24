
use super::{
    MonteCarlo,
    MonteCarloDefault,
    super::{
        super::{
            Real,
            Complex,
            field::{
                LinkMatrix,
            },
            su3,
            lattice::{
                LatticePoint,
                LatticeLinkCanonical,
                Direction,
                LatticeElementToIndex,
                LatticeLink,
                LatticeCyclique,
                DirectionList,
            }
        },
        state::{
            LatticeState,
            LatticeStateNew,
            LatticeStateDefault,
        },
        SimulationError,
    },
};
use std::marker::PhantomData;
use rand_distr::Distribution;
use na::ComplexField;
use rayon::prelude::*;
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};

/// Metropolis Hastings algorithm. Very slow, use [`MetropolisHastingsDeltaDiagnostic`] instead.
pub struct MetropolisHastings<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    number_of_update: usize,
    spread: Real,
    _phantom: PhantomData<(State, D)>,
}

impl<State, D> MetropolisHastings<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            _phantom: PhantomData,
        })
    }
}

impl<State, D> MonteCarloDefault<State, D> for MetropolisHastings<State, D>
    where State: LatticeState<D> + LatticeStateNew<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(state.lattice().clone(), state.beta(), LinkMatrix::new(link_matrix))
    }
}

/// Metropolis Hastings algorithm with diagnostics. Very slow, use [`MetropolisHastingsDeltaDiagnostic`] instead.
pub struct MetropolisHastingsDiagnostic<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    number_of_update: usize,
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
    _phantom: PhantomData<(State, D)>,
}

impl<State, D> MetropolisHastingsDiagnostic<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
            _phantom: PhantomData,
        })
    }
    
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
}

impl<State, D> MonteCarloDefault<State, D> for MetropolisHastingsDiagnostic<State, D>
    where State: LatticeState<D> + LatticeStateNew<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(state.lattice().clone(), state.beta(), LinkMatrix::new(link_matrix))
    }
    
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            self.has_replace_last = true;
            return Ok(potential_next);
        }
        else{
            self.has_replace_last = false;
            return Ok(state);
        }
    }
}

#[inline]
fn get_delta_s_old_new_cmp(
    link_matrix: &LinkMatrix,
    lattice: &LatticeCyclique<na::U4>,
    link: &LatticeLinkCanonical<na::U4>,
    new_link: &na::Matrix3<Complex>,
    beta : Real,
    old_matrix: &na::Matrix3<Complex>,
) -> Real
{
    let dir_j = link.dir();
    let a: na::Matrix3<na::Complex<Real>> = Direction::<na::U4>::get_all_positive_directions().par_iter()
    .filter(|dir_i| *dir_i != dir_j ).map(|dir_i| {
        let el_1 = link_matrix.get_sij(link.pos(), dir_j, &dir_i, lattice).unwrap().adjoint();
        let l_1 = LatticeLink::new(lattice.add_point_direction(*link.pos(), dir_j), - dir_i);
        let u1 = link_matrix.get_matrix(&l_1, lattice).unwrap();
        let l_2 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &- dir_i), *dir_j);
        let u2 = link_matrix.get_matrix(&l_2, lattice).unwrap().adjoint();
        let l_3 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &- dir_i), *dir_i);
        let u3 = link_matrix.get_matrix(&l_3, lattice).unwrap();
        el_1 + u1 * u2 * u3
    }).sum();
    -((new_link - old_matrix) * a).trace().real() * beta / LatticeStateDefault::CA
}

/// Metropolis Hastings algorithm with diagnostics.
pub struct MetropolisHastingsDeltaDiagnostic<Rng>
    where Rng: rand::Rng,
{
    number_of_update: usize,
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
    delta_s: Real,
    rng: Rng
}

impl<Rng> MetropolisHastingsDeltaDiagnostic<Rng>
    where Rng: rand::Rng,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real, rng: Rng) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
            delta_s: 0_f64,
            rng,
        })
    }
    
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
    
    pub fn delta_s(&self) -> Real {
        self.delta_s
    }
    
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    #[inline]
    fn get_delta_s(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<na::U4>,
        link: &LatticeLinkCanonical<na::U4>,
        new_link: &na::Matrix3<Complex>,
        beta : Real,
        previous_modification: &[(LatticeLinkCanonical<na::U4>, na::Matrix3<Complex>)],
    ) -> Real {
        let old_matrix_opt = previous_modification.iter().filter(|(link_m, _)| link_m == link ).last();
        let old_matrix;
        match old_matrix_opt {
            Some((_, m)) => old_matrix = *m,
            None => old_matrix = link_matrix.get_matrix(&LatticeLink::from(*link), lattice).unwrap(),
        }
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }
    
    #[inline]
    fn get_potential_modif(&mut self, state: &LatticeStateDefault) -> Vec<(LatticeLinkCanonical<na::U4>, na::Matrix3<Complex>)>
    {
        self.delta_s = 0_f64;
        let d_p = rand::distributions::Uniform::new(0, state.lattice().dim());
        let d_d = rand::distributions::Uniform::new(0, LatticeCyclique::<na::U4>::dim_st());
        let mut return_val = Vec::with_capacity(self.number_of_update);
        
        (0..self.number_of_update).for_each(|_| {
            let point = LatticePoint::from_fn(|_| d_p.sample(&mut self.rng));
            let direction = Direction::get_all_positive_directions()[d_d.sample(&mut self.rng)];
            let link = LatticeLinkCanonical::new(point, direction).unwrap();
            let index = link.to_index(state.lattice());
            
            let old_link_m = state.link_matrix()[index];
            let rand_m = su3::orthonormalize_matrix(&su3::get_random_su3_close_to_unity(self.spread, &mut self.rng));
            let new_link = rand_m * old_link_m;
            
            self.delta_s += Self::get_delta_s(state.link_matrix(), state.lattice(), &link, &new_link, state.beta(), &return_val);
            return_val.push((link, new_link));
        });
        
        return_val
    }
    
    #[inline]
    fn get_next_element_default(&mut self, mut state: LatticeStateDefault) -> LatticeStateDefault {
        
        let potential_modif = self.get_potential_modif(&state);
        let proba = (-self.delta_s).exp().min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(&mut self.rng) {
            self.has_replace_last = true;
            potential_modif.iter().for_each(|(link, matrix)| *state.get_link_mut(link).unwrap() = *matrix);
            state
        }
        else{
            self.has_replace_last = false;
            state
        }
    }
}

impl<Rng> MonteCarlo<LatticeStateDefault, na::U4> for MetropolisHastingsDeltaDiagnostic<Rng>
    where Rng: rand::Rng,
{
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault) -> Result<LatticeStateDefault, SimulationError>{
        Ok(self.get_next_element_default(state))
    }
}

#[cfg(test)]
#[test]
fn test_mh_delta(){
    let mut rng = rand::thread_rng();
    
    let size = 1_000_f64;
    let number_of_pts = 4;
    let beta = 2_f64;
    let mut simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
    
    let mut mcd = MetropolisHastingsDeltaDiagnostic::new(1, 0.01, rng).unwrap();
    for _ in 0..10 {
        let mut simulation2 = simulation.clone();
        mcd.get_potential_modif(&simulation)
            .iter()
            .for_each(|(link, matrix)| *simulation2.get_link_mut(link).unwrap() = *matrix);
        println!("ds {}, dh {}", mcd.delta_s(), -simulation.get_hamiltonian_links() + simulation2.get_hamiltonian_links());
        let prob_of_replacement = (simulation.get_hamiltonian_links() - simulation2.get_hamiltonian_links()).exp()
            .min(1_f64)
            .max(0_f64);
        assert!(
            ((-mcd.delta_s()).exp().min(1_f64).max(0_f64) -
            prob_of_replacement).abs() < 1E-8_f64
        );
        simulation = simulation2;
    }
}


/// Metropolis Hastings algorithm optimize to do one step.
pub struct MetropolisHastingsDeltaOneDiagnostic<Rng>
    where Rng: rand::Rng,
{
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
    rng: Rng
}

impl<Rng> MetropolisHastingsDeltaOneDiagnostic<Rng>
    where Rng: rand::Rng,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(spread: Real, rng: Rng) -> Option<Self> {
        if spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
            rng,
        })
    }
    
    getter_copy!(prob_replace_last, Real);
    getter_copy!(has_replace_last, bool);
    
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    #[inline]
    fn get_delta_s(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<na::U4>,
        link: &LatticeLinkCanonical<na::U4>,
        new_link: &na::Matrix3<Complex>,
        beta : Real,
    ) -> Real {
        let old_matrix = link_matrix.get_matrix(&LatticeLink::from(*link), lattice).unwrap();
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }
    
    #[inline]
    fn get_potential_modif(&mut self, state: &LatticeStateDefault) -> (LatticeLinkCanonical<na::U4>, na::Matrix3<Complex>)
    {
        let d_p = rand::distributions::Uniform::new(0, state.lattice().dim());
        let d_d = rand::distributions::Uniform::new(0, 4);
        
        let point = LatticePoint::from_fn(|_| d_p.sample(&mut self.rng));
        let direction = Direction::get_all_positive_directions()[d_d.sample(&mut self.rng)];
        let link = LatticeLinkCanonical::new(point, direction).unwrap();
        let index = link.to_index(state.lattice());
        
        let old_link_m = state.link_matrix()[index];
        let rand_m = su3::orthonormalize_matrix(&su3::get_random_su3_close_to_unity(self.spread, &mut self.rng));
        let new_link = rand_m * old_link_m;
        (link, new_link)
    }
    
    #[inline]
    fn get_next_element_default(&mut self, mut state: LatticeStateDefault) -> LatticeStateDefault {
        let (link, matrix) = self.get_potential_modif(&state);
        let delta_s = Self::get_delta_s(state.link_matrix(), state.lattice(), &link, &matrix, state.beta());
        let proba = (-delta_s).exp().min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(&mut self.rng) {
            self.has_replace_last = true;
            *state.get_link_mut(&link).unwrap() = matrix;
            state
        }
        else {
            self.has_replace_last = false;
            state
        }
    }
}

impl<Rng> MonteCarlo<LatticeStateDefault, na::U4> for MetropolisHastingsDeltaOneDiagnostic<Rng>
    where Rng: rand::Rng,
{
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault) -> Result<LatticeStateDefault, SimulationError>{
        Ok(self.get_next_element_default(state))
    }
}
