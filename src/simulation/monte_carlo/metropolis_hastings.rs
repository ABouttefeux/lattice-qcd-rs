
//! Metropolis Hastings methode

use super::{
    MonteCarlo,
    MonteCarloDefault,
    get_delta_s_old_new_cmp,
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
            },
            error::{
                Never,
                ErrorWithOnwnedValue
            },
        },
        state::{
            LatticeState,
            LatticeStateNew,
            LatticeStateDefault,
        },
    },
};
use std::marker::PhantomData;
use rand_distr::Distribution;
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
    type Error = ErrorWithOnwnedValue<State::Error, LinkMatrix>;
    
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, Self::Error> {
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
    
    /// Get the last probably of acceptance of the random change.
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    /// Get if last step has accepted the replacement.
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
    
    type Error = ErrorWithOnwnedValue<State::Error, LinkMatrix>;
    
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, Self::Error> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(state.lattice().clone(), state.beta(), LinkMatrix::new(link_matrix))
    }
    
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, Self::Error> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            self.has_replace_last = true;
            Ok(potential_next)
        }
        else{
            self.has_replace_last = false;
            Ok(state)
        }
    }
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
    
    /// Get the last probably of acceptance of the random change.
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    /// Get if last step has accepted the replacement.
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
    
    /// Get the last step delta of energy.
    pub fn delta_s(&self) -> Real {
        self.delta_s
    }
    
    /// Absorbe self and return the RNG as owned. It essentialy deconstruct the structure.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    #[inline]
    fn get_delta_s<D>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &na::Matrix3<Complex>,
        beta : Real,
        previous_modification: &[(LatticeLinkCanonical<D>, na::Matrix3<Complex>)],
    ) -> Real
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        // TODO corect error staple
        let old_matrix_opt = previous_modification.iter().filter(|(link_m, _)| link_m == link ).last();
        let old_matrix;
        match old_matrix_opt {
            Some((_, m)) => old_matrix = *m,
            None => old_matrix = link_matrix.get_matrix(&LatticeLink::from(*link), lattice).unwrap(),
        }
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }
    
    #[inline]
    fn get_potential_modif<D>(&mut self, state: &LatticeStateDefault<D>) -> Vec<(LatticeLinkCanonical<D>, na::Matrix3<Complex>)>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        self.delta_s = 0_f64;
        let d_p = rand::distributions::Uniform::new(0, state.lattice().dim());
        let d_d = rand::distributions::Uniform::new(0, LatticeCyclique::<D>::dim_st());
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
    fn get_next_element_default<D>(&mut self, mut state: LatticeStateDefault<D>) -> LatticeStateDefault<D>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        
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

impl<Rng, D> MonteCarlo<LatticeStateDefault<D>, D> for MetropolisHastingsDeltaDiagnostic<Rng>
    where Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    // todo review
    type Error = Never;
    
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault<D>) -> Result<LatticeStateDefault<D>, Never>{
        Ok(self.get_next_element_default(state))
    }
}

#[cfg(test)]
#[test]
fn test_mh_delta(){
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x45_78_93_f4_4a_b0_67_f0);
    
    let size = 1_000_f64;
    let number_of_pts = 4;
    let beta = 2_f64;
    let mut simulation = LatticeStateDefault::<na::U4>::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
    
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
    
    getter_copy!(
        /// Get the last probably of acceptance of the random change.
        prob_replace_last, Real
    );
    getter_copy!(
        /// Get if last step has accepted the replacement.
        has_replace_last, bool
    );
    
    /// Absorbe self and return the RNG as owned. It essentialy deconstruct the structure.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    #[inline]
    fn get_delta_s<D>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &na::Matrix3<Complex>,
        beta : Real,
    ) -> Real
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let old_matrix = link_matrix.get_matrix(&LatticeLink::from(*link), lattice).unwrap();
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }
    
    #[inline]
    fn get_potential_modif<D>(&mut self, state: &LatticeStateDefault<D>) -> (LatticeLinkCanonical<D>, na::Matrix3<Complex>)
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let d_p = rand::distributions::Uniform::new(0, state.lattice().dim());
        let d_d = rand::distributions::Uniform::new(0, LatticeCyclique::<D>::dim_st());
        
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
    fn get_next_element_default<D>(&mut self, mut state: LatticeStateDefault<D>) -> LatticeStateDefault<D>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
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

impl<Rng, D> MonteCarlo<LatticeStateDefault<D>, D> for MetropolisHastingsDeltaOneDiagnostic<Rng>
    where Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    // todo review
    type Error = Never;
    
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault<D>) -> Result<LatticeStateDefault<D>, Never>{
        Ok(self.get_next_element_default(state))
    }
}
