
use super::{
    Real,
    CMatrix3,
    lattice::{
        LatticeLinkCanonical,
        LatticePoint,
        LatticeCyclique,
        PositiveF64,
        LatticeLink,
        Direction,
        LatticeElementToIndex,
    },
    Vector8,
    su3,
    su3::{
        GENERATORS,
    },
    I,
    ZERO,
    Complex,
    thread::{
        ThreadError,
        run_pool_parallel_vec_with_initialisation_mutable,
    },
    integrator::Integrator,
};
use na::{
    Vector3,
    Matrix3,
    ComplexField
};
use  std::{
    ops::{Index, IndexMut, Mul, Add},
    vec::Vec,
};

use crossbeam::thread;

// easy switch for the hash map used
//type HashMapUse<K,V> = HashMap<K,V>;

/// Ajdoint representation of SU(3), it is su(3) (i.e. the lie algebra).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Su3Adjoint {
    data: Vector8<Real> // use an [Real ;8] instead ?
}

impl Su3Adjoint {
    
    pub fn new(data: Vector8<Real>)-> Self {
        Self {data}
    }
    
    pub fn data(&self) -> &Vector8<Real> {
        &self.data
    }
    
    /// return the su(3) (Lie algebra) matix.
    pub fn to_matrix(&self) -> Matrix3<na::Complex<Real>> {
        let mut mat = Matrix3::from_element(ZERO);
        for i in 0..self.data.len() {
            mat += *GENERATORS[i] * na::Complex::<Real>::from(self.data[i]);
        }
        return mat;
    }
    
    /// Return the SU(3) matrix associtaed with this generator.
    /// Note that the function consume self.
    pub fn to_su3(self) -> Matrix3<na::Complex<Real>> {
        // TODO should it consume ? the user can manually clone and there is use because
        // where the value is not necessary anymore.
        //(self.to_matrix() * na::Complex::<Real>::i() ).exp()
        su3::su3_exp_i(self)
    }
    
    /// return exp( T^a v^a) where v is self.
    /// Note that the function consume self.
    pub fn exp(self) -> Matrix3<na::Complex<Real>> {
        su3::su3_exp_r(self)
    }
    
    /// create a new random SU3 adjoint.
    pub fn random(rng: &mut impl rand::Rng, d: &impl rand_distr::Distribution<Real>) -> Self {
        Self {
            data : Vector8::<Real>::from_fn(|_,_| d.sample(rng))
        }
    }
    
    /// Return the t coeff `t = 1/2 * Tr(X^2).
    /// Used for [`su3::su3_exp_i`]
    pub fn t(&self) -> na::Complex<Real> {
        // todo optimize
        let m = self.to_matrix();
        - na::Complex::from(0.5_f64) * (m * m).trace()
    }
    
    /// Return the t coeff `d = i * det(X).
    /// Used for [`su3::su3_exp_i`]
    pub fn d(&self) -> na::Complex<Real> {
        self.to_matrix().determinant() * I
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Mul<Real> for Su3Adjoint {
    type Output = Self;
    fn mul(mut self, rhs: Real) -> Self::Output {
        self.data *= rhs;
        self
    }
}

impl Mul<Su3Adjoint> for Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Add<Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output{
        self.data += rhs.data();
        self
    }
}

impl Default for Su3Adjoint {
    fn default() -> Self{
        Su3Adjoint::new(Vector8::from_element(0_f64))
    }
}

impl Index<usize> for Su3Adjoint {
    type Output = Real;
    
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl IndexMut<usize> for Su3Adjoint {
        
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

impl From<Vector8<Real>> for Su3Adjoint {
    fn from(v: Vector8<Real>) -> Self {
        Su3Adjoint::new(v)
    }
}

impl From<Su3Adjoint> for Vector8<Real> {
    fn from(v: Su3Adjoint) -> Self {
        v.data
    }
}

/// Reoresent the link matrics
#[derive(Debug)]
pub struct LinkMatrix {
    max_time : usize,
    data: Vec<Matrix3<na::Complex<Real>>>,
}

impl LinkMatrix {
    
    pub fn new (data: Vec<Matrix3<na::Complex<Real>>>) -> Self{
        Self{data, max_time: 0}
    }
    
    pub fn data(&self) -> &Vec<Matrix3<na::Complex<Real>>> {
        &self.data
    }
    
    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data but slower than [`LinkMatrix::new_random_threaded`]
    pub fn new_deterministe(
        l: &LatticeCyclique,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut data = Vec::with_capacity(l.get_number_of_canonical_links_space());
        for _ in l.get_links_space(0) {
            // the iterator *should* be in order
            let matrix = Su3Adjoint::random(rng, d).to_su3();
            data.push(matrix);
        }
        Self {
            max_time: 0,
            data,
        }
    }
    
    
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be repoduce esealy, In that case use [`LinkMatrix::new_random_threaded`].
    pub fn new_random_threaded<Distribution>(
        l: &LatticeCyclique,
        d: &Distribution,
        number_of_thread: usize,
        //s: &thread::Scope,
    ) -> Result<Self, ThreadError>
        where Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(ThreadError::ThreadNumberIncorect);
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Ok(LinkMatrix::new_deterministe(l, &mut rng, d));
        }
        let data = run_pool_parallel_vec_with_initialisation_mutable(
            l.get_links_space(0),
            d,
            &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
            || rand::thread_rng(),
            number_of_thread,
            l.get_number_of_canonical_links_space(),
            l,
            CMatrix3::zeros(),
        )?;
        Ok(Self {
            max_time: 0,
            data,
        })
    }
    
    /// get the link matrix associtate to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    pub fn get_matrix(&self, l: &LatticeCyclique, link: &LatticeLink)-> Option<Matrix3<na::Complex<Real>>> {
        let link_c = l.get_canonical(link);
        let matrix_o = self.data.get(link_c.to_index(l));
        match matrix_o {
            Some(matrix) => {
                let m_return = matrix.clone();
                if link_c != *link {
                    // that means the the link was in the negative direction
                    return Some(m_return.adjoint());
                }
                return Some(m_return);
            },
            None => None,
        }
    }
    
}

/// represent an electric field.
#[derive(Debug)]
pub struct EField
{
    max_time : usize,
    data: Vec<Vector3<Su3Adjoint>>, // use a [Su3Adjoint; 4] instead ?
}

impl EField {
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn new (data: Vec<Vector3<Su3Adjoint>>) -> Self{
        Self{data, max_time: 0}
    }
    
    pub fn data(&self) -> &Vec<Vector3<Su3Adjoint>> {
        &self.data
    }
    
    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data but slower than [`EField::new_random_threaded`]
    pub fn new_deterministe(
        l: &LatticeCyclique,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut data = Vec::with_capacity(l.get_number_of_points());
        for _ in l.get_points(0) {
            // iterator *should* be ordoned
            let p1 = Su3Adjoint::random(rng, d);
            let p2 = Su3Adjoint::random(rng, d);
            let p3 = Su3Adjoint::random(rng, d);
            data.push(Vector3::new(p1, p2, p3));
        }
        Self {
            max_time: 0,
            data,
        }
    }
    
    /// Single thread generation by seeding a new rng number. Used in [`LatticeSimulation::new_random_threaded`]
    /// To create a seedable and reproducible set use [`EField::new_random_threaded`].
    pub fn new_random(
        l: &LatticeCyclique,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        EField::new_deterministe(l, &mut rng, d)
    }
    
    pub fn get_e_vec(&self, point: &LatticePoint, l: &LatticeCyclique) -> Option<&Vector3<Su3Adjoint>> {
        self.data.get(point.to_index(l))
    }
    
    pub fn get_e_field(&self, point: &LatticePoint, dir: &Direction, l: &LatticeCyclique) -> Option<&Su3Adjoint> {
        let value = self.get_e_vec(point, l);
        match value {
            Some(vec) => Some(&vec[dir.to_index()]),
            None => None,
        }
    }
}

#[derive(Debug)]
pub enum SimulationError {
    ThreadingError(ThreadError),
    InitialisationError,
}

#[derive(Debug)]
pub struct LatticeSimulation {
    lattice : LatticeCyclique,
    e_field: EField,
    link_matrix: LinkMatrix,
    t: usize,
}

impl LatticeSimulation {
    
    pub fn new(lattice: LatticeCyclique, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Self {
        Self {lattice, e_field, link_matrix, t}
    }
    
    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data but slower than [`LatticeSimulation::new_random_threaded`]
    pub fn new_deterministe(
        size: PositiveF64,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Option<Self> {
        let lattice_option = LatticeCyclique::new(size, number_of_points);
        if let None = lattice_option {
            return None;
        }
        let lattice = lattice_option.unwrap();
        let e_field = EField::new_deterministe(&lattice, rng, d);
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng, d);
        Some(Self {
            lattice,
            e_field,
            link_matrix,
            t: 0,
        })
    }
    
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be repoduce esealy, In that case use [`LatticeSimulation::new_random_threaded`].
    pub fn new_random_threaded<Distribution>(
        size: PositiveF64,
        number_of_points: usize,
        d: &Distribution,
        number_of_thread : usize
    ) -> Result<Self, SimulationError>
        where Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(SimulationError::ThreadingError(ThreadError::ThreadNumberIncorect)); //
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return LatticeSimulation::new_deterministe(size, number_of_points, &mut rng, d)
                .ok_or(SimulationError::InitialisationError);
        }
        let lattice_option = LatticeCyclique::new(size, number_of_points);
        if let None = lattice_option {
            return Err(SimulationError::InitialisationError);
        }
        let lattice = lattice_option.unwrap();
        let result = thread::scope(|s| {
            let lattice_clone = lattice.clone();
            let handel = s.spawn(move |_| {
                EField::new_random(&lattice_clone, d)
            });
            let link_matrix = LinkMatrix::new_random_threaded(&lattice, d, number_of_points - 1)
                .map_err(|err| SimulationError::ThreadingError(err))?;
            
            let e_field = handel.join().map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
            Ok(Self {
                lattice,
                e_field,
                link_matrix,
                t: 0,
            })
        }).map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
        return result;
    }
    
    pub fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    pub fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    pub fn lattice(&self) -> &LatticeCyclique {
        &self.lattice
    }
    
    pub fn t(&self) -> usize {
        self.t
    }
    
    const CA: Real = 3_f64;
    
    
    pub fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * LatticeSimulation::CA ).sqrt();
        let u_i = self.link_matrix().get_matrix(self.lattice(), &LatticeLink::from(link.clone()));
        match u_i {
            Some(u_i) => {
                let e_i = self.e_field().get_e_field(link.pos(), link.dir(), self.lattice());
                match e_i {
                    Some(e_i) => {
                        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / self.lattice().size()));
                    }
                    None => return None,
                }
            }
            None => return None,
        }
        
    }
    
    pub fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        let c = - Complex::new(2_f64 / LatticeSimulation::CA, 0_f64).sqrt();
        let mut derivative = Vector3::from_element(Su3Adjoint::default());
        for dir in Direction::POSITIVES_SPACE.iter() {
            let derivative_i = &mut derivative[dir.to_index()];
            let u_i = self.link_matrix().get_matrix(self.lattice(), &LatticeLink::new(point.clone(), dir.clone()))?;
            let mut sum_s = CMatrix3::zeros();
            for dir_2 in Direction::DIRECTIONS_SPACE.iter() {
                if dir_2.to_positive() != *dir {
                    let u_j = self.link_matrix().get_matrix(self.lattice(), &LatticeLink::new(point.clone(), dir_2.clone()))?;
                    let mut point_pj = point.clone();
                    point_pj[dir_2.to_index()] += 1;
                    let u_i_p_j = self.link_matrix().get_matrix(self.lattice(), &LatticeLink::new(point_pj, dir.clone()))?;
                    let mut point_pi = point.clone();
                    point_pi[dir.to_index()] += 1;
                    let u_j_d = self.link_matrix().get_matrix(self.lattice(), &LatticeLink::new(point_pi, dir.clone()))?.adjoint();
                    let s_ij = u_j * u_i_p_j * u_j_d;
                    sum_s += s_ij.adjoint();
                }
            }
            for b in 0..derivative_i.len(){
                let derivative_i_b = &mut derivative_i[b];
                *derivative_i_b += (c * (su3::GENERATORS[b] * u_i * sum_s).trace().im).re / self.lattice().size();
            }
        }
        return Some(derivative);
    }
    
    pub fn simulate<I>(&self, delta_t: Real, number_of_thread: usize) -> Result<Self, SimulationError> where I: Integrator {
        I::integrate(&self, delta_t, number_of_thread).map_err(|err| SimulationError::ThreadingError(err))
    }
}
