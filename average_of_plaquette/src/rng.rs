use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn get_rand_from_entropy() -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::from_entropy()
}

pub fn get_rand_from_seed(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}
