//! defines macro
//!
//! # getter!
//! getter that return a reference
//! ## Examples
//! ```ignore
//! struct a {b: usize}
//! impl a {
//!     getter!(b, usize);
//! }
//! ```
//! ```ignore
//! struct a {b: usize}
//! impl a {
//!     getter!(const, b, usize);
//! }
//! ```
//! # getter_trait!
//! getter for trait (without the `pub` ident)
//! ## Example
//! ```ignore
//! pub trait get_b {
//!     fn b(&self) -> usize
//! }
//! struct a {b: usize}
//! impl get_b for a {
//!     getter_trait!(b, usize);
//! }
//! ```
//! # getter_copy!
//! create a getter that copy the value.
//! ## Examples
//! ```ignore
//! struct a {b: usize}
//! impl a {
//!     getter_copy!(b, usize);
//! }
//! ```
//! ```ignore
//! struct a {b: usize}
//! impl a {
//!     getter_copy!(const, b, usize);
//! }
//! ```
//! # project!
//! project an methode on one of the composant to the structur.
//! ## Example
//! ```ignore
//! struct A{}
//! struct B{ a: A}
//! impl A {
//!     pub const foo(&self, var: f64, var2: &[usize; 5]) -> Bar {
//!         ...
//!     }
//! }
//! impl B {
//!     project!(pub const, foo, a, Bar, var: f64, var2: &[usize; 5]);
//! }
//! ```
//! # project_mut!
//! same as `project!` but with `&mut self` in the signature of the methode replacing `&sefl`
//! ## Example
//! ```ignore
//! struct A{}
//! struct B{ a: A}
//! impl A {
//!     pub foo(&mut self, var: f64, var2: &[usize; 5]) -> Bar {
//!         ...
//!     }
//! }
//! impl B {
//!     project_mut!(pub, foo, a, Bar, var: f64, var2: &[usize; 5]);
//! }
//! ```

macro_rules! getter {
    ($(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        pub fn $i(&self) -> &$t {
            &self.$i
        }
    };
    (const, $(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        pub const fn $i(&self) -> &$t {
            &self.$i
        }
    }
}

/// Getter in a trait (without the pub )
macro_rules! getter_trait {
    ($(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        fn $i(&self) -> &$t {
            &self.$i
        }
    };
}

macro_rules! getter_copy {
    ($(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        pub fn $i(&self) -> $t {
            self.$i
        }
    };
    (const, $(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        pub const fn $i(&self) -> $t {
            self.$i
        }
    }
}

macro_rules! getter_copy_trait {
    ($(#[$meta:meta])* $i:ident, $t:ty) => {
        $(#[$meta])*
        fn $i(&self) -> $t {
            self.$i
        }
    }
}

macro_rules! project {
    (pub, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        pub fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    (pub const, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        pub const fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    (const, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        const fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    ($(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
}

macro_rules! project_mut {
    (pub, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        pub fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    (pub const, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        pub const fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    (const, $(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        const fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    ($(#[$meta:meta])* $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        $(#[$meta])*
        fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
}

#[macro_export]
/// assert if two matrices are approximatively the same
macro_rules! assert_eq_matrix {
    ($e:expr, $e2:expr, $epsilon:expr) => {
        assert!(($e - $e2).norm() < $epsilon, "assertion failed: norm `{} > {}`", ($e - $e2).norm(), $epsilon)
    };
    ($e:expr, $e2:expr, $epsilon:expr, $($arg:tt)+) => {
        assert!(($e - $e2).norm() < $epsilon, "assertion failed: norm `{} > {}` : {}", ($e - $e2).norm(), $epsilon, format_args!($($arg)*))
    };
}

#[macro_export]
/// assert if two complex are approximatively the same
macro_rules! assert_eq_complex {
    ($e:expr, $e2:expr, $epsilon:expr) => {
        {
            use nalgebra::ComplexField;
            assert!(($e - $e2).modulus() < $epsilon, "assertion failed: `({} - {}).modulus() > {}", $e, $e2, $epsilon);
        }
    };
    ($e:expr, $e2:expr, $epsilon:expr, $($arg:tt)+) => {
        {
            use nalgebra::ComplexField;
            assert!(($e - $e2).modulus() < $epsilon, "assertion failed: `({} - {}).modulus() > {} : {}", $e, $e2, $epsilon, format_args!($($arg)*));
        }
    };
}

#[macro_export]
/// assert if the matrix is U(2) ( unitary 2 x 2)
macro_rules! assert_matrix_is_unitary_2 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        assert!(
            ($m.determinant().modulus() - 1_f64).abs() < $epsilon,
            "determinant {} of {:?} is not of norm 1",
            $m.determinant().modulus(),
            $m
        );
        assert!(
            ($m * $m.adjoint() - nalgebra::Matrix2::identity()).norm() < $epsilon,
            "The matrix is not unitary"
        );
    }};
}

#[macro_export]
/// assert if the matrix is U(3) (unitary 3 x 3)
macro_rules! assert_matrix_is_unitary_3 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        assert!(
            ($m.determinant().modulus() - 1_f64).abs() < $epsilon,
            "determinant {} of {} is not of norm 1",
            $m.determinant().modulus(),
            $m
        );
        assert!(
            ($m * $m.adjoint() - nalgebra::Matrix3::identity()).norm() < $epsilon,
            "The matrix {} is not unitary",
            $m
        );
    }};
}

#[macro_export]
/// assert if the matrix is SU(2) (special unitary)
macro_rules! assert_matrix_is_su_2 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        assert!(
            ($m.determinant() - nalgebra::Complex::from(1_f64)).modulus() < $epsilon,
            "determinant {} of {} is not of norm 1",
            $m.determinant().modulus(),
            $m
        );
        assert!(
            ($m * $m.adjoint() - nalgebra::Matrix2::identity()).norm() < $epsilon,
            "The matrix {} is not unitary",
            $m
        );
    }};
}

#[macro_export]
/// assert if the matrix is SU(3) (special unitary)
macro_rules! assert_matrix_is_su_3 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        assert!(
            ($m.determinant() - nalgebra::Complex::from(1_f64)).modulus() < $epsilon,
            "determinant {} of {} is not of norm 1",
            $m.determinant().modulus(),
            $m
        );
        assert!(
            ($m * $m.adjoint() - nalgebra::Matrix3::identity()).norm() < $epsilon,
            "The matrix is not unitary {}",
            $m
        );
    }};
}
