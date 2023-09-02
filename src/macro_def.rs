//! defines macro
//!
//! # [`getter!`]
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
//! # [`getter_copy!`]
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
//! # [`project!`]
//! project an methods on one of the corposant to the structure.
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
//! # [`project_mut!`]
//! same as `project!` but with `&mut self` in the signature of the method replacing `&self`
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
    ($(#[$meta:meta])* $v:vis $(,)? $i:ident, $t:ty) => {
        getter!($(#[$meta])* $v $i() -> $t);
    };
    ($(#[$meta:meta])* $v:vis $(,)? const, $i:ident, $t:ty) => {
        getter!($(#[$meta])* $v const $i() -> $t);
    };
    ($(#[$meta:meta])* $v:vis $i:ident() -> $t:ty) => {
        $(#[$meta])*
        $v fn $i(&self) -> &$t {
            &self.$i
        }
    };
    ($(#[$meta:meta])* $v:vis const $i:ident() -> $t:ty) => {
        $(#[$meta])*
        $v const fn $i(&self) -> &$t {
            &self.$i
        }
    };
}

macro_rules! getter_copy {
    ($(#[$meta:meta])* $v:vis $(,)? $i:ident, $t:ty) => {
        getter_copy!($(#[$meta])* $v $i() -> $t);
    };
    ($(#[$meta:meta])* $v:vis $(,)? const, $i:ident, $t:ty) => {
        getter_copy!($(#[$meta])* $v const $i() -> $t);
    };
    ($(#[$meta:meta])* $v:vis $i:ident() -> $t:ty) => {
        $(#[$meta])*
        $v fn $i(&self) -> $t {
            self.$i
        }
    };
    ($(#[$meta:meta])* $v:vis const $i:ident() -> $t:ty) => {
        $(#[$meta])*
        $v const fn $i(&self) -> $t {
            self.$i
        }
    };
}

macro_rules! project {
    ($(#[$meta:meta])* $v:vis $(,)? $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        project!($(#[$meta])* $v $data.$i($( $arg : $type_arg ),*) -> $t);
    };
    ($(#[$meta:meta])* $v:vis $(,)? const, $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        project!($(#[$meta])* $v const $data.$i($( $arg : $type_arg ),*) -> $t);
    };
    ($(#[$meta:meta])* $v:vis $data:ident.$i:ident($($arg:ident : $type_arg:ty ),*) -> $t:ty) => {
        $(#[$meta])*
        $v fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    ($(#[$meta:meta])* $v:vis const $data:ident.$i:ident($($arg:ident : $type_arg:ty ),*) -> $t:ty) => {
        $(#[$meta])*
        $v const fn $i(&self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
}

macro_rules! project_mut {
    ($(#[$meta:meta])* $v:vis $(,)? $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        project_mut!($(#[$meta])* $v $data.$i($( $arg: $type_arg ),*) -> $t);
    };
    ($(#[$meta:meta])* $v:vis $(,)? const, $i:ident, $data:ident, $t:ty $(, $arg:ident : $type_arg:ty )* ) => {
        project_mut!($(#[$meta])* $v const $data.$i($( $arg : $type_arg ),*) -> $t);
    };
    ($(#[$meta:meta])* $v:vis $data:ident.$i:ident($( $arg:ident : $type_arg:ty ),*) -> $t:ty) => {
        $(#[$meta])*
        $v fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
    ($(#[$meta:meta])* $v:vis const $data:ident.$i:ident($( $arg:ident : $type_arg:ty ),*) -> $t:ty) => {
        $(#[$meta])*
        $v const fn $i(&mut self $(, $arg : $type_arg)*) -> $t {
            self.$data.$i($($arg, )*)
        }
    };
}

#[macro_export]
/// assert if two matrices are approximately the same
// TODO example
macro_rules! assert_eq_matrix {
    ($e:expr, $e2:expr, $epsilon:expr) => {
        let e = $e;
        let e2 = $e2;
        assert!((e - e2).norm() < $epsilon, "assertion failed: norm `{} > {}`", (e - e2).norm(), $epsilon)
    };
    ($e:expr, $e2:expr, $epsilon:expr, $($arg:tt)+) => {
        let e = $e;
        let e2 = $e2;
        assert!((e - e2).norm() < $epsilon, "assertion failed: norm `{} > {}` : {}", (e - e2).norm(), $epsilon, format_args!($($arg)*))
    };
}

#[macro_export]
/// assert if two complex are approximately the same
// TODO example
macro_rules! assert_eq_complex {
    ($e:expr, $e2:expr, $epsilon:expr) => {
        {
            use nalgebra::ComplexField;
            let e = $e;
            let e2 = $e2;
            assert!((e - e2).modulus() < $epsilon, "assertion failed: `({} - {}).modulus() > {}", e, e2, $epsilon);
        }
    };
    ($e:expr, $e2:expr, $epsilon:expr, $($arg:tt)+) => {
        {
            use nalgebra::ComplexField;
            let e = $e;
            let e2 = $e2;
            assert!((e - e2).modulus() < $epsilon, "assertion failed: `({} - {}).modulus() > {} : {}", e, e2, $epsilon, format_args!($($arg)*));
        }
    };
}

#[macro_export]
/// assert if the matrix is U(2) ( unitary 2 x 2)
// TODO example
macro_rules! assert_matrix_is_unitary_2 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        let m = $m;
        assert!(
            (m.determinant().modulus() - 1_f64).abs() < $epsilon,
            "determinant {} of {} is not of norm 1",
            m.determinant().modulus(),
            m
        );
        assert!(
            (m * m.adjoint() - nalgebra::Matrix2::identity()).norm() < $epsilon,
            "The matrix is not unitary"
        );
    }};
}

#[macro_export]
/// assert if the matrix is U(3) (unitary 3 x 3)
// TODO example
macro_rules! assert_matrix_is_unitary_3 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        let m = $m;
        assert!(
            (m.determinant().modulus() - 1_f64).abs() < $epsilon,
            "determinant {} of {} is not of norm 1",
            m.determinant().modulus(),
            m
        );
        assert!(
            (m * m.adjoint() - nalgebra::Matrix3::identity()).norm() < $epsilon,
            "The matrix {} is not unitary",
            m
        );
    }};
}

#[macro_export]
/// assert if the matrix is SU(2) (special unitary)
// TODO examples
macro_rules! assert_matrix_is_su_2 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        let m = $m;
        assert!(
            (m.determinant() - nalgebra::Complex::from(1_f64)).modulus() < $epsilon,
            "determinant {} of {} is not of norm 1",
            m.determinant().modulus(),
            m
        );
        assert!(
            (m * m.adjoint() - nalgebra::Matrix2::identity()).norm() < $epsilon,
            "The matrix {} is not unitary",
            m
        );
    }};
}

#[macro_export]
/// assert if the matrix is SU(3) (special unitary)
// TODO examples
macro_rules! assert_matrix_is_su_3 {
    ($m:expr, $epsilon:expr) => {{
        use nalgebra::ComplexField;
        let m = $m;
        assert!(
            (m.determinant() - nalgebra::Complex::from(1_f64)).modulus() < $epsilon,
            "determinant {} of {} is not of norm 1",
            m.determinant().modulus(),
            m
        );
        assert!(
            (m * m.adjoint() - nalgebra::Matrix3::identity()).norm() < $epsilon,
            "The matrix is not unitary {}",
            m
        );
    }};
}
