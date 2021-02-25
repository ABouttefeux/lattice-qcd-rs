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