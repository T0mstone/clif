//! # Clif
//! A Clifford Algebra (Geometric algebra, GA) library
//!
//! Predefined 2D and 3D GA types are available via `2d` and `3d` features respectively (both enabled by default).
//!
//! To create and use new GA types, use the [`register_multivector`](macro.register_multivector.html) macro

mod field;

pub use crate::field::Field;
pub use clif_proc_macro::register_multivector;
pub use num_traits::{One, Zero};

pub trait Trig: Sized {
    /// Returns the `cos` of `f` (where `f` is in radians)
    fn cos(f: f64) -> Self;

    /// Returns the `sin` of `f` (where `f` is in radians)
    fn sin(f: f64) -> Self;
}

macro_rules! impl_trig {
    ($($t:ty),*) => {
        $(
        impl Trig for $t {
            fn cos(f: f64) -> Self {
                f.cos() as $t
            }

            fn sin(f: f64) -> Self {
                f.sin() as $t
            }
        }
        )*
    };
}

impl_trig!(f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

/// Pre-defined types for 2 Dimensions
#[cfg(feature = "2d")]
pub mod clif2d {
    use super::*;

    register_multivector!(pub 2; pub 1, pub 2);
}

/// Pre-defined types for 3 Dimensions
#[cfg(feature = "3d")]
pub mod clif3d {
    use super::*;

    register_multivector!(pub 3; pub 1, pub 2, pub 3);
}
