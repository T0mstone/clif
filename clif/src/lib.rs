mod field;

pub use crate::field::Field;
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
