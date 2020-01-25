use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Sub};

mod field_seal {
    pub trait FieldSeal {}
}

pub trait Field:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Zero
    + One
    + self::field_seal::FieldSeal
{
    fn minus_one() -> Self {
        Self::zero() - Self::one()
    }
}

impl<
        T: Sized
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Zero
            + One,
    > self::field_seal::FieldSeal for T
{
}

impl<
        T: Sized
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Zero
            + One,
    > Field for T
{
}
