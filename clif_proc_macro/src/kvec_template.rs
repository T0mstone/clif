use crate::template::_def_multivector_Zdz::MultivectorZdz;
use clif::{Field, Zero};
use core::cmp::{Eq, PartialEq};
use core::convert::TryFrom;
use core::mem::MaybeUninit;
use core::ops::{Add, Div, Mul, Sub};
use core::ptr;

#[allow(non_upper_case_globals)]
const Zdz: usize = 3;
#[allow(non_upper_case_globals)]
const Zkz: usize = 2;
#[allow(non_upper_case_globals)]
const Zd_choose_kz: usize = 3;
//rm_all_before

/// A Zk_vectorz in Zdz dimensionZs(d)z
#[allow(non_camel_case_types)] //rm
#[derive(Copy, Clone)]
#[rustfmt::skip] //rm
/* @KVPub */ struct Zk_vectorzZdz<T: Field + Clone>([T; Zd_choose_kz]);

impl<T: Field + Clone + PartialEq> PartialEq for Zk_vectorzZdz<T> {
    fn eq(&self, other: &Self) -> bool {
        (0..Zd_choose_kz).all(|i| self.0[i] == other.0[i])
    }
}

impl<T: Field + Clone + Eq> Eq for Zk_vectorzZdz<T> {}

impl<T: Field + Clone> Zero for Zk_vectorzZdz<T> {
    fn zero() -> Self {
        let mut data: MaybeUninit<[T; Zd_choose_kz]> = MaybeUninit::uninit();

        let p = data.as_mut_ptr();

        for i in 0..Zd_choose_kz {
            let elt = unsafe { &mut (*p)[i] as *mut T };
            unsafe { elt.write(T::zero()) }
        }

        Self(unsafe { data.assume_init() })
    }

    fn is_zero(&self) -> bool {
        (0..Zd_choose_kz).all(|i| self.0[i].is_zero())
    }
}

impl<T: Field + Clone> Add for Zk_vectorzZdz<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        for i in 0..Zd_choose_kz {
            unsafe { ptr::write(&mut self.0[i], ptr::read(&self.0[i]) + ptr::read(&rhs.0[i])) }
        }
        self
    }
}

impl<T: Field + Clone> Sub for Zk_vectorzZdz<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (rhs * T::minus_one())
    }
}

impl<T: Field + Clone> Mul<T> for Zk_vectorzZdz<T> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        for i in 0..Zd_choose_kz {
            unsafe { ptr::write(&mut self.0[i], ptr::read(&self.0[i]) * rhs.clone()) }
        }
        self
    }
}

impl<T: Field + Clone> Div<T> for Zk_vectorzZdz<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        self * (T::one() / rhs)
    }
}

#[rustfmt::skip] //rm
fn permutation(_n: usize, _k: usize, _i: usize) -> usize { unimplemented!() } //rm
#[rustfmt::skip] //rm
/* @BinCoeffFn */ //only_if first
#[rustfmt::skip] //rm
/* @PermFn */ //only_if first

impl<T: Field + Clone> Into<MultivectorZdz<T>> for Zk_vectorzZdz<T> {
    fn into(self) -> MultivectorZdz<T> {
        let mut res = MultivectorZdz::zero().0;
        for i in 0..Zd_choose_kz {
            res[permutation(Zdz, Zkz, i)] = self.0[i].clone();
        }
        MultivectorZdz(res)
    }
}

impl<T: Field + Clone> TryFrom<MultivectorZdz<T>> for Zk_vectorzZdz<T> {
    type Error = ();

    fn try_from(value: MultivectorZdz<T>) -> Result<Self, Self::Error> {
        if value.grade() == Some(Zkz) {
            let mut res = Self::zero().0;
            for i in 0..Zd_choose_kz {
                res[i] = value.0[permutation(Zdz, Zkz, i)].clone();
            }
            Ok(Self(res))
        } else {
            Err(())
        }
    }
}

/// The geometric product
impl<T: Field + Clone> Mul<Zk_vectorzZdz<T>> for MultivectorZdz<T> {
    type Output = Self;

    fn mul(self, rhs: Zk_vectorzZdz<T>) -> Self {
        self * Into::<Self>::into(rhs)
    }
}

/// The geometric product
impl<T: Field + Clone> Mul<MultivectorZdz<T>> for Zk_vectorzZdz<T> {
    type Output = MultivectorZdz<T>;

    fn mul(self, rhs: MultivectorZdz<T>) -> MultivectorZdz<T> {
        Into::<MultivectorZdz<T>>::into(self) * rhs
    }
}

/// The geometric product
impl<T: Field + Clone> Mul for Zk_vectorzZdz<T> {
    type Output = MultivectorZdz<T>;

    fn mul(self, rhs: Self) -> MultivectorZdz<T> {
        Into::<MultivectorZdz<T>>::into(self) * Into::<MultivectorZdz<T>>::into(rhs)
    }
}
