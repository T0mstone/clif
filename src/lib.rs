use num_traits::Zero;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::{Add, Deref, Mul};

mod field;

pub use self::field::Field;

/*
Encoding for N Dimensions
1   =^= 0
e_n =^= 1 << n
a = product_i e_i; b = product_j e_j; a * b =^= a xor b // ascending order is implied; factors of (-1) computed separately

KEEP IN MIND: basis vectors are numbered from 0
*/

fn lowest_bit_pos(i: NonZeroUsize) -> usize {
    let mut r = 0;
    let mut i = i.get();

    loop {
        if i & 1 == 1 {
            return r;
        }

        i >>= 1;
        r += 1;
    }
}

mod repr {
    use super::lowest_bit_pos;
    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex};

    #[inline]
    pub fn grade(e_x: usize) -> u32 {
        e_x.count_ones()
    }

    pub fn make_basis_str(e_x: usize) -> String {
        let mut res = String::new();

        for i in 0.. {
            if e_x >> i == 0 {
                break;
            }

            if e_x & (1 << i) != 0 {
                res += &format!("e{}", i + 1);
            }
        }

        res
    }

    #[inline]
    fn fac_neg1(e_x: usize, e_y: usize) -> bool {
        match grade(e_y) {
            0 => false,
            // now e_y is guaranteed > 0
            1 => {
                let b = lowest_bit_pos(unsafe { NonZeroUsize::new_unchecked(e_y) }); // get the `n` from `e_n`
                let num_swaps = e_x & !(((1 << b) - 1) + (1 << b)); // an ugly version of `e_x & !(1 << (b + 1) - 1)` to avoid overflows
                num_swaps % 2 != 0
            }
            _ => {
                let b = lowest_bit_pos(unsafe { NonZeroUsize::new_unchecked(e_y) });
                let e_z = 1 << b;
                fac_neg1(e_x, e_z) ^ fac_neg1(e_x, e_y ^ e_z)
            }
        }
    }

    struct FacNeg1Buf(HashMap<(usize, usize), bool>);

    impl FacNeg1Buf {
        pub fn new() -> Self {
            Self(HashMap::new())
        }

        pub fn get(&mut self, i: usize, j: usize) -> bool {
            *self.0.entry((i, j)).or_insert_with(|| fac_neg1(i, j))
        }
    }

    static FAC_NEG1_BUFFER: Lazy<Arc<Mutex<FacNeg1Buf>>> =
        Lazy::new(|| Arc::new(Mutex::new(FacNeg1Buf::new())));

    #[inline]
    pub fn get_fac_neg_1(e_x: usize, e_y: usize) -> bool {
        let mut r = FAC_NEG1_BUFFER.lock().unwrap();
        r.get(e_x, e_y)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Pow2LengthList<T>(Vec<T>);

impl<T> Pow2LengthList<T> {
    pub fn new(v: Vec<T>) -> Option<Self> {
        if v.len().is_power_of_two() {
            Some(Self(v))
        } else {
            None
        }
    }

    pub fn len(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.0.len()) }
    }
}

impl<T> Deref for Pow2LengthList<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Multivector<T: Field>(Pow2LengthList<T>);

impl<T: Field + fmt::Debug> fmt::Debug for Multivector<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let d = self.dim();

        f.pad("Multivector[")?;
        f.pad(&format!("{}", d))?;
        f.pad("](")?;

        let mut strs = self
            .0
            .iter()
            .enumerate()
            .map(|(i, t)| (i, if t.is_zero() { None } else { Some(t) }))
            .collect::<Vec<_>>();

        strs.sort_by(
            |&(e_x, _), &(e_y, _)| match repr::grade(e_x).cmp(&repr::grade(e_y)) {
                Ordering::Equal => e_x.cmp(&e_y),
                r => r,
            },
        );

        for (not_first, (i, v)) in strs
            .into_iter()
            .filter_map(|(i, t)| t.map(|t| (i, t)))
            .enumerate()
            .map(|(idx, t)| (idx != 0, t))
        {
            if not_first {
                f.pad(" + ")?;
            }
            let basis_str = repr::make_basis_str(i);
            if basis_str.is_empty() {
                f.pad(&format!("{:?}", v))?;
            } else {
                f.pad(&format!("{:?}*{}", v, basis_str))?;
            }
        }

        f.pad(")")
    }
}

impl<T: Field> Multivector<T> {
    #[inline]
    pub fn from_data(v: Vec<T>) -> Option<Self> {
        Some(Self(Pow2LengthList::new(v)?))
    }

    pub fn build() -> MultivectorBuilder<T>
    where
        T: Clone,
    {
        MultivectorBuilder(HashMap::new())
    }

    #[inline]
    pub fn scalar(t: T) -> Self {
        Self::from_data(vec![t]).unwrap()
    }

    #[inline]
    pub fn vector(v: Vec<T>) -> Self
    where
        T: Clone,
    {
        let mut res = Self::build();
        for (i, t) in v.into_iter().enumerate() {
            res = res.with_component(BasisMultivector::one().switch_vector(i), t);
        }
        res.finish()
    }

    pub fn data(&self) -> &[T] {
        &self.0[..]
    }

    pub fn dim(&self) -> usize {
        lowest_bit_pos(self.0.len())
    }

    fn into_iter(self) -> std::vec::IntoIter<T> {
        let Self(Pow2LengthList(v)) = self;
        v.into_iter()
    }
}

impl<T: Field> Zero for Multivector<T> {
    fn zero() -> Self {
        Self(Pow2LengthList(vec![T::zero()]))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(Zero::is_zero)
    }
}

impl<T: Field + Clone> Mul for Multivector<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let len = self.0.len().max(rhs.0.len()).get(); // always: len > 0
        let mut res = vec![T::zero(); len];

        for (i, a) in self.into_iter().enumerate() {
            for (j, b) in rhs.clone().into_iter().enumerate() {
                let fac_n1 = self::repr::get_fac_neg_1(i, j);
                let c = a.clone() * b;
                let idx = i ^ j;
                if fac_n1 {
                    unsafe { std::ptr::write(&mut res[idx], std::ptr::read(&res[idx]) - c) };
                } else {
                    unsafe { std::ptr::write(&mut res[idx], std::ptr::read(&res[idx]) + c) };
                }
            }
        }

        // ok since `dim` is definetely a power of 2
        Self(Pow2LengthList(res))
    }
}

impl<T: Field + Clone> Mul<T> for Multivector<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let res = self.into_iter().map(|t| rhs.clone() * t).collect();
        Self(Pow2LengthList(res))
    }
}

impl<T: Field> Add for Multivector<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let res = self
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect();
        Self(Pow2LengthList(res))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BasisMultivector(pub HashSet<usize>);

impl BasisMultivector {
    #[inline]
    pub fn one() -> Self {
        Self(HashSet::new())
    }

    #[inline]
    pub fn from_vec(basis_vectors: Vec<usize>) -> Self {
        basis_vectors
            .into_iter()
            .fold(Self::one(), |s, i| s.switch_vector(i))
    }

    pub fn switch_vector(mut self, i: usize) -> Self {
        if self.0.contains(&i) {
            self.0.remove(&i);
        } else {
            self.0.insert(i);
        }
        self
    }

    #[inline]
    pub fn grade(&self) -> usize {
        self.0.len()
    }

    fn data_index(&self) -> usize {
        self.0.iter().fold(0, |acc, &i| acc | (1 << i))
    }

    fn min_dimension(&self) -> usize {
        // if the largest item is e6 (represented as `5`), then it needs at least 6 = 5 + 1 dimensions
        // if self.0 is empty, i.e. the represented basis multivector is 1, the dimensionality is 0
        self.0.iter().max().map_or(0, |i| i + 1)
    }
}

impl Hash for BasisMultivector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        let mut v = self.0.iter().collect::<Vec<_>>();
        v.sort();
        for u in v {
            u.hash(state);
        }
    }
}

pub struct MultivectorBuilder<T: Field + Clone>(HashMap<BasisMultivector, T>);

impl<T: Field + Clone> MultivectorBuilder<T> {
    pub fn with_component(mut self, basis: BasisMultivector, value: T) -> Self {
        self.0.insert(basis, value);
        self
    }

    pub fn finish(self) -> Multivector<T> {
        if self.0.is_empty() {
            return Multivector::zero();
        }

        let dim = self.0.iter().map(|(b, _)| b.min_dimension()).max().unwrap();
        let mut res = vec![T::zero(); 1 << dim];

        for (b, t) in self.0 {
            res[b.data_index()] = t;
        }

        Multivector(Pow2LengthList(res))
    }
}

#[macro_export]
macro_rules! multivector {
    (@inner {$($res:tt)*}) => {
        $crate::Multivector::build()
        $($res)*
        .finish()
    };
    (@inner {$($res:tt)*} [] $x:expr $(, [$($l:literal)*] $e:expr)*) => {
        multivector!(@inner {$($res)* .with_component($crate::BasisMultivector::one(), $x)} $([$($l)*] $e),*);
    };
    (@inner {$($res:tt)*} [$($li:literal)*] $x:expr $(, [$($l:literal)*] $e:expr)*) => {
        multivector!(@inner {$($res)* .with_component($crate::BasisMultivector::from_vec(vec![$($li - 1),*]), $x)} $([$($l)*] $e),*);
    };
    ($([$($l:literal)*] => $e:expr),+) => {
        multivector!(@inner {} $([$($l)*] $e),+);
    };
}
