use num_traits::Zero;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::{Add, Deref, Div, Mul, Sub};

// TODO: add support for non-orthonormal bases?

mod field;

pub use self::field::Field;

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

/*
Encoding for N Dimensional Basis Multivectors (Products of Basis Vectors in ascending Order; The Vector Basis is assumed to be Orthonormal)
1   =^= 0
e_n =^= 1 << n
a = product_i e_i; b = product_j e_j; a * b =^= a xor b // ascending order is implied; factors of (-1) computed separately

KEEP IN MIND: basis vectors are numbered from 0
*/
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
        if e_x == 0 || e_y == 0 {
            return false;
        }
        let b = lowest_bit_pos(unsafe { NonZeroUsize::new_unchecked(e_y) }); // get the `n` from `e_n` (0-based)
        match grade(e_y) {
            0 => unreachable!(),
            // now e_y is guaranteed > 0
            1 => {
                // an ugly version of `e_x & !(1 << (b + 1) - 1)` to avoid overflows
                // it extracts all bits higher than the `b`th of e_x
                let num_swaps = (e_x & !(((1 << b) - 1) + (1 << b))).count_ones();
                num_swaps % 2 != 0
            }
            _ => {
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

/// A `Vec` whose length is a power of two (this also implies it to not be 0)
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Pow2LengthVec<T>(Vec<T>);

impl<T> Pow2LengthVec<T> {
    /// Create a new `Pow2LengthVec` from a regular `Vec`. Returns `None` iff the `Vec`'s length is not a power of 2.
    pub fn new(v: Vec<T>) -> Option<Self> {
        if v.len().is_power_of_two() {
            Some(Self(v))
        } else {
            None
        }
    }

    /// The length of the `Pow2LengthVec`
    pub fn len(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.0.len()) }
    }
}

impl<T> Deref for Pow2LengthVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

/// A Multivector.
///
/// Its dimensionality is only runtime-known
/// (This is due to a limitation of Rust's const generics. This may be rewritten once expressions like `{1 << N}` can be used as array lengths)
#[derive(Clone, Eq, PartialEq)]
pub struct Multivector<T: Field + Clone>(Pow2LengthVec<T>);

impl<T: Field + Clone> Multivector<T> {
    #[inline]
    fn from_data(v: Vec<T>) -> Option<Self> {
        Some(Self(Pow2LengthVec::new(v)?))
    }

    /// Creates an empty [`MultivectorBuilder`](#struct.MultivectorBuilder)
    pub fn build() -> MultivectorBuilder<T> {
        MultivectorBuilder(HashMap::new())
    }

    /// A shortcut to create a Multivector which is only a scalar
    #[inline]
    pub fn scalar(t: T) -> Self {
        Self::from_data(vec![t]).unwrap()
    }

    /// A shortcut to create a Multivector which is only a vector
    #[inline]
    pub fn vector(v: Vec<T>) -> Self {
        let mut res = Self::build();
        for (i, t) in v.into_iter().enumerate() {
            res = res.with_component(BasisMultivector::one().switch_vector(i), t);
        }
        res.finish()
    }

    /// The dimensionality of the Multivector
    pub fn dim(&self) -> usize {
        lowest_bit_pos(self.0.len())
    }

    /// The geometric product of the Multivector with itself (which always is a scalar, so only `T` is returned)
    pub fn square(&self) -> T {
        (self.clone() * self.clone()).scalar_part()
    }

    /// The multiplicative inverse of the Multivector
    pub fn inverse(self) -> Self {
        let sq = self.square();
        self / sq
    }

    fn extract_grades(&self, f: impl Fn(usize) -> bool) -> Self {
        let mut res = vec![T::zero(); self.0.len().get()];
        for (i, t) in self.0.iter().enumerate() {
            if f(i.count_ones() as usize) {
                res[i] = t.clone();
            }
        }
        Self(Pow2LengthVec(res))
    }

    /// Extracts only the part of `self` that has the specified grade
    pub fn grade_part(&self, grade: isize) -> Self {
        if grade < 0 {
            return Self::zero();
        }
        let grade = grade as usize;

        self.extract_grades(|g| g == grade)
    }

    /// Extracts only the scalar (grade 0) part of `self`
    #[inline]
    pub fn scalar_part(&self) -> T {
        self.grade_part(0).0[0].clone()
    }

    /// Extracts only the part of `self` that has an even grade
    #[inline]
    pub fn even_grade_part(&self) -> Self {
        self.extract_grades(|g| g % 2 == 0)
    }

    /// Extracts only the part of `self` that has an odd grade
    #[inline]
    pub fn odd_grade_part(&self) -> Self {
        self.extract_grades(|g| g % 2 == 1)
    }

    fn product_helper(self, rhs: Self, f: impl Fn(usize, usize) -> isize) -> Self {
        let mut res = Self::zero();

        for r in 0..self.dim() {
            let s_r = self.grade_part(r as _);
            if s_r.is_zero() {
                continue;
            }
            for s in 0..rhs.dim() {
                let r_s = rhs.grade_part(s as _);
                if r_s.is_zero() {
                    continue;
                }
                let delta = (s_r.clone() * r_s).grade_part(f(r, s));
                res = res + delta;
            }
        }

        res
    }

    /// The left inner product, defined as A.left_inner_prod(B) =
    /// sum over r,s with 0 <= r < A.dim() and 0 <= s < B.dim() of (A.grade(r) * B.grade(s)).grade(s - r)
    #[inline]
    pub fn left_inner_prod(self, rhs: Self) -> Self {
        self.product_helper(rhs, |r, s| s as isize - r as isize)
    }

    /// The right inner product, defined as A.left_inner_prod(B) =
    /// sum over r,s with 0 <= r < A.dim() and 0 <= s < B.dim() of (A.grade(r) * B.grade(s)).grade(r - s)
    #[inline]
    pub fn right_inner_prod(self, rhs: Self) -> Self {
        self.product_helper(rhs, |r, s| r as isize - s as isize)
    }

    /// The right inner product, defined as A.left_inner_prod(B) =
    /// sum over r,s with 0 <= r < A.dim() and 0 <= s < B.dim() of (A.grade(r) * B.grade(s)).grade(r + s)
    #[inline]
    pub fn outer_prod(self, rhs: Self) -> Self {
        self.product_helper(rhs, |r, s| r as isize + s as isize)
    }

    /// The grade involution of the Multivector
    #[inline]
    pub fn invol(self) -> Self {
        self.even_grade_part() - self.odd_grade_part()
    }

    /// The reversion of the Multivector
    pub fn rev(mut self) -> Self {
        for (i, t) in (self.0).0.iter_mut().enumerate() {
            let r = i.count_ones();
            let x = r * (r - 1) / 2;
            if x % 2 == 1 {
                unsafe { std::ptr::write(t, std::ptr::read(t) * T::minus_one()) }
            }
        }

        self
    }

    /// The Clifford conjugation of the Multivector
    #[inline]
    pub fn conjug(self) -> Self {
        self.invol().rev()
    }

    /// The scalar product of two Multivectors.
    #[inline]
    pub fn scalar_prod(self, rhs: Self) -> T {
        (self.rev() * rhs).scalar_part()
    }

    /// The squared magnitude of a Multivector
    #[inline]
    pub fn magsqr(&self) -> T {
        self.clone().scalar_prod(self.clone())
    }

    /// The commutator operation on two Multivectors
    #[inline]
    pub fn commut(self, rhs: Self) -> Self {
        (self.clone() * rhs.clone() - rhs * self) / T::two()
    }

    fn into_iter(self) -> std::vec::IntoIter<T> {
        let Self(Pow2LengthVec(v)) = self;
        v.into_iter()
    }
}

impl<T: Field + Clone + fmt::Debug> fmt::Debug for Multivector<T> {
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

impl<T: Field + Clone> Zero for Multivector<T> {
    fn zero() -> Self {
        Self(Pow2LengthVec(vec![T::zero()]))
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
            if a.is_zero() {
                continue;
            }
            for (j, b) in rhs.clone().into_iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
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
        Self(Pow2LengthVec(res))
    }
}

impl<T: Field + Clone> Mul<T> for Multivector<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let res = self.into_iter().map(|t| rhs.clone() * t).collect();
        Self(Pow2LengthVec(res))
    }
}

impl<T: Field + Clone> Div<T> for Multivector<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let res = self.into_iter().map(|t| rhs.clone() / t).collect();
        Self(Pow2LengthVec(res))
    }
}

impl<T: Field + Clone> Add for Multivector<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let res = self
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect();
        Self(Pow2LengthVec(res))
    }
}

impl<T: Field + Clone> Sub for Multivector<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + rhs * T::minus_one()
    }
}

/// A Basis Multivector, which is the geometric/outer product of some number of basis vectors
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BasisMultivector(HashSet<usize>);

impl BasisMultivector {
    /// The Basis Multivector for Scalars
    #[inline]
    pub fn one() -> Self {
        Self(HashSet::new())
    }

    /// Takes a list of basis vectors and returns their geometric/outer product
    ///
    /// # Caution
    /// The index for the basis vectors is zero-based, so e1 would be represented as `0`, e2 as `1` and so on
    #[inline]
    pub fn from_vec(basis_vectors: Vec<usize>) -> Self {
        basis_vectors
            .into_iter()
            .fold(Self::one(), |s, i| s.switch_vector(i))
    }

    /// Switches if a basis vector should be included in the final product
    ///
    /// # Caution
    /// The index for the basis vector is zero-based, so e1 would be represented as `0`, e2 as `1` and so on
    pub fn switch_vector(mut self, i: usize) -> Self {
        if self.0.contains(&i) {
            self.0.remove(&i);
        } else {
            self.0.insert(i);
        }
        self
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

/// A Struct to build a Multivector
pub struct MultivectorBuilder<T: Field + Clone>(HashMap<BasisMultivector, T>);

impl<T: Field + Clone> MultivectorBuilder<T> {
    /// Sets the component of the Multivector which corresponds to `basis`` to the specified value
    pub fn with_component(mut self, basis: BasisMultivector, value: T) -> Self {
        self.0.insert(basis, value);
        self
    }

    /// Extracts the contained Multivector
    pub fn finish(self) -> Multivector<T> {
        if self.0.is_empty() {
            return Multivector::zero();
        }

        let dim = self.0.iter().map(|(b, _)| b.min_dimension()).max().unwrap();
        let mut res = vec![T::zero(); 1 << dim];

        for (b, t) in self.0 {
            res[b.data_index()] = t;
        }

        Multivector(Pow2LengthVec(res))
    }
}

/// A macro to simplify creating a Multivector.
///
/// Syntax:
/// ```no_run
/// multivector!([] => t); // creates a scalar with value t
/// multivector!([1] => t); // creates a vector with its e1 component being t
/// multivector!([1] => t, [2] => u); // creates a vector with its e1 component being t and its e2 component being u
/// multivector!([1 2] => t, [2] => u); // creates a vector with its e1e2 component being t and its e2 component being u
/// ```
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
