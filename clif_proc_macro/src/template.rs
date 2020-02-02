#[allow(non_snake_case)] //rm

//insert mod _def_multivector_Zdz {
//rm_nextline
pub mod _def_multivector_Zdz {
    use clif::{Field, Trig, Zero};
    use core::cmp::{Eq, PartialEq};
    #[allow(unused)]
    use core::convert::TryFrom;
    use core::fmt;
    use core::iter::Iterator;
    use core::mem::MaybeUninit;
    use core::ops::{Add, Div, Mul, Sub};
    use core::ptr;

    #[allow(non_upper_case_globals)] //rm
    const Zlenz: usize = 8; //rm
    #[allow(non_upper_case_globals)] //rm
    const Zdz: u32 = 3; //rm
    #[allow(non_upper_case_globals)] //rm
    const Zd_usizez: usize = 3; //rm

    #[inline]
    #[allow(unused)] //rm
    fn calc_fac_n1(i: usize, j: usize) -> bool {
        unimplemented!() //rm

        /* @Fac-1Mat */
    }

    /// A Multivector in Zdz dimensionZs(d)z, using an orthonormal basis.
    ///
    /// The API is modeled after the Geometric Algebra as defined in ["Geometric Algebra" by Eric Chisolm](https://arxiv.org/pdf/1205.5935.pdf)
    #[derive(Copy, Clone)]
    //insert pub struct MultivectorZdz<T: Field + Clone>([T; Zlenz]);
    //rm_nextline
    pub struct MultivectorZdz<T: Field + Clone>(pub [T; Zlenz]);

    impl<T: Field + Clone + fmt::Debug> fmt::Debug for MultivectorZdz<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.pad("MultivectorZdz(")?;

            let mut not_first = false;
            let mut any = false;

            for grade in 0..=Zdz {
                for idx in (0usize..Zlenz).filter(|i| i.count_ones() == grade) {
                    let t = &self.0[idx];
                    if t.is_zero() {
                        continue;
                    }

                    if not_first {
                        f.pad(" + ")?;
                    } else {
                        not_first = true;
                    }

                    t.fmt(f)?;
                    any = true;

                    if grade == 0 {
                        continue;
                    }

                    f.pad("*")?;

                    for j in 0..Zdz {
                        if idx & (1 << j) != 0 {
                            f.pad("e")?;
                            (j + 1).fmt(f)?;
                        }
                    }
                }
            }

            if !any {
                T::zero().fmt(f)?;
            }

            f.pad(")")
        }
    }

    impl<T: Field + Clone + PartialEq> PartialEq for MultivectorZdz<T> {
        fn eq(&self, other: &Self) -> bool {
            (0..Zlenz).all(|i| self.0[i] == other.0[i])
        }
    }

    impl<T: Field + Clone + Eq> Eq for MultivectorZdz<T> {}

    impl<T: Field + Clone> Zero for MultivectorZdz<T> {
        fn zero() -> Self {
            let mut data: MaybeUninit<[T; Zlenz]> = MaybeUninit::uninit();

            let p = data.as_mut_ptr();

            for i in 0..Zlenz {
                let elt = unsafe { &mut (*p)[i] as *mut T };
                unsafe { elt.write(T::zero()) }
            }

            Self(unsafe { data.assume_init() })
        }

        fn is_zero(&self) -> bool {
            (0..Zlenz).all(|i| self.0[i].is_zero())
        }
    }

    impl<T: Field + Clone> Add for MultivectorZdz<T> {
        type Output = Self;

        fn add(mut self, rhs: Self) -> Self {
            for i in 0..Zlenz {
                unsafe { ptr::write(&mut self.0[i], ptr::read(&self.0[i]) + ptr::read(&rhs.0[i])) }
            }
            self
        }
    }

    impl<T: Field + Clone> Sub for MultivectorZdz<T> {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self + (rhs * T::minus_one())
        }
    }

    impl<T: Field + Clone> Mul<T> for MultivectorZdz<T> {
        type Output = Self;

        fn mul(mut self, rhs: T) -> Self {
            for i in 0..Zlenz {
                unsafe { ptr::write(&mut self.0[i], ptr::read(&self.0[i]) * rhs.clone()) }
            }
            self
        }
    }

    impl<T: Field + Clone> Div<T> for MultivectorZdz<T> {
        type Output = Self;

        fn div(mut self, rhs: T) -> Self {
            for i in 0..Zlenz {
                unsafe { ptr::write(&mut self.0[i], ptr::read(&self.0[i]) / rhs.clone()) }
            }
            self
        }
    }

    /// The geometric product
    impl<T: Field + Clone> Mul for MultivectorZdz<T> {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            let Self(mut res) = Self::zero();

            for i in 0..Zlenz {
                let a = &self.0[i];
                if a.is_zero() {
                    continue;
                }
                for j in 0..Zlenz {
                    let b = &rhs.0[j];
                    if b.is_zero() {
                        continue;
                    }

                    let fac_n1 = calc_fac_n1(i, j);

                    let c: T = a.clone() * b.clone();
                    let idx = i ^ j;
                    if fac_n1 {
                        unsafe { ptr::write(&mut res[idx], ptr::read(&res[idx]) - c) };
                    } else {
                        unsafe { ptr::write(&mut res[idx], ptr::read(&res[idx]) + c) };
                    }
                }
            }

            Self(res)
        }
    }

    #[allow(unused)] //rm
    impl<T: Field + Clone> MultivectorZdz<T> {
        /// Creates an empty `MultivectorZdzBuilder`
        pub fn build() -> MultivectorZdzBuilder<T> {
            MultivectorZdzBuilder(Self::zero().0)
        }

        /// A shortcut to create a Multivector which is only a scalar
        #[inline]
        pub fn scalar(t: T) -> Self {
            let Self(mut data) = Self::zero();
            data[0] = t;
            Self(data)
        }

        /// A shortcut to create a Multivector which is only a vector
        #[inline]
        pub fn vector(v: [T; Zd_usizez]) -> Self {
            let Self(mut data) = Self::zero();
            for i in 0..Zd_usizez {
                data[1 << i] = v[i].clone();
            }
            Self(data)
        }

        /// A shortcut to create a Multivector which is only a pseudoscalar
        #[inline]
        pub fn pseudoscalar(t: T) -> Self {
            let Self(mut data) = Self::zero();
            data[Zlenz - 1] = t;
            Self(data)
        }

        fn extract_grades(&self, f: impl Fn(usize) -> bool) -> Self {
            let Self(mut data) = Self::zero();

            for i in 0..Zlenz {
                if f(i.count_ones() as usize) {
                    data[i] = self.0[i].clone();
                }
            }

            Self(data)
        }

        fn lowest_grade(&self) -> usize {
            (0..Zlenz)
                .filter(|&i| !self.0[i].is_zero())
                .map(|i| i.count_ones() as usize)
                .next()
                // for if `self` is zero
                .unwrap_or(0)
        }

        /// Returns if the Multivector is made up of only parts of the same grade
        pub fn is_single_graded(&self) -> bool {
            let mut iter = (0..Zlenz).filter(|&i| !self.0[i].is_zero());
            let n = match iter.next() {
                Some(i) => i.count_ones(),
                None => return true,
            };
            iter.all(|i| i.count_ones() == n)
        }

        /// If the Multivector is made up of only parts of the same grade, that grade is returned; otherwise `None` is returned
        pub fn grade(&self) -> Option<usize> {
            let mut iter = (0..Zlenz).filter(|&i| !self.0[i].is_zero());
            let n = match iter.next() {
                Some(i) => i.count_ones(),
                None => return Some(0),
            };
            if iter.all(|i| i.count_ones() == n) {
                Some(n as usize)
            } else {
                None
            }
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

        /// The geometric product of the Multivector with itself
        pub fn square(self) -> Self {
            self.clone() * self
        }

        fn product_helper(self, rhs: Self, f: impl Fn(usize, usize) -> isize) -> Self {
            let mut res = Self::zero();

            for r in 0..=Zd_usizez {
                let s_r = self.grade_part(r as _);
                if s_r.is_zero() {
                    continue;
                }
                for s in 0..=Zd_usizez {
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

        /// The left inner product, defined as
        /// ```A.left_inner_prod(B) = sum over r,s with 0 <= r,s <= Zdz of (A.grade(r) * B.grade(s)).grade(s - r)```
        #[inline]
        pub fn left_inner_prod(self, rhs: Self) -> Self {
            self.product_helper(rhs, |r, s| s as isize - r as isize)
        }

        /// The right inner product, defined as
        /// ```A.left_inner_prod(B) = sum over r,s with 0 <= r,s <= Zdz of (A.grade(r) * B.grade(s)).grade(r - s)```
        #[inline]
        pub fn right_inner_prod(self, rhs: Self) -> Self {
            self.product_helper(rhs, |r, s| r as isize - s as isize)
        }

        /// The right inner product, defined as
        /// ```A.left_inner_prod(B) = sum over r,s with 0 <= r,s <= Zdz of (A.grade(r) * B.grade(s)).grade(r + s)```
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
            for i in 0..Zlenz {
                let r = i.count_ones();
                // (-1)^(x + 2n) = (-1)^x
                // => (-1)^(r (r-1) / 2) = (-1)^( (r (r-1) / 2) + 2r) = (-1)^(r (r + 3) / 2)
                let x = r * (r + 3) / 2;
                if x % 2 == 1 {
                    unsafe {
                        std::ptr::write(&mut self.0[i], std::ptr::read(&self.0[i]) * T::minus_one())
                    }
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

        /// The squared magnitude (scalar product with itself) of a Multivector
        #[inline]
        pub fn norm_sqr(&self) -> T {
            self.clone().scalar_prod(self.clone())
        }

        /// The multiplicative inverse of the Multivector
        #[inline]
        pub fn inverse(self) -> Self {
            let sq = self.norm_sqr();
            self.rev() / sq
        }

        /// The dual of `self` by `rhs`
        ///
        /// # Caution
        /// `rhs` should be an invertible blade! If it is not, the result has no meaning.
        pub fn dual_by(self, rhs: Self) -> Self {
            self.left_inner_prod(rhs.inverse())
        }

        /// The dual of `self` by the unit pseudoscalar
        pub fn dual(self, rhs: Self) -> Self {
            self.left_inner_prod(Self::pseudoscalar(T::one()))
        }

        /// The commutator operation on two Multivectors
        #[inline]
        pub fn commut(self, rhs: Self) -> Self {
            (self.clone() * rhs.clone() - rhs * self) / T::two()
        }

        /// Projects `self` onto the subspace defined by `rhs`.
        ///
        /// # Caution
        /// `rhs` should be an invertible blade! If it is not, the result has no meaning.
        pub fn project(self, rhs: Self) -> Self {
            self.left_inner_prod(rhs.clone()) * rhs.inverse()
        }

        /// Calculates the rejection of `self` from the subspace defined by `rhs`.
        ///
        /// # Caution
        /// `rhs` should be an invertible blade! If it is not, the result has no meaning.
        pub fn reject(self, rhs: Self) -> Self {
            self.outer_prod(rhs.clone()) * rhs.inverse()
        }

        /// Calculates the Reflection of `self` in the subspace defined by `rhs`
        ///
        /// # Caution
        /// `rhs` should be an invertible blade! If it is not, the result has no meaning.
        pub fn reflect(self, rhs: Self) -> Self {
            let r = rhs.lowest_grade();

            if r % 2 == 0 {
                rhs.clone() * self * rhs.inverse()
            } else {
                rhs.clone() * self.invol() * rhs.inverse()
            }
        }

        /// Rotates `self` by `angle` (in `plane`). `plane` is automatically normalized
        ///
        /// Returns `None` if `plane` is not a Bivector
        pub fn rotate(self, plane: Self, angle: f64) -> Option<Self>
        where
            T: Trig,
        {
            if plane.grade() != Some(2) {
                return None;
            }

            let half_angle = angle / 2.0;
            let nsq = plane.norm_sqr();
            let rotor = Self::scalar(T::cos(half_angle)) - (plane * T::sin(half_angle) / nsq);
            if rotor.is_zero() {
                panic!("T::sin(half_angle) and T::cos(half_angle) both returned zero.");
            }
            let inv = rotor.clone().inverse();
            Some(rotor * self * inv)
        }
    }

    /// A Basis Multivector, which is the geometric/outer product of some number of basis vectors
    #[derive(Clone)]
    pub struct BasisMultivectorZdz([bool; Zlenz]);

    #[allow(unused)] //rm
    impl BasisMultivectorZdz {
        /// The Basis Multivector for Scalars
        #[inline]
        pub fn one() -> Self {
            Self([false; Zlenz])
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
        #[inline]
        pub fn switch_vector(mut self, i: usize) -> Self {
            self.0[i] ^= true;
            self
        }

        fn data_index(&self) -> usize {
            let mut res = 0;
            for i in 0..Zlenz {
                if self.0[i] {
                    res |= 1 << i;
                }
            }
            res
        }
    }

    /// A Struct to build a Multivector
    pub struct MultivectorZdzBuilder<T: Field + Clone>([T; Zlenz]);

    #[allow(unused)] //rm
    impl<T: Field + Clone> MultivectorZdzBuilder<T> {
        /// Sets the component of the Multivector which corresponds to `basis` to the specified value
        pub fn with_component(mut self, basis: BasisMultivectorZdz, value: T) -> Self {
            self.0[basis.data_index()] = value;
            self
        }

        /// Extracts the contained Multivector
        #[inline]
        pub fn finish(self) -> MultivectorZdz<T> {
            MultivectorZdz(self.0)
        }
    }

    /// A macro to simplify creating a Multivector in Zdz dimensionZs(d)z.
    ///
    /// Syntax:
    /// ```no_run
    /// multivectorZdz!([] => t); // creates a scalar with value t
    /// multivectorZdz!([1] => t); // creates a vector with its e1 component being t //only_if dgt0
    /// multivectorZdz!([1] => t, [2] => u); // creates a vector with its e1 component being t and its e2 component being u //only_if dgt0
    /// multivectorZdz!([1 2] => t, [2] => u); // creates a vector with its e1e2 component being t and its e2 component being u //only_if dgt1
    /// multivectorZdz!([2 1] => t, [2] => u); // creates a vector with its e1e2 component being t and its e2 component being u //only_if dgt1
    /// ```
    //insert #[macro_export]
    #[allow(unused)] //rm
    macro_rules! multivectorZdz {
        (@inner {$($res:tt)*}) => {
            $crate::MultivectorZdz::build()
            $($res)*
            .finish()
        };
        (@inner {$($res:tt)*} [] $x:expr $(, [$($l:literal)*] $e:expr)*) => {
            multivectorZdz!(@inner {$($res)* .with_component($crate::BasisMultivectorZdz::one(), $x)} $([$($l)*] $e),*);
        };
        (@inner {$($res:tt)*} [$($li:literal)*] $x:expr $(, [$($l:literal)*] $e:expr)*) => {
            multivectorZdz!(@inner {$($res)* .with_component($crate::BasisMultivectorZdz::from_vec(vec![$($li - 1),*]), $x)} $([$($l)*] $e),*);
        };
        ($([$($l:literal)*] => $e:expr),+) => {
            multivectorZdz!(@inner {} $([$($l)*] $e),+);
        };
    }

    /* @Kvecs */
}

#[allow(unused)] //rm
#[rustfmt::skip] //rm
/* @MVpub */ use self::_def_multivector_Zdz::*;
