use crate::{Adaptor, CrossProductAdaptor, ScalarAdaptor, TrigonometryAdaptor};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DVec<const DIM: usize>(pub [f64; DIM]);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec<const DIM: usize>(pub [f32; DIM]);

pub type DVec3 = DVec<3>;
pub type DVec2 = DVec<2>;
pub type Vec3 = Vec<3>;
pub type Vec2 = Vec<2>;

impl<const DIM: usize> DVec<DIM> {
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        self.0
            .iter()
            .zip(rhs.0.iter())
            .fold(0.0, |acc, (a, b)| a.mul_add(*b, acc))
    }

    #[inline(always)]
    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 { self } else { self / len }
    }
}

impl DVec<3> {
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }
}

impl<const DIM: usize> Index<usize> for DVec<DIM> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl<const DIM: usize> IndexMut<usize> for DVec<DIM> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl<const DIM: usize> Add for DVec<DIM> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        DVec(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const DIM: usize> Neg for DVec<DIM> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        DVec(std::array::from_fn(|i| -self.0[i]))
    }
}

impl<const DIM: usize> Sub for DVec<DIM> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        DVec(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const DIM: usize> Mul<f64> for DVec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        DVec(std::array::from_fn(|i| self.0[i] * rhs))
    }
}

impl<const DIM: usize> Mul<DVec<DIM>> for f64 {
    type Output = DVec<DIM>;
    #[inline(always)]
    fn mul(self, rhs: DVec<DIM>) -> DVec<DIM> {
        DVec(std::array::from_fn(|i| self * rhs.0[i]))
    }
}

impl<const DIM: usize> Div<f64> for DVec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        DVec(std::array::from_fn(|i| self.0[i] / rhs))
    }
}

impl<const DIM: usize> AddAssign for DVec<DIM> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<const DIM: usize> SubAssign for DVec<DIM> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl<const DIM: usize> MulAssign<f64> for DVec<DIM> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..DIM {
            self.0[i] *= rhs;
        }
    }
}

impl<const DIM: usize> DivAssign<f64> for DVec<DIM> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..DIM {
            self.0[i] /= rhs;
        }
    }
}

impl<const DIM: usize> Vec<DIM> {
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.0
            .iter()
            .zip(rhs.0.iter())
            .fold(0.0, |acc, (a, b)| a.mul_add(*b, acc))
    }

    #[inline(always)]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 { self } else { self / len }
    }
}

impl Vec<3> {
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }
}

impl<const DIM: usize> Index<usize> for Vec<DIM> {
    type Output = f32;
    #[inline(always)]
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

impl<const DIM: usize> IndexMut<usize> for Vec<DIM> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.0[i]
    }
}

impl<const DIM: usize> Add for Vec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Vec(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const DIM: usize> Neg for Vec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Vec(std::array::from_fn(|i| -self.0[i]))
    }
}

impl<const DIM: usize> Sub for Vec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Vec(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const DIM: usize> Mul<f32> for Vec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Vec(std::array::from_fn(|i| self.0[i] * rhs))
    }
}

impl<const DIM: usize> Mul<Vec<DIM>> for f32 {
    type Output = Vec<DIM>;
    #[inline(always)]
    fn mul(self, rhs: Vec<DIM>) -> Vec<DIM> {
        Vec(std::array::from_fn(|i| self * rhs.0[i]))
    }
}

impl<const DIM: usize> Div<f32> for Vec<DIM> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Vec(std::array::from_fn(|i| self.0[i] / rhs))
    }
}

impl<const DIM: usize> AddAssign for Vec<DIM> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<const DIM: usize> SubAssign for Vec<DIM> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..DIM {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl<const DIM: usize> MulAssign<f32> for Vec<DIM> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..DIM {
            self.0[i] *= rhs;
        }
    }
}

impl<const DIM: usize> DivAssign<f32> for Vec<DIM> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        for i in 0..DIM {
            self.0[i] /= rhs;
        }
    }
}

// -- Default Adaptor implementations --------------------------

#[derive(Clone)]
pub struct F32Adaptor;

#[derive(Clone)]
pub struct F64Adaptor;

impl ScalarAdaptor for F32Adaptor {
    type Float = f32;

    #[inline(always)]
    fn abs(x: Self::Float) -> Self::Float {
        x.abs()
    }

    #[inline(always)]
    fn sqrt(x: Self::Float) -> Self::Float {
        x.sqrt()
    }

    #[inline(always)]
    fn is_finite(x: Self::Float) -> bool {
        x.is_finite()
    }

    #[inline(always)]
    fn mul_add(x: Self::Float, a: Self::Float, b: Self::Float) -> Self::Float {
        x.mul_add(a, b)
    }

    #[inline(always)]
    fn scalar(val: f64) -> Self::Float {
        val as f32
    }

    #[inline(always)]
    fn epsilon() -> Self::Float {
        f32::EPSILON
    }

    #[inline(always)]
    fn min(x: Self::Float, other: Self::Float) -> Self::Float {
        x.min(other)
    }

    #[inline(always)]
    fn max(x: Self::Float, other: Self::Float) -> Self::Float {
        x.max(other)
    }

    #[inline(always)]
    fn clamp(x: Self::Float, lo: Self::Float, hi: Self::Float) -> Self::Float {
        x.clamp(lo, hi)
    }

    #[inline(always)]
    fn ceil(x: Self::Float) -> Self::Float {
        x.ceil()
    }

    #[inline(always)]
    fn to_usize(x: Self::Float) -> usize {
        x as usize
    }
}

impl TrigonometryAdaptor for F32Adaptor {
    #[inline(always)]
    fn acos(x: Self::Float) -> Self::Float {
        x.acos()
    }

    #[inline(always)]
    fn sin_cos(x: Self::Float) -> (Self::Float, Self::Float) {
        x.sin_cos()
    }

    #[inline(always)]
    fn sin(x: Self::Float) -> Self::Float {
        x.sin()
    }

    #[inline(always)]
    fn cos(x: Self::Float) -> Self::Float {
        x.cos()
    }

    #[inline(always)]
    fn tan(x: Self::Float) -> Self::Float {
        x.tan()
    }
}

impl ScalarAdaptor for F64Adaptor {
    type Float = f64;

    #[inline(always)]
    fn abs(x: Self::Float) -> Self::Float {
        x.abs()
    }

    #[inline(always)]
    fn sqrt(x: Self::Float) -> Self::Float {
        x.sqrt()
    }

    #[inline(always)]
    fn is_finite(x: Self::Float) -> bool {
        x.is_finite()
    }

    #[inline(always)]
    fn mul_add(x: Self::Float, a: Self::Float, b: Self::Float) -> Self::Float {
        x.mul_add(a, b)
    }

    #[inline(always)]
    fn scalar(val: f64) -> Self::Float {
        val as f64
    }

    #[inline(always)]
    fn epsilon() -> Self::Float {
        f64::EPSILON
    }

    #[inline(always)]
    fn min(x: Self::Float, other: Self::Float) -> Self::Float {
        x.min(other)
    }

    #[inline(always)]
    fn max(x: Self::Float, other: Self::Float) -> Self::Float {
        x.max(other)
    }

    #[inline(always)]
    fn clamp(x: Self::Float, lo: Self::Float, hi: Self::Float) -> Self::Float {
        x.clamp(lo, hi)
    }

    #[inline(always)]
    fn ceil(x: Self::Float) -> Self::Float {
        x.ceil()
    }

    #[inline(always)]
    fn to_usize(x: Self::Float) -> usize {
        x as usize
    }
}

impl TrigonometryAdaptor for F64Adaptor {
    #[inline(always)]
    fn acos(x: Self::Float) -> Self::Float {
        x.acos()
    }

    #[inline(always)]
    fn sin_cos(x: Self::Float) -> (Self::Float, Self::Float) {
        x.sin_cos()
    }

    #[inline(always)]
    fn sin(x: Self::Float) -> Self::Float {
        x.sin()
    }

    #[inline(always)]
    fn cos(x: Self::Float) -> Self::Float {
        x.cos()
    }

    #[inline(always)]
    fn tan(x: Self::Float) -> Self::Float {
        x.tan()
    }
}

impl<const DIM: usize> Adaptor<DIM> for F64Adaptor {
    type Vector = DVec<DIM>;
    type Scalar = f64;

    #[inline(always)]
    fn zero_vector() -> Self::Vector {
        DVec([0.0; DIM])
    }

    #[inline(always)]
    fn vector(coords: [Self::Scalar; DIM]) -> Self::Vector {
        DVec(coords)
    }

    #[inline(always)]
    fn vector_coord(v: &Self::Vector, i: usize) -> Self::Scalar {
        v[i]
    }

    #[inline(always)]
    fn vector_length(v: &Self::Vector) -> Self::Scalar {
        v.length()
    }

    #[inline(always)]
    fn vector_length_sq(v: &Self::Vector) -> Self::Scalar {
        v.length_sq()
    }

    #[inline(always)]
    fn normalize(v: Self::Vector) -> Self::Vector {
        v.normalize()
    }

    #[inline(always)]
    fn dot_product(a: &Self::Vector, b: &Self::Vector) -> Self::Scalar {
        a.dot(*b)
    }

    #[inline(always)]
    fn coord_arr(v: &Self::Vector) -> [Self::Scalar; DIM] {
        v.0
    }
}

impl CrossProductAdaptor<3> for F64Adaptor {
    #[inline(always)]
    fn cross(a: &Self::Vector, b: &Self::Vector) -> Self::Vector {
        DVec([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}

impl<const DIM: usize> Adaptor<DIM> for F32Adaptor {
    type Vector = Vec<DIM>;
    type Scalar = f32;

    #[inline(always)]
    fn zero_vector() -> Self::Vector {
        Vec([0.0; DIM])
    }

    #[inline(always)]
    fn vector(coords: [Self::Scalar; DIM]) -> Self::Vector {
        Vec(coords)
    }

    #[inline(always)]
    fn vector_coord(v: &Self::Vector, i: usize) -> Self::Scalar {
        v[i]
    }

    #[inline(always)]
    fn vector_length(v: &Self::Vector) -> Self::Scalar {
        v.length()
    }

    #[inline(always)]
    fn vector_length_sq(v: &Self::Vector) -> Self::Scalar {
        v.length_sq()
    }

    #[inline(always)]
    fn normalize(v: Self::Vector) -> Self::Vector {
        v.normalize()
    }

    #[inline(always)]
    fn dot_product(a: &Self::Vector, b: &Self::Vector) -> Self::Scalar {
        a.dot(*b)
    }

    #[inline(always)]
    fn coord_arr(v: &Self::Vector) -> [Self::Scalar; DIM] {
        v.0
    }
}

impl CrossProductAdaptor<3> for F32Adaptor {
    #[inline(always)]
    fn cross(a: &Self::Vector, b: &Self::Vector) -> Self::Vector {
        Vec([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}
