use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait ScalarTraits:
    'static
    + Sized
    + Copy
    + Clone
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + SubAssign
    + DivAssign
    + PartialEq
    + PartialOrd
{
}

pub trait VectorTraits:
    'static
    + Sized
    + Copy
    + std::fmt::Debug
    + Clone
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + Neg<Output = Self>
{
}

pub trait ScalarAdaptor {
    type Float: ScalarTraits;

    // Float ops.
    fn abs(x: Self::Float) -> Self::Float;
    fn sqrt(x: Self::Float) -> Self::Float;
    fn is_finite(x: Self::Float) -> bool;
    fn mul_add(x: Self::Float, a: Self::Float, b: Self::Float) -> Self::Float;
    fn scalar(val: f64) -> Self::Float;
    fn epsilon() -> Self::Float;
    fn min(x: Self::Float, other: Self::Float) -> Self::Float;
    fn max(x: Self::Float, other: Self::Float) -> Self::Float;
    fn clamp(x: Self::Float, lo: Self::Float, hi: Self::Float) -> Self::Float;
    fn ceil(x: Self::Float) -> Self::Float;
    fn to_usize(x: Self::Float) -> usize;
}

pub trait TrigonometryAdaptor: ScalarAdaptor {
    fn acos(x: Self::Float) -> Self::Float;
    fn sin_cos(x: Self::Float) -> (Self::Float, Self::Float);
    fn sin(x: Self::Float) -> Self::Float;
    fn cos(x: Self::Float) -> Self::Float;
    fn tan(x: Self::Float) -> Self::Float;
}

pub trait Adaptor<const DIM: usize>: Clone + ScalarAdaptor<Float = Self::Scalar> {
    type Vector: VectorTraits
        // Arithmetic with scalars.
        + Mul<Self::Scalar, Output = Self::Vector>
        + Div<Self::Scalar, Output = Self::Vector>
        + DivAssign<Self::Scalar>
        + MulAssign<Self::Scalar>;
    type Scalar: ScalarTraits + Mul<Self::Vector, Output = Self::Vector>;

    // Vector ops.
    fn zero_vector() -> Self::Vector;
    fn vector(coords: [Self::Scalar; DIM]) -> Self::Vector;
    fn vector_coord(v: Self::Vector, i: usize) -> Self::Scalar;
    fn vector_length(v: Self::Vector) -> Self::Scalar;
    fn vector_length_sq(v: Self::Vector) -> Self::Scalar;
    fn normalize(v: Self::Vector) -> Self::Vector;
    fn dot_product(a: Self::Vector, b: Self::Vector) -> Self::Scalar;
    fn coord_arr(v: Self::Vector) -> [Self::Scalar; DIM];
}

/// Functions to serialize/deserialize scalars.
pub trait SerialAdaptor {
    type Value;

    fn write(val: Self::Value, w: impl std::io::Write) -> Result<(), std::io::Error>;
    fn read(src: impl std::io::Read) -> Result<Self::Value, std::io::Error>;
}

// Helper implementations for types:

impl<T> ScalarTraits for T where
    T: 'static
        + Sized
        + Copy
        + Clone
        + std::fmt::Debug
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + MulAssign
        + SubAssign
        + DivAssign
        + PartialEq
        + PartialOrd
{
}

impl<T> VectorTraits for T where
    T: 'static
        + Sized
        + Copy
        + Clone
        + std::fmt::Debug
        + PartialEq
        + Add<Output = Self>
        + Sub<Output = Self>
        + AddAssign
        + SubAssign
        + Neg<Output = Self>
{
}
