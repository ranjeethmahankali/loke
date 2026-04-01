mod adaptor;
mod arc;
mod error;
mod spline;
mod vec;

#[doc(hidden)]
pub mod polynomial;

pub use adaptor::{Adaptor, CrossProductAdaptor, ScalarAdaptor, TrigonometryAdaptor};
pub use arc::{Arc, Arc2d, Arc2f, Arc3d, Arc3f};
pub use error::Error;
pub use spline::{Spline, Spline2d, Spline2f, Spline3d, Spline3f};
pub use vec::{DVec, DVec2, DVec3, F32Adaptor, F64Adaptor, Vec, Vec2, Vec3};
