use std::fmt::Display;

#[derive(Debug)]
pub enum Error {
    OutOfBounds(&'static str),
    InvalidParameter,
    IncorrectKnotCount,
    InsufficientControlPoints,
    RadiusTooSmall,
    PointsCollinear,
    ArcCannotBeCircle,
    DegenerateValue,
    InvalidArgument,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OutOfBounds(msg) => write!(f, "Value out of bounds: {msg}"),
            Error::InvalidParameter => write!(f, "Invalid parameter"),
            Error::IncorrectKnotCount => write!(f, "Incorrect knot count"),
            Error::InsufficientControlPoints => {
                write!(f, "Insufficient control points to construct curve")
            }
            Error::RadiusTooSmall => write!(f, "Radius too small"),
            Error::PointsCollinear => write!(f, "Points cannot be collinear"),
            Error::ArcCannotBeCircle => write!(f, "Arc cannot be a full circle"),
            Error::DegenerateValue => write!(f, "Value is degenerate"),
            Error::InvalidArgument => write!(f, "Invalid argument"),
        }
    }
}
