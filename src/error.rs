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
