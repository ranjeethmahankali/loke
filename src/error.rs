#[derive(Debug)]
pub enum Error {
    OutOfBounds(&'static str),
    InvalidParameter,
    IncorrectKnotCount,
    InsufficientControlPoints,
}
