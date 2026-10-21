use crate::Adaptor;

// TODO: This should be generic over an adaptor that is properly bounded to support all the curve
// types.  Each variant of this enum should be fixed size, without any heap allocations of it's
// own. If it needs variable length data, we store them in the parent projector and store offsets
// like the splines example. Or something in that spirit, to avoid memory fragmentation.
enum SegmentProjector<const DIM: usize, A: Adaptor<DIM>> {
    Polynomial {
        /*
        TODO:

        Cache the (P.P') polynomial, and the P' polynomial separately here.

        Those can be used to compute the (q.P') polynomial on the fly, and subtract it from P.P'.

        We then find roots of that, plus the ends of the segment to find the closest point.

        For memory compactness, this should only store the offsets that can be used to get the actual coefficients.

        We store one offset per dimension. From those offsets, we read coefficients, and use them as polynomials.
         */
        degree: usize,                                  // Of the original spline.
        position_dot_deriv_coeff_offsets: [usize; DIM], // Offsets to coefficients of degree = (2 * degree - 1) polynomial.
        deriv_coeff_offsets: [usize; DIM], // Offsets to coefficients of degree = degree - 1 polynomial.
    },
    Line {
        // TODO: Just cache the end points, and use them to compute the closest point.
        start: A::Vector,
        end: A::Vector,
    },
    Arc {
        // TODO: Do it per the task plan in PROJECT.org.

        // We should only store the reduced closed form dot product expression, and the end points
        // of the arc. That is all we need to do the projection. We don't need to copy the whole
        // arc.
    },
    EllipticArc {
        // TODO: Do it per the task plan in PROJECT.org.
    },
}

// TODO: In the future we will have a polycurve type. So this projector needs to work over arbitrary
// collections of segments. We should also break up the spline segments and flatten those into a
// single list with arcs, and lines etc all together.

pub struct Projector<const DIM: usize, A: Adaptor<DIM>> {
    // TODO: To start, we should just iterate over all segments, but later we can think about using a BVH.
    segment_bounds: Box<[(A::Vector, A::Vector)]>,
    polynomial_coefficients: Box<[A::Scalar]>,
    projectors: Box<[SegmentProjector<DIM, A>]>,
}
