use crate::{Adaptor, Error, F32Adaptor, F64Adaptor, ScalarAdaptor, SerialAdaptor, polynomial};
use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub struct Spline<const DIM: usize, A>
where
    A: Adaptor<DIM>,
{
    knots: Vec<A::Scalar>,
    control_points: Vec<A::Vector>,
    power_basis_coeff: Vec<A::Scalar>,
    unique_knots: Vec<A::Scalar>,
}

/// A 3D spline using the built-in f64 vector type.
pub type Spline3d = Spline<3, F64Adaptor>;

/// A 2D spline using the built-in f64 vector type.
pub type Spline2d = Spline<2, F64Adaptor>;

/// A 3D spline using the built-in f32 vector type.
pub type Spline3f = Spline<3, F32Adaptor>;

/// A 2D spline using the built-in f32 vector type.
pub type Spline2f = Spline<2, F32Adaptor>;

impl<const DIM: usize, A> Spline<DIM, A>
where
    A: Adaptor<DIM>,
{
    pub fn create<Points, Knots>(
        control_points: Points,
        knots: Knots,
        degree: usize,
    ) -> Result<Self, Error>
    where
        Points: Into<Vec<A::Vector>>,
        Knots: Into<Vec<A::Scalar>>,
    {
        let control_points = control_points.into();
        let knots = knots.into();
        if control_points.len() + degree + 1 != knots.len() {
            return Err(Error::IncorrectKnotCount);
        }
        let power_basis_coeff = compute_polynomial_coeff::<DIM, A>(&knots, &control_points);
        let mut unique_knots = knots.clone();
        unique_knots.dedup();
        assert_eq!(
            power_basis_coeff.len(),
            (degree + 1) * (unique_knots.len() - 1) * DIM
        );
        Ok(Self {
            knots,
            control_points,
            power_basis_coeff,
            unique_knots,
        })
    }

    /// Create a clamped spline with uniform interior knots.
    /// Requires `control_points.len() >= degree + 1`.
    pub fn create_clamped<Points>(control_points: Points, degree: usize) -> Result<Self, Error>
    where
        Points: Into<Vec<A::Vector>>,
    {
        let control_points = control_points.into();
        let nclamp = degree + 1;
        if control_points.len() < nclamp {
            return Err(Error::InsufficientControlPoints);
        }
        let ntotal = control_points.len() + nclamp;
        let nmiddle = ntotal - 2 * nclamp;
        let mut knots = Vec::with_capacity(ntotal);
        knots.extend(
            std::iter::repeat_n(A::scalar(0.0), nclamp)
                .chain((0..nmiddle).map(|i| A::scalar((i + 1) as f64)))
                .chain(std::iter::repeat_n(A::scalar((nmiddle + 1) as f64), nclamp)),
        );
        assert_eq!(knots.len(), ntotal);
        let power_basis_coeff = compute_polynomial_coeff::<DIM, A>(&knots, &control_points);
        let mut unique_knots = knots.clone();
        unique_knots.dedup();
        assert_eq!(
            power_basis_coeff.len(),
            (degree + 1) * (unique_knots.len() - 1) * DIM
        );
        Ok(Self {
            knots,
            control_points,
            power_basis_coeff,
            unique_knots,
        })
    }

    pub fn degree(&self) -> usize {
        self.knots.len() - self.control_points.len() - 1
    }

    pub fn control_points(&self) -> &[A::Vector] {
        &self.control_points
    }

    pub fn domain(&self) -> (A::Scalar, A::Scalar) {
        let degree = self.degree();
        (
            self.knots[degree],
            self.knots[self.knots.len() - degree - 1],
        )
    }

    #[inline(always)]
    fn valid_param(&self, u: A::Scalar) -> bool {
        let (lo, hi) = self.domain();
        u >= lo && u <= hi
    }

    pub fn point(&self, u: A::Scalar) -> Option<A::Vector> {
        // Check if point is in domain.
        if !self.valid_param(u) {
            return None;
        }
        // Compute points with borrowed buffers.
        let degree = self.degree();
        let span = find_span::<A>(&self.knots, degree, u);
        Some(with_thread_local_buffers::<_, A::Scalar, _, _>(
            [degree + 1, degree + 1, degree + 1],
            |[basis, left, right]| {
                calc_basis::<A>(span, u, degree, &self.knots, [basis, left, right]);
                basis
                    .iter()
                    .zip(self.control_points[span - degree..].iter())
                    .fold(A::zero_vector(), |acc, (b, v)| acc + *b * *v)
            },
        ))
    }

    pub fn point_with_derivs(&self, u: A::Scalar, results: &mut [A::Vector]) -> Result<(), Error> {
        if results.is_empty() {
            return Ok(()); // Nothing to evaluate.
        }
        if !self.valid_param(u) {
            return Err(Error::InvalidParameter);
        }
        let degree = self.degree();
        let n_derivs = (results.len() - 1).min(degree); // Because higher order derivatives over degree-th are all zero.
        let span = find_span::<A>(&self.knots, degree, u);
        with_thread_local_buffers::<_, A::Scalar, _, _>(
            [
                (n_derivs + 1) * (degree + 1),
                (degree + 1).pow(2),
                (degree + 1) * 2,
                degree + 1,
                degree + 1,
            ],
            |[basis, ndu, alt_coeff, left, right]| {
                calc_ders_basis::<A>(
                    span,
                    u,
                    degree,
                    n_derivs,
                    &self.knots,
                    [basis, ndu, alt_coeff, left, right],
                );
                assert_eq!(basis.len(), (n_derivs + 1) * (degree + 1));
                let nders = View2D::create(basis, n_derivs + 1, degree + 1);
                let span = span - degree;
                results.fill(A::zero_vector());
                for (k, dst) in results.iter_mut().take(n_derivs + 1).enumerate() {
                    let nders = &nders[k];
                    *dst = nders
                        .iter()
                        .zip(self.control_points[span..].iter())
                        .fold(A::zero_vector(), |acc, (b, cp)| acc + *b * *cp)
                }
            },
        );
        Ok(())
    }

    pub fn tangent(&self, u: A::Scalar) -> Option<A::Vector> {
        let mut results = [A::zero_vector(); 2];
        self.point_with_derivs(u, &mut results)
            .ok()
            .map(|()| results[1])
    }

    pub fn curvature(&self, u: A::Scalar) -> Option<A::Vector> {
        let mut results = [A::zero_vector(); 3];
        self.point_with_derivs(u, &mut results)
            .ok()
            .map(|()| results[2])
    }

    pub fn start(&self) -> A::Vector {
        self.point(self.domain().0)
            .expect("Internal error, can never fail")
    }

    pub fn end(&self) -> A::Vector {
        self.point(self.domain().1)
            .expect("Internal error, can never fail")
    }

    pub fn bounds(&self) -> (A::Vector, A::Vector) {
        let degree = self.degree();
        with_thread_local_buffers([degree, degree.saturating_sub(1)], |[deriv, roots]| {
            let n_segs = self.unique_knots.len() - 1;
            assert!(n_segs > 0, "INTERNAL ERROR: Should never fail");
            let mut lo_arr = [A::scalar(0.0); DIM];
            let mut hi_arr = [A::scalar(0.0); DIM];
            let n_coeff = degree + 1;
            let mut poly: &[A::Scalar] = &self.power_basis_coeff;
            for ci in 0..DIM {
                let val = polynomial::eval::<A>(&poly[..n_coeff], A::scalar(0.0));
                let mut lo = val;
                let mut hi = val;
                for _ in 0..n_segs {
                    let coeff = &poly[..n_coeff];
                    let val = polynomial::eval::<A>(coeff, A::scalar(1.0));
                    lo = A::min(lo, val);
                    hi = A::max(hi, val);
                    if degree > 1 {
                        polynomial::differentiate::<A>(coeff, deriv);
                        let n_roots = polynomial::polynomial_roots_in_range::<A>(
                            deriv,
                            roots,
                            A::scalar(0.0),
                            A::scalar(1.0),
                            A::epsilon(),
                        )
                        .expect("This is an internal error. This should never happen");
                        for r in &roots[0..n_roots] {
                            let val = polynomial::eval::<A>(coeff, *r);
                            lo = A::min(lo, val);
                            hi = A::max(hi, val);
                        }
                    }
                    poly = &poly[n_coeff..]; // Stride to the next polynomial coefficients.
                }
                lo_arr[ci] = lo;
                hi_arr[ci] = hi;
            }
            (A::vector(lo_arr), A::vector(hi_arr))
        })
    }

    pub fn adaptive_samples(&self, tolerance: A::Scalar) -> impl Iterator<Item = A::Vector> {
        SplineAdaptiveSamples::new(self, tolerance).map(|AdaptiveSample { point, .. }| point)
    }

    pub fn length(&self, tolerance: A::Scalar) -> A::Scalar {
        let mut samples = SplineAdaptiveSamples::new(self, tolerance);
        let first = match samples.next() {
            Some(first) => first,
            None => return A::scalar(0.0),
        };
        samples
            .fold((first, A::scalar(0.0)), |(prev, total), s| {
                let pt = s.point;
                (
                    s,
                    total
                        + curvature_adjusted_arc_length::<DIM, A>(
                            prev.point,
                            pt,
                            prev.curvature_magnitude_sq,
                            prev.param_step,
                        ),
                )
            })
            .1
    }

    pub fn uniform_samples(
        &self,
        start: A::Scalar,
        step: A::Scalar,
        tolerance: A::Scalar,
    ) -> impl Iterator<Item = A::Vector> {
        SplineUniformSamples::new(self, start, step, tolerance).map(|s| s.point)
    }

    pub fn reversed(mut self) -> Self {
        // Section 6.5 from The NURBS Book.
        let (dom_start, dom_end) = self.domain();
        let ksum = dom_start + dom_end;
        self.control_points.reverse();
        for k in self.knots.iter_mut() {
            *k = ksum - *k;
        }
        self.knots.reverse();
        self.unique_knots = self.knots.clone();
        self.unique_knots.dedup();
        self.power_basis_coeff =
            compute_polynomial_coeff::<DIM, A>(&self.knots, &self.control_points);
        self
    }

    pub fn with_control_points<F>(mut self, func: F) -> Self
    where
        F: Fn(&A::Vector) -> A::Vector,
    {
        for p in self.control_points.iter_mut() {
            *p = func(p);
        }
        self.power_basis_coeff =
            compute_polynomial_coeff::<DIM, A>(&self.knots, &self.control_points);
        self
    }

    pub fn is_closed(&self) -> bool {
        self.start() == self.end()
    }

    fn power_basis_polynomial(&self, segment: usize, coord: usize) -> &[A::Scalar] {
        /*
        The power basis coeffients are stored in the following order:
        - The coefficients of the x polynomial of the first segment.
        - The coefficients of the x polynomial of the second segment.
        ...
        - The coefficients of the x polynomial of the last segment.
        - The coefficients of the y polynomial of the first segment.
        ...
        - The coefficients of the y polynomial of the last segment.
        - The coefficients of the z polynomial of the first segment.
        ...
        - The coefficients of the z polynomial of the last segment.
        ...
        And so on for however many dimensions there are.
         */
        let n_segs = self.unique_knots.len() - 1;
        let n_coeff = self.degree() + 1;
        let offset = (n_coeff * n_segs * coord) + (n_coeff * segment);
        &self.power_basis_coeff[offset..(offset + n_coeff)]
    }

    pub fn serialize(&self, mut w: impl std::io::Write) -> Result<(), std::io::Error>
    where
        A: SerialAdaptor<Value = A::Scalar>,
    {
        // Write n_knots, then n_control_points as a 64 bit integer each.  Then write n_knots
        // scalars via the serial adaptor.  Then write DIM x n_control_points scalars (all coords of
        // one vector one after another in x, y, z, x, y, z, pattern except dimension agnostic).
        w.write_all(&(self.knots.len() as u64).to_ne_bytes())?;
        w.write_all(&(self.control_points.len() as u64).to_ne_bytes())?;
        for &k in &self.knots {
            A::write(k, &mut w)?;
        }
        for &cp in &self.control_points {
            for coord in A::coord_arr(cp) {
                A::write(coord, &mut w)?;
            }
        }
        Ok(())
    }

    pub fn deserialize(mut src: impl std::io::Read) -> Result<Self, std::io::Error>
    where
        A: SerialAdaptor<Value = A::Scalar>,
    {
        // Deserialize the same order what was written out by serialize.
        let read_u64 = |src: &mut dyn std::io::Read| -> Result<u64, std::io::Error> {
            let mut buf = [0u8; std::mem::size_of::<u64>()];
            src.read_exact(&mut buf)?;
            Ok(u64::from_ne_bytes(buf))
        };
        let n_knots = read_u64(&mut src)? as usize;
        let n_control_points = read_u64(&mut src)? as usize;
        let mut knots = Vec::with_capacity(n_knots);
        for _ in 0..n_knots {
            knots.push(A::read(&mut src)?);
        }
        let mut control_points = Vec::with_capacity(n_control_points);
        for _ in 0..n_control_points {
            let mut coords = [A::scalar(0.0); DIM];
            for c in coords.iter_mut() {
                *c = A::read(&mut src)?;
            }
            control_points.push(A::vector(coords));
        }
        let mut unique_knots = knots.clone();
        unique_knots.dedup();
        let power_basis_coeff = compute_polynomial_coeff::<DIM, A>(&knots, &control_points);
        Ok(Self {
            knots,
            control_points,
            power_basis_coeff,
            unique_knots,
        })
    }
}

fn curvature_adjusted_arc_length<const DIM: usize, A: Adaptor<DIM>>(
    prev_pt: A::Vector,
    next_pt: A::Vector,
    curv_mag2: A::Scalar,
    param_delta: A::Scalar,
) -> A::Scalar {
    let mut len = A::vector_length(next_pt - prev_pt); // Chord length
    if len > A::scalar(0.0) {
        // Sagitta (midpoint deviation) from the curvature at the previous sample: h ≈ |C''(t)| * Δt² / 8
        // Taylor expansion of circular arc length given chord c and sagitta h: arc ≈ c + 8h²/(3c)
        let h: A::Scalar = A::sqrt(curv_mag2) * param_delta * param_delta / A::scalar(8.0);
        len += A::scalar(8.0) * h * h / (A::scalar(3.0) * len);
    }
    len
}

fn power_basis_param_to_spline_basis<A: ScalarAdaptor>(
    local_param: A::Float,
    segment_index: usize,
    unique_knots: &[A::Float],
) -> Option<A::Float> {
    if let (Some(left), Some(right)) = (
        unique_knots.get(segment_index),
        unique_knots.get(segment_index + 1),
    ) && local_param >= A::scalar(0.0)
        && local_param <= A::scalar(1.0)
    {
        Some(*left * (A::scalar(1.0) - local_param) + *right * local_param)
    } else {
        None
    }
}

struct SplineAdaptiveSamples<'a, const DIM: usize, A: Adaptor<DIM>> {
    curve: &'a Spline<DIM, A>,
    n_coeff: usize,
    n_segments: usize,
    curvature_magnitude_sq: Box<[A::Scalar]>,
    tolerance: A::Scalar,
    segment_index: usize,
    param: A::Scalar,
}

impl<'a, const DIM: usize, A: Adaptor<DIM>> SplineAdaptiveSamples<'a, DIM, A> {
    fn new(curve: &'a Spline<DIM, A>, tolerance: A::Scalar) -> Self {
        let degree = curve.degree();
        let n_segments = curve.unique_knots.len() - 1;
        // Compute the polynomials for the curvature magnitude squared.
        // Compute second derivative polynomials with a temporary buffer for the first derivative.
        let n_coeff_cmag2 = degree.saturating_sub(2) * 2 + 1;
        let mut curvature_magnitude_sq =
            vec![A::scalar(0.0); n_coeff_cmag2 * n_segments].into_boxed_slice();
        with_thread_local_buffers(
            [degree, degree.saturating_sub(1)],
            |[deriv_buf, curv_buf]| {
                for ci in 0..DIM {
                    for (si, dst) in curvature_magnitude_sq
                        .chunks_exact_mut(n_coeff_cmag2)
                        .enumerate()
                    {
                        deriv_buf.fill(A::scalar(0.0));
                        polynomial::differentiate::<A>(
                            curve.power_basis_polynomial(si, ci),
                            deriv_buf,
                        );
                        curv_buf.fill(A::scalar(0.0));
                        polynomial::differentiate::<A>(deriv_buf, curv_buf);
                        polynomial::mul_add::<A>(curv_buf, curv_buf, dst);
                    }
                }
            },
        );
        SplineAdaptiveSamples {
            curve,
            n_coeff: n_coeff_cmag2,
            n_segments,
            curvature_magnitude_sq,
            tolerance: A::abs(tolerance) / A::scalar(degree as f64),
            segment_index: 0,
            param: A::scalar(0.0),
        }
    }
}

struct AdaptiveSample<const DIM: usize, A: Adaptor<DIM>> {
    point: A::Vector,
    local_param: A::Scalar,
    segment_index: usize,
    param_step: A::Scalar,
    curvature_magnitude_sq: A::Scalar,
}

struct SplineSample<const DIM: usize, A: Adaptor<DIM>> {
    point: A::Vector,
}

impl<'a, const DIM: usize, A: Adaptor<DIM>> Iterator for SplineAdaptiveSamples<'a, DIM, A> {
    type Item = AdaptiveSample<DIM, A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segment_index >= self.n_segments {
            return None;
        }
        // We assume the previous iteration left us in a clean state, and attempt to evaluate a point.
        let coords: [A::Scalar; DIM] = std::array::from_fn(|ci| {
            polynomial::eval::<A>(
                self.curve.power_basis_polynomial(self.segment_index, ci),
                self.param,
            )
        });
        let out_local_param = self.param;
        let out_segment_idx = self.segment_index;
        // Try to advance.
        if self.param == A::scalar(1.0) {
            self.segment_index += 1;
            self.param = A::scalar(0.0);
        }
        let (cmag2, step) = if self.segment_index < self.n_segments {
            let offset = self.n_coeff * self.segment_index;
            let cmag2 = A::abs(polynomial::eval::<A>(
                &self.curvature_magnitude_sq[offset..(offset + self.n_coeff)],
                self.param,
            ));
            let step = if cmag2 == A::scalar(0.0) {
                A::scalar(2.0) // Just a large enough value to snap to the end of this segment.
            } else {
                const MIN_STEP: f64 = 1e-4; // 10K samples per segment is already an overkill.
                A::max(
                    A::sqrt(A::scalar(8.0) * self.tolerance / A::sqrt(cmag2)),
                    A::scalar(MIN_STEP),
                )
            };
            self.param += step; // Always increment no matter how small the step.
            (cmag2, step)
        } else {
            (A::scalar(0.0), A::scalar(0.0))
        };
        if self.param > A::scalar(1.0) {
            self.param = A::scalar(1.0);
        }
        Some(AdaptiveSample {
            point: A::vector(coords),
            local_param: out_local_param,
            segment_index: out_segment_idx,
            param_step: step,
            curvature_magnitude_sq: cmag2,
        })
    }
}

struct SplineUniformSamples<'a, const DIM: usize, A: Adaptor<DIM>> {
    finished: bool,
    sampler: SplineAdaptiveSamples<'a, DIM, A>,
    tprev: A::Scalar,
    tnext: A::Scalar,
    prev: AdaptiveSample<DIM, A>,
    next: AdaptiveSample<DIM, A>,
    dist: A::Scalar,
    step: A::Scalar,
    lspan: A::Scalar,
}

impl<'a, const DIM: usize, A: Adaptor<DIM>> SplineUniformSamples<'a, DIM, A> {
    pub fn new(
        curve: &'a Spline<DIM, A>,
        start: A::Scalar,
        step: A::Scalar,
        tolerance: A::Scalar,
    ) -> Self {
        let mut sampler = SplineAdaptiveSamples::new(curve, tolerance);
        let (tprev, tnext, prev, next) = match (sampler.next(), sampler.next()) {
            (Some(p), Some(n)) => {
                match (
                    power_basis_param_to_spline_basis::<A>(
                        p.local_param,
                        p.segment_index,
                        &curve.unique_knots,
                    ),
                    power_basis_param_to_spline_basis::<A>(
                        n.local_param,
                        n.segment_index,
                        &curve.unique_knots,
                    ),
                ) {
                    (Some(tprev), Some(tnext)) => (tprev, tnext, p, n),
                    _ => return Self::empty(sampler),
                }
            }
            _ => {
                return Self::empty(sampler);
            }
        };
        let lspan = curvature_adjusted_arc_length::<DIM, A>(
            prev.point,
            next.point,
            prev.curvature_magnitude_sq,
            prev.param_step,
        );
        Self {
            finished: false,
            sampler,
            tprev,
            tnext,
            prev,
            next,
            dist: start,
            step,
            lspan,
        }
    }

    pub fn empty(sampler: SplineAdaptiveSamples<'a, DIM, A>) -> Self {
        Self {
            finished: true,
            sampler,
            tprev: A::scalar(0.0),
            tnext: A::scalar(0.0),
            prev: AdaptiveSample {
                point: A::zero_vector(),
                local_param: A::scalar(0.0),
                segment_index: 0,
                param_step: A::scalar(0.0),
                curvature_magnitude_sq: A::scalar(0.0),
            },
            next: AdaptiveSample {
                point: A::zero_vector(),
                local_param: A::scalar(0.0),
                segment_index: 0,
                param_step: A::scalar(0.0),
                curvature_magnitude_sq: A::scalar(0.0),
            },
            dist: A::scalar(0.0),
            step: A::scalar(0.0),
            lspan: A::scalar(0.0),
        }
    }
}

impl<'a, const DIM: usize, A: Adaptor<DIM>> Iterator for SplineUniformSamples<'a, DIM, A> {
    type Item = SplineSample<DIM, A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        while self.lspan <= A::scalar(0.0) || self.dist > self.lspan {
            self.dist -= self.lspan;
            self.prev = std::mem::replace(
                &mut self.next,
                match self.sampler.next() {
                    Some(s) => s,
                    None => {
                        self.finished = true;
                        return None;
                    }
                },
            );
            self.tprev = std::mem::replace(
                &mut self.tnext,
                match power_basis_param_to_spline_basis::<A>(
                    self.next.local_param,
                    self.next.segment_index,
                    &self.sampler.curve.unique_knots,
                ) {
                    Some(tnext) => tnext,
                    None => {
                        self.finished = true;
                        return None;
                    }
                },
            );
            self.lspan = curvature_adjusted_arc_length::<DIM, A>(
                self.prev.point,
                self.next.point,
                self.prev.curvature_magnitude_sq,
                self.prev.param_step,
            );
        }
        let t = A::clamp(
            (self.dist * self.tnext + (self.lspan - self.dist) * self.tprev) / self.lspan,
            self.tprev,
            self.tnext,
        );
        self.dist += self.step;
        let point = match self.sampler.curve.point(t) {
            Some(pt) => pt,
            None => {
                self.finished = true;
                return None;
            }
        };
        Some(SplineSample { point })
    }
}

/*
The NURBS book uses the following conventions:
- p is the degree
- n + 1 is the number of control points (so `n` is the last control point index)
- m + 1 is the number of knots, with `m = n + p + 1`

The '+ 1' thing is to make it easier in the notation. U_m is easier than U_(m-1) as the last knot. Same with control points.

The relationship m = n + p + 1 holds true.
 */

/// This is based on algorithm A2.1 from the book.
fn find_span<A: ScalarAdaptor>(knots: &[A::Float], degree: usize, u: A::Float) -> usize {
    let n = knots.len() - degree - 2;
    if u == knots[n + 1] {
        // Special case at the end of the domain
        return n;
    }
    let low = degree;
    let hi = n + 1;
    // The book manually implements the binary search, here we use the std function that does the same.
    low + knots[low..hi].partition_point(|k| k <= &u) - 1
}

/// Compute the power-basis polynomial coefficients from the piecewise Bézier
/// decomposition. The output is a flat buffer of dimensions
/// `(degree+1) × n_segments × DIM`, packed so that the first `degree+1` values
/// are the x polynomial of segment 0, followed by segment 1, etc., then all y
/// polynomials, then all z polynomials.
fn compute_polynomial_coeff<const DIM: usize, A: Adaptor<DIM>>(
    knots: &[A::Scalar],
    control_points: &[A::Vector],
) -> Vec<A::Scalar> {
    let mut bezier_cps: Vec<A::Vector> = Vec::new();
    piecewise_bezier::<DIM, A>(knots, control_points, &mut bezier_cps);
    let p = knots.len() - control_points.len() - 1;
    let msize = p + 1;
    let n_seg = bezier_cps.len() / msize;
    assert_eq!(bezier_cps.len() % msize, 0);
    // Use the cached bezier matrix cache to compute the coefficients.
    let mut coeff = vec![A::scalar(0.0); DIM * n_seg * msize];
    with_bezier_matrix(p, |bmat| {
        // For each segment, multiply bezierMat by the column of scalar Bézier CP
        // values for each component. The C++ transposes the segment matrix to make
        // columns be segments; we skip the transpose and just read from rows.
        for ci in 0..DIM {
            let comp_offset = ci * n_seg * msize;
            for seg in 0..n_seg {
                let cps = &bezier_cps[seg * msize..(seg + 1) * msize];
                let out_base = comp_offset + seg * msize;
                for i in 0..msize {
                    let mut val = A::scalar(0.0);
                    // bezierMat is lower triangular, so k only goes up to i.
                    for k in 0..=i {
                        val += A::scalar(bmat[i][k]) * A::vector_coord(cps[k], ci);
                    }
                    coeff[out_base + i] = val;
                }
            }
        }
    });
    coeff
}

#[inline(always)]
fn with_bezier_matrix<F, R>(degree: usize, callback: F) -> R
where
    F: FnOnce(View2D<'_, f64>) -> R,
{
    thread_local! {
        static BEZIER_MATS: RefCell<Vec<Vec<f64>>> = RefCell::new(Default::default());
    }
    // First ensure the matrix has been computed.
    BEZIER_MATS.with_borrow_mut(|cache| {
        if degree >= cache.len() {
            cache.resize(degree + 1, Vec::default());
        }
        let mat = &mut cache[degree];
        if mat.is_empty() {
            // We haven't calculated the bezier matrix for this degree before. So we calculate it now, and cache it.
            let msize = degree + 1;
            mat.resize(msize * msize, 0.0);
            let mut mat = View2DMut::create(mat, msize, msize);
            for i in 0..msize {
                for k in 0..=i {
                    let sign = if ((i - k) & 1) != 0 { -1.0 } else { 1.0 };
                    mat[i][k] =
                        (binomial_coeff(degree, i) as f64) * (binomial_coeff(i, k) as f64) * sign;
                }
            }
        }
        let msize = degree + 1;
        callback(View2D::create(mat, msize, msize))
    })
}

pub(crate) fn binomial_coeff(n: usize, k: usize) -> usize {
    struct PascalTriangle {
        entries: Vec<usize>,
        n_rows: usize,
    }

    // Thread local cache to store lazily computed binomial coefficients.
    thread_local! {
        static PASCAL_TRIANGLE: RefCell<PascalTriangle> = RefCell::new(PascalTriangle { entries: vec![1], n_rows: 1 });
    }

    PASCAL_TRIANGLE.with_borrow_mut(|triangle| {
        while triangle.n_rows <= n {
            let irow = triangle.n_rows;
            triangle.entries.push(1);
            let prev_offset = (irow * (irow - 1)) / 2;
            for k in 1..irow {
                let next =
                    triangle.entries[prev_offset + k] + triangle.entries[prev_offset + k - 1];
                triangle.entries.push(next);
            }
            triangle.entries.push(1);
            triangle.n_rows += 1;
        }
        triangle.entries[((n * (n + 1)) / 2) + k]
    })
}

fn piecewise_bezier<const DIM: usize, A: Adaptor<DIM>>(
    knots: &[A::Scalar],
    control_points: &[A::Vector],
    dst: &mut Vec<A::Vector>,
) {
    let degree = knots.len() - control_points.len() - 1;
    let n_segments = knots
        .iter()
        .zip(knots.iter().skip(1))
        .filter(|&(a, b)| a != b)
        .count();
    dst.clear();
    dst.resize(n_segments * (degree + 1), A::zero_vector());
    let mut alphas = Vec::<A::Scalar>::new();
    // Algorithm A5.6 from the NURBS book. Matching the variable names
    // within reason for legibility.
    let n = control_points.len() - 1;
    let p = degree;
    let u_vec: &[A::Scalar] = knots;
    let p_vec: &[A::Vector] = control_points;
    let mut q_mat = View2DMut::create(dst, n_segments, p + 1);
    let m = n + p + 1;
    let mut a = p;
    let mut b = p + 1;
    let mut nb = 0;
    q_mat[nb].copy_from_slice(&p_vec[..p + 1]);
    while b < m {
        let i = b;
        while b < m && u_vec[b + 1] == u_vec[b] {
            b += 1;
        }
        let mult = b - i + 1;
        if mult < p {
            let numer = u_vec[b] - u_vec[a]; // Numerator of alpha.
            // Compute and store alphas.
            alphas.resize(p - mult, A::scalar(0.0));
            for j in ((mult + 1)..=p).rev() {
                alphas[j - mult - 1] = numer / (u_vec[a + j] - u_vec[a]);
            }
            let r = p - mult; // Insert the knot r times.
            for j in 1..=r {
                let save = r - j;
                let s = mult + j; // This many new points.
                for k in (s..=p).rev() {
                    let alpha = alphas[k - s];
                    q_mat[nb][k] =
                        alpha * q_mat[nb][k] + (A::scalar(1.0) - alpha) * q_mat[nb][k - 1];
                }
                if b < m {
                    // Control point of.
                    q_mat[nb + 1][save] = q_mat[nb][p]; // Next segment.
                }
            }
        }
        nb += 1; // Bezier segment completed.
        if b < m {
            // Initialize for next segment.
            for i in (p.saturating_sub(mult))..=p {
                q_mat[nb][i] = p_vec[b - p + i];
            }
            a = b;
            b += 1;
        }
    }
}

struct View2DMut<'a, T> {
    data: &'a mut [T],
    cols: usize,
}

impl<'a, T> View2DMut<'a, T> {
    fn create(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { data, cols }
    }
}

impl<'a, T> Index<usize> for View2DMut<'a, T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = index * self.cols;
        &self.data[offset..(offset + self.cols)]
    }
}

impl<'a, T> IndexMut<usize> for View2DMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let offset = index * self.cols;
        &mut self.data[offset..(offset + self.cols)]
    }
}

struct View2D<'a, T> {
    data: &'a [T],
    cols: usize,
}

impl<'a, T> View2D<'a, T> {
    fn create(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { data, cols }
    }
}

impl<'a, T> Index<usize> for View2D<'a, T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = index * self.cols;
        &self.data[offset..(offset + self.cols)]
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<u128>> = RefCell::new(Vec::default());
}

/// Lend `N` disjoint `&mut [T]` scratch buffers from a single type-erased
/// thread-local `Vec<_>`.  `T: Copy` guarantees no `Drop` glue, so the
/// raw-byte backing store is always safe to reinterpret and reuse.
#[inline(always)]
fn with_thread_local_buffers<const N: usize, T: Copy + Sized, F, R>(
    lens: [usize; N],
    callback: F,
) -> R
where
    F: FnOnce([&mut [T]; N]) -> R,
{
    const BLOCK_SIZE: usize = std::mem::size_of::<u128>();
    const {
        // Compile time checks.
        assert!(
            std::mem::align_of::<T>() <= std::mem::align_of::<u128>(),
            "Scalar alignment exceeds u128 alignment"
        );
        assert!(
            std::mem::size_of::<T>() <= BLOCK_SIZE,
            "At this time, the scalar types cannot be larger than 16 bytes"
        );
    }
    let elem_size = std::mem::size_of::<T>();
    let total_elems: usize = lens.iter().sum();
    let buf_size = (total_elems * elem_size).div_ceil(BLOCK_SIZE);
    SCRATCH.with_borrow_mut(|raw| {
        raw.resize(buf_size, 0);
        let base = raw.as_mut_ptr();
        assert!(total_elems * elem_size <= raw.len() * BLOCK_SIZE);
        // SAFETY:
        // - `base.add(offset)` is aligned to `align_of::<T>()` and within the allocation.
        // - `total_elems * elem_size` bytes fit (we allocated with room for alignment).
        // - `T: Copy` so all-zero bytes are a valid bit pattern (no padding invariants
        //   for primitive numeric types), and no destructors run when the buffer is reused.
        // - The sub-slices handed to the callback are disjoint (split_at_mut).
        // - The slices do not outlive this scope (FnOnce runs and returns here).
        let all = unsafe {
            std::slice::from_raw_parts_mut(base as *mut T, total_elems)
        };
        let mut offset = 0;
        let ranges = lens.map(|len| {
            let start = offset;
            offset += len;
            start..offset
        });
        let bufs = all.get_disjoint_mut(ranges)
            .expect("INTERNAL ERROR: This should never fail, since we're responsible allocating the buffers above.");
        callback(bufs)
    })
}

/// This is based on algorithm A2.2 from the book.
fn calc_basis<A: ScalarAdaptor>(
    i: usize,
    u: A::Float,
    degree: usize,
    knots: &[A::Float],
    [basis, left, right]: [&mut [A::Float]; 3],
) {
    basis[0] = A::scalar(1.0);
    for j in 1..=degree {
        left[j] = u - knots[i + 1 - j];
        right[j] = knots[i + j] - u;
        let mut saved = A::scalar(0.0f64);
        for r in 0..j {
            // This part is rewritten a little differently from the book to
            // improve floating point accuracy. As a consequence it also resuled
            // in a small performance improvement as per early benchmarks.
            let denom = right[r + 1] + left[j - r];
            let right_frac = right[r + 1] / denom;
            let left_frac = left[j - r] / denom;
            let old = basis[r];
            basis[r] = saved + old * right_frac;
            saved = old * left_frac;
        }
        basis[j] = saved;
    }
}

/// This is algorithm A2.3 from the book.
fn calc_ders_basis<A: ScalarAdaptor>(
    span: usize,
    u: A::Float,
    degree: usize,
    n_derivs: usize,
    knots: &[A::Float],
    [basis, ndu, alt_coeff, left, right]: [&mut [A::Float]; 5],
) {
    // Prep convenient views into buffers:
    let mut ders = View2DMut::create(basis, n_derivs + 1, degree + 1);
    let mut ndu = View2DMut::create(ndu, degree + 1, degree + 1);
    let mut a = View2DMut::create(alt_coeff, 2, degree + 1);
    // Implement the algo.
    ndu[0][0] = A::scalar(1.0_f64);
    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = A::scalar(0.0_f64);
        for r in 0..j {
            // This part is rewritten a little differently from the book to
            // improve floating point accuracy. As a consequence it also resuled
            // in a small performance improvement as per early benchmarks.
            let denom = right[r + 1] + left[j - r];
            ndu[j][r] = denom;
            let right_frac = right[r + 1] / denom;
            let left_frac = left[j - r] / denom;
            let old = ndu[r][j - 1];
            ndu[r][j] = saved + old * right_frac;
            saved = old * left_frac;
        }
        ndu[j][j] = saved;
    }
    for j in 0..=degree {
        // Load the basis functions
        ders[0][j] = ndu[j][degree];
    }
    // This section computes the derivatives.
    for r in 0..=degree {
        // Loop over the function index.
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = A::scalar(1.);
        // Loop to compute the kth derivative.
        for k in 1..=n_derivs {
            let mut d = A::scalar(0.0_f64);
            let rk: isize = (r as isize) - (k as isize);
            let pk: isize = (degree as isize) - (k as isize);
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1: isize = if rk >= -1 { 1 } else { -rk };
            let j2: isize = if (r as isize - 1) <= pk {
                k as isize - 1
            } else {
                (degree as isize) - (r as isize)
            };
            for j in j1..=j2 {
                a[s2][j as usize] = (a[s1][j as usize] - a[s1][(j - 1) as usize])
                    / ndu[(pk + 1) as usize][(rk + j) as usize];
                d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
            }
            if r as isize <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                d += a[s2][k] * ndu[r][pk as usize];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2); // Switch rows.
        }
    }
    // Multiply through by the correct factors.
    let mut r = A::scalar(degree as f64);
    for k in 1..=n_derivs {
        for j in 0..=degree {
            ders[k][j] *= r;
        }
        r *= A::scalar(degree as f64 - k as f64);
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::{E, FRAC_1_PI, LN_2, LOG2_E, PI, SQRT_2, TAU};

    use super::*;
    use crate::{DVec, DVec3, polynomial};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    // -- Helpers -----------------------------------------------------------

    fn vec3(x: f64, y: f64, z: f64) -> DVec3 {
        DVec([x, y, z])
    }

    fn make(cps: &[DVec3], knots: &[f64], degree: usize) -> Spline3d {
        Spline3d::create(cps, knots, degree).unwrap()
    }

    fn make_clamped(cps: &[DVec3], degree: usize) -> Spline3d {
        Spline3d::create_clamped(cps, degree).unwrap()
    }

    // ======================================================================
    // Tests
    // ======================================================================

    #[test]
    fn t_binomial_coefficients() {
        assert_eq!(binomial_coeff(0, 0), 1);
        assert_eq!(binomial_coeff(1, 0), 1);
        assert_eq!(binomial_coeff(1, 1), 1);
        assert_eq!(binomial_coeff(4, 0), 1);
        assert_eq!(binomial_coeff(4, 1), 4);
        assert_eq!(binomial_coeff(4, 2), 6);
        assert_eq!(binomial_coeff(4, 3), 4);
        assert_eq!(binomial_coeff(4, 4), 1);
        assert_eq!(binomial_coeff(10, 5), 252);
        assert_eq!(binomial_coeff(12, 6), 924);
        for n in 0..=8 {
            for k in 0..=n {
                assert_eq!(binomial_coeff(n, k), binomial_coeff(n, n - k));
            }
        }
    }

    #[test]
    fn t_find_span_clamped_cubic() {
        let degree = 3;
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.0), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 4.0), 6);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0), 4);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 2.0), 5);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 3.0), 6);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.5), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.5), 4);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 2.5), 5);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 3.5), 6);
        let eps = 1e-14;
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.0 + eps), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0 - eps), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0 + eps), 4);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 4.0 - eps), 6);
        // Clamped linear.
        let degree = 1;
        let knots = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.0), 1);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.5), 1);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0), 2);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 2.0), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 2.5), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 3.0), 3);
        // Single span quadratic.
        let degree = 2;
        let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.0), 2);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.5), 2);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0), 2);
        // Non-uniform interior knots.
        let degree = 3;
        let knots = [0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.0), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.15), 3);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.3), 4);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.5), 4);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.7), 5);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 0.85), 5);
        assert_eq!(find_span::<F64Adaptor>(&knots, degree, 1.0), 5);
    }

    #[test]
    fn t_create_validates_knot_count() {
        let cps = [vec3(0., 0., 0.), vec3(1., 0., 0.), vec3(2., 0., 0.)];
        assert!(Spline3d::create(&cps, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2).is_ok());
        assert!(matches!(
            Spline3d::create(&cps, &[0.0, 0.0, 0.0, 1.0, 1.0], 2),
            Err(Error::IncorrectKnotCount)
        ));
        assert!(matches!(
            Spline3d::create(&cps, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 2),
            Err(Error::IncorrectKnotCount)
        ));
    }

    #[test]
    fn t_create_clamped_validates_control_points() {
        assert!(matches!(
            Spline3d::create_clamped(&[vec3(0., 0., 0.), vec3(1., 0., 0.)], 2),
            Err(Error::InsufficientControlPoints)
        ));
        assert!(matches!(
            Spline3d::create_clamped(Vec::<DVec3>::new(), 0),
            Err(Error::InsufficientControlPoints)
        ));
        let spline = make_clamped(&[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)], 2);
        assert_eq!(spline.degree(), 2);
        assert_eq!(spline.domain(), (0.0, 1.0));
    }

    #[test]
    fn t_create_clamped_knot_generation() {
        let spline = make_clamped(
            &(0..7).map(|i| vec3(i as f64, 0., 0.)).collect::<Vec<_>>(),
            3,
        );
        assert_eq!(spline.degree(), 3);
        assert_eq!(spline.domain(), (0.0, 4.0));
        let spline = make_clamped(
            &(0..4).map(|i| vec3(i as f64, 0., 0.)).collect::<Vec<_>>(),
            1,
        );
        assert_eq!(spline.degree(), 1);
        assert_eq!(spline.domain(), (0.0, 3.0));
    }

    #[test]
    fn t_start_end_linear() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(3., 1., 0.),
                vec3(4., 4., 0.),
            ],
            &[0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        );
        assert_eq!(spline.start(), vec3(0., 0., 0.));
        assert_eq!(spline.end(), vec3(4., 4., 0.));
    }

    #[test]
    fn t_start_end_quadratic_bezier() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        assert_eq!(spline.start(), vec3(0., 0., 0.));
        assert_eq!(spline.end(), vec3(2., 0., 0.));
    }

    #[test]
    fn t_start_end_cubic_bezier() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        assert_eq!(spline.start(), vec3(0., 0., 0.));
        assert_eq!(spline.end(), vec3(1., 0., 0.));
    }

    #[test]
    fn t_start_end_clamped_cubic_multi_segment() {
        let spline = make(
            &[
                vec3(1., 2., 3.),
                vec3(4., 5., 6.),
                vec3(7., 8., 9.),
                vec3(10., 11., 12.),
                vec3(13., 14., 15.),
                vec3(16., 17., 18.),
                vec3(19., 20., 21.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        assert_eq!(spline.start(), vec3(1., 2., 3.));
        assert_eq!(spline.end(), vec3(19., 20., 21.));
    }

    #[test]
    fn t_start_end_matches_eval_point() {
        let cases: [(&[DVec3], usize); _] = [
            (&[vec3(0., 0., 0.), vec3(1., 1., 1.)], 1),
            (&[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)], 2),
            (
                &[
                    vec3(0., 0., 0.),
                    vec3(1., 5., -2.),
                    vec3(2., -3., 4.),
                    vec3(3., 5., -2.),
                    vec3(4., 0., 0.),
                ],
                3,
            ),
        ];
        for (cps, degree) in cases {
            let spline = make_clamped(cps, degree);
            let (dom_start, dom_end) = spline.domain();
            assert_eq!(spline.start(), spline.point(dom_start).unwrap());
            assert_eq!(spline.end(), spline.point(dom_end).unwrap());
        }
    }

    #[test]
    fn t_eval_point_out_of_domain() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., 0.),
                vec3(2., 0., 0.),
                vec3(3., 2., 0.),
                vec3(4., 0., 0.),
                vec3(5., 2., 0.),
                vec3(6., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        assert_eq!(spline.domain(), (0.0, 4.0));
        assert!(spline.point(-1.0).is_none());
        assert!(spline.point(-f64::EPSILON).is_none());
        assert!(spline.point(4.0_f64.next_up()).is_none());
        assert!(spline.point(100.0).is_none());
        assert!(spline.point(f64::NEG_INFINITY).is_none());
        assert!(spline.point(f64::INFINITY).is_none());
        assert!(spline.point(f64::NAN).is_none());
        assert!(spline.point(0.0).is_some());
        assert!(spline.point(2.0).is_some());
        assert!(spline.point(4.0).is_some());
    }

    #[test]
    fn t_eval_point_linear() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(3., 1., 0.),
                vec3(4., 4., 0.),
            ],
            &[0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        );
        assert_eq!(spline.point(0.0).unwrap(), vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), vec3(1., 3., 0.));
        assert_eq!(spline.point(2.0).unwrap(), vec3(3., 1., 0.));
        assert_eq!(spline.point(3.0).unwrap(), vec3(4., 4., 0.));
        assert_eq!(spline.point(0.5).unwrap(), vec3(0.5, 1.5, 0.));
        assert_eq!(spline.point(1.5).unwrap(), vec3(2., 2., 0.));
        assert_eq!(spline.point(2.5).unwrap(), vec3(3.5, 2.5, 0.));
    }

    #[test]
    fn t_eval_point_quadratic_bezier() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        assert_eq!(spline.point(0.0).unwrap(), vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), vec3(2., 0., 0.));
        assert_eq!(spline.point(0.5).unwrap(), vec3(1., 1., 0.));
        assert_eq!(spline.point(0.25).unwrap(), vec3(0.5, 0.75, 0.));
        assert_eq!(spline.point(0.75).unwrap(), vec3(1.5, 0.75, 0.));
    }

    #[test]
    fn t_eval_point_cubic_bezier() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        assert_eq!(spline.point(0.0).unwrap(), vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), vec3(1., 0., 0.));
        assert_eq!(spline.point(0.5).unwrap(), vec3(0.5, 0.75, 0.));
    }

    #[test]
    fn t_eval_derivs_out_of_domain() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., 0.),
                vec3(2., 0., 0.),
                vec3(3., 2., 0.),
                vec3(4., 0., 0.),
                vec3(5., 2., 0.),
                vec3(6., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        let mut results = [vec3(0., 0., 0.); 2];
        assert!(spline.point_with_derivs(-1.0, &mut results).is_err());
        assert!(spline.point_with_derivs(5.0, &mut results).is_err());
        assert!(spline.point_with_derivs(f64::NAN, &mut results).is_err());
        assert!(spline.point_with_derivs(-1.0, &mut []).is_ok());
    }

    #[test]
    fn t_eval_derivs_linear() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(3., 1., 0.),
                vec3(4., 4., 0.),
            ],
            &[0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        );
        let mut results = [vec3(0., 0., 0.); 2];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], vec3(1., 3., 0.));
        spline.point_with_derivs(1.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(1.5).unwrap());
        assert_eq!(results[1], vec3(2., -2., 0.));
        spline.point_with_derivs(2.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(2.5).unwrap());
        assert_eq!(results[1], vec3(1., 3., 0.));
        let mut results = [vec3(0., 0., 0.); 1];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
    }

    #[test]
    fn t_eval_derivs_quadratic_bezier() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        let mut results = [vec3(0., 0., 0.); 3];
        spline.point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], vec3(0., 0., 0.));
        assert_eq!(results[1], vec3(2., 4., 0.));
        assert_eq!(results[2], vec3(0., -8., 0.));
        spline.point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], vec3(2., 0., 0.));
        assert_eq!(results[1], vec3(2., -4., 0.));
        assert_eq!(results[2], vec3(0., -8., 0.));
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], vec3(2., 0., 0.));
        assert_eq!(results[2], vec3(0., -8., 0.));
        let mut results = [vec3(0., 0., 0.); 4];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[3], vec3(0., 0., 0.));
    }

    #[test]
    fn t_eval_derivs_cubic_bezier() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        let mut results = [vec3(0., 0., 0.); 3];
        spline.point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], vec3(0., 0., 0.));
        assert_eq!(results[1], vec3(0., 3., 0.));
        spline.point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], vec3(1., 0., 0.));
        assert_eq!(results[1], vec3(0., -3., 0.));
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], vec3(1.5, 0., 0.));
    }

    #[test]
    fn t_eval_tangent_out_of_domain() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 1., 0.), vec3(3., 0., 0.)],
            &[0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        );
        assert!(spline.tangent(-0.1).is_none());
        assert!(spline.tangent(2.1).is_none());
        assert!(spline.tangent(f64::NAN).is_none());
    }

    #[test]
    fn t_eval_tangent_linear() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(2., 6., 0.), vec3(5., 3., 0.)],
            &[0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        );
        assert_eq!(spline.tangent(0.0).unwrap(), vec3(2., 6., 0.));
        assert_eq!(spline.tangent(0.5).unwrap(), vec3(2., 6., 0.));
        assert_eq!(spline.tangent(1.5).unwrap(), vec3(3., -3., 0.));
        assert_eq!(spline.tangent(2.0).unwrap(), vec3(3., -3., 0.));
    }

    #[test]
    fn t_eval_tangent_quadratic_bezier() {
        let spline = make(
            &[vec3(1., 0., 0.), vec3(1., 1., 0.), vec3(0., 1., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        assert_eq!(spline.tangent(0.0).unwrap(), vec3(0., 2., 0.));
        assert_eq!(spline.tangent(1.0).unwrap(), vec3(-2., 0., 0.));
        assert_eq!(spline.tangent(0.5).unwrap(), vec3(-1., 1., 0.));
        assert_eq!(spline.tangent(0.25).unwrap(), vec3(-0.5, 1.5, 0.));
    }

    #[test]
    fn t_eval_tangent_cubic_bezier() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 0., 0.),
                vec3(1., 1., 0.),
                vec3(0., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        assert_eq!(spline.tangent(0.0).unwrap(), vec3(3., 0., 0.));
        assert_eq!(spline.tangent(1.0).unwrap(), vec3(-3., 0., 0.));
        assert_eq!(spline.tangent(0.5).unwrap(), vec3(0., 1.5, 0.));
    }

    #[test]
    fn t_eval_point_clamped_cubic_endpoints() {
        let spline = make(
            &[
                vec3(1., 2., 3.),
                vec3(4., 5., 6.),
                vec3(7., 8., 9.),
                vec3(10., 11., 12.),
                vec3(13., 14., 15.),
                vec3(16., 17., 18.),
                vec3(19., 20., 21.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        assert_eq!(spline.point(0.0).unwrap(), vec3(1., 2., 3.));
        assert_eq!(spline.point(4.0).unwrap(), vec3(19., 20., 21.));
    }

    /// De Casteljau evaluation of a Bézier curve.
    fn de_casteljau(cps: &[DVec3], t: f64) -> DVec3 {
        let mut work: Vec<DVec3> = cps.to_vec();
        for level in 1..cps.len() {
            for i in 0..cps.len() - level {
                work[i] = work[i] * (1.0 - t) + work[i + 1] * t;
            }
        }
        work[0]
    }

    fn check_bezier_decomposition(spline: &Spline3d) {
        let mut bezier_cps = Vec::new();
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut bezier_cps);
        let p = spline.degree();
        let n_segments = bezier_cps.len() / (p + 1);
        assert_eq!(bezier_cps.len() % (p + 1), 0);
        let knots = &spline.knots;
        let mut spans: Vec<(f64, f64)> = Vec::new();
        for i in 0..knots.len() - 1 {
            if knots[i] != knots[i + 1] {
                spans.push((knots[i], knots[i + 1]));
            }
        }
        assert_eq!(spans.len(), n_segments);
        let view = View2D::create(&bezier_cps, n_segments, p + 1);
        for (seg, &(u_lo, u_hi)) in spans.iter().enumerate() {
            let n_samples = 17;
            for si in 0..=n_samples {
                let t = si as f64 / n_samples as f64;
                let u = u_lo + t * (u_hi - u_lo);
                let from_spline = spline.point(u).unwrap();
                let from_bezier = de_casteljau(&view[seg], t);
                let diff = from_spline - from_bezier;
                let err = (0..3).fold(0.0f64, |acc, i| acc.max(diff[i].abs()));
                assert!(
                    err < 1e-12,
                    "seg={seg}, t={t}, u={u}: spline={from_spline:?} bezier={from_bezier:?} err={err}"
                );
            }
        }
    }

    #[test]
    fn t_piecewise_bezier_all_cases() {
        // Single segment, degree 1.
        let spline = make(
            &[vec3(0., 0., 0.), vec3(3., 4., 0.)],
            &[0.0, 0.0, 1.0, 1.0],
            1,
        );
        let mut dst = Vec::new();
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 2);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[1], vec3(3., 4., 0.));
        check_bezier_decomposition(&spline);
        // Single segment, degree 2.
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[1], vec3(1., 2., 0.));
        assert_eq!(dst[2], vec3(2., 0., 0.));
        check_bezier_decomposition(&spline);
        // Single segment, degree 3.
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 4);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[1], vec3(0., 1., 0.));
        assert_eq!(dst[2], vec3(1., 1., 0.));
        assert_eq!(dst[3], vec3(1., 0., 0.));
        check_bezier_decomposition(&spline);
        // Multi-segment, degree 1.
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(3., 1., 0.),
                vec3(4., 4., 0.),
            ],
            &[0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 6);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[1], vec3(1., 3., 0.));
        assert_eq!(dst[2], vec3(1., 3., 0.));
        assert_eq!(dst[3], vec3(3., 1., 0.));
        assert_eq!(dst[4], vec3(3., 1., 0.));
        assert_eq!(dst[5], vec3(4., 4., 0.));
        check_bezier_decomposition(&spline);
        // Multi-segment, degree 3.
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., 0.),
                vec3(2., -1., 3.),
                vec3(3., 2., 1.),
                vec3(4., 0., -1.),
                vec3(5., 3., 2.),
                vec3(6., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 4 * 4);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[15], vec3(6., 1., 0.));
        for seg in 0..3 {
            assert_eq!(
                dst[seg * 4 + 3],
                dst[(seg + 1) * 4],
                "segments {seg} and {} don't share endpoint",
                seg + 1
            );
        }
        check_bezier_decomposition(&spline);
        // Multi-segment, degree 2.
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 4., 0.),
                vec3(3., -2., 1.),
                vec3(5., 1., 3.),
                vec3(7., 0., 0.),
            ],
            2,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3 * 3);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[8], vec3(7., 0., 0.));
        for seg in 0..2 {
            assert_eq!(
                dst[seg * 3 + 2],
                dst[(seg + 1) * 3],
                "segments {seg} and {} don't share endpoint",
                seg + 1
            );
        }
        check_bezier_decomposition(&spline);
        // Interior knot with multiplicity 2.
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(2., 0., 1.),
                vec3(3., 2., -1.),
                vec3(4., -1., 2.),
                vec3(5., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            3,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 2 * 4);
        assert_eq!(dst[0], vec3(0., 0., 0.));
        assert_eq!(dst[7], vec3(5., 1., 0.));
        assert_eq!(dst[3], dst[4]);
        check_bezier_decomposition(&spline);
        // 3D all nonzero.
        let spline = make(
            &[
                vec3(1., 2., 3.),
                vec3(4., 5., 6.),
                vec3(7., 8., 9.),
                vec3(10., 11., 12.),
                vec3(13., 14., 15.),
                vec3(16., 17., 18.),
                vec3(19., 20., 21.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        );
        check_bezier_decomposition(&spline);
        // Non-uniform interior knots.
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0.5, 2., 1.),
                vec3(1.5, -1., 2.),
                vec3(2.5, 3., -1.),
                vec3(3.5, 0., 1.),
                vec3(4., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3 * 4);
        check_bezier_decomposition(&spline);
        // Reuse of dst vector.
        let spline = make(
            &[vec3(0., 0., 0.), vec3(1., 1., 1.)],
            &[0.0, 0.0, 1.0, 1.0],
            1,
        );
        piecewise_bezier::<3, F64Adaptor>(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 2);
    }

    fn check_power_basis(spline: &Spline3d) {
        let unique = &spline.unique_knots;
        let n_segments = unique.len() - 1;
        let p = spline.degree();
        let (domain_lo, domain_hi) = spline.domain();
        let mut deriv_polys = vec![0.0; 3 * n_segments * p];
        for ci in 0..3 {
            for seg in 0..n_segments {
                let offset = (ci * n_segments + seg) * p;
                polynomial::differentiate::<F64Adaptor>(
                    spline.power_basis_polynomial(seg, ci),
                    &mut deriv_polys[offset..offset + p],
                );
            }
        }
        let n_samples = 51;
        for si in 0..=n_samples {
            let t = si as f64 / n_samples as f64;
            let u = domain_lo + t * (domain_hi - domain_lo);
            let seg = {
                let mut s = match unique.partition_point(|&k| k <= u) {
                    0 => 0,
                    i => i - 1,
                };
                if s >= n_segments {
                    s = n_segments - 1;
                }
                s
            };
            let u_lo = unique[seg];
            let u_hi = unique[seg + 1];
            let t_local = (u - u_lo) / (u_hi - u_lo);
            let x = polynomial::eval::<F64Adaptor>(spline.power_basis_polynomial(seg, 0), t_local);
            let y = polynomial::eval::<F64Adaptor>(spline.power_basis_polynomial(seg, 1), t_local);
            let z = polynomial::eval::<F64Adaptor>(spline.power_basis_polynomial(seg, 2), t_local);
            let from_poly = vec3(x, y, z);
            let mut results = [vec3(0., 0., 0.); 2];
            spline.point_with_derivs(u, &mut results).unwrap();
            let from_spline = results[0];
            let tangent_spline = results[1];
            let diff = from_spline - from_poly;
            let err = (0..3).fold(0.0f64, |acc, i| acc.max(diff[i].abs()));
            assert!(
                err < 1e-10,
                "point: seg={seg}, u={u}, t_local={t_local}: spline={from_spline:?} poly={from_poly:?} err={err}"
            );
            let span = u_hi - u_lo;
            let dpoly = |ci: usize| {
                let offset = (ci * n_segments + seg) * p;
                &deriv_polys[offset..offset + p]
            };
            let dx = polynomial::eval::<F64Adaptor>(dpoly(0), t_local) / span;
            let dy = polynomial::eval::<F64Adaptor>(dpoly(1), t_local) / span;
            let dz = polynomial::eval::<F64Adaptor>(dpoly(2), t_local) / span;
            let tangent_poly = vec3(dx, dy, dz);
            let diff = tangent_spline - tangent_poly;
            let err = (0..3).fold(0.0f64, |acc, i| acc.max(diff[i].abs()));
            assert!(
                err < 1e-8,
                "tangent: seg={seg}, u={u}, t_local={t_local}: spline={tangent_spline:?} poly={tangent_poly:?} err={err}"
            );
        }
    }

    #[test]
    fn t_power_basis_matches_eval() {
        check_power_basis(&make(
            &[vec3(0., 0., 0.), vec3(3., 4., 5.)],
            &[0.0, 0.0, 1.0, 1.0],
            1,
        ));
        check_power_basis(&make(
            &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        ));
        check_power_basis(&make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        ));
        check_power_basis(&make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., 0.),
                vec3(2., -1., 3.),
                vec3(3., 2., 1.),
                vec3(4., 0., -1.),
                vec3(5., 3., 2.),
                vec3(6., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        ));
        check_power_basis(&make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 4., 0.),
                vec3(3., -2., 1.),
                vec3(5., 1., 3.),
                vec3(7., 0., 0.),
            ],
            2,
        ));
        check_power_basis(&make(
            &[
                vec3(0., 0., 0.),
                vec3(0.5, 2., 1.),
                vec3(1.5, -1., 2.),
                vec3(2.5, 3., -1.),
                vec3(3.5, 0., 1.),
                vec3(4., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0],
            3,
        ));
        check_power_basis(&make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(2., 0., 1.),
                vec3(3., 2., -1.),
                vec3(4., -1., 2.),
                vec3(5., 1., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            3,
        ));
        check_power_basis(&make(
            &[
                vec3(1., 2., 3.),
                vec3(4., 5., 6.),
                vec3(7., 8., 9.),
                vec3(10., 11., 12.),
                vec3(13., 14., 15.),
                vec3(16., 17., 18.),
                vec3(19., 20., 21.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        ));
    }

    fn verify_bounds(spline: &Spline3d, n_samples: usize) {
        let (bmin, bmax) = spline.bounds();
        let b_lo = bmin.0;
        let b_hi = bmax.0;
        let (lo, hi) = spline.domain();
        let mut observed_min = [f64::INFINITY; 3];
        let mut observed_max = [f64::NEG_INFINITY; 3];
        for i in 0..=n_samples {
            let t = lo + (hi - lo) * (i as f64) / (n_samples as f64);
            let pt = spline.point(t).unwrap();
            let coords = pt.0;
            for c in 0..3 {
                assert!(
                    coords[c] >= b_lo[c] - 1e-12,
                    "coord {c} value {} below min bound {} at t={t}",
                    coords[c],
                    b_lo[c]
                );
                assert!(
                    coords[c] <= b_hi[c] + 1e-12,
                    "coord {c} value {} above max bound {} at t={t}",
                    coords[c],
                    b_hi[c]
                );
                observed_min[c] = observed_min[c].min(coords[c]);
                observed_max[c] = observed_max[c].max(coords[c]);
            }
        }
        for c in 0..3 {
            assert!(
                (observed_min[c] - b_lo[c]).abs() < 1e-6,
                "coord {c} min bound {} not tight, observed min {}",
                b_lo[c],
                observed_min[c]
            );
            assert!(
                (observed_max[c] - b_hi[c]).abs() < 1e-6,
                "coord {c} max bound {} not tight, observed max {}",
                b_hi[c],
                observed_max[c]
            );
        }
    }

    #[test]
    fn bounds_degree_0() {
        let spline = make(
            &[vec3(1., 0., 0.), vec3(5., 0., 0.), vec3(2., 0., 0.)],
            &[0.0, 1.0, 2.0, 3.0],
            0,
        );
        let (lo, hi) = spline.bounds();
        assert_eq!(lo[0], 1.0);
        assert_eq!(hi[0], 5.0);
        let spline = make(&[vec3(3., 7., -2.)], &[0.0, 1.0], 0);
        let (lo, hi) = spline.bounds();
        assert!((lo[0] - 3.0).abs() < 1e-12);
        assert!((lo[1] - 7.0).abs() < 1e-12);
        assert!((lo[2] - -2.0).abs() < 1e-12);
        assert!((hi[0] - 3.0).abs() < 1e-12);
        assert!((hi[1] - 7.0).abs() < 1e-12);
        assert!((hi[2] - -2.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_degree_1() {
        let spline = make_clamped(&[vec3(0., 0., 0.), vec3(1., 3., 0.), vec3(2., 0., 0.)], 1);
        let (lo, hi) = spline.bounds();
        assert!((lo[0] - 0.0).abs() < 1e-12);
        assert!((hi[0] - 2.0).abs() < 1e-12);
        assert!((lo[1] - 0.0).abs() < 1e-12);
        assert!((hi[1] - 3.0).abs() < 1e-12);
        verify_bounds(&spline, 1000);
        let spline = make_clamped(&[vec3(0., 2., 0.), vec3(1., -1., 0.), vec3(2., 2., 0.)], 1);
        let (lo, hi) = spline.bounds();
        assert!((lo[1] - -1.0).abs() < 1e-12);
        assert!((hi[1] - 2.0).abs() < 1e-12);
        verify_bounds(&spline, 1000);
        let spline = make_clamped(&[vec3(1., 2., 3.), vec3(4., 5., 6.)], 1);
        let (lo, hi) = spline.bounds();
        assert!((lo[0] - 1.0).abs() < 1e-12);
        assert!((lo[1] - 2.0).abs() < 1e-12);
        assert!((lo[2] - 3.0).abs() < 1e-12);
        assert!((hi[0] - 4.0).abs() < 1e-12);
        assert!((hi[1] - 5.0).abs() < 1e-12);
        assert!((hi[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_degree_2() {
        let spline = make(
            &[vec3(0., 0., 0.), vec3(0.5, 2., 0.), vec3(1., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        let (lo, hi) = spline.bounds();
        assert!((hi[1] - 1.0).abs() < 1e-10);
        assert!((lo[1] - 0.0).abs() < 1e-10);
        verify_bounds(&spline, 1000);
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 3., 0.),
                vec3(2., -1., 0.),
                vec3(3., 0., 0.),
            ],
            2,
        );
        let (lo, hi) = spline.bounds();
        assert!(lo[1] < 0.0);
        assert!(hi[1] > 0.0);
        verify_bounds(&spline, 10000);
    }

    #[test]
    fn bounds_degree_3() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 5., -2.),
                vec3(2., -3., 4.),
                vec3(3., 5., -2.),
                vec3(4., 0., 0.),
            ],
            3,
        );
        verify_bounds(&spline, 10000);
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., -1.),
                vec3(2., -1., 3.),
                vec3(3., 4., -2.),
                vec3(4., -3., 1.),
                vec3(5., 1., -1.),
                vec3(6., 0., 0.),
            ],
            3,
        );
        verify_bounds(&spline, 10000);
    }

    fn verify_reversed(spline: &Spline3d, n_samples: usize) {
        let (lo, hi) = spline.domain();
        let ksum = lo + hi;
        let rev = spline.clone().reversed();
        assert_eq!(rev.domain(), spline.domain());
        assert_eq!(rev.degree(), spline.degree());
        assert_eq!(rev.start(), spline.end());
        assert_eq!(rev.end(), spline.start());
        for i in 0..=n_samples {
            let u = lo + (hi - lo) * (i as f64) / (n_samples as f64);
            let pt_orig = spline.point(ksum - u).unwrap();
            let pt_rev = rev.point(u).unwrap();
            let diff = (0..3).fold(0.0f64, |acc, i| acc + (pt_orig[i] - pt_rev[i]).abs());
            assert!(
                diff < 1e-10,
                "Mismatch at u={u}: orig({})={:?}, rev={:?}",
                ksum - u,
                pt_orig,
                pt_rev,
            );
        }
    }

    #[test]
    fn t_reversed_linear() {
        verify_reversed(
            &make(
                &[
                    vec3(0., 0., 0.),
                    vec3(1., 3., 0.),
                    vec3(3., 1., 0.),
                    vec3(4., 4., 0.),
                ],
                &[0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
                1,
            ),
            1000,
        );
    }

    #[test]
    fn t_reversed_quadratic_bezier() {
        verify_reversed(
            &make(
                &[vec3(0., 0., 0.), vec3(1., 2., 0.), vec3(2., 0., 0.)],
                &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                2,
            ),
            1000,
        );
    }

    #[test]
    fn t_reversed_cubic_bezier() {
        verify_reversed(
            &make(
                &[
                    vec3(0., 0., 0.),
                    vec3(0., 1., 0.),
                    vec3(1., 1., 0.),
                    vec3(1., 0., 0.),
                ],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                3,
            ),
            1000,
        );
    }

    #[test]
    fn t_reversed_cubic_multi_segment() {
        verify_reversed(
            &make(
                &[
                    vec3(1., 2., 3.),
                    vec3(4., 5., 6.),
                    vec3(7., 8., 9.),
                    vec3(10., 11., 12.),
                    vec3(13., 14., 15.),
                    vec3(16., 17., 18.),
                    vec3(19., 20., 21.),
                ],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                3,
            ),
            1000,
        );
    }

    #[test]
    fn t_reversed_clamped_3d() {
        verify_reversed(
            &make_clamped(
                &[
                    vec3(0., 0., 0.),
                    vec3(1., 5., -2.),
                    vec3(2., -3., 4.),
                    vec3(3., 5., -2.),
                    vec3(4., 0., 0.),
                ],
                3,
            ),
            1000,
        );
    }

    #[test]
    fn t_reversed_twice_is_identity() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., -1.),
                vec3(2., -1., 3.),
                vec3(3., 4., -2.),
                vec3(4., -3., 1.),
                vec3(5., 1., -1.),
                vec3(6., 0., 0.),
            ],
            3,
        );
        let (lo, hi) = spline.domain();
        let roundtrip = spline.clone().reversed().reversed();
        for i in 0..=1000 {
            let u = lo + (hi - lo) * (i as f64) / 1000.0;
            let pt_orig = spline.point(u).unwrap();
            let pt_rt = roundtrip.point(u).unwrap();
            let diff = (0..3).fold(0.0f64, |acc, i| acc + (pt_orig[i] - pt_rt[i]).abs());
            assert!(diff < 1e-10, "Round-trip mismatch at u={u}");
        }
    }

    #[test]
    fn t_reversed_single_segment() {
        verify_reversed(&make_clamped(&[vec3(1., 2., 3.), vec3(4., 5., 6.)], 1), 100);
    }

    #[test]
    fn t_length_approx_degree_1_straight_line() {
        let spline = make_clamped(&[vec3(0., 0., 0.), vec3(3., 4., 0.)], 1);
        let len = spline.length(1e-6);
        assert!((len - 5.0).abs() < 1e-6, "Expected 5.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_1_polyline_3d() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 0., 0.),
                vec3(1., 1., 0.),
                vec3(1., 1., 1.),
            ],
            1,
        );
        let len = spline.length(1e-6);
        assert!((len - 3.0).abs() < 1e-6, "Expected 3.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_1_zero_length() {
        let spline = make_clamped(&[vec3(1., 2., 3.), vec3(1., 2., 3.)], 1);
        let len = spline.length(1e-6);
        assert!(len.abs() < 1e-12, "Expected 0.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_2_bezier() {
        let expected = {
            let f = |u: f64| (u * (1.0 + u * u).sqrt() + u.asinh()) / 2.0;
            (f(2.0) - f(-2.0)) / 4.0
        };
        let spline = make(
            &[vec3(0., 0., 0.), vec3(0.5, 1., 0.), vec3(1., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-6);
        assert!(
            (fine - expected).abs() < 1e-4,
            "Expected {expected}, got {fine}"
        );
        assert!(
            (fine - expected).abs() <= (coarse - expected).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - expected).abs(),
            (coarse - expected).abs()
        );
    }

    #[test]
    fn t_length_approx_degree_3_bezier_exact() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(0., 1., 0.),
                vec3(1., 1., 0.),
                vec3(1., 0., 0.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-6);
        assert!((fine - 2.0).abs() < 1e-4, "Expected 2.0, got {fine}");
        assert!(
            (fine - 2.0).abs() <= (coarse - 2.0).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - 2.0).abs(),
            (coarse - 2.0).abs()
        );
    }

    #[test]
    fn t_length_approx_cubic_multi_segment() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., -1.),
                vec3(2., -1., 3.),
                vec3(3., 4., -2.),
                vec3(4., 0., 0.),
            ],
            3,
        );
        let reference = spline.length(1e-6);
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-4);
        assert!(
            (fine - reference).abs() < 1e-4,
            "fine {fine} not close to reference {reference}"
        );
        assert!(
            (fine - reference).abs() <= (coarse - reference).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - reference).abs(),
            (coarse - reference).abs()
        );
        assert!(fine > 0.0);
        let polygon_len: f64 = spline
            .control_points()
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum();
        assert!(fine <= polygon_len + 1e-12);
    }

    #[test]
    fn t_length_approx_3d_cubic() {
        let spline = make(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., 3.),
                vec3(3., 1., -1.),
                vec3(4., 0., 2.),
            ],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        );
        let reference = spline.length(1e-6);
        let l1 = spline.length(1e-1);
        let l2 = spline.length(1e-3);
        let l3 = spline.length(1e-5);
        let e1 = (l1 - reference).abs();
        let e2 = (l2 - reference).abs();
        let e3 = (l3 - reference).abs();
        assert!(e3 <= e2 + 1e-12, "l3 err {e3} > l2 err {e2}");
        assert!(e2 <= e1 + 1e-12, "l2 err {e2} > l1 err {e1}");
        assert!(e3 < 1e-4, "l3 err {e3} too large");
        let chord = (spline.end() - spline.start()).length();
        let polygon_len: f64 = spline
            .control_points()
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum();
        assert!(l3 >= chord - 1e-12);
        assert!(l3 <= polygon_len + 1e-12);
    }

    #[test]
    fn t_length_approx_reversed_same() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 5., -2.),
                vec3(2., -3., 4.),
                vec3(3., 5., -2.),
                vec3(4., 0., 0.),
            ],
            3,
        );
        let len_fwd = spline.length(1e-6);
        let len_rev = spline.reversed().length(1e-6);
        assert!(
            (len_fwd - len_rev).abs() < 1e-6,
            "Forward {len_fwd} != reversed {len_rev}"
        );
    }

    // ======================================================================
    // uniform_walk tests
    // ======================================================================

    #[test]
    fn t_uniform_walk_straight_line() {
        // 3-4-5 right-triangle polyline: three legs of lengths 3, 4, 5 → total 12.
        // For a piecewise-linear spline the uniform walk is exact (no curvature correction
        // needed), so we can verify positions to floating-point precision.
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(3., 0., 0.),
                vec3(3., 4., 0.),
                vec3(-2., 4., 0.),
            ],
            1,
        );
        // start=0, step=1 → 13 points at arc-distances 0, 1, …, 12.
        let pts: Vec<_> = spline.uniform_samples(0.0, 1.0, 1e-6).collect();
        let expected = [
            vec3(0., 0., 0.),
            vec3(1., 0., 0.),
            vec3(2., 0., 0.),
            vec3(3., 0., 0.),
            vec3(3., 1., 0.),
            vec3(3., 2., 0.),
            vec3(3., 3., 0.),
            vec3(3., 4., 0.),
            vec3(2., 4., 0.),
            vec3(1., 4., 0.),
            vec3(0., 4., 0.),
            vec3(-1., 4., 0.),
            vec3(-2., 4., 0.),
        ];
        assert_eq!(
            pts.len(),
            expected.len(),
            "expected {} points",
            expected.len()
        );
        for (i, (pt, exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*pt - *exp).length() < 1e-9,
                "pt[{i}]: expected {exp:?}, got {pt:?}"
            );
        }
        // start=0.5, step=1 → 12 points at arc-distances 0.5, 1.5, …, 11.5.
        let pts: Vec<_> = spline.uniform_samples(0.5, 1.0, 1e-6).collect();
        assert_eq!(pts.len(), 12, "expected 12 points with offset start");
        dbg!(&pts);
        let expected = [
            vec3(0.5, 0.0, 0.0),
            vec3(1.5, 0.0, 0.0),
            vec3(2.5, 0.0, 0.0),
            vec3(3.0, 0.5, 0.0),
            vec3(3.0, 1.5, 0.0),
            vec3(3.0, 2.5, 0.0),
            vec3(3.0, 3.5, 0.0),
            vec3(2.5, 4.0, 0.0),
            vec3(1.5, 4.0, 0.0),
            vec3(0.5, 4.0, 0.0),
            vec3(-0.5, 4.0, 0.0),
            vec3(-1.5, 4.0, 0.0),
        ];
        assert_eq!(
            pts.len(),
            expected.len(),
            "expected {} points",
            expected.len()
        );
        for (i, (pt, exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*pt - *exp).length() < 1e-9,
                "pt[{i}]: expected {exp:?}, got {pt:?}"
            );
        }
    }

    #[test]
    fn t_uniform_walk_zero_length_segment() {
        // Degree-1 spline where the middle segment has zero length (two consecutive
        // identical control points). This exercises the lspan == 0 guard.
        // Segments: (0,0,0)→(1,0,0) [len=1], (1,0,0)→(1,0,0) [len=0], (1,0,0)→(2,0,0) [len=1].
        // Expected: same as a simple 2-unit line — the zero-length segment is skipped.
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 0., 0.),
                vec3(1., 0., 0.),
                vec3(2., 0., 0.),
            ],
            1,
        );
        let pts: Vec<_> = spline.uniform_samples(0.0, 0.5, 1e-6).collect();
        let expected = [
            vec3(0., 0., 0.),
            vec3(0.5, 0., 0.),
            vec3(1., 0., 0.),
            vec3(1.5, 0., 0.),
            vec3(2., 0., 0.),
        ];
        assert_eq!(pts.len(), expected.len(), "wrong point count");
        for (i, (pt, exp)) in pts.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*pt - *exp).length() < 1e-9,
                "pt[{i}]: expected {exp:?}, got {pt:?}"
            );
        }
    }

    #[test]
    fn t_uniform_walk_curved_uniformity() {
        // Quadratic Bezier: (0,0,0)→(0.5,1,0)→(1,0,0).  Arc length ≈ 1.479.
        // Verify that consecutive walk points are approximately equidistant in arc
        // length, and that the number of points is consistent with curve length.
        let spline = make(
            &[vec3(0., 0., 0.), vec3(0.5, 1., 0.), vec3(1., 0., 0.)],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        );
        let total_len = spline.length(1e-6);
        let step = 0.1_f64;
        let pts: Vec<_> = spline.uniform_samples(0.0, step, 1e-6).collect();
        // Point count must satisfy (n-1)*step ≤ total_len < n*step.
        let n = pts.len() as f64;
        assert!(
            (n - 1.0) * step <= total_len + 1e-9,
            "too many points: {n} for length {total_len:.4}"
        );
        assert!(
            n * step > total_len - 1e-9,
            "too few points: {n} for length {total_len:.4}"
        );
        // Consecutive chord distances must be nearly equal (arc-length uniformity).
        let dists: Vec<f64> = pts.windows(2).map(|w| (w[1] - w[0]).length()).collect();
        let max_d = dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_d = dists.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_d / min_d < 1.05,
            "spacing non-uniform: min={min_d:.5}, max={max_d:.5}"
        );
        // All chord distances should be close to step in magnitude.
        assert!(
            max_d < step * 1.02,
            "chord distance {max_d:.5} exceeds step {step}"
        );
        // Finer tolerance produces more uniform spacing (convergence check).
        let pts_fine: Vec<_> = spline.uniform_samples(0.0, step / 2.0, 1e-6).collect();
        let dists_fine: Vec<f64> = pts_fine
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .collect();
        let max_fine = dists_fine.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_fine = dists_fine.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_fine / min_fine <= max_d / min_d + 1e-9,
            "finer step should be at least as uniform: coarse ratio {:.5}, fine ratio {:.5}",
            max_d / min_d,
            max_fine / min_fine
        );
    }

    #[test]
    fn t_uniform_walk_edge_cases() {
        // Zero-length curve → no points emitted.
        let zero = make_clamped(&[vec3(1., 2., 3.), vec3(1., 2., 3.)], 1);
        assert_eq!(
            zero.uniform_samples(0.0, 0.5, 1e-6).count(),
            0,
            "zero-length curve should yield 0 points"
        );

        let line = make_clamped(&[vec3(0., 0., 0.), vec3(5., 0., 0.)], 1);

        // start beyond total length → no points.
        assert_eq!(
            line.uniform_samples(6.0, 1.0, 1e-6).count(),
            0,
            "start past end should yield 0 points"
        );

        // step larger than total length → exactly one point (the start).
        let pts: Vec<_> = line.uniform_samples(0.0, 100.0, 1e-6).collect();
        assert_eq!(pts.len(), 1, "oversized step should yield 1 point");
        assert!((pts[0] - vec3(0., 0., 0.)).length() < 1e-9);

        // step equals total length → exactly two points (start and end).
        let pts2: Vec<_> = line.uniform_samples(0.0, 5.0, 1e-6).collect();
        assert_eq!(pts2.len(), 2, "step==length should yield 2 points");
        assert!((pts2[0] - vec3(0., 0., 0.)).length() < 1e-9);
        assert!((pts2[1] - vec3(5., 0., 0.)).length() < 1e-9);

        // Multi-segment cubic: uniform_walk output matches length() estimate.
        let cubic = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., -1.),
                vec3(2., -1., 3.),
                vec3(3., 0., 0.),
            ],
            3,
        );
        let total = cubic.length(1e-6);
        let step = total / 7.0;
        let n = cubic.uniform_samples(0.0, step, 1e-6).count() as f64;
        assert!((n - 1.0) * step <= total + 1e-6, "too many points: {n}");
        assert!(n * step > total - 1e-6, "too few points: {n}");
    }

    #[test]
    fn t_clamped_endpoints_match_control_points() {
        let control_points = [
            vec3(0.7123, PI, -2.8081),
            vec3(-SQRT_2, 0.5772, TAU),
            vec3(E, -LN_2, LOG2_E),
            vec3(-3.3691, 4.6692, -FRAC_1_PI),
            vec3(0.1103, -2.5029, 5.7722),
            vec3(8.3144, 1.6180, -4.1888),
            vec3(-0.9033, 7.3891, 0.6137),
        ];
        for degree in 1..=6 {
            let pts = &control_points[..degree + 1];
            let spline = make_clamped(pts, degree);
            let first = *pts.first().unwrap();
            let last = *pts.last().unwrap();
            assert_eq!(
                spline.start(),
                first,
                "degree {degree}: start {:?} != first control point {first:?}",
                spline.start()
            );
            assert_eq!(
                spline.end(),
                last,
                "degree {degree}: end {:?} != last control point {last:?}",
                spline.end()
            );
        }
    }

    fn assert_clamped_basis_exact_at_endpoints(knots: &[f64], degree: usize) {
        let u_start = knots[0];
        let u_max = knots[knots.len() - 1];
        // At domain start.
        let span_start = find_span::<F64Adaptor>(knots, degree, u_start);
        with_thread_local_buffers::<_, f64, _, _>(
            [degree + 1, degree + 1, degree + 1],
            |[basis, left, right]| {
                calc_basis::<F64Adaptor>(span_start, u_start, degree, knots, [basis, left, right]);
                assert_eq!(
                    basis[0], 1.0,
                    "degree={degree}: basis[0] at start = {} (expected 1.0); knots={knots:?}",
                    basis[0]
                );
                assert!(
                    basis[1..=degree].iter().all(|c| *c == 0.0),
                    "degree={degree}: non-zero basis at start: {:?}",
                    &basis[..degree + 1]
                );
            },
        );
        // At domain end.
        let span_end = find_span::<F64Adaptor>(knots, degree, u_max);
        with_thread_local_buffers::<_, f64, _, _>(
            [degree + 1, degree + 1, degree + 1],
            |[basis, left, right]| {
                calc_basis::<F64Adaptor>(span_end, u_max, degree, knots, [basis, left, right]);
                assert_eq!(
                    basis[degree], 1.0,
                    "degree={degree}: basis[{degree}] at end = {} (expected 1.0)",
                    basis[degree]
                );
                assert!(
                    basis[..degree].iter().all(|c| *c == 0.0),
                    "degree={degree}: non-zero basis at end: {:?}",
                    &basis[..degree + 1]
                );
            },
        );
    }

    fn make_clamped_knots(degree: usize, u_start: f64, u_max: f64, interior: &[f64]) -> Vec<f64> {
        let mut knots = Vec::new();
        knots.extend(std::iter::repeat_n(u_start, degree + 1));
        knots.extend(interior.iter().map(|&k| u_start + k));
        knots.extend(std::iter::repeat_n(u_max, degree + 1));
        knots
    }

    const TEST_KNOT_SCALES: &[f64] = &[
        1e-15,
        1e-10,
        1e-6,
        1.0 / 7.0,
        1.0 / 3.0,
        0.5,
        1.0,
        std::f64::consts::PI,
        7.0,
        100.0,
        1e6,
        1e10,
        1e15,
    ];

    #[test]
    fn t_clamped_basis_endpoints_uniform_knots() {
        let mut rng = StdRng::seed_from_u64(42);
        for degree in 1..=6 {
            let n_ctrl = degree + 1;
            let n_middle = n_ctrl + degree + 1 - 2 * (degree + 1);
            for &scale in TEST_KNOT_SCALES {
                let step = 0.1 + rng.random::<f64>() * 10.0;
                let interior: Vec<f64> = (0..n_middle).map(|i| (i + 1) as f64 * step).collect();
                let u_start = -scale * rng.random::<f64>();
                let u_max = u_start
                    + interior.last().copied().unwrap_or(1.0)
                    + scale * rng.random::<f64>().max(1e-20);
                let knots = make_clamped_knots(degree, u_start, u_max, &interior);
                assert_clamped_basis_exact_at_endpoints(&knots, degree);
            }
        }
    }

    #[test]
    fn t_clamped_basis_endpoints_random_increment_knots() {
        let mut rng = StdRng::seed_from_u64(42);
        for degree in 1..=6 {
            let n_ctrl = degree + 1;
            let n_middle = n_ctrl + degree + 1 - 2 * (degree + 1);
            for &scale in TEST_KNOT_SCALES {
                let mut val = rng.random::<f64>() * 0.01;
                let interior: Vec<f64> = (0..n_middle)
                    .map(|_| {
                        val += rng.random::<f64>() * rng.random::<f64>() * 100.0 + 1e-12;
                        val
                    })
                    .collect();
                let u_start = -scale * rng.random::<f64>();
                let u_max = u_start
                    + interior.last().copied().unwrap_or(1.0)
                    + scale * rng.random::<f64>().max(1e-20);
                let knots = make_clamped_knots(degree, u_start, u_max, &interior);
                assert_clamped_basis_exact_at_endpoints(&knots, degree);
            }
        }
    }

    #[test]
    fn t_clamped_basis_endpoints_exponential_knots() {
        let mut rng = StdRng::seed_from_u64(42);
        for degree in 1..=6 {
            let n_ctrl = degree + 1;
            let n_middle = n_ctrl + degree + 1 - 2 * (degree + 1);
            for &scale in TEST_KNOT_SCALES {
                let base = 1.01 + rng.random::<f64>() * 5.0;
                let interior: Vec<f64> = (0..n_middle).map(|i| base.powi(i as i32 + 1)).collect();
                let u_start = -scale * rng.random::<f64>();
                let u_max = u_start
                    + interior.last().copied().unwrap_or(1.0)
                    + scale * rng.random::<f64>().max(1e-20);
                let knots = make_clamped_knots(degree, u_start, u_max, &interior);
                assert_clamped_basis_exact_at_endpoints(&knots, degree);
            }
        }
    }

    #[test]
    fn t_clamped_basis_endpoints_tightly_clustered_knots() {
        let mut rng = StdRng::seed_from_u64(42);
        for degree in 1..=6 {
            let n_ctrl = degree + 1;
            let n_middle = n_ctrl + degree + 1 - 2 * (degree + 1);
            for &scale in TEST_KNOT_SCALES {
                let mut val = 1.0;
                let interior: Vec<f64> = (0..n_middle)
                    .map(|_| {
                        val += f64::EPSILON * (1.0 + rng.random::<f64>() * 1e6);
                        val
                    })
                    .collect();
                let u_start = -scale * rng.random::<f64>();
                let u_max = u_start
                    + interior.last().copied().unwrap_or(1.0)
                    + scale * rng.random::<f64>().max(1e-20);
                let knots = make_clamped_knots(degree, u_start, u_max, &interior);
                assert_clamped_basis_exact_at_endpoints(&knots, degree);
            }
        }
    }

    // ── curvature tests ───────────────────────────────────────────────────

    #[test]
    fn t_curvature_linear_spline_is_zero() {
        // Degree-1 spline: all higher-order derivatives are zero.
        let spline = make_clamped(&[vec3(0., 0., 0.), vec3(3., 4., 0.)], 1);
        let (lo, hi) = spline.domain();
        assert_eq!(spline.curvature(lo), Some(vec3(0., 0., 0.)));
        assert_eq!(spline.curvature((lo + hi) / 2.0), Some(vec3(0., 0., 0.)));
        assert_eq!(spline.curvature(hi), Some(vec3(0., 0., 0.)));
        assert!(spline.curvature(lo - 0.001).is_none());
        assert!(spline.curvature(hi + 0.001).is_none());
    }

    #[test]
    fn t_curvature_quadratic_constant_second_derivative() {
        // Quadratic Bezier with p0=(0,0,0), p1=(0.5,1,0), p2=(1,0,0).
        // P(t) = (t, 2t-2t², 0), so P''(t) = (0,-4,0) everywhere.
        let spline = make_clamped(&[vec3(0., 0., 0.), vec3(0.5, 1., 0.), vec3(1., 0., 0.)], 2);
        let expected = vec3(0., -4., 0.);
        let (lo, hi) = spline.domain();
        for i in 0..=4 {
            let t = lo + (hi - lo) * i as f64 / 4.0;
            let curv = spline.curvature(t).unwrap();
            assert!(
                (curv - expected).length() < 1e-10,
                "curvature at t={t}: got {curv:?}"
            );
        }
    }

    // ── is_closed tests ───────────────────────────────────────────────────

    #[test]
    fn t_is_closed() {
        // Open spline: first and last control points differ.
        let open = make_clamped(&[vec3(0., 0., 0.), vec3(1., 1., 0.), vec3(2., 0., 0.)], 2);
        assert!(!open.is_closed());

        // Closed spline: first and last control points are identical.
        let closed = make_clamped(
            &[
                vec3(1., 0., 0.),
                vec3(0., 1., 0.),
                vec3(-1., 0., 0.),
                vec3(0., -1., 0.),
                vec3(1., 0., 0.),
            ],
            3,
        );
        assert!(closed.is_closed());
    }

    #[test]
    fn t_serialize_deserialize_roundtrip() {
        let spline = make_clamped(
            &[
                vec3(0., 0., 0.),
                vec3(1., 2., -1.),
                vec3(2., -3., 4.),
                vec3(3., 5., -2.),
                vec3(4., 0., 0.),
            ],
            3,
        );
        let mut bytes = Vec::new();
        spline.serialize(&mut bytes).unwrap();
        let restored = Spline3d::deserialize(&bytes[..]).unwrap();
        assert_eq!(restored.knots, spline.knots);
        assert_eq!(restored.control_points, spline.control_points);
        assert_eq!(restored.unique_knots, spline.unique_knots);
        assert_eq!(restored.power_basis_coeff, spline.power_basis_coeff);
        assert_eq!(restored.degree(), spline.degree());
        assert_eq!(restored.domain(), spline.domain());
    }
}
