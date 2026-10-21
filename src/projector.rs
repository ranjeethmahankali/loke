use crate::{Adaptor, Arc, EllipticArc, Error, LineSeg, Spline, TrigonometryAdaptor, polynomial};

/// Root tolerance for the dot product polynomials. Tighter tolerances are slower and no more
/// accurate, because the last Newton step already lands far inside this (see tn8bmt).
const ROOT_TOLERANCE: f64 = 6e-7;

/// The dot product polynomial of a spline segment has degree `2 * degree - 1`, and the root
/// finder supports up to 16.
const MAX_SPLINE_DEGREE: usize = 8;
const MAX_COEFFS: usize = 2 * MAX_SPLINE_DEGREE;

/// Fixed size data to project onto one segment. Variable length data (polynomial
/// coefficients) lives in the parent projector, and is referred to by offsets.
enum SegmentProjector<const DIM: usize, A: Adaptor<DIM>> {
    /// One knot span of a spline, in the local parameter `τ` in `[0, 1]`. The roots of
    /// `(P - q)·P' = P·P' - q·P'` are the critical points.
    Polynomial {
        degree: usize,
        position_dot_deriv_coeff_offset: usize, // `2 * degree` coefficients of P·P'.
        deriv_coeff_offset: [usize; DIM],       // `degree` coefficients of P', per coordinate.
        position_coeff_offset: [usize; DIM],    // `degree + 1` coefficients of P, per coordinate.
    },
    /// The end points in the segment info are all we need.
    Line,
    /// Frame at the midpoint of the arc: `m` toward the midpoint, `n` the unit tangent there.
    /// The closest point is in the direction of `q - center` projected onto the plane, clamped
    /// to the sweep.
    Arc {
        center: A::Vector,
        m: A::Vector,
        n: A::Vector,
        radius: A::Scalar,
        cos_half: A::Scalar, // Cosine of half the sweep.
    },
    /// Frame at the midpoint of the arc, `P(α) - center = cos α U + sin α V` for `|α| <= θ/2`.
    /// With `τ = tan(α/2)`, `(P - q)·P' (1 + τ²)²` is a quartic in `τ`.
    EllipticArc {
        center: A::Vector,
        u: A::Vector,
        v: A::Vector,
        b_minus_a: A::Scalar, // V·V - U·U.
        w: A::Scalar,         // U·V.
        tau_max: A::Scalar,   // tan(θ/4).
    },
}

struct SegmentInfo<const DIM: usize, A: Adaptor<DIM>> {
    projector: SegmentProjector<DIM, A>,
    start: A::Vector,
    end: A::Vector,
}

/// Finds the closest point on a curve to a query point.
///
/// Curves are broken into segments (a spline into its knot spans), each with cached data
/// for fast queries. Each segment's bounding box is used to skip segments that cannot
/// contain a closer point than the best found so far.
pub struct Projector<const DIM: usize, A: Adaptor<DIM>> {
    polynomial_coefficients: Box<[A::Scalar]>,
    infos: Box<[SegmentInfo<DIM, A>]>,
    // Parallel to `infos`, kept separate so the culling scan stays dense.
    bounds: Box<[(A::Vector, A::Vector)]>,
}

impl<const DIM: usize, A: Adaptor<DIM> + TrigonometryAdaptor> Projector<DIM, A> {
    fn from_single(info: SegmentInfo<DIM, A>, bounds: (A::Vector, A::Vector)) -> Self {
        Self {
            polynomial_coefficients: Box::new([]),
            infos: Box::new([info]),
            bounds: Box::new([bounds]),
        }
    }

    pub fn from_line(line: &LineSeg<DIM, A>) -> Self {
        Self::from_single(
            SegmentInfo {
                projector: SegmentProjector::Line,
                start: line.start(),
                end: line.end(),
            },
            line.bounds(),
        )
    }

    pub fn from_arc(arc: &Arc<DIM, A>) -> Self {
        let projector = if arc.angle < A::epsilon() {
            SegmentProjector::Line // Same as `slerp`, which treats such an arc as linear.
        } else {
            SegmentProjector::Arc {
                center: arc.center,
                m: arc.mid_dir,
                n: A::normalize(arc.end_dir - arc.start_dir), // ⟂ m, by symmetry.
                radius: arc.radius,
                cos_half: A::cos(arc.angle * A::scalar(0.5)),
            }
        };
        Self::from_single(
            SegmentInfo {
                projector,
                start: arc.start(),
                end: arc.end(),
            },
            arc.bounds(),
        )
    }

    pub fn from_elliptic_arc(arc: &EllipticArc<DIM, A>) -> Self {
        let (sin_h, cos_h) = A::sin_cos(arc.angle * A::scalar(0.5));
        let projector = if sin_h < A::epsilon() {
            SegmentProjector::Line
        } else {
            // V = P'(φ = h), from §1 of ELLIPSE ARC MATH with the frame moved to the midpoint.
            let u = arc.mid_vec;
            let v = (u * cos_h - arc.start_vec) / sin_h;
            SegmentProjector::EllipticArc {
                center: arc.center,
                u,
                v,
                b_minus_a: A::dot_product(v, v) - A::dot_product(u, u),
                w: A::dot_product(u, v),
                tau_max: A::tan(arc.angle * A::scalar(0.25)),
            }
        };
        Self::from_single(
            SegmentInfo {
                projector,
                start: arc.start(),
                end: arc.end(),
            },
            arc.bounds(),
        )
    }

    /// Fails with [`Error::OutOfBounds`] for splines of degree higher than 8.
    pub fn from_spline(spline: &Spline<DIM, A>) -> Result<Self, Error> {
        let degree = spline.degree();
        if degree > MAX_SPLINE_DEGREE {
            return Err(Error::OutOfBounds(
                "spline degree exceeds maximum supported degree of 8 for projection",
            ));
        }
        let n_segs = spline.num_segments();
        // Per segment: P·P', then P' and P for each coordinate.
        let stride = 2 * degree + DIM * (2 * degree + 1);
        let mut coeffs = vec![A::scalar(0.0); n_segs * stride];
        let mut infos = Vec::with_capacity(n_segs);
        let mut bounds = Vec::with_capacity(n_segs);
        let mut roots = [A::scalar(0.0); MAX_COEFFS];
        for si in 0..n_segs {
            let base = si * stride;
            let deriv_coeff_offset = std::array::from_fn(|ci| base + 2 * degree + ci * degree);
            let position_coeff_offset =
                std::array::from_fn(|ci| base + 2 * degree + DIM * degree + ci * (degree + 1));
            let mut lo = [A::scalar(0.0); DIM];
            let mut hi = [A::scalar(0.0); DIM];
            for ci in 0..DIM {
                let pos = spline.power_basis_polynomial(si, ci);
                let mut deriv = [A::scalar(0.0); MAX_SPLINE_DEGREE];
                let deriv = &mut deriv[..degree];
                polynomial::differentiate::<A>(pos, deriv);
                polynomial::mul_add::<A>(pos, deriv, &mut coeffs[base..base + 2 * degree]);
                let (p0, d0) = (position_coeff_offset[ci], deriv_coeff_offset[ci]);
                coeffs[p0..p0 + degree + 1].copy_from_slice(pos);
                coeffs[d0..d0 + degree].copy_from_slice(deriv);
                // Coordinate extrema are at the ends, or at the roots of P'.
                let (v0, v1) = (
                    polynomial::eval::<A>(pos, A::scalar(0.0)),
                    polynomial::eval::<A>(pos, A::scalar(1.0)),
                );
                (lo[ci], hi[ci]) = (A::min(v0, v1), A::max(v0, v1));
                let n_roots = polynomial::polynomial_roots_in_range::<A>(
                    deriv,
                    &mut roots,
                    A::scalar(0.0),
                    A::scalar(1.0),
                    A::epsilon(),
                )
                .expect("INTERNAL ERROR: Degree is checked above");
                for &r in &roots[..n_roots] {
                    let val = polynomial::eval::<A>(pos, r);
                    (lo[ci], hi[ci]) = (A::min(lo[ci], val), A::max(hi[ci], val));
                }
            }
            let eval = |t: A::Scalar| {
                A::vector(std::array::from_fn(|ci| {
                    polynomial::eval::<A>(spline.power_basis_polynomial(si, ci), t)
                }))
            };
            infos.push(SegmentInfo {
                projector: SegmentProjector::Polynomial {
                    degree,
                    position_dot_deriv_coeff_offset: base,
                    deriv_coeff_offset,
                    position_coeff_offset,
                },
                start: eval(A::scalar(0.0)),
                end: eval(A::scalar(1.0)),
            });
            bounds.push((A::vector(lo), A::vector(hi)));
        }
        Ok(Self {
            polynomial_coefficients: coeffs.into_boxed_slice(),
            infos: infos.into_boxed_slice(),
            bounds: bounds.into_boxed_slice(),
        })
    }

    /// The closest point on the curve to `q`.
    #[inline]
    pub fn project(&self, q: A::Vector) -> A::Vector {
        if self.infos.len() == 1 {
            return self.project_segment(&self.infos[0], q);
        }
        // Seeding with the nearest segment end lets the box test skip most segments before any
        // roots are found: 2.3 instead of 15 of 30 spline segments solved per query.
        let mut best = (A::scalar(f64::INFINITY), q);
        for info in self.infos.iter() {
            best = closer::<DIM, A>(q, best, info.start);
            best = closer::<DIM, A>(q, best, info.end);
        }
        for (info, &(lo, hi)) in self.infos.iter().zip(self.bounds.iter()) {
            if box_distance_sq::<DIM, A>(q, lo, hi) < best.0 {
                best = closer::<DIM, A>(q, best, self.project_segment(info, q));
            }
        }
        best.1
    }

    /// The closest point on the segment to `q`.
    ///
    /// Inlined all the way into the caller. Otherwise `q` and the result go through memory at
    /// every call boundary, and that chain of store forwards makes a line projection several
    /// times slower.
    #[inline(always)]
    fn project_segment(&self, seg_info: &SegmentInfo<DIM, A>, q: A::Vector) -> A::Vector {
        match &seg_info.projector {
            SegmentProjector::Polynomial {
                degree,
                position_dot_deriv_coeff_offset,
                deriv_coeff_offset,
                position_coeff_offset,
            } => {
                let degree = *degree;
                let all = &self.polynomial_coefficients;
                let n_coeff = 2 * degree;
                let mut coef = [A::scalar(0.0); MAX_COEFFS];
                let coef = &mut coef[..n_coeff];
                coef.copy_from_slice(&all[*position_dot_deriv_coeff_offset..][..n_coeff]);
                for (ci, &offset) in deriv_coeff_offset.iter().enumerate() {
                    let qi = A::vector_coord(q, ci);
                    for (c, &d) in coef.iter_mut().zip(&all[offset..offset + degree]) {
                        *c -= qi * d;
                    }
                }
                let mut roots = [A::scalar(0.0); MAX_COEFFS];
                let n_roots = polynomial::polynomial_roots_in_range::<A>(
                    coef,
                    &mut roots,
                    A::scalar(0.0),
                    A::scalar(1.0),
                    A::scalar(ROOT_TOLERANCE),
                )
                .expect("INTERNAL ERROR: Degree is checked on construction");
                let mut best =
                    closer::<DIM, A>(q, start_candidate::<DIM, A>(q, seg_info), seg_info.end);
                for &t in &roots[..n_roots] {
                    let p = A::vector(std::array::from_fn(|ci| {
                        let offset = position_coeff_offset[ci];
                        polynomial::eval::<A>(&all[offset..offset + degree + 1], t)
                    }));
                    best = closer::<DIM, A>(q, best, p);
                }
                best.1
            }
            SegmentProjector::Line => {
                let dir = seg_info.end - seg_info.start;
                let len_sq = A::vector_length_sq(dir);
                let t = if len_sq > A::scalar(0.0) {
                    A::clamp(
                        A::dot_product(q - seg_info.start, dir) / len_sq,
                        A::scalar(0.0),
                        A::scalar(1.0),
                    )
                } else {
                    A::scalar(0.0)
                };
                seg_info.start + dir * t
            }
            SegmentProjector::Arc {
                center,
                m,
                n,
                radius,
                cos_half,
            } => {
                // TODO: We need a variant of projection that returns the parameter value too. We
                // might need to bring back the atan2. We can have two codepaths, or just go with
                // the atan2. What is the comparison with and without atan2?

                // The distance grows with the angle from the direction of d in the plane, and
                // the arc is symmetric about m, so outside the sweep the nearer end is on the
                // side of d·n.
                let d = q - *center;
                let (dm, dn) = (A::dot_product(d, *m), A::dot_product(d, *n));
                let rho = A::sqrt(dm * dm + dn * dn);
                if rho > A::scalar(0.0) && dm >= *cos_half * rho {
                    *center + (*m * dm + *n * dn) * (*radius / rho)
                } else if dn < A::scalar(0.0) {
                    seg_info.start
                } else {
                    seg_info.end
                }
            }
            SegmentProjector::EllipticArc {
                center,
                u,
                v,
                b_minus_a,
                w,
                tau_max,
            } => {
                let d = q - *center;
                let (du, dv) = (A::dot_product(d, *u), A::dot_product(d, *v));
                let two = A::scalar(2.0);
                let coef = [
                    *w - dv,
                    two * (*b_minus_a + du),
                    A::scalar(-6.0) * *w,
                    two * (du - *b_minus_a),
                    *w + dv,
                ];
                let mut roots = [A::scalar(0.0); 4];
                let n_roots = polynomial::polynomial_roots_in_range::<A>(
                    &coef,
                    &mut roots,
                    -*tau_max,
                    *tau_max,
                    A::scalar(ROOT_TOLERANCE),
                )
                .expect("INTERNAL ERROR: A quartic is always supported");
                let mut best =
                    closer::<DIM, A>(q, start_candidate::<DIM, A>(q, seg_info), seg_info.end);
                for &tau in &roots[..n_roots] {
                    let k = A::scalar(1.0) / (A::scalar(1.0) + tau * tau);
                    let (cos_a, sin_a) = ((A::scalar(1.0) - tau * tau) * k, two * tau * k);
                    best = closer::<DIM, A>(q, best, *center + *u * cos_a + *v * sin_a);
                }
                best.1
            }
        }
    }
}

/// `(squared distance, point)` of the segment start, to seed a search over candidates.
#[inline(always)]
fn start_candidate<const DIM: usize, A: Adaptor<DIM>>(
    q: A::Vector,
    info: &SegmentInfo<DIM, A>,
) -> (A::Scalar, A::Vector) {
    (A::vector_length_sq(info.start - q), info.start)
}

/// `best` (squared distance, point), or `p` if that is closer to `q`.
#[inline(always)]
fn closer<const DIM: usize, A: Adaptor<DIM>>(
    q: A::Vector,
    best: (A::Scalar, A::Vector),
    p: A::Vector,
) -> (A::Scalar, A::Vector) {
    let dsq = A::vector_length_sq(p - q);
    if dsq < best.0 { (dsq, p) } else { best }
}

/// Squared distance from `q` to the box `[lo, hi]`, zero inside.
fn box_distance_sq<const DIM: usize, A: Adaptor<DIM>>(
    q: A::Vector,
    lo: A::Vector,
    hi: A::Vector,
) -> A::Scalar {
    let (q, lo, hi) = (A::coord_arr(q), A::coord_arr(lo), A::coord_arr(hi));
    let mut sum = A::scalar(0.0);
    for ci in 0..DIM {
        let gap = A::max(A::max(lo[ci] - q[ci], q[ci] - hi[ci]), A::scalar(0.0));
        sum += gap * gap;
    }
    sum
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{DVec, F32Adaptor, F64Adaptor, Vec as FVec};
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::f64::consts::TAU;

    fn rand_vec<const D: usize>(rng: &mut StdRng, scale: f64) -> DVec<D> {
        DVec(std::array::from_fn(|_| rng.random_range(-scale..scale)))
    }

    /// Distance from `q` to the curve `f` on `[lo, hi]`: dense samples, then golden section
    /// search around every local minimum.
    fn brute_force_dist<const D: usize>(
        f: impl Fn(f64) -> DVec<D>,
        (lo, hi): (f64, f64),
        q: DVec<D>,
    ) -> f64 {
        const N: usize = 4000;
        let param = |i: f64| lo + (hi - lo) * (i / N as f64).clamp(0.0, 1.0);
        let dist = |u: f64| (f(u) - q).length();
        let samples: std::vec::Vec<f64> = (0..=N).map(|i| dist(param(i as f64))).collect();
        let mut best = f64::MAX;
        for i in 0..=N {
            let left = if i > 0 { samples[i - 1] } else { f64::MAX };
            let right = if i < N { samples[i + 1] } else { f64::MAX };
            if samples[i] > left || samples[i] > right {
                continue;
            }
            let (mut a, mut b) = (param(i as f64 - 1.0), param(i as f64 + 1.0));
            for _ in 0..100 {
                let (x0, x1) = (a + (b - a) * 0.381966, b - (b - a) * 0.381966);
                if dist(x0) < dist(x1) {
                    b = x1;
                } else {
                    a = x0;
                }
            }
            best = best.min(samples[i]).min(dist((a + b) * 0.5));
        }
        best
    }

    /// Projects random queries, and checks that the result is on the curve and as close as
    /// the brute force distance.
    fn check<const D: usize>(
        proj: &Projector<D, F64Adaptor>,
        f: impl Fn(f64) -> DVec<D> + Copy,
        domain: (f64, f64),
        queries: impl IntoIterator<Item = DVec<D>>,
    ) {
        for q in queries {
            let p = proj.project(q);
            let reference = brute_force_dist(f, domain, q);
            let found = (p - q).length();
            let scale = 1.0 + q.length();
            assert!(
                (found - reference).abs() <= 1e-11 * scale,
                "q = {q:?}, p = {p:?}, found {found}, reference {reference}"
            );
            // The result is a point on the curve.
            assert!(
                brute_force_dist(f, domain, p) <= 1e-9 * scale,
                "p = {p:?} is off the curve"
            );
        }
    }

    fn rand_queries<const D: usize>(rng: &mut StdRng, n: usize) -> std::vec::Vec<DVec<D>> {
        (0..n).map(|_| rand_vec::<D>(rng, 6.0)).collect()
    }

    #[test]
    fn t_line() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..20 {
            let line =
                LineSeg::<3, F64Adaptor>::create(rand_vec(&mut rng, 3.0), rand_vec(&mut rng, 3.0));
            let queries = rand_queries::<3>(&mut rng, 10);
            check(
                &Projector::from_line(&line),
                |t| line.point(t).unwrap(),
                (0.0, 1.0),
                queries,
            );
        }
        // Zero length.
        let p = DVec([1.0, 2.0]);
        let proj = Projector::from_line(&LineSeg::<2, F64Adaptor>::create(p, p));
        assert_eq!(proj.project(DVec([5.0, -1.0])), p);
    }

    fn rand_arc<const D: usize>(rng: &mut StdRng) -> Arc<D, F64Adaptor> {
        loop {
            let [a, b, c] = std::array::from_fn(|_| rand_vec::<D>(rng, 3.0));
            if let Ok(arc) = Arc::from_three_points(a, b, c) {
                return arc;
            }
        }
    }

    fn check_arcs<const D: usize>(seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..30 {
            let arc = rand_arc::<D>(&mut rng);
            let mut queries = rand_queries::<D>(&mut rng, 10);
            // On the axis through the center: every point is equally close.
            queries.push(arc.center());
            check(
                &Projector::from_arc(&arc),
                |t| arc.point(t).unwrap(),
                (0.0, 1.0),
                queries,
            );
        }
    }

    #[test]
    fn t_arc_2d() {
        check_arcs::<2>(2);
    }

    #[test]
    fn t_arc_3d() {
        check_arcs::<3>(3);
    }

    #[test]
    fn t_arc_5d() {
        check_arcs::<5>(5);
    }

    /// A random circular arc of the given sweep, squashed onto a random plane.
    fn rand_elliptic_arc(
        rng: &mut StdRng,
        max_ratio: f64,
        sweep: f64,
    ) -> EllipticArc<3, F64Adaptor> {
        loop {
            let center = rand_vec::<3>(rng, 2.0);
            let normal = rand_vec::<3>(rng, 1.0).normalize();
            let x = rand_vec::<3>(rng, 1.0);
            let x = (x - normal * x.dot(normal)).normalize();
            // Tilting the arc by acos(1 / ratio) against the plane projects it to that B:A ratio.
            let tilt = (1.0 / rng.random_range(1.0..max_ratio)).acos();
            let y = normal.cross(x) * tilt.cos() + normal * tilt.sin();
            let r = rng.random_range(0.5..3.0);
            let a0 = rng.random_range(0.0..TAU);
            let p = |a: f64| center + (x * a.cos() + y * a.sin()) * r;
            let Ok(arc) = Arc::from_three_points(p(a0), p(a0 + sweep * 0.5), p(a0 + sweep)) else {
                continue;
            };
            if let Ok(e) = EllipticArc::from_projected_arc(&arc, center, normal) {
                return e;
            }
        }
    }

    #[test]
    fn t_elliptic_arc() {
        let mut rng = StdRng::seed_from_u64(4);
        for (max_ratio, sweep_lo, sweep_hi) in [(4.0, 0.1, TAU - 1e-2), (100.0, 0.1, TAU - 1e-2)] {
            for _ in 0..30 {
                let sweep = rng.random_range(sweep_lo..sweep_hi);
                let arc = rand_elliptic_arc(&mut rng, max_ratio, sweep);
                let queries: std::vec::Vec<_> = (0..10)
                    .map(|_| arc.center() + rand_vec::<3>(&mut rng, 4.0))
                    .chain([arc.center()])
                    .collect();
                check(
                    &Projector::from_elliptic_arc(&arc),
                    |t| arc.point(t).unwrap(),
                    (0.0, 1.0),
                    queries,
                );
            }
        }
    }

    fn check_splines<const D: usize>(seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for degree in 1..=MAX_SPLINE_DEGREE {
            for _ in 0..4 {
                let n_cps = degree + 1 + rng.random_range(0..6);
                let cps: std::vec::Vec<DVec<D>> =
                    (0..n_cps).map(|_| rand_vec::<D>(&mut rng, 3.0)).collect();
                // Random increasing knots, with some interior knots repeated up to `degree` times.
                let mut knots = vec![0.0; degree + 1];
                let mut mult = 0;
                for _ in 0..(n_cps - degree - 1) {
                    let last = *knots.last().unwrap();
                    if mult > 0 && mult < degree && rng.random_bool(0.3) {
                        knots.push(last);
                        mult += 1;
                    } else {
                        knots.push(last + rng.random_range(0.1..2.0));
                        mult = 1;
                    }
                }
                let end = *knots.last().unwrap() + rng.random_range(0.1..2.0);
                knots.extend(std::iter::repeat_n(end, degree + 1));
                let spline = Spline::<D, F64Adaptor>::create(cps, knots, degree).unwrap();
                let proj = Projector::from_spline(&spline).unwrap();
                let queries = rand_queries::<D>(&mut rng, 8);
                check(
                    &proj,
                    |u| spline.point(u).unwrap(),
                    spline.domain(),
                    queries,
                );
            }
        }
    }

    #[test]
    fn t_spline_2d() {
        check_splines::<2>(6);
    }

    #[test]
    fn t_spline_3d() {
        check_splines::<3>(7);
    }

    #[test]
    fn t_spline_degree_too_high() {
        let cps: std::vec::Vec<DVec<2>> =
            (0..10).map(|i| DVec([i as f64, (i * i) as f64])).collect();
        let spline = Spline::<2, F64Adaptor>::create_clamped(cps, 9).unwrap();
        assert!(matches!(
            Projector::from_spline(&spline),
            Err(Error::OutOfBounds(_))
        ));
    }

    #[test]
    fn t_f32() {
        let arc = Arc::<3, F32Adaptor>::from_three_points(
            FVec([1.0, 0.0, 0.0]),
            FVec([0.0, 1.0, 0.0]),
            FVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let p = Projector::from_arc(&arc).project(FVec([0.0, 3.0, 2.0]));
        assert!((p - FVec([0.0, 1.0, 0.0])).length() < 1e-6);
        let p = Projector::from_arc(&arc).project(FVec([0.5, -3.0, 0.0]));
        assert!((p - FVec([1.0, 0.0, 0.0])).length() < 1e-6);
    }
}
