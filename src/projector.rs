use crate::{
    Adaptor, Arc, EllipticArc, Error, LineSeg, ScalarAdaptor, Spline, TrigonometryAdaptor,
    polynomial,
};

/// Root tolerance for the dot product polynomials. Tighter tolerances are slower and no more
/// accurate, because the last Newton step already lands far inside this (see tn8bmt).
const ROOT_TOLERANCE: f64 = 6e-7;

/// The dot product polynomial of a spline segment has degree `2 * degree - 1`, and the root
/// finder supports up to 16.
const MAX_SPLINE_DEGREE: usize = 8;
const MAX_COEFFS: usize = 2 * MAX_SPLINE_DEGREE;

/// Samples are placed so the tangent turns by at most this angle (radians) between neighbours.
/// Fewer samples mean looser bounding balls, more mean more balls to test: 1 was fastest of
/// 0.25, 0.5, 1, 1.5 and 2.5. Exactness doesn't depend on it.
const MAX_TURN: f64 = 1.0;
/// Elliptic arc samples are at most this far apart in angle (radians), which keeps the local
/// half-angle parameter of each interval small and well conditioned.
const MAX_ANGLE_STEP: f64 = 1.0;
/// Limits how often an interval is halved while placing samples, e.g. around a cusp.
const MAX_SAMPLE_DEPTH: usize = 12;
/// Newton (or bisection) iterations per bracket, at most.
const MAX_NEWTON: usize = 12;
/// How often an interval's Bernstein coefficients are split to isolate the roots, at most.
const MAX_SPLIT_DEPTH: usize = 8;

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
        samples: Samples,
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
        cos_half: A::Scalar,  // Cosine of half the sweep.
        inv_angle: A::Scalar, // 1 / sweep, to turn the angle from the midpoint into `t`.
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
        inv_angle: A::Scalar, // 1 / θ.
        samples: Samples,     // Sample parameters are angles `α`.
    },
}

/// Where a segment's samples live in the projector's buffers.
#[derive(Clone, Copy)]
struct Samples {
    params: (usize, usize), // Range of sample parameters, and of bounding balls.
    bernstein: usize,       // Start of the Bernstein blocks, one per interval.
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
/// contain a closer point than the best found so far. Lines and arcs have closed forms. Spline
/// spans and elliptic arcs are sampled, and the closest point is found with Newton from the
/// samples. The result is exact, not an approximation.
pub struct Projector<const DIM: usize, A: Adaptor<DIM>> {
    polynomial_coefficients: Box<[A::Scalar]>,
    // Sample parameters along the segments. The curve between neighbouring samples lies in a
    // ball, stored at the index of the interval's first sample (the last sample's is unused).
    sample_params: Box<[A::Scalar]>,
    ball_centers: Box<[A::Vector]>,
    ball_radii: Box<[A::Scalar]>,
    // Per interval, the Bernstein coefficients of the dot polynomial's parts: see `refine`.
    bernstein: Box<[A::Scalar]>,
    // Per elliptic arc interval, its local frame: see `from_elliptic_arc`.
    ellipse_frames: Box<[EllipseFrame<A::Scalar>]>,
    infos: Box<[SegmentInfo<DIM, A>]>,
    // Parallel to `infos`, kept separate so the culling scan stays dense.
    bounds: Box<[(A::Vector, A::Vector)]>,
    // Each segment starts where the previous one ends, as in a spline.
    contiguous: bool,
}

impl<const DIM: usize, A: Adaptor<DIM> + TrigonometryAdaptor> Projector<DIM, A> {
    fn from_single(info: SegmentInfo<DIM, A>, bounds: (A::Vector, A::Vector)) -> Self {
        Self {
            polynomial_coefficients: Box::new([]),
            sample_params: Box::new([]),
            ball_centers: Box::new([]),
            ball_radii: Box::new([]),
            bernstein: Box::new([]),
            ellipse_frames: Box::new([]),
            infos: Box::new([info]),
            bounds: Box::new([bounds]),
            contiguous: true,
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
                inv_angle: A::scalar(1.0) / arc.angle,
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
        let half = arc.angle * A::scalar(0.5);
        let (sin_h, cos_h) = A::sin_cos(half);
        let mut buffers = SampleBuffers::<DIM, A>::default();
        let mut frames = Vec::new();
        let projector = if sin_h < A::epsilon() {
            SegmentProjector::Line
        } else {
            // V = P'(φ = h), from §1 of ELLIPSE ARC MATH with the frame moved to the midpoint.
            let u = arc.mid_vec;
            let v = (u * cos_h - arc.start_vec) / sin_h;
            let point = |a: A::Scalar| {
                let (s, c) = A::sin_cos(a);
                arc.center + u * c + v * s
            };
            let samples = buffers.push(
                -half,
                half,
                A::scalar(MAX_ANGLE_STEP),
                |a| {
                    let (s, c) = A::sin_cos(a);
                    v * c - u * s
                },
                |a, b| {
                    // The arc lies in the triangle of its ends and the meeting point of their
                    // tangents, since an ellipse is an affine image of a circle.
                    let m = (a + b) * A::scalar(0.5);
                    let apex = point(m) - arc.center;
                    let apex = arc.center + apex / A::cos((b - a) * A::scalar(0.5));
                    [point(a), point(b), apex]
                },
            );
            // Each interval gets its own frame at its middle angle αc, rotated from U and V.
            // In it, the arc is P = c + cos β Uc + sin β Vc with β = α - αc, and with
            // τ = tan(β/2) the dot polynomial is the same quartic as in `closest_on_segment`,
            // with Uc and Vc. τ stays within tan(MAX_ANGLE_STEP / 4), where the quartic is well
            // conditioned. The quartic is c + (d·Uc) cu + (d·Vc) cv, and d·Uc, d·Vc are rotations
            // of d·U, d·V.
            let two = A::scalar(2.0);
            let (zero, one) = (A::scalar(0.0), A::scalar(1.0));
            let (s0, s1) = samples.params;
            for k in s0..s1 - 1 {
                let (a, b) = (buffers.params[k], buffers.params[k + 1]);
                let (sin_c, cos_c) = A::sin_cos((a + b) * A::scalar(0.5));
                let (uc, vc) = (u * cos_c + v * sin_c, v * cos_c - u * sin_c);
                let frame = EllipseFrame {
                    cos: cos_c,
                    sin: sin_c,
                    tau_max: A::tan((b - a) * A::scalar(0.25)),
                    b_minus_a: A::dot_product(vc, vc) - A::dot_product(uc, uc),
                    w: A::dot_product(uc, vc),
                };
                let bern =
                    |row: &[A::Scalar]| to_bernstein::<A>(row, 4, -frame.tau_max, frame.tau_max);
                let (w, bma) = (frame.w, frame.b_minus_a);
                let constant = bern(&[w, two * bma, A::scalar(-6.0) * w, -two * bma, w]);
                let cu = bern(&[zero, two, zero, two, zero]);
                let cv = bern(&[-one, zero, zero, zero, one]);
                buffers.bernstein.extend_from_slice(&constant[..5]);
                // d·Uc = cos d·U + sin d·V, and d·Vc = cos d·V - sin d·U.
                buffers
                    .bernstein
                    .extend((0..5).map(|j| cos_c * cu[j] - sin_c * cv[j]));
                buffers
                    .bernstein
                    .extend((0..5).map(|j| sin_c * cu[j] + cos_c * cv[j]));
                frames.push(frame);
            }
            // Keep the frames aligned with the parameters.
            frames.push(EllipseFrame {
                cos: zero,
                sin: zero,
                tau_max: zero,
                b_minus_a: zero,
                w: zero,
            });
            let (b_minus_a, w) = (
                A::dot_product(v, v) - A::dot_product(u, u),
                A::dot_product(u, v),
            );
            SegmentProjector::EllipticArc {
                center: arc.center,
                u,
                v,
                b_minus_a,
                w,
                tau_max: A::tan(arc.angle * A::scalar(0.25)),
                inv_angle: A::scalar(1.0) / arc.angle,
                samples,
            }
        };
        Self {
            sample_params: buffers.params.into_boxed_slice(),
            ball_centers: buffers.centers.into_boxed_slice(),
            ball_radii: buffers.radii.into_boxed_slice(),
            bernstein: buffers.bernstein.into_boxed_slice(),
            ellipse_frames: frames.into_boxed_slice(),
            ..Self::from_single(
                SegmentInfo {
                    projector,
                    start: arc.start(),
                    end: arc.end(),
                },
                arc.bounds(),
            )
        }
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
        let mut buffers = SampleBuffers::<DIM, A>::default();
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
            let samples = buffers.push(
                A::scalar(0.0),
                A::scalar(1.0),
                A::scalar(f64::INFINITY),
                |t| {
                    A::vector(std::array::from_fn(|ci| {
                        let offset = deriv_coeff_offset[ci];
                        polynomial::eval::<A>(&coeffs[offset..offset + degree], t)
                    }))
                },
                // The piece lies in the convex hull of its Bézier control points.
                |a, b| -> [A::Vector; MAX_SPLINE_DEGREE + 1] {
                    let ctrl: [_; DIM] = std::array::from_fn(|ci| {
                        to_bernstein::<A>(spline.power_basis_polynomial(si, ci), degree, a, b)
                    });
                    // Ends first. Unused entries repeat the start.
                    std::array::from_fn(|i| {
                        let i = match i {
                            0 => 0,
                            1 => degree,
                            i if i <= degree => i - 1,
                            _ => 0,
                        };
                        A::vector(std::array::from_fn(|ci| ctrl[ci][i]))
                    })
                },
            );
            // The dot polynomial is P·P' - Σ q_i P'_i.
            let rows: Vec<&[A::Scalar]> = std::iter::once(&coeffs[base..base + 2 * degree])
                .chain(deriv_coeff_offset.iter().map(|&o| &coeffs[o..o + degree]))
                .collect();
            buffers.push_bernstein(samples, 2 * degree - 1, &rows);
            infos.push(SegmentInfo {
                projector: SegmentProjector::Polynomial {
                    degree,
                    position_dot_deriv_coeff_offset: base,
                    deriv_coeff_offset,
                    position_coeff_offset,
                    samples,
                },
                start: eval(A::scalar(0.0)),
                end: eval(A::scalar(1.0)),
            });
            bounds.push((A::vector(lo), A::vector(hi)));
        }
        Ok(Self {
            polynomial_coefficients: coeffs.into_boxed_slice(),
            sample_params: buffers.params.into_boxed_slice(),
            ball_centers: buffers.centers.into_boxed_slice(),
            ball_radii: buffers.radii.into_boxed_slice(),
            bernstein: buffers.bernstein.into_boxed_slice(),
            ellipse_frames: Box::new([]),
            infos: infos.into_boxed_slice(),
            bounds: bounds.into_boxed_slice(),
            contiguous: true,
        })
    }

    /// The closest point on the curve to `q`.
    #[inline]
    pub fn closest_point(&self, q: A::Vector) -> A::Vector {
        self.closest::<false, false>(q).point
    }

    /// The closest point found with the root finder instead of sampled Newton: the reference in
    /// tests.
    #[cfg(test)]
    fn closest_point_exact(&self, q: A::Vector) -> A::Vector {
        self.closest::<false, true>(q).point
    }

    /// The closest point on the curve to `q`, its parameter, and the index of its segment in
    /// this projector. The parameter is local to the segment, in `[0, 1]`: for a spline that
    /// is the parameter within the knot span, not the spline's own parameter.
    #[inline]
    pub fn closest_point_with_param(&self, q: A::Vector) -> (A::Vector, A::Scalar, usize) {
        let best = self.closest::<true, false>(q);
        (best.point, best.param, best.segment)
    }

    /// Shared by the public functions. Without `WANT_PARAM` the parameter code is dead, so
    /// [`Self::closest_point`] stays small enough for the compiler to inline it into the
    /// caller, which matters: see [`Self::closest_on_segment`].
    #[inline(always)]
    fn closest<const WANT_PARAM: bool, const EXACT: bool>(
        &self,
        q: A::Vector,
    ) -> Candidate<DIM, A> {
        if self.infos.len() == 1 {
            return self.closest_on_segment::<WANT_PARAM, EXACT>(0, q);
        }
        // Seeding with the nearest segment end lets the box test skip most segments before any
        // roots are found: 2.3 instead of 15 of 30 spline segments solved per query.
        let last = self.infos.len() - 1;
        let mut best = Candidate::new(q, self.infos[last].end, A::scalar(1.0), last);
        for (i, info) in self.infos.iter().enumerate() {
            best = best.closer(q, info.start, A::scalar(0.0), i);
            if !self.contiguous {
                best = best.closer(q, info.end, A::scalar(1.0), i);
            }
        }
        for (i, &(lo, hi)) in self.bounds.iter().enumerate() {
            if box_distance_sq::<DIM, A>(q, lo, hi) < best.dist_sq {
                let candidate = self.closest_on_segment::<WANT_PARAM, EXACT>(i, q);
                if candidate.dist_sq < best.dist_sq {
                    best = candidate;
                }
            }
        }
        best
    }

    /// The closest point on segment `si` to `q`.
    ///
    /// Inlined all the way into the caller. Otherwise `q` and the result go through memory at
    /// every call boundary, and that chain of store forwards makes a line projection several
    /// times slower.
    #[inline(always)]
    fn closest_on_segment<const WANT_PARAM: bool, const EXACT: bool>(
        &self,
        si: usize,
        q: A::Vector,
    ) -> Candidate<DIM, A> {
        let seg_info = &self.infos[si];
        let (zero, one) = (A::scalar(0.0), A::scalar(1.0));
        match &seg_info.projector {
            SegmentProjector::Polynomial {
                degree,
                position_dot_deriv_coeff_offset,
                deriv_coeff_offset,
                position_coeff_offset,
                samples,
            } => {
                let degree = *degree;
                let all = &self.polynomial_coefficients;
                let n_coeff = 2 * degree;
                let mut coef = [zero; MAX_COEFFS];
                let coef = &mut coef[..n_coeff];
                coef.copy_from_slice(&all[*position_dot_deriv_coeff_offset..][..n_coeff]);
                for (ci, &offset) in deriv_coeff_offset.iter().enumerate() {
                    let qi = A::vector_coord(q, ci);
                    for (c, &d) in coef.iter_mut().zip(&all[offset..offset + degree]) {
                        *c -= qi * d;
                    }
                }
                let point_at = |t: A::Scalar| {
                    A::vector(std::array::from_fn(|ci| {
                        let offset = position_coeff_offset[ci];
                        polynomial::eval::<A>(&all[offset..offset + degree + 1], t)
                    }))
                };
                if !EXACT {
                    let weights: [A::Scalar; DIM] =
                        std::array::from_fn(|ci| -A::vector_coord(q, ci));
                    let params = &self.sample_params[samples.params.0..samples.params.1];
                    let coef = &*coef;
                    return self.refine(
                        *samples,
                        2 * degree - 1,
                        &weights,
                        q,
                        (seg_info, zero, one, si),
                        |k| (params[k], params[k + 1]),
                        |_, a, b, ga, gb| {
                            newton_in_bracket::<A>(coef, a, b, ga, gb).map(|t| (t, point_at(t)))
                        },
                    );
                }
                let mut roots = [zero; MAX_COEFFS];
                let n_roots = polynomial::polynomial_roots_in_range::<A>(
                    coef,
                    &mut roots,
                    zero,
                    one,
                    A::scalar(ROOT_TOLERANCE),
                )
                .expect("INTERNAL ERROR: Degree is checked on construction");
                let mut best =
                    Candidate::new(q, seg_info.start, zero, si).closer(q, seg_info.end, one, si);
                for &t in &roots[..n_roots] {
                    best = best.closer(q, point_at(t), t, si);
                }
                best
            }
            SegmentProjector::Line => {
                let dir = seg_info.end - seg_info.start;
                let len_sq = A::vector_length_sq(dir);
                let t = if len_sq > zero {
                    A::clamp(A::dot_product(q - seg_info.start, dir) / len_sq, zero, one)
                } else {
                    zero
                };
                Candidate::new(q, seg_info.start + dir * t, t, si)
            }
            SegmentProjector::Arc {
                center,
                m,
                n,
                radius,
                cos_half,
                inv_angle,
            } => {
                // The distance grows with the angle from the direction of d in the plane, and
                // the arc is symmetric about m, so outside the sweep the nearer end is on the
                // side of d·n.
                let d = q - *center;
                let (dm, dn) = (A::dot_product(d, *m), A::dot_product(d, *n));
                let rho = A::sqrt(dm * dm + dn * dn);
                if rho > zero && dm >= *cos_half * rho {
                    // Only the parameter needs trig, at about 10 ns.
                    let t = if WANT_PARAM {
                        A::clamp(A::scalar(0.5) + A::atan2(dn, dm) * *inv_angle, zero, one)
                    } else {
                        zero
                    };
                    Candidate::new(q, *center + (*m * dm + *n * dn) * (*radius / rho), t, si)
                } else if dn < zero {
                    Candidate::new(q, seg_info.start, zero, si)
                } else {
                    Candidate::new(q, seg_info.end, one, si)
                }
            }
            SegmentProjector::EllipticArc {
                center,
                u,
                v,
                b_minus_a,
                w,
                tau_max,
                inv_angle,
                samples,
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
                if !EXACT {
                    let (s0, s1) = samples.params;
                    let (params, frames) =
                        (&self.sample_params[s0..s1], &self.ellipse_frames[s0..s1]);
                    let half = params[params.len() - 1];
                    let mut best = self.refine(
                        *samples,
                        4,
                        &[du, dv],
                        q,
                        (seg_info, -half, half, si),
                        |k| (-frames[k].tau_max, frames[k].tau_max),
                        |k, a, b, ga, gb| {
                            // The quartic in the interval's frame, see `from_elliptic_arc`.
                            let f = &frames[k];
                            let (uc, vc) = (*u * f.cos + *v * f.sin, *v * f.cos - *u * f.sin);
                            let (duc, dvc) = (f.cos * du + f.sin * dv, f.cos * dv - f.sin * du);
                            let coef = [
                                f.w - dvc,
                                two * (f.b_minus_a + duc),
                                A::scalar(-6.0) * f.w,
                                two * (duc - f.b_minus_a),
                                f.w + dvc,
                            ];
                            let tau = newton_in_bracket::<A>(&coef, a, b, ga, gb)?;
                            let r = one / (one + tau * tau);
                            let point =
                                *center + uc * ((one - tau * tau) * r) + vc * (two * tau * r);
                            // The angle, from the middle of the interval: β = 2 atan(τ).
                            let alpha = if WANT_PARAM {
                                (params[k] + params[k + 1]) * A::scalar(0.5)
                                    + two * A::atan2(tau, one)
                            } else {
                                zero
                            };
                            Some((alpha, point))
                        },
                    );
                    if WANT_PARAM {
                        best.param = A::clamp(A::scalar(0.5) + best.param * *inv_angle, zero, one);
                    }
                    return best;
                }
                let mut roots = [zero; 4];
                let n_roots = polynomial::polynomial_roots_in_range::<A>(
                    &coef,
                    &mut roots,
                    -*tau_max,
                    *tau_max,
                    A::scalar(ROOT_TOLERANCE),
                )
                .expect("INTERNAL ERROR: A quartic is always supported");
                let mut best =
                    Candidate::new(q, seg_info.start, zero, si).closer(q, seg_info.end, one, si);
                // Roots carry τ as their parameter until the end, so atan only runs once.
                let mut best_is_root = false;
                for &tau in &roots[..n_roots] {
                    let k = one / (one + tau * tau);
                    let (cos_a, sin_a) = ((one - tau * tau) * k, two * tau * k);
                    let candidate = Candidate::new(q, *center + *u * cos_a + *v * sin_a, tau, si);
                    if candidate.dist_sq < best.dist_sq {
                        (best, best_is_root) = (candidate, true);
                    }
                }
                if WANT_PARAM && best_is_root {
                    // α = 2 atan(τ), and t = 1/2 + α/θ.
                    let t = A::scalar(0.5) + two * A::atan2(best.param, one) * *inv_angle;
                    best.param = A::clamp(t, zero, one);
                }
                best
            }
        }
    }
}

/// Sample parameters, the balls around the curve between them, and the Bernstein blocks,
/// while building.
struct SampleBuffers<const DIM: usize, A: Adaptor<DIM>> {
    params: Vec<A::Scalar>,
    centers: Vec<A::Vector>,
    radii: Vec<A::Scalar>,
    bernstein: Vec<A::Scalar>,
}

impl<const DIM: usize, A: Adaptor<DIM>> Default for SampleBuffers<DIM, A> {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            centers: Vec::new(),
            radii: Vec::new(),
            bernstein: Vec::new(),
        }
    }
}

impl<const DIM: usize, A: Adaptor<DIM> + TrigonometryAdaptor> SampleBuffers<DIM, A> {
    /// Samples a segment on `[t0, t1]`. Intervals are halved until the tangent turns by at most
    /// [`MAX_TURN`] across each, so tight bends get more samples, and until they are at most
    /// `max_width` wide. `hull(a, b)` gives points whose
    /// convex hull contains the curve on `[a, b]`, ends first.
    fn push<const N: usize>(
        &mut self,
        t0: A::Scalar,
        t1: A::Scalar,
        max_width: A::Scalar,
        tangent: impl Fn(A::Scalar) -> A::Vector,
        hull: impl Fn(A::Scalar, A::Scalar) -> [A::Vector; N],
    ) -> Samples {
        let start = self.params.len();
        let cos_max = A::cos(A::scalar(MAX_TURN));
        let turns = |a: A::Vector, b: A::Vector| {
            A::dot_product(a, b)
                < cos_max * A::sqrt(A::vector_length_sq(a) * A::vector_length_sq(b))
        };
        self.params.push(t0);
        // Intervals still to place, left to right from the top of the stack.
        let mut stack = vec![(t0, tangent(t0), t1, tangent(t1), 0)];
        while let Some((a, da, b, db, depth)) = stack.pop() {
            let m = (a + b) * A::scalar(0.5);
            let dm = tangent(m);
            if depth < MAX_SAMPLE_DEPTH && (b - a > max_width || turns(da, dm) || turns(dm, db)) {
                stack.push((m, dm, b, db, depth + 1));
                stack.push((a, da, m, dm, depth + 1));
            } else {
                self.params.push(b);
            }
        }
        for k in start..self.params.len() - 1 {
            let pts = hull(self.params[k], self.params[k + 1]);
            let center = (pts[0] + pts[1]) * A::scalar(0.5);
            let radius = pts.iter().fold(A::scalar(0.0), |r, p| {
                A::max(r, A::vector_length(*p - center))
            });
            self.centers.push(center);
            self.radii.push(radius);
        }
        // Keep the balls aligned with the parameters.
        self.centers.push(A::zero_vector());
        self.radii.push(A::scalar(0.0));
        Samples {
            params: (start, self.params.len()),
            bernstein: self.bernstein.len(),
        }
    }

    /// Appends, for every interval of `samples`, the Bernstein coefficients (degree `m`) of each
    /// polynomial in `rows`, given in the power basis of the segment parameter.
    fn push_bernstein(&mut self, samples: Samples, m: usize, rows: &[&[A::Scalar]]) {
        let (s0, s1) = samples.params;
        for k in s0..s1 - 1 {
            for row in rows {
                let (a, b) = (self.params[k], self.params[k + 1]);
                self.bernstein
                    .extend_from_slice(&to_bernstein::<A>(row, m, a, b)[..=m]);
            }
        }
    }
}

/// Bernstein coefficients (degree `m`) on `[a, b]` of the polynomial with power basis
/// coefficients `power` (degree at most `m`). The polynomial on `[a, b]` lies between the
/// smallest and largest, and the first and last are its values at `a` and `b`.
fn to_bernstein<A: ScalarAdaptor>(
    power: &[A::Float],
    m: usize,
    a: A::Float,
    b: A::Float,
) -> [A::Float; MAX_COEFFS] {
    // Coefficients in s, for the parameter a + (b - a) s: shift by a, then scale.
    let mut c = [A::scalar(0.0); MAX_COEFFS];
    c[..power.len()].copy_from_slice(power);
    for i in 0..m {
        for j in (i..m).rev() {
            let next = c[j + 1];
            c[j] += a * next;
        }
    }
    let mut scale = A::scalar(1.0);
    for cj in c.iter_mut().take(m + 1) {
        *cj *= scale;
        scale *= b - a;
    }
    // Power to Bernstein basis: b_i = Σ over j <= i of C(i, j) / C(m, j) c_j.
    let mut out = [A::scalar(0.0); MAX_COEFFS];
    for (i, oi) in out.iter_mut().enumerate().take(m + 1) {
        for (j, cj) in c.iter().enumerate().take(i + 1) {
            *oi += A::scalar(binomial(i, j) / binomial(m, j)) * *cj;
        }
    }
    out
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

impl<const DIM: usize, A: Adaptor<DIM> + TrigonometryAdaptor> Projector<DIM, A> {
    /// The closest point on a sampled segment.
    ///
    /// Intervals whose ball is farther than the best point so far are skipped. For the rest, the
    /// dot polynomial `g = (P - q)·P'` is a Bernstein polynomial: the interval's block, row 0
    /// plus `weights[i]` times row `i + 1`. Its coefficients bound the number of roots in the
    /// interval by their sign changes. A minimum of distance is a root where `g` goes from
    /// negative to positive. With one sign change that root is bracketed by `bracket(k)`, the
    /// range of the Bernstein variable, and `solve(k, a, b, g(a), g(b))` finds it, giving its
    /// parameter and point. With more, the coefficients are split in half until each piece has
    /// at most one.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn refine(
        &self,
        samples: Samples,
        m: usize,
        weights: &[A::Scalar],
        q: A::Vector,
        (seg_info, t_start, t_end, si): (&SegmentInfo<DIM, A>, A::Scalar, A::Scalar, usize),
        bracket: impl Fn(usize) -> (A::Scalar, A::Scalar),
        solve: impl Fn(
            usize,
            A::Scalar,
            A::Scalar,
            A::Scalar,
            A::Scalar,
        ) -> Option<(A::Scalar, A::Vector)>,
    ) -> Candidate<DIM, A> {
        let zero = A::scalar(0.0);
        let mut best =
            Candidate::new(q, seg_info.start, t_start, si).closer(q, seg_info.end, t_end, si);
        let mut best_dist = A::sqrt(best.dist_sq);
        let (s0, s1) = samples.params;
        let block = (weights.len() + 1) * (m + 1);
        for k in 0..s1 - s0 - 1 {
            let reach = best_dist + self.ball_radii[s0 + k];
            if A::vector_length_sq(q - self.ball_centers[s0 + k]) >= reach * reach {
                continue;
            }
            let rows = &self.bernstein[samples.bernstein + k * block..][..block];
            let mut beta = [zero; MAX_COEFFS];
            beta[..=m].copy_from_slice(&rows[..=m]);
            for (w, row) in weights.iter().zip(rows[m + 1..].chunks_exact(m + 1)) {
                for (bj, rj) in beta.iter_mut().zip(row) {
                    *bj += *w * *rj;
                }
            }
            let (a, b) = bracket(k);
            let mut consider = |found: Option<(A::Scalar, A::Vector)>| {
                if let Some((t, point)) = found {
                    let candidate = Candidate::new(q, point, t, si);
                    if candidate.dist_sq < best.dist_sq {
                        best = candidate;
                        best_dist = A::sqrt(best.dist_sq);
                    }
                }
            };
            match sign_changes::<A>(&beta[..=m]) {
                0 => {}
                1 => consider(solve(k, a, b, beta[0], beta[m])),
                _ => {
                    let mut brackets = [(zero, zero, zero, zero); MAX_COEFFS];
                    let n = isolate::<A>(&beta, m, a, b, &mut brackets);
                    for &(a, b, ga, gb) in &brackets[..n] {
                        consider(solve(k, a, b, ga, gb));
                    }
                }
            }
        }
        best
    }
}

/// The number of sign changes in Bernstein coefficients, skipping zeros. It bounds the number
/// of roots in the interval, and has the same parity.
#[inline(always)]
fn sign_changes<A: ScalarAdaptor>(beta: &[A::Float]) -> usize {
    let zero = A::scalar(0.0);
    let mut changes = 0;
    let mut neg = None;
    for &bj in beta {
        if bj != zero {
            let is_neg = bj < zero;
            if neg.is_some_and(|n| n != is_neg) {
                changes += 1;
            }
            neg = Some(is_neg);
        }
    }
    changes
}

/// The root of `coef` in `[a, b]` where it goes from negative to positive, when `ga = g(a) < 0`
/// and `gb = g(b) >= 0`, and the interval has just that one root. Starts where the secant
/// crosses zero, then Newton, bisecting when a step leaves the bracket.
#[inline(always)]
fn newton_in_bracket<A: ScalarAdaptor>(
    coef: &[A::Float],
    a: A::Float,
    b: A::Float,
    ga: A::Float,
    gb: A::Float,
) -> Option<A::Float> {
    let (zero, half) = (A::scalar(0.0), A::scalar(0.5));
    if !(ga < zero && gb >= zero) {
        return None; // Distance has a maximum here, not a minimum.
    }
    let (mut lo, mut hi) = (a, b);
    let mut t = a - ga * (b - a) / (gb - ga);
    let tol = A::sqrt(A::epsilon()) * (b - a);
    for _ in 0..MAX_NEWTON {
        let (g, slope) = polynomial::eval_with_deriv::<A>(coef, t);
        if g == zero {
            break;
        } else if g < zero {
            lo = t;
        } else {
            hi = t;
        }
        let mut next = t - g / slope;
        if !(next >= lo && next <= hi) {
            next = (lo + hi) * half;
        }
        let done = A::abs(next - t) <= tol;
        t = next;
        if done {
            break;
        }
    }
    Some(t)
}

/// For Bernstein coefficients on `[a, b]` with more than one sign change: splits them in half
/// (de Casteljau) until each piece has at most one, and writes the pieces that hold a single
/// root to `brackets`, as `(a, b, g(a), g(b))`. Returns their count. Rare, so kept out of line.
#[cold]
#[inline(never)]
fn isolate<A: ScalarAdaptor>(
    beta: &[A::Float; MAX_COEFFS],
    m: usize,
    a: A::Float,
    b: A::Float,
    brackets: &mut [Bracket<A::Float>; MAX_COEFFS],
) -> usize {
    let (zero, half) = (A::scalar(0.0), A::scalar(0.5));
    let mut n = 0;
    // Pieces still to check: coefficients, parameter range, and split depth.
    let mut stack = vec![(*beta, a, b, 0usize)];
    while let Some((beta, a, b, depth)) = stack.pop() {
        let changes = sign_changes::<A>(&beta[..=m]);
        if changes == 0 {
            continue;
        }
        if changes == 1 || depth == MAX_SPLIT_DEPTH {
            if n < MAX_COEFFS {
                brackets[n] = (a, b, beta[0], beta[m]);
                n += 1;
            }
            continue;
        }
        // De Casteljau at the middle: the left piece takes the first entry of each level, the
        // right piece the last.
        let (mut left, mut right) = ([zero; MAX_COEFFS], [zero; MAX_COEFFS]);
        let mut level = beta;
        for r in 0..=m {
            left[r] = level[0];
            right[m - r] = level[m - r];
            for j in 0..m - r {
                level[j] = (level[j] + level[j + 1]) * half;
            }
        }
        let mid = (a + b) * half;
        stack.push((right, mid, b, depth + 1));
        stack.push((left, a, mid, depth + 1));
    }
    n
}

/// A piece `[a, b]` of an interval holding one root, with the dot polynomial at its ends:
/// `(a, b, g(a), g(b))`.
type Bracket<S> = (S, S, S, S);

/// The frame of one elliptic arc interval: see `Projector::from_elliptic_arc`.
#[derive(Clone, Copy)]
struct EllipseFrame<S> {
    cos: S, // Of the middle angle.
    sin: S,
    tau_max: S,   // tan(width / 4), the range of the local half-angle parameter.
    b_minus_a: S, // Vc·Vc - Uc·Uc.
    w: S,         // Uc·Vc.
}

/// A point on a segment, with its squared distance from the query, its parameter on the
/// segment, and the segment index.
struct Candidate<const DIM: usize, A: Adaptor<DIM>> {
    dist_sq: A::Scalar,
    point: A::Vector,
    param: A::Scalar,
    segment: usize,
}

impl<const DIM: usize, A: Adaptor<DIM>> Candidate<DIM, A> {
    #[inline(always)]
    fn new(q: A::Vector, point: A::Vector, param: A::Scalar, segment: usize) -> Self {
        Self {
            dist_sq: A::vector_length_sq(point - q),
            point,
            param,
            segment,
        }
    }

    /// `self`, or the given point if that is closer to `q`.
    #[inline(always)]
    fn closer(self, q: A::Vector, point: A::Vector, param: A::Scalar, segment: usize) -> Self {
        let other = Self::new(q, point, param, segment);
        if other.dist_sq < self.dist_sq {
            other
        } else {
            self
        }
    }
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
    /// the brute force distance. `on_segment` evaluates the curve at a segment parameter, to
    /// check the parameter and segment index from [`Projector::closest_point_with_param`].
    fn check<const D: usize>(
        proj: &Projector<D, F64Adaptor>,
        f: impl Fn(f64) -> DVec<D> + Copy,
        domain: (f64, f64),
        on_segment: impl Fn(f64, usize) -> DVec<D>,
        queries: impl IntoIterator<Item = DVec<D>>,
    ) {
        for q in queries {
            let p = proj.closest_point(q);
            let (p_param, t, segment) = proj.closest_point_with_param(q);
            assert_eq!(p, p_param, "q = {q:?}: the two functions disagree");
            assert!(
                (0.0..=1.0).contains(&t),
                "q = {q:?}: t = {t} is outside [0, 1]"
            );
            let at_t = on_segment(t, segment);
            assert!(
                (at_t - p).length() <= 1e-9 * (1.0 + p.length()),
                "q = {q:?}, p = {p:?}, but segment {segment} at t = {t} is {at_t:?}"
            );
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
                |t, segment| {
                    assert_eq!(segment, 0);
                    line.point(t).unwrap()
                },
                queries,
            );
        }
        // Zero length.
        let p = DVec([1.0, 2.0]);
        let proj = Projector::from_line(&LineSeg::<2, F64Adaptor>::create(p, p));
        assert_eq!(proj.closest_point(DVec([5.0, -1.0])), p);
        assert_eq!(
            proj.closest_point_with_param(DVec([5.0, -1.0])),
            (p, 0.0, 0)
        );
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
                |t, segment| {
                    assert_eq!(segment, 0);
                    arc.point(t).unwrap()
                },
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
                    |t, segment| {
                        assert_eq!(segment, 0);
                        arc.point(t).unwrap()
                    },
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
                // Segment i is the i-th non-empty knot span.
                let mut spans = spline.knots().to_vec();
                spans.dedup();
                check(
                    &proj,
                    |u| spline.point(u).unwrap(),
                    spline.domain(),
                    |t, segment| {
                        let (u0, u1) = (spans[segment], spans[segment + 1]);
                        spline.point(u0 + t * (u1 - u0)).unwrap()
                    },
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

    /// Sampled Newton finds the same closest points as the root finder, including for elliptic
    /// arcs that sweep almost a full turn, where the half-angle parameter of the whole arc is huge.
    #[test]
    fn t_matches_root_finder() {
        let mut rng = StdRng::seed_from_u64(13);
        let check = |proj: &Projector<3, F64Adaptor>, center: DVec<3>, rng: &mut StdRng| {
            for _ in 0..40 {
                let q = center + rand_vec::<3>(rng, 8.0);
                let fast = (proj.closest_point(q) - q).length();
                let exact = (proj.closest_point_exact(q) - q).length();
                assert!(
                    (fast - exact).abs() <= 1e-12 * (1.0 + q.length()),
                    "q = {q:?}: {fast} instead of {exact}"
                );
            }
        };
        for degree in 1..=5 {
            for _ in 0..20 {
                let n_cps = degree + 1 + rng.random_range(0..6);
                let cps: std::vec::Vec<DVec<3>> =
                    (0..n_cps).map(|_| rand_vec::<3>(&mut rng, 3.0)).collect();
                let spline = Spline::<3, F64Adaptor>::create_clamped(cps, degree).unwrap();
                check(
                    &Projector::from_spline(&spline).unwrap(),
                    DVec([0.0; 3]),
                    &mut rng,
                );
            }
        }
        for (max_ratio, lo, hi) in [
            (4.0, 0.1, TAU - 0.1),
            (100.0, 0.1, TAU - 0.1),
            (4.0, TAU - 3e-3, TAU - 1e-6),
        ] {
            for _ in 0..30 {
                let sweep = rng.random_range(lo..hi);
                let arc = rand_elliptic_arc(&mut rng, max_ratio, sweep);
                check(&Projector::from_elliptic_arc(&arc), arc.center(), &mut rng);
            }
        }
    }

    #[test]
    fn t_f32() {
        let arc = Arc::<3, F32Adaptor>::from_three_points(
            FVec([1.0, 0.0, 0.0]),
            FVec([0.0, 1.0, 0.0]),
            FVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let p = Projector::from_arc(&arc).closest_point(FVec([0.0, 3.0, 2.0]));
        assert!((p - FVec([0.0, 1.0, 0.0])).length() < 1e-6);
        let p = Projector::from_arc(&arc).closest_point(FVec([0.5, -3.0, 0.0]));
        assert!((p - FVec([1.0, 0.0, 0.0])).length() < 1e-6);
    }
}
