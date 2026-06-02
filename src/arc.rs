use crate::{
    Adaptor, ScalarAdaptor, TrigonometryAdaptor,
    error::Error,
    vec::{F32Adaptor, F64Adaptor},
};
use core::f64;
use std::f64::consts::{PI, TAU};

pub type Arc2d = Arc<2, F64Adaptor>;
pub type Arc3d = Arc<3, F64Adaptor>;
pub type Arc2f = Arc<2, F32Adaptor>;
pub type Arc3f = Arc<3, F32Adaptor>;

#[derive(Clone, Debug)]
pub struct Arc<const DIM: usize, A>
where
    A: Adaptor<DIM>,
{
    center: A::Vector,
    start_dir: A::Vector, // Unit vector from center toward start.
    mid_dir: A::Vector,   // Unit vector from center toward midpoint (used for antipodal SLERP).
    end_dir: A::Vector,   // Unit vector from center toward end.
    radius: A::Scalar,
    angle: A::Scalar, // Signed sweep angle from start to end through mid.
}

impl<const DIM: usize, A> Arc<DIM, A>
where
    A: Adaptor<DIM>,
{
    /// Construct an arc from three points: start, a point along the arc, and end.
    pub fn from_three_points(
        start: A::Vector,
        middle: A::Vector,
        end: A::Vector,
    ) -> Result<Self, Error>
    where
        A: TrigonometryAdaptor,
    {
        let (center, radius) =
            circumcircle::<DIM, A>(start, middle, end).ok_or(Error::PointsCollinear)?;
        let start_dir = (start - center) / radius;
        let end_dir = (end - center) / radius;
        let mid_guess = (middle - center) / radius;
        let mid_dir = calc_arc_middle::<DIM, A>(start_dir, mid_guess, end_dir)
            .ok_or(Error::PointsCollinear)?;
        let angle = A::acos(A::clamp(
            A::dot_product(start_dir, mid_dir),
            A::scalar(-1.0),
            A::scalar(1.0),
        )) + A::acos(A::clamp(
            A::dot_product(mid_dir, end_dir),
            A::scalar(-1.0),
            A::scalar(1.0),
        ));
        check_radius_and_angle::<A>(radius, angle)?;
        Ok(Arc {
            center,
            radius,
            start_dir,
            mid_dir,
            end_dir,
            angle,
        })
    }

    /// Construct an arc from a start point, tangent direction at start, and end point.
    pub fn from_start_tangent_end(
        start: A::Vector,
        tangent: A::Vector,
        end: A::Vector,
    ) -> Result<Self, Error>
    where
        A: TrigonometryAdaptor,
    {
        let tangent = A::normalize(tangent);
        let chord = end - start;
        let chord_dir = A::normalize(chord);
        let dot = A::clamp(
            A::dot_product(chord_dir, tangent),
            A::scalar(-1.0),
            A::scalar(1.0),
        );
        let angle = A::acos(dot);
        if angle <= A::epsilon() || angle >= (A::scalar(PI) - A::epsilon()) {
            return Err(Error::PointsCollinear);
        }
        let sin_angle = A::sin(angle);
        let halfchord = A::vector_length(chord) * A::scalar(0.5);
        let radius = halfchord / sin_angle;
        let sweep = A::scalar(2.0) * angle;
        check_radius_and_angle::<A>(radius, sweep)?;
        // Component of tangent perpendicular to chord; |tangent| = 1 so |t_perp| = sin(angle).
        let t_perp_dir = (tangent - chord_dir * dot) / sin_angle;
        // Center lies on the perpendicular bisector of the chord. tan(angle) is positive
        // for minor arcs (angle < π/2) and negative for major arcs (angle > π/2), which
        // automatically places the center on the correct side in both cases.
        let shift = if A::abs(angle - A::scalar(PI * 0.5)) < A::epsilon() {
            A::scalar(0.0)
        } else {
            halfchord / A::tan(angle)
        };
        let center = (start + end) * A::scalar(0.5) - t_perp_dir * shift;
        let start_dir = A::normalize(start - center);
        let end_dir = A::normalize(end - center);
        // Rotate start_dir by half the sweep within the arc's plane. The tangent is the
        // in-plane basis perpendicular to start_dir; project out any start_dir component
        // caused by floating-point error in the user-supplied tangent.
        let tangent_start = A::normalize(tangent - start_dir * A::dot_product(tangent, start_dir));
        let mid_dir = start_dir * A::cos(angle) + tangent_start * sin_angle;
        Ok(Arc {
            center,
            radius,
            start_dir,
            mid_dir,
            end_dir,
            angle: sweep,
        })
    }

    /// Parameter domain: `[0, arc_length]`.
    pub fn domain(&self) -> (A::Scalar, A::Scalar) {
        (A::scalar(0.0), self.length())
    }

    /// Total arc length.
    pub fn length(&self) -> A::Scalar {
        self.radius * self.angle
    }

    pub fn radius(&self) -> A::Scalar {
        self.radius
    }

    pub fn center(&self) -> A::Vector {
        self.center
    }

    /// Sweep angle (signed).
    pub fn angle(&self) -> A::Scalar {
        self.angle
    }

    pub fn start(&self) -> A::Vector {
        self.center + self.start_dir * self.radius
    }

    pub fn end(&self) -> A::Vector {
        self.center + self.end_dir * self.radius
    }

    /// Evaluate a point on the arc at arc-length parameter `t` in `[0, length]`.
    pub fn point(&self, t: A::Scalar) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        let len = self.length();
        if t < A::scalar(0.0) || t > len {
            return None;
        }
        let coeff = slerp::<A>(self.angle, t / len);
        Some(
            self.center
                + self.start_dir * coeff[0] * self.radius
                + self.mid_dir * coeff[1] * self.radius
                + self.end_dir * coeff[2] * self.radius,
        )
    }

    /// Evaluate the unit tangent at arc-length parameter `t` in `[0, length]`.
    pub fn tangent(&self, t: A::Scalar) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        let len = self.length();
        if t < A::scalar(0.0) || t > len || self.radius < A::epsilon() {
            return None;
        }
        let coeff = slerp_deriv::<A>(self.angle, t / len);
        // Arc-length parameterization: scale by ds/dt = 1/length to get unit tangent.
        Some(
            (self.start_dir * coeff[0] + self.mid_dir * coeff[1] + self.end_dir * coeff[2])
                / self.angle,
        )
    }

    pub fn curvature(&self, t: A::Scalar) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        match self.point(t) {
            Some(p) if self.radius > A::epsilon() => {
                Some((self.center - p) / (self.radius * self.radius))
            }
            _ => None,
        }
    }

    pub fn point_with_derivs(&self, t: A::Scalar, results: &mut [A::Vector]) -> Result<(), Error>
    where
        A: TrigonometryAdaptor,
    {
        if results.is_empty() {
            return Ok(());
        }
        let len = self.length();
        if t < A::scalar(0.0) || t > len {
            return Err(Error::InvalidParameter);
        }
        let frac = t / len;
        if results.len() == 1 {
            let coeff = slerp::<A>(self.angle, frac);
            results[0] = self.center
                + self.start_dir * coeff[0] * self.radius
                + self.mid_dir * coeff[1] * self.radius
                + self.end_dir * coeff[2] * self.radius;
            Ok(())
        } else {
            let [pos_coeff, deriv_coeff] = slerp_with_deriv::<A>(self.angle, frac);
            let pos = self.center
                + self.start_dir * pos_coeff[0] * self.radius
                + self.mid_dir * pos_coeff[1] * self.radius
                + self.end_dir * pos_coeff[2] * self.radius;
            results[0] = pos;
            results[1] = (self.start_dir * deriv_coeff[0]
                + self.mid_dir * deriv_coeff[1]
                + self.end_dir * deriv_coeff[2])
                / self.angle;
            if results.len() > 2 {
                results[2] = (self.center - pos) / (self.radius * self.radius);
                results[3..].fill(A::vector([A::scalar(0.0); DIM]));
            }
            Ok(())
        }
    }

    /// Uniformly sample the arc with the given tolerance (max chord deviation).
    /// For a circle, the chord error at step angle α is `r(1 - cos(α/2))`.
    /// Solving for α: `α = 2 * acos(1 - tolerance/r)`.
    pub fn adaptive_samples(&self, tolerance: A::Scalar) -> impl Iterator<Item = A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        let ang_step = if self.radius > A::epsilon() && tolerance > A::scalar(0.0) {
            A::scalar(2.0)
                * A::acos(A::scalar(1.0) - A::min(tolerance / self.radius, A::scalar(1.0)))
        } else {
            A::scalar(0.1) // fallback
        };
        let half_angle = A::scalar(0.5) * self.angle;
        let n = if self.angle < A::epsilon() {
            1
        } else {
            (A::to_usize(A::ceil(half_angle / ang_step))).max(1)
        };
        let nf = A::scalar(n as f64);
        let inv_sin = A::scalar(1.0) / A::sin(half_angle);
        let zero = A::scalar(0.0);
        let one = A::scalar(1.0);
        std::iter::once([A::scalar(1.0), A::scalar(0.0), A::scalar(0.0)])
            .chain((1..=n).map(move |i| {
                let coeff = slerp_raw_no_adjust::<A>(half_angle, A::scalar(i as f64) / nf);
                [inv_sin * coeff[0], inv_sin * coeff[1], zero]
            }))
            .chain((1..n).map(move |i| {
                let coeff = slerp_raw_no_adjust::<A>(half_angle, A::scalar(i as f64) / nf);
                [zero, inv_sin * coeff[0], inv_sin * coeff[1]]
            }))
            .chain(std::iter::once([zero, zero, one]))
            .map(|coeff| {
                self.center
                    + self.start_dir * coeff[0] * self.radius
                    + self.mid_dir * coeff[1] * self.radius
                    + self.end_dir * coeff[2] * self.radius
            })
    }

    pub fn uniform_samples(
        &self,
        start: A::Scalar,
        step: A::Scalar,
        _tolerance: A::Scalar,
    ) -> impl Iterator<Item = A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        let zero = A::scalar(0.0);
        let one = A::scalar(1.0);
        let half_angle = A::scalar(0.5) * self.angle;
        let inv_sin = A::scalar(1.0) / A::sin(half_angle);
        let two = A::scalar(2.0);
        let ang_step = step / self.radius;
        let mut t = A::max((start / self.radius) / half_angle, zero);
        let step = ang_step / half_angle;
        std::iter::from_fn(move || {
            if t <= one {
                let coeff = slerp_raw_no_adjust::<A>(half_angle, t);
                t += step;
                Some([inv_sin * coeff[0], inv_sin * coeff[1], zero])
            } else if t > one && t <= two {
                let coeff = slerp_raw_no_adjust::<A>(half_angle, t - one);
                t += step;
                Some([zero, inv_sin * coeff[0], inv_sin * coeff[1]])
            } else {
                None
            }
        })
        .map(|coeff| {
            self.center
                + self.start_dir * coeff[0] * self.radius
                + self.mid_dir * coeff[1] * self.radius
                + self.end_dir * coeff[2] * self.radius
        })
    }

    /// Compute the axis-aligned bounding box of the arc.
    ///
    /// Returns `(min_corner, max_corner)`.
    pub fn bounds(&self) -> (A::Vector, A::Vector)
    where
        A: TrigonometryAdaptor,
    {
        arc_bounds::<DIM, A>(
            self.center,
            self.radius,
            self.start_dir,
            self.mid_dir,
            self.end_dir,
            self.angle,
        )
    }

    /// Reverse the arc direction.
    pub fn reversed(self) -> Self {
        Arc {
            center: self.center,
            radius: self.radius,
            start_dir: self.end_dir,
            end_dir: self.start_dir,
            mid_dir: self.mid_dir,
            angle: self.angle,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.start() == self.end()
    }
}

impl<A> Arc<3, A>
where
    A: Adaptor<3> + TrigonometryAdaptor,
{
    /// Construct an arc from center, normal, start point on the circle, and sweep angle.
    /// Angle is signed: positive = CCW around normal.
    pub fn from_axis_start_angle(
        axis: (A::Vector, A::Vector),
        start: A::Vector,
        angle: A::Scalar,
    ) -> Result<Self, Error> {
        let (mut ax0, mut ax1) = axis;
        let mut normal = ax1 - ax0;
        let mut angle = angle;
        if angle < A::scalar(0.0) {
            normal = -normal;
            angle = -angle;
            std::mem::swap(&mut ax0, &mut ax1);
        }
        let norm_len = A::vector_length(normal);
        if norm_len < A::epsilon() {
            return Err(Error::DegenerateValue);
        }
        normal /= norm_len;
        let vdiff = start - ax0;
        let plane_dist = A::dot_product(vdiff, normal);
        let vproject = plane_dist * normal;
        let vrad = vdiff - vproject;
        let center = start - vrad;
        let radius = A::vector_length(vrad);
        check_radius_and_angle::<A>(radius, angle)?;
        let start_dir = A::normalize(start - center);
        let end_dir = rotate_vec::<A>(start_dir, normal, angle);
        let mid_dir = rotate_vec::<A>(start_dir, normal, angle * A::scalar(0.5));
        Ok(Arc {
            center,
            radius,
            start_dir,
            mid_dir,
            end_dir,
            angle,
        })
    }

    pub fn intrinsic_normal(&self) -> A::Vector {
        let [ax, ay, az] = A::coord_arr(self.start_dir);
        let [bx, by, bz] = A::coord_arr(self.mid_dir);
        A::normalize(A::vector([
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        ]))
    }
}

/// Rodrigues' rotation formula for 3D: rotate `v` around unit `axis` by `angle`.
#[inline(always)]
fn rotate_vec<A: Adaptor<3> + TrigonometryAdaptor>(
    v: A::Vector,
    axis: A::Vector,
    angle: A::Scalar,
) -> A::Vector {
    let (s, c) = A::sin_cos(angle);
    let [ax, ay, az] = A::coord_arr(axis);
    let [vx, vy, vz] = A::coord_arr(v);
    let cross = A::vector([ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx]);
    v * c + cross * s + axis * A::dot_product(axis, v) * (A::scalar(1.0) - c)
}

fn arc_bounds<const DIM: usize, A: Adaptor<DIM> + TrigonometryAdaptor>(
    center: A::Vector,
    radius: A::Scalar,
    start_dir: A::Vector,
    mid_dir: A::Vector,
    end_dir: A::Vector,
    angle: A::Scalar,
) -> (A::Vector, A::Vector) {
    let (min, max) = {
        let start = center + start_dir * radius;
        let end = center + end_dir * radius;
        (
            A::vector(std::array::from_fn(|i| {
                A::min(A::vector_coord(start, i), A::vector_coord(end, i))
            })),
            A::vector(std::array::from_fn(|i| {
                A::max(A::vector_coord(start, i), A::vector_coord(end, i))
            })),
        )
    };
    if angle < A::epsilon() {
        // Near degenerate arc.
        return (min, max);
    }
    if A::abs(angle - A::scalar(PI)) < A::epsilon() {
        // Near semicircle.
        let half = angle * A::scalar(0.5);
        let new_mid = A::normalize(start_dir + mid_dir);
        let (lmin, lmax): (A::Vector, A::Vector) =
            arc_bounds::<DIM, A>(center, radius, start_dir, new_mid, mid_dir, half);
        let new_mid = A::normalize(mid_dir + end_dir);
        let (rmin, rmax) = arc_bounds::<DIM, A>(center, radius, mid_dir, new_mid, end_dir, half);
        return (
            A::vector(std::array::from_fn(|i| {
                A::min(A::vector_coord(lmin, i), A::vector_coord(rmin, i))
            })),
            A::vector(std::array::from_fn(|i| {
                A::max(A::vector_coord(lmax, i), A::vector_coord(rmax, i))
            })),
        );
    }
    // Find the roots (zeros of the derivative) and check if they lie within
    // the arc. At the extremum angle u for component i, the direction vector
    // satisfies (cos u, sin u) ∝ (denom_i, numer_i), and the extremal slerp
    // value is ±h_i / sin(θ) where h_i = sqrt(numer_i² + denom_i²).
    // Instead of computing u = atan2(n, d) and checking u ∈ [0, θ), we test
    // sector membership with cross products: sin(u) > 0 ↔ numer_i > 0, and
    // sin(u−θ) < 0 ↔ numer_i·cosθ − denom_i·sinθ < 0.
    let (sin, cos) = A::sin_cos(angle);
    let numer = end_dir - start_dir * cos;
    let denom = start_dir * sin;
    let inv_sin = A::scalar(1.0) / sin;
    let numer = A::coord_arr(numer);
    let denom = A::coord_arr(denom);
    let center = A::coord_arr(center);
    let mut min = A::coord_arr(min);
    let mut max = A::coord_arr(max);
    let short = angle < A::scalar(PI);
    for ci in 0..DIM {
        let n = numer[ci];
        let d = denom[ci];
        let cross_end = n * cos - d * sin; // ∝ sin(u - θ)
        let ext = radius * A::sqrt(n * n + d * d) * inv_sin;
        // u root: (cos u, sin u) ∝ (d, n).
        let u_in = if short {
            n > A::scalar(0.0) && cross_end < A::scalar(0.0)
        } else {
            n > A::scalar(0.0) || cross_end < A::scalar(0.0)
        };
        // u + π root: (cos u, sin u) ∝ (-d, -n). Flips both signs.
        let u_pi_in = if short {
            n < A::scalar(0.0) && cross_end > A::scalar(0.0)
        } else {
            n < A::scalar(0.0) || cross_end > A::scalar(0.0)
        };
        if u_in {
            let val = center[ci] + ext;
            min[ci] = A::min(val, min[ci]);
            max[ci] = A::max(val, max[ci]);
        }
        if u_pi_in {
            let val = center[ci] - ext;
            min[ci] = A::min(val, min[ci]);
            max[ci] = A::max(val, max[ci]);
        }
    }
    (A::vector(min), A::vector(max))
}

/// Check radius and angle invariants. Call early, as soon as both are known.
fn check_radius_and_angle<A: ScalarAdaptor>(
    radius: A::Float,
    angle: A::Float,
) -> Result<(), Error> {
    assert!(
        angle >= A::scalar(0.) && angle <= A::scalar(TAU),
        "INTERNAL ERROR: angle is outside bounds. Should never happen.",
    );
    if radius < A::epsilon() {
        Err(Error::RadiusTooSmall)
    } else if angle >= (A::scalar(TAU) - A::epsilon()) {
        Err(Error::ArcCannotBeCircle)
    } else {
        Ok(())
    }
}

/// Circumcircle of three points in any dimension. Returns (center, radius).
///
/// Solves the 2×2 system: writing P = a + s·u + t·v (u = b-a, v = c-a),
/// equidistance from a, b, c gives:
///   |u|²·s + (u·v)·t = |u|²/2
///   (u·v)·s + |v|²·t = |v|²/2
/// The determinant |u|²|v|² − (u·v)² equals |u×v|² (Lagrange identity), so it
/// is zero iff the three points are collinear.
fn circumcircle<const DIM: usize, A>(
    a: A::Vector,
    b: A::Vector,
    c: A::Vector,
) -> Option<(A::Vector, A::Scalar)>
where
    A: Adaptor<DIM>,
{
    let u = b - a;
    let v = c - a;
    let uu = A::dot_product(u, u);
    let uv = A::dot_product(u, v);
    let vv = A::dot_product(v, v);
    let det = uu * vv - uv * uv;
    if det < A::scalar(f64::EPSILON) {
        return None;
    }
    let s = vv * (uu - uv) / (A::scalar(2.0) * det);
    let t = uu * (vv - uv) / (A::scalar(2.0) * det);
    let rvec = u * s + v * t;
    let center = a + rvec;
    let radius = A::vector_length(rvec);
    Some((center, radius))
}

fn calc_arc_middle<const DIM: usize, A>(
    start: A::Vector,
    mid_guess: A::Vector,
    end: A::Vector,
) -> Option<A::Vector>
where
    A: Adaptor<DIM>,
{
    let mut mid = start + end;
    let len_mid = A::vector_length(mid);
    if len_mid < A::epsilon() {
        // This is a semi circle. We need to figure out the perpendicular direcion.
        let vperp = mid_guess - start * A::dot_product(mid_guess, start);
        let len = A::vector_length(vperp);
        if len < A::epsilon() {
            // mid_bias is collinear.
            None
        } else {
            Some(vperp / len)
        }
    } else {
        mid /= len_mid;
        let dir = A::normalize(end - start);
        let mid_perp = {
            let diff = mid - start;
            diff - dir * A::dot_product(diff, dir)
        };
        let bias_perp = {
            let diff = mid_guess - start;
            diff - dir * A::dot_product(diff, dir)
        };
        if A::vector_length(bias_perp) < A::epsilon() {
            // Collinear.
            return None;
        }
        if A::dot_product(mid_perp, bias_perp) < A::scalar(0.0) {
            mid = -mid;
        }
        Some(mid)
    }
}

#[inline(always)]
fn slerp<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [A::Float; 3] {
    if angle < A::epsilon() {
        return [A::scalar(1.0) - t, A::scalar(0.0), t];
    }
    let half = angle * A::scalar(0.5);
    if t <= A::scalar(0.5) {
        let coeff = slerp_raw::<A>(half, t * A::scalar(2.0));
        [coeff[0], coeff[1], A::scalar(0.0)]
    } else {
        let coeff = slerp_raw::<A>(half, (t - A::scalar(0.5)) * A::scalar(2.0));
        [A::scalar(0.0), coeff[0], coeff[1]]
    }
}

#[inline(always)]
fn slerp_raw<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [A::Float; 2] {
    let inv_sin = A::scalar(1.0) / A::sin(angle);
    slerp_raw_no_adjust::<A>(angle, t).map(|c| c * inv_sin)
}

#[inline(always)]
fn slerp_raw_no_adjust<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [A::Float; 2] {
    [A::sin((A::scalar(1.0) - t) * angle), A::sin(t * angle)]
}

/// Derivative of SLERP with respect to the fractional parameter `t` in [0, 1].
/// d/dt [sin((1-t)θ)/sin(θ) · a + sin(tθ)/sin(θ) · b]
///    = θ/sin(θ) · [-cos((1-t)θ) · a + cos(tθ) · b]
#[inline(always)]
fn slerp_deriv<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [A::Float; 3] {
    if angle < A::epsilon() {
        return [A::scalar(-1.0), A::scalar(0.0), A::scalar(1.0)];
    }
    let half = angle / A::scalar(2.0);
    // Chain rule: d/dt f(2t) = 2 * f'(2t).
    if t <= A::scalar(0.5) {
        let coeff =
            slerp_deriv_unchecked::<A>(half, t * A::scalar(2.0)).map(|c| c * A::scalar(2.0));
        [coeff[0], coeff[1], A::scalar(0.0)]
    } else {
        let coeff = slerp_deriv_unchecked::<A>(half, (t - A::scalar(0.5)) * A::scalar(2.0))
            .map(|c| c * A::scalar(2.0));
        [A::scalar(0.0), coeff[0], coeff[1]]
    }
}

#[inline(always)]
fn slerp_deriv_unchecked<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [A::Float; 2] {
    let theta_over_sin = angle / A::sin(angle);
    let w0 = -A::cos((A::scalar(1.0) - t) * angle) * theta_over_sin;
    let w1 = A::cos(t * angle) * theta_over_sin;
    [w0, w1]
}

/// Combined SLERP value and derivative, sharing branching and trig.
#[inline(always)]
fn slerp_with_deriv<A: TrigonometryAdaptor>(angle: A::Float, t: A::Float) -> [[A::Float; 3]; 2] {
    if angle < A::epsilon() {
        return [
            [A::scalar(1.0) - t, A::scalar(0.0), t],
            [A::scalar(-1.0), A::scalar(0.0), A::scalar(1.0)],
        ];
    }
    let half = angle / A::scalar(2.0);
    if t <= A::scalar(0.5) {
        let [val_coeff, deriv_coeff] = slerp_with_deriv_unchecked::<A>(half, t * A::scalar(2.0));
        [
            [val_coeff[0], val_coeff[1], A::scalar(0.0)],
            [
                deriv_coeff[0] * A::scalar(2.0),
                deriv_coeff[1] * A::scalar(2.0),
                A::scalar(0.0),
            ], // Chain rule.
        ]
    } else {
        let [val_coeff, deriv_coeff] =
            slerp_with_deriv_unchecked::<A>(half, (t - A::scalar(0.5)) * A::scalar(2.0));
        [
            [A::scalar(0.0), val_coeff[0], val_coeff[1]],
            [
                A::scalar(0.0),
                deriv_coeff[0] * A::scalar(2.0),
                deriv_coeff[1] * A::scalar(2.0),
            ], // Chain rule.
        ]
    }
}

#[inline(always)]
fn slerp_with_deriv_unchecked<A: TrigonometryAdaptor>(
    angle: A::Float,
    t: A::Float,
) -> [[A::Float; 2]; 2] {
    let inv_sin = A::scalar(1.0) / A::sin(angle);
    let arg0 = (A::scalar(1.0) - t) * angle;
    let arg1 = t * angle;
    let (sin0, cos0) = A::sin_cos(arg0);
    let (sin1, cos1) = A::sin_cos(arg1);
    [
        [(sin0 * inv_sin), (sin1 * inv_sin)],
        [(-cos0 * angle * inv_sin), (cos1 * angle * inv_sin)],
    ]
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{DVec, DVec3};
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn t_semicircle() {
        // Semicircle from (-1,0,0) through (0,1,0) to (1,0,0).
        let arc = Arc3d::from_three_points(
            DVec([-1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let expected_len = PI; // radius=1, angle=π
        assert!((arc.length() - expected_len).abs() < 1e-10);
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        // Midpoint should be at (0, 1, 0).
        let mid = arc.point(expected_len / 2.0).unwrap();
        assert!((mid[0]).abs() < 1e-6);
        assert!((mid[1] - 1.0).abs() < 1e-6);
        // Endpoints exact.
        assert!((arc.start()[0] + 1.0).abs() < 1e-10);
        assert!((arc.end()[0] - 1.0).abs() < 1e-10);
        // CW in XY plane (angles π → π/2 → 0) → normal = -Z.
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, -1.0])).length() < 1e-10);
    }

    #[test]
    fn t_quarter_circle() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let expected_len = PI / 2.0;
        assert!(
            (arc.length() - expected_len).abs() < 1e-6,
            "length {} != {}",
            arc.length(),
            expected_len
        );
        // CCW in XY plane → normal = +Z.
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, 1.0])).length() < 1e-10);
    }

    #[test]
    fn t_major_arc() {
        // Arc3d going the long way (> π) from (1,0,0) through (0,-1,0) to (-1,0,0).
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let expected_len = PI; // Still π, but going the other way.
        assert!(
            (arc.length() - expected_len).abs() < 1e-6,
            "length {} != {}",
            arc.length(),
            expected_len
        );
        // Midpoint should be at (0, -1, 0).
        let mid = arc.point(expected_len / 2.0).unwrap();
        assert!((mid[0]).abs() < 1e-6);
        assert!((mid[1] + 1.0).abs() < 1e-6, "mid.y = {}", mid[1]);
        // CW in XY plane → normal = -Z.
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, -1.0])).length() < 1e-10);
    }

    #[test]
    fn t_major_arc_obtuse_triangle() {
        // Unit circle. start=0°, middle=270°, end=60°.
        // Going CW from 0° through 270° to 60° is the major arc (sweep = 300° = 5π/3).
        // The three points form an obtuse triangle (angle at start ≈ 105°).
        let start = DVec([1.0, 0.0, 0.0]);
        let middle = DVec([0.0, -1.0, 0.0]); // 270° — on the CW/major arc
        let end = DVec([0.5, 3f64.sqrt() / 2.0, 0.0]); // 60°
        let arc = Arc3d::from_three_points(start, middle, end).unwrap();
        let expected_sweep = 5.0 * PI / 3.0; // 300°
        assert!(
            arc.angle() > PI,
            "expected major arc (sweep > π), got angle = {}",
            arc.angle()
        );
        assert!(
            (arc.angle() - expected_sweep).abs() < 1e-10,
            "expected sweep = 5π/3, got {}",
            arc.angle()
        );
        assert!(
            (arc.radius() - 1.0).abs() < 1e-10,
            "expected radius = 1, got {}",
            arc.radius()
        );
        // True midpoint of a 300° arc starting at 0° going CW is at 0° − 150° = 210°.
        let arc_mid = arc.point(arc.length() / 2.0).unwrap();
        let expected_mid = DVec([-(3f64.sqrt() / 2.0), -0.5, 0.0]); // cos210°, sin210°
        assert!(
            (arc_mid[0] - expected_mid[0]).abs() < 1e-6,
            "mid.x: {} != {}",
            arc_mid[0],
            expected_mid[0]
        );
        assert!(
            (arc_mid[1] - expected_mid[1]).abs() < 1e-6,
            "mid.y: {} != {}",
            arc_mid[1],
            expected_mid[1]
        );
    }

    #[test]
    fn t_reversed() {
        let arc = Arc3d::from_three_points(
            DVec([-1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let rev = arc.clone().reversed();
        let len = arc.length();
        assert!((len - rev.length()).abs() < 1e-12);
        assert!((arc.start()[0] - rev.end()[0]).abs() < 1e-10);
        assert!((arc.end()[0] - rev.start()[0]).abs() < 1e-10);
        // Points along the forward and reversed arcs should be identical
        // when sampled at mirrored parameters.
        let n = 20;
        for i in 0..=n {
            let t = len * i as f64 / n as f64;
            let fwd = arc.point(t).unwrap();
            let bwd = rev.point(len - t).unwrap();
            let err = (fwd - bwd).length();
            assert!(err < 1e-10, "mismatch at t={t}: err={err}");
        }
    }

    #[test]
    fn t_tangent_perpendicular_to_radius() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let len = arc.length();
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            let p = arc.point(t).unwrap();
            let tan = arc.tangent(t).unwrap();
            let radial = (p - arc.center()).normalize();
            let dot = radial.dot(tan).abs();
            assert!(dot < 1e-6, "tangent not perpendicular at t={t}: dot={dot}");
            // Arc3d-length parameterization ⇒ unit tangent.
            let mag = tan.length();
            assert!(
                (mag - 1.0).abs() < 1e-6,
                "tangent not unit length at t={t}: |tan|={mag}"
            );
        }
    }

    #[test]
    fn t_unit_speed_parameterization() {
        // Arc3d-length parameterization means |dp/dt| = 1 everywhere.
        let arc = Arc3d::from_three_points(
            DVec([2.0, 0.0, 0.0]),
            DVec([0.0, 2.0, 0.0]),
            DVec([-2.0, 0.0, 0.0]),
        )
        .unwrap();
        let len = arc.length();
        let dt = 1e-7;
        for i in 0..10 {
            let t = len * i as f64 / 10.0;
            // Numerical: finite difference speed should be 1.
            let p0 = arc.point(t).unwrap();
            let p1 = arc.point(t + dt).unwrap();
            let speed = (p1 - p0).length() / dt;
            assert!(
                (speed - 1.0).abs() < 1e-4,
                "numerical speed at t={t}: {speed} != 1.0"
            );
            // Analytical: tangent magnitude should be 1.
            let tan = arc.tangent(t).unwrap();
            let mag = tan.length();
            assert!(
                (mag - 1.0).abs() < 1e-6,
                "tangent magnitude at t={t}: {mag} != 1.0"
            );
        }
    }

    #[test]
    fn t_collinear_returns_none() {
        assert!(
            Arc3d::from_three_points(
                DVec([0.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([2.0, 0.0, 0.0]),
            )
            .is_err()
        );
    }

    #[test]
    fn t_3d_arc() {
        // Arc3d not in a coordinate plane.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 1.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        // All points should be equidistant from center.
        let len = arc.length();
        let r = arc.radius();
        for i in 0..=20 {
            let t = len * i as f64 / 20.0;
            let p = arc.point(t).unwrap();
            let dist = (p - arc.center()).length();
            assert!(
                (dist - r).abs() < 1e-6,
                "point at t={t} dist={dist} != radius={r}"
            );
        }
    }

    #[test]
    fn t_out_of_domain_returns_none() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!(arc.point(-1.0).is_none());
        assert!(arc.point(arc.length() + 1.0).is_none());
    }

    #[test]
    fn t_adaptive_samples_deviation() {
        // For a circular arc, the midpoint of each chord deviates from the circle
        // by the sagitta: h = r(1 - cos(α/2)), where α is the step angle.
        // With tolerance t, max_angle = 2*acos(1 - t/r), so h ≤ t.
        const CASES: &[(DVec3, DVec3, DVec3, f64)] = &[
            // Semicircle, r=3, XY plane
            (
                DVec([3.0, 0.0, 0.0]),
                DVec([0.0, 3.0, 0.0]),
                DVec([-3.0, 0.0, 0.0]),
                0.01,
            ),
            // Small quarter-circle, r=1, XY plane
            (
                DVec([1.0, 0.0, 0.0]),
                DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
                DVec([0.0, 1.0, 0.0]),
                0.001,
            ),
            // Large radius, r=100, tight tolerance
            (
                DVec([100.0, 0.0, 0.0]),
                DVec([0.0, 100.0, 0.0]),
                DVec([-100.0, 0.0, 0.0]),
                0.1,
            ),
            // Tiny radius, r≈0.1
            (
                DVec([0.1, 0.0, 0.0]),
                DVec([0.0, 0.1, 0.0]),
                DVec([-0.1, 0.0, 0.0]),
                0.001,
            ),
            // 3D arc, not in a coordinate plane
            (
                DVec([1.0, 0.0, 0.0]),
                DVec([0.0, 1.0, 1.0]),
                DVec([-1.0, 0.0, 0.0]),
                0.01,
            ),
            // Major arc (going the long way)
            (
                DVec([1.0, 0.0, 0.0]),
                DVec([0.0, -1.0, 0.0]),
                DVec([-1.0, 0.0, 0.0]),
                0.01,
            ),
            // Offset center
            (
                DVec([10.0, 5.0, 0.0]),
                DVec([7.0, 8.0, 0.0]),
                DVec([4.0, 5.0, 0.0]),
                0.005,
            ),
        ];
        for (i, &(p0, p1, p2, tol)) in CASES.iter().enumerate() {
            let arc = Arc3d::from_three_points(p0, p1, p2).unwrap();
            let r = arc.radius();
            let center = arc.center();
            let samples: Vec<DVec3> = arc.adaptive_samples(tol).collect();
            assert!(
                samples.len() > 2,
                "case {i}: too few samples: {}",
                samples.len()
            );
            for pair in samples.windows(2) {
                let mid = (pair[0] + pair[1]) * 0.5;
                let dist = (mid - center).length();
                let deviation = (dist - r).abs();
                assert!(
                    deviation <= tol + 1e-10,
                    "case {i}: deviation {deviation} > tolerance {tol} (r={r}, dist={dist})"
                );
            }
            // Analytical check: step sagitta ≤ tolerance.
            let n = (samples.len() - 1) as f64;
            let step_angle = arc.angle().abs() / n;
            let expected_sagitta = r * (1.0 - (step_angle / 2.0).cos());
            assert!(
                expected_sagitta <= tol + 1e-10,
                "case {i}: analytical sagitta {expected_sagitta} > tolerance {tol}"
            );
        }
    }

    #[test]
    fn t_point_with_derivs_matches_point_and_tangent() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let len = arc.length();
        let mut results = [DVec([0.0, 0.0, 0.0]); 3];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            arc.point_with_derivs(t, &mut results).unwrap();
            let p = arc.point(t).unwrap();
            let tan = arc.tangent(t).unwrap();
            let err_p = (results[0] - p).length();
            let err_t = (results[1] - tan).length();
            assert!(err_p < 1e-12, "point mismatch at t={t}: {err_p}");
            assert!(err_t < 1e-12, "tangent mismatch at t={t}: {err_t}");
        }
    }

    #[test]
    fn t_point_with_derivs_curvature() {
        // d²p/ds² = (1/r) · inward_normal. Magnitude = 1/r, points toward center.
        let arc = Arc3d::from_three_points(
            DVec([3.0, 0.0, 0.0]),
            DVec([0.0, 3.0, 0.0]),
            DVec([-3.0, 0.0, 0.0]),
        )
        .unwrap();
        let r = arc.radius();
        let len = arc.length();
        let mut results = [DVec([0.0, 0.0, 0.0]); 3];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            arc.point_with_derivs(t, &mut results).unwrap();
            let pos = results[0];
            let curv = results[2];
            let mag = curv.length();
            assert!(
                (mag - 1.0 / r).abs() < 1e-6,
                "curvature magnitude at t={t}: {mag} != {}",
                1.0 / r
            );
            let inward = (arc.center() - pos).normalize();
            let dot = inward.dot(curv.normalize());
            assert!(
                (dot - 1.0).abs() < 1e-6,
                "curvature not pointing inward at t={t}: dot={dot}"
            );
        }
    }

    #[test]
    fn t_point_with_derivs_out_of_domain() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut results = [DVec([0.0, 0.0, 0.0]); 2];
        assert!(arc.point_with_derivs(-1.0, &mut results).is_err());
        assert!(
            arc.point_with_derivs(arc.length() + 1.0, &mut results)
                .is_err()
        );
        // Empty slice is a no-op, even out of domain.
        assert!(arc.point_with_derivs(-1.0, &mut []).is_ok());
    }

    #[test]
    fn t_point_with_derivs_point_only() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut results = [DVec([0.0, 0.0, 0.0]); 1];
        arc.point_with_derivs(0.0, &mut results).unwrap();
        let p = arc.point(0.0).unwrap();
        assert!((results[0] - p).length() < 1e-12);
    }

    #[test]
    fn t_point_with_derivs_higher_derivs_zero() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut results = [DVec([1.0, 1.0, 1.0]); 5]; // pre-fill with non-zero
        arc.point_with_derivs(arc.length() / 2.0, &mut results)
            .unwrap();
        for (i, r) in results[3..].iter().enumerate() {
            assert_eq!(*r, DVec([0.0, 0.0, 0.0]), "results[{}] not zero", i + 3);
        }
    }

    #[test]
    fn t_point_with_derivs_3d_arc() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 1.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let r = arc.radius();
        let len = arc.length();
        let mut results = [DVec([0.0, 0.0, 0.0]); 3];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            arc.point_with_derivs(t, &mut results).unwrap();
            let mag = results[2].length();
            assert!(
                (mag - 1.0 / r).abs() < 1e-6,
                "3D curvature magnitude at t={t}: {mag} != {}",
                1.0 / r
            );
        }
    }

    // --- Helper: check arc geometric invariants ---
    fn assert_arc_valid(arc: &Arc3d, expected_start: DVec3, expected_end: DVec3, tol: f64) {
        let r = arc.radius();
        let len = arc.length();
        // Start and end match.
        assert!(
            (arc.start() - expected_start).length() < tol,
            "start mismatch: {:?} vs {:?}",
            arc.start(),
            expected_start
        );
        assert!(
            (arc.end() - expected_end).length() < tol,
            "end mismatch: {:?} vs {:?}",
            arc.end(),
            expected_end
        );
        // All sampled points equidistant from center.
        for i in 0..=20 {
            let t = len * i as f64 / 20.0;
            let p = arc.point(t).unwrap();
            let dist = (p - arc.center()).length();
            assert!(
                (dist - r).abs() < tol,
                "point at t={t} dist={dist} != radius={r}"
            );
        }
        // Unit speed.
        let dt = 1e-7;
        for i in 0..10 {
            let t = len * i as f64 / 10.0;
            let p0 = arc.point(t).unwrap();
            let p1 = arc.point(t + dt).unwrap();
            let speed = (p1 - p0).length() / dt;
            assert!((speed - 1.0).abs() < 1e-4, "speed at t={t}: {speed} != 1.0");
        }
    }

    // ===================== from_start_tangent_end =====================

    #[test]
    fn t_start_tangent_end_semicircle() {
        // Tangent at (1,0,0) pointing up = (0,1,0), end at (-1,0,0).
        // Should produce a semicircle of radius 1.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6);
        assert!((arc.length() - PI).abs() < 1e-6);
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([-1.0, 0.0, 0.0]), 1e-6);
    }

    #[test]
    fn t_start_tangent_end_quarter_circle() {
        // Tangent at (1,0,0) pointing up, end at (0,1,0) → quarter circle.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6, "r={}", arc.radius());
        assert!(
            (arc.length() - PI / 2.0).abs() < 1e-6,
            "len={}",
            arc.length()
        );
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([0.0, 1.0, 0.0]), 1e-6);
    }

    #[test]
    fn t_start_tangent_end_matches_from_three_points() {
        // Build an arc from three points, get its start tangent, then reconstruct.
        let arc1 = Arc3d::from_three_points(
            DVec([2.0, 0.0, 0.0]),
            DVec([0.0, 2.0, 0.0]),
            DVec([-2.0, 0.0, 0.0]),
        )
        .unwrap();
        let tan = arc1.tangent(0.0).unwrap();
        let arc2 =
            Arc3d::from_start_tangent_end(DVec([2.0, 0.0, 0.0]), tan, DVec([-2.0, 0.0, 0.0]))
                .unwrap();
        assert!((arc1.radius() - arc2.radius()).abs() < 1e-6);
        assert!((arc1.length() - arc2.length()).abs() < 1e-6);
        assert!((arc1.center() - arc2.center()).length() < 1e-6);
    }

    #[test]
    fn t_start_tangent_end_collinear_fails() {
        // Tangent parallel to chord.
        assert!(
            Arc3d::from_start_tangent_end(
                DVec([0.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([2.0, 0.0, 0.0]),
            )
            .is_err()
        );
    }

    #[test]
    fn t_start_tangent_end_coincident_fails() {
        // Start == end.
        assert!(
            Arc3d::from_start_tangent_end(
                DVec([1.0, 0.0, 0.0]),
                DVec([0.0, 1.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
            )
            .is_err()
        );
    }

    #[test]
    fn t_start_tangent_end_major_arc() {
        // Unit circle. start=(1,0,0), CW tangent=(0,-1,0), end=(0,1,0).
        // CW from 0° through 270° to 90° → sweep = 3π/2, midpoint at 225°.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        assert!(
            (arc.radius() - 1.0).abs() < 1e-10,
            "radius = {}",
            arc.radius()
        );
        assert!(
            arc.angle() > PI,
            "expected major arc, got angle = {}",
            arc.angle()
        );
        assert!(
            (arc.angle() - 3.0 * PI / 2.0).abs() < 1e-10,
            "sweep = {}",
            arc.angle()
        );
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([0.0, 1.0, 0.0]), 1e-6);
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!((mid[0] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.x = {}", mid[0]);
        assert!((mid[1] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.y = {}", mid[1]);
    }

    #[test]
    fn t_start_tangent_end_off_origin_center() {
        // Circle radius 2, center at (3,4,0). Quarter circle CCW from (5,4,0) to (3,6,0).
        let arc = Arc3d::from_start_tangent_end(
            DVec([5.0, 4.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([3.0, 6.0, 0.0]),
        )
        .unwrap();
        assert!(
            (arc.radius() - 2.0).abs() < 1e-10,
            "radius = {}",
            arc.radius()
        );
        assert!(
            (arc.angle() - PI / 2.0).abs() < 1e-10,
            "sweep = {}",
            arc.angle()
        );
        let c = arc.center();
        assert!((c[0] - 3.0).abs() < 1e-10, "center.x = {}", c[0]);
        assert!((c[1] - 4.0).abs() < 1e-10, "center.y = {}", c[1]);
        assert_arc_valid(&arc, DVec([5.0, 4.0, 0.0]), DVec([3.0, 6.0, 0.0]), 1e-6);
        // Midpoint of quarter arc is at 45° from center: (3+√2, 4+√2, 0).
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!(
            (mid[0] - (3.0 + 2f64.sqrt())).abs() < 1e-6,
            "mid.x = {}",
            mid[0]
        );
        assert!(
            (mid[1] - (4.0 + 2f64.sqrt())).abs() < 1e-6,
            "mid.y = {}",
            mid[1]
        );
    }

    #[test]
    fn t_start_tangent_end_non_unit_tangent() {
        // Same geometry as the semicircle test but with a non-normalised tangent.
        // The function normalises internally so the result must be identical.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 5.0, 0.0]), // length 5, same direction as (0,1,0)
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6);
        assert!((arc.length() - PI).abs() < 1e-6);
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([-1.0, 0.0, 0.0]), 1e-6);
    }

    #[test]
    fn t_2d_start_tangent_end_quarter_circle() {
        // 2D unit circle, CCW quarter arc from (1,0) to (0,1).
        let arc =
            Arc2d::from_start_tangent_end(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([0.0, 1.0]))
                .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!((arc.angle() - PI / 2.0).abs() < 1e-10);
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!((mid[0] - FRAC_1_SQRT_2).abs() < 1e-6, "mid.x = {}", mid[0]);
        assert!((mid[1] - FRAC_1_SQRT_2).abs() < 1e-6, "mid.y = {}", mid[1]);
    }

    #[test]
    fn t_2d_start_tangent_end_major_arc() {
        // 2D unit circle. CW from (1,0) through 270° to (0,1) → sweep = 3π/2.
        let arc =
            Arc2d::from_start_tangent_end(DVec([1.0, 0.0]), DVec([0.0, -1.0]), DVec([0.0, 1.0]))
                .unwrap();
        assert!(
            (arc.radius() - 1.0).abs() < 1e-10,
            "radius = {}",
            arc.radius()
        );
        assert!(
            arc.angle() > PI,
            "expected major arc, got angle = {}",
            arc.angle()
        );
        assert!(
            (arc.angle() - 3.0 * PI / 2.0).abs() < 1e-10,
            "sweep = {}",
            arc.angle()
        );
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!((mid[0] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.x = {}", mid[0]);
        assert!((mid[1] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.y = {}", mid[1]);
    }

    // ===================== from_axis_start_angle =====================

    #[test]
    fn t_axis_start_angle_quarter() {
        // Axis along Z through origin, start at (1,0,0), sweep π/2 CCW.
        let arc = Arc3d::from_axis_start_angle(
            (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
            DVec([1.0, 0.0, 0.0]),
            PI / 2.0,
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!((arc.length() - PI / 2.0).abs() < 1e-10);
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([0.0, 1.0, 0.0]), 1e-6);
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, 1.0])).length() < 1e-10);
    }

    #[test]
    fn t_axis_start_angle_negative() {
        // Negative angle = CW.
        let arc = Arc3d::from_axis_start_angle(
            (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
            DVec([1.0, 0.0, 0.0]),
            -PI / 2.0,
        )
        .unwrap();
        assert!((arc.length() - PI / 2.0).abs() < 1e-10);
        assert!((arc.end() - DVec([0.0, -1.0, 0.0])).length() < 1e-6);
        assert_arc_valid(&arc, DVec([1.0, 0.0, 0.0]), DVec([0.0, -1.0, 0.0]), 1e-6);
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, -1.0])).length() < 1e-6);
    }

    #[test]
    fn t_axis_start_angle_large_arc() {
        // 270 degrees, radius 5.
        let arc = Arc3d::from_axis_start_angle(
            (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
            DVec([5.0, 0.0, 0.0]),
            3.0 * PI / 2.0,
        )
        .unwrap();
        assert!((arc.radius() - 5.0).abs() < 1e-10);
        assert!((arc.length() - 5.0 * 3.0 * PI / 2.0).abs() < 1e-6);
        assert_arc_valid(&arc, DVec([5.0, 0.0, 0.0]), DVec([0.0, -5.0, 0.0]), 1e-6);
    }

    #[test]
    fn t_axis_start_angle_full_circle_fails() {
        assert!(
            Arc3d::from_axis_start_angle(
                (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
                DVec([1.0, 0.0, 0.0]),
                TAU,
            )
            .is_err()
        );
    }

    #[test]
    fn t_axis_start_angle_start_on_axis_fails() {
        // Start lies exactly on the axis → radius = 0.
        assert!(
            Arc3d::from_axis_start_angle(
                (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
                DVec([0.0, 0.0, 0.5]),
                PI / 2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn t_axis_start_angle_degenerate_axis_fails() {
        // Both axis points identical → zero-length normal.
        assert!(
            Arc3d::from_axis_start_angle(
                (DVec([1.0, 2.0, 3.0]), DVec([1.0, 2.0, 3.0])),
                DVec([1.0, 0.0, 0.0]),
                PI / 2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn t_axis_start_angle_off_plane_projects() {
        // Start has a Z component — the function should project it to the plane
        // and still produce a valid arc, not fail.
        let arc = Arc3d::from_axis_start_angle(
            (DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0])),
            DVec([1.0, 0.0, 1.0]), // not perpendicular to Z axis
            PI / 2.0,
        )
        .unwrap();
        // Projected start is (1,0,0) at z=1, center is (0,0,1), radius = 1.
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        let c = arc.center();
        assert!((c[0]).abs() < 1e-10, "center.x = {}", c[0]);
        assert!((c[1]).abs() < 1e-10, "center.y = {}", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-10, "center.z = {}", c[2]);
        // End should be at (0,1,1).
        assert!((arc.end() - DVec([0.0, 1.0, 1.0])).length() < 1e-6);
    }

    #[test]
    fn t_axis_start_angle_3d_tilted() {
        // Tilted axis through (5,5,5) in direction (1,1,1).
        let center = DVec([5.0, 5.0, 5.0]);
        let axis_dir = DVec([1.0, 1.0, 1.0]);
        let start_dir = DVec([1.0, -1.0, 0.0]).normalize();
        let radius = 3.0;
        let start = center + start_dir * radius;
        let arc =
            Arc3d::from_axis_start_angle((center, center + axis_dir), start, PI / 3.0).unwrap();
        assert!((arc.radius() - radius).abs() < 1e-6);
        assert_arc_valid(&arc, start, arc.end(), 1e-6);
        let n = arc.intrinsic_normal();
        let expected_normal = axis_dir.normalize();
        assert!(
            (n - expected_normal).length() < 1e-6,
            "intrinsic normal = {:?}",
            n
        );
    }

    #[test]
    fn t_axis_start_angle_offset_center() {
        // Axis does NOT pass through origin. Axis along Z through (3,4,0).
        // Start at (5,4,0) → radius = 2, center inferred at (3,4,0).
        let arc = Arc3d::from_axis_start_angle(
            (DVec([3.0, 4.0, 0.0]), DVec([3.0, 4.0, 1.0])),
            DVec([5.0, 4.0, 0.0]),
            PI / 2.0,
        )
        .unwrap();
        assert!((arc.radius() - 2.0).abs() < 1e-10);
        let c = arc.center();
        assert!((c[0] - 3.0).abs() < 1e-10);
        assert!((c[1] - 4.0).abs() < 1e-10);
        assert!((c[2]).abs() < 1e-10);
        assert!((arc.end() - DVec([3.0, 6.0, 0.0])).length() < 1e-6);
    }

    // ── intrinsic_normal dedicated tests ─────────────────────────────

    #[test]
    fn t_intrinsic_normal_perpendicular_to_arc() {
        // The normal must be perpendicular to every tangent and every radial direction.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 1.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let n = arc.intrinsic_normal();
        let len = arc.length();
        for i in 0..=20 {
            let t = len * i as f64 / 20.0;
            let p = arc.point(t).unwrap();
            let radial = (p - arc.center()).normalize();
            assert!(
                n.dot(radial).abs() < 1e-6,
                "normal not perpendicular to radial at t={t}: dot={}",
                n.dot(radial)
            );
            let tan = arc.tangent(t).unwrap();
            assert!(
                n.dot(tan).abs() < 1e-6,
                "normal not perpendicular to tangent at t={t}: dot={}",
                n.dot(tan)
            );
        }
    }

    #[test]
    fn t_intrinsic_normal_is_unit() {
        let arc = Arc3d::from_three_points(
            DVec([3.0, 0.0, 0.0]),
            DVec([0.0, 3.0, 0.0]),
            DVec([-3.0, 0.0, 0.0]),
        )
        .unwrap();
        let n = arc.intrinsic_normal();
        assert!(
            (n.length() - 1.0).abs() < 1e-10,
            "|normal| = {}",
            n.length()
        );
    }

    #[test]
    fn t_intrinsic_normal_reversed_flips() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let rev = arc.clone().reversed();
        let n_fwd = arc.intrinsic_normal();
        let n_rev = rev.intrinsic_normal();
        // Reversing an arc reverses the sweep direction, so the normal must flip.
        assert!(
            (n_fwd + n_rev).length() < 1e-10,
            "forward normal {:?} + reversed normal {:?} should be zero",
            n_fwd,
            n_rev
        );
    }

    #[test]
    fn t_intrinsic_normal_from_start_tangent_end() {
        // CCW quarter circle in XY → normal = +Z.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, 1.0])).length() < 1e-10);

        // CW major arc → normal = -Z.
        let arc = Arc3d::from_start_tangent_end(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let n = arc.intrinsic_normal();
        assert!((n - DVec([0.0, 0.0, -1.0])).length() < 1e-10);
    }

    // ── bounds tests ─────────────────────────────────────────────────

    /// Verify bounds by sampling: every sampled point must lie within bounds,
    /// and at least one sample must be near each face.
    fn verify_bounds(arc: &Arc3d, n_samples: usize) {
        let (lo, hi) = arc.bounds();
        let lo = [lo[0], lo[1], lo[2]];
        let hi = [hi[0], hi[1], hi[2]];
        let len = arc.length();
        let mut observed_min = [f64::INFINITY; 3];
        let mut observed_max = [f64::NEG_INFINITY; 3];
        for i in 0..=n_samples {
            let t = len * i as f64 / n_samples as f64;
            let p = arc.point(t).unwrap();
            let coords = [p[0], p[1], p[2]];
            for j in 0..3 {
                assert!(
                    coords[j] >= lo[j] - 1e-10 && coords[j] <= hi[j] + 1e-10,
                    "sample at t={t} coord {j}: {} not in [{}, {}]",
                    coords[j],
                    lo[j],
                    hi[j],
                );
                observed_min[j] = observed_min[j].min(coords[j]);
                observed_max[j] = observed_max[j].max(coords[j]);
            }
        }
        // Bounds should be tight: observed extremes should be close to the box.
        for j in 0..3 {
            assert!(
                (observed_min[j] - lo[j]).abs() < 1e-4,
                "bound lo[{j}]={} but observed min={} (gap={})",
                lo[j],
                observed_min[j],
                (observed_min[j] - lo[j]).abs(),
            );
            assert!(
                (observed_max[j] - hi[j]).abs() < 1e-4,
                "bound hi[{j}]={} but observed max={} (gap={})",
                hi[j],
                observed_max[j],
                (observed_max[j] - hi[j]).abs(),
            );
        }
    }

    #[test]
    fn t_bounds_quarter_circle_xy() {
        // Quarter circle in XY plane from (1,0,0) to (0,1,0), center at origin.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        assert!((lo[0] - 0.0).abs() < 1e-10);
        assert!((lo[1] - 0.0).abs() < 1e-10);
        assert!((hi[0] - 1.0).abs() < 1e-10);
        assert!((hi[1] - 1.0).abs() < 1e-10);
        assert!((lo[2]).abs() < 1e-10);
        assert!((hi[2]).abs() < 1e-10);
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_semicircle() {
        // Semicircle from (-1,0,0) through (0,1,0) to (1,0,0).
        let arc = Arc3d::from_three_points(
            DVec([-1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        assert!((lo[0] - -1.0).abs() < 1e-10);
        assert!((hi[0] - 1.0).abs() < 1e-10);
        assert!((lo[1] - 0.0).abs() < 1e-10);
        assert!((hi[1] - 1.0).abs() < 1e-10);
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_major_arc() {
        // 270-degree arc: from (1,0,0) through (0,-1,0) to (-1,0,0)... actually
        // through (0,-1,0) to get the "long way around".
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        // This arc goes through (0,-1,0) and (1,0,0) to (-1,0,0), sweeping CW.
        // It should include y=-1 and x=1, x=-1.
        assert!(lo[1] <= -1.0 + 1e-10);
        assert!(hi[0] >= 1.0 - 1e-10);
        verify_bounds(&arc, 2000);
    }

    #[test]
    fn t_bounds_3d_arc() {
        // Arc3d in 3D: not axis-aligned.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 1.0]).normalize(),
            DVec([0.0, 0.0, 1.0]),
        )
        .unwrap();
        verify_bounds(&arc, 2000);
    }

    #[test]
    fn t_bounds_offset_center() {
        // Arc3d with center not at origin.
        let arc = Arc3d::from_three_points(
            DVec([10.0, 5.0, 0.0]),
            DVec([10.5, 5.5, 0.0]),
            DVec([11.0, 5.0, 0.0]),
        )
        .unwrap();
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_reversed_matches() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let (lo1, hi1) = arc.bounds();
        let (lo2, hi2) = arc.reversed().bounds();
        assert!((lo1[0] - lo2[0]).abs() < 1e-10);
        assert!((lo1[1] - lo2[1]).abs() < 1e-10);
        assert!((lo1[2] - lo2[2]).abs() < 1e-10);
        assert!((hi1[0] - hi2[0]).abs() < 1e-10);
        assert!((hi1[1] - hi2[1]).abs() < 1e-10);
        assert!((hi1[2] - hi2[2]).abs() < 1e-10);
    }

    #[test]
    fn arc_bounds_failing_special_case() {
        // This test case came up during manual testing and exposed a bug.
        let arc = Arc3d::from_three_points(
            DVec([-4.3480377197265625, 2.220446049250313e-16, 0.0]),
            DVec([-2.3053700923919678, 1.0424094200134277, 0.35949984192848206]),
            DVec([-2.5, 2.220446049250313e-16, 0.0]),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        // Sample densely and check all points are within bounds.
        let len = arc.length();
        let n = 10000;
        let mut min_x = f64::INFINITY;
        for i in 0..=n {
            let t = len * (i as f64) / (n as f64);
            let pt = arc.point(t).unwrap();
            min_x = min_x.min(pt[0]);
            assert!(
                pt[0] >= lo[0] - f64::EPSILON && pt[0] <= hi[0] + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt[0],
                lo[0],
                hi[0]
            );
            assert!(
                pt[1] >= lo[1] - f64::EPSILON && pt[1] <= hi[1] + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt[1],
                lo[1],
                hi[1]
            );
            assert!(
                pt[2] >= lo[2] - f64::EPSILON && pt[2] <= hi[2] + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt[2],
                lo[2],
                hi[2]
            );
        }
        eprintln!("bounds lo.x = {}, actual min_x = {}", lo[0], min_x);
        eprintln!("angle = {}, radius = {}", arc.angle(), arc.radius());
        eprintln!("center = {:?}", arc.center());
    }

    // ── 2D arc tests ──────────────────────────────────────────────────────────

    #[test]
    fn t_2d_quarter_circle() {
        // Unit circle, CCW from (1,0) through (√2/2, √2/2) to (0,1). Sweep = π/2.
        let arc = Arc2d::from_three_points(
            DVec([1.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
            DVec([0.0, 1.0]),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!((arc.angle() - PI / 2.0).abs() < 1e-10);
        assert!((arc.length() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn t_2d_semicircle() {
        // Unit circle, CCW from (-1,0) through (0,1) to (1,0). Sweep = π.
        let arc = Arc2d::from_three_points(DVec([-1.0, 0.0]), DVec([0.0, 1.0]), DVec([1.0, 0.0]))
            .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!((arc.angle() - PI).abs() < 1e-10);
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!(mid[0].abs() < 1e-10);
        assert!((mid[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_2d_major_arc() {
        // Unit circle, CW from (1,0) through (0,-1) to (0,1). Sweep = 3π/2.
        // (1,0) and (0,1) are not antipodal, so this is a genuine major arc.
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, -1.0]), DVec([0.0, 1.0]))
            .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!(
            arc.angle() > PI,
            "expected major arc, got angle = {}",
            arc.angle()
        );
        assert!((arc.angle() - 3.0 * PI / 2.0).abs() < 1e-10);
        // True midpoint at 0° − 135° = 225°.
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!((mid[0] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.x = {}", mid[0]);
        assert!((mid[1] + FRAC_1_SQRT_2).abs() < 1e-6, "mid.y = {}", mid[1]);
    }

    #[test]
    fn t_2d_major_arc_obtuse_triangle() {
        // Unit circle. start=0°, middle=270°, end=60°. Sweep = 300° = 5π/3.
        let arc = Arc2d::from_three_points(
            DVec([1.0, 0.0]),
            DVec([0.0, -1.0]),
            DVec([0.5, 3f64.sqrt() / 2.0]),
        )
        .unwrap();
        let expected_sweep = 5.0 * PI / 3.0;
        assert!(
            arc.angle() > PI,
            "expected major arc, got angle = {}",
            arc.angle()
        );
        assert!(
            (arc.angle() - expected_sweep).abs() < 1e-10,
            "sweep: {} != 5π/3",
            arc.angle()
        );
        // True midpoint at 0° − 150° = 210°.
        let arc_mid = arc.point(arc.length() / 2.0).unwrap();
        assert!(
            (arc_mid[0] - (-(3f64.sqrt() / 2.0))).abs() < 1e-6,
            "mid.x = {}",
            arc_mid[0]
        );
        assert!((arc_mid[1] + 0.5).abs() < 1e-6, "mid.y = {}", arc_mid[1]);
    }

    #[test]
    fn t_2d_off_origin_center() {
        // Circle of radius 3 centered at (5, -2). Three points on it.
        let cx = 5.0_f64;
        let cy = -2.0_f64;
        let r = 3.0_f64;
        let arc =
            Arc2d::from_three_points(DVec([cx + r, cy]), DVec([cx, cy + r]), DVec([cx - r, cy]))
                .unwrap();
        assert!(
            (arc.radius() - r).abs() < 1e-10,
            "radius = {}",
            arc.radius()
        );
        assert!((arc.angle() - PI).abs() < 1e-10, "angle = {}", arc.angle());
        let start = arc.start();
        assert!((start[0] - (cx + r)).abs() < 1e-10);
        assert!((start[1] - cy).abs() < 1e-10);
    }

    #[test]
    fn t_2d_endpoints() {
        // Arc endpoints must match the input start and end points.
        let start = DVec([1.0, 0.0]);
        let end = DVec([0.0, 1.0]);
        let arc =
            Arc2d::from_three_points(start, DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2]), end).unwrap();
        assert!((arc.start()[0] - start[0]).abs() < 1e-10);
        assert!((arc.start()[1] - start[1]).abs() < 1e-10);
        assert!((arc.end()[0] - end[0]).abs() < 1e-10);
        assert!((arc.end()[1] - end[1]).abs() < 1e-10);
    }

    #[test]
    fn t_2d_collinear_returns_err() {
        // Three collinear points must return an error.
        let result = Arc2d::from_three_points(DVec([0.0, 0.0]), DVec([1.0, 0.0]), DVec([2.0, 0.0]));
        assert!(result.is_err());
    }

    #[test]
    fn t_2d_reversed() {
        let arc = Arc2d::from_three_points(DVec([-1.0, 0.0]), DVec([0.0, 1.0]), DVec([1.0, 0.0]))
            .unwrap();
        let rev = arc.clone().reversed();
        let len = arc.length();
        assert!((len - rev.length()).abs() < 1e-12);
        assert!((arc.start()[0] - rev.end()[0]).abs() < 1e-10);
        assert!((arc.end()[0] - rev.start()[0]).abs() < 1e-10);
        let n = 20;
        for i in 0..=n {
            let t = len * i as f64 / n as f64;
            let fwd = arc.point(t).unwrap();
            let bwd = rev.point(len - t).unwrap();
            let err = (fwd - bwd).length();
            assert!(err < 1e-10, "mismatch at t={t}: err={err}");
        }
    }

    #[test]
    fn t_2d_tangent_perpendicular_to_radius() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        let len = arc.length();
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            let p = arc.point(t).unwrap();
            let tan = arc.tangent(t).unwrap();
            let radial = (p - arc.center()).normalize();
            let dot = radial.dot(tan).abs();
            assert!(dot < 1e-6, "tangent not perpendicular at t={t}: dot={dot}");
            let mag = tan.length();
            assert!(
                (mag - 1.0).abs() < 1e-6,
                "tangent not unit length at t={t}: |tan|={mag}"
            );
        }
    }

    #[test]
    fn t_2d_unit_speed_parameterization() {
        let arc = Arc2d::from_three_points(DVec([2.0, 0.0]), DVec([0.0, 2.0]), DVec([-2.0, 0.0]))
            .unwrap();
        let len = arc.length();
        let dt = 1e-7;
        for i in 0..10 {
            let t = len * i as f64 / 10.0;
            let p0 = arc.point(t).unwrap();
            let p1 = arc.point(t + dt).unwrap();
            let speed = (p1 - p0).length() / dt;
            assert!(
                (speed - 1.0).abs() < 1e-4,
                "numerical speed at t={t}: {speed} != 1.0"
            );
            let mag = arc.tangent(t).unwrap().length();
            assert!(
                (mag - 1.0).abs() < 1e-6,
                "tangent magnitude at t={t}: {mag} != 1.0"
            );
        }
    }

    #[test]
    fn t_2d_out_of_domain_returns_none() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        assert!(arc.point(-1.0).is_none());
        assert!(arc.point(arc.length() + 1.0).is_none());
    }

    #[test]
    fn t_2d_adaptive_samples_deviation() {
        // Test cases as three points and a radius per arc.
        const CASES: &[([[f64; 2]; 3], f64)] = &[
            ([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], 0.01), // semicircle r=1
            ([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0]], 0.01), // semicircle r=3
            (
                [[1.0, 0.0], [FRAC_1_SQRT_2, FRAC_1_SQRT_2], [0.0, 1.0]], // quarter circle
                0.001,
            ),
            ([[100.0, 0.0], [0.0, 100.0], [-100.0, 0.0]], 0.1), // large radius
            ([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], 0.01),     // other semicircle
            ([[10.0, 5.0], [7.0, 8.0], [4.0, 5.0]], 0.005),     // off-origin center
        ];
        for (i, &([p0, p1, p2], tol)) in CASES.iter().enumerate() {
            let arc = Arc2d::from_three_points(DVec(p0), DVec(p1), DVec(p2)).unwrap();
            let r = arc.radius();
            let center = arc.center();
            let samples: Vec<_> = arc.adaptive_samples(tol).collect();
            assert!(
                samples.len() > 2,
                "case {i}: too few samples: {}",
                samples.len()
            );
            for pair in samples.windows(2) {
                let mid = (pair[0] + pair[1]) * 0.5;
                let deviation = ((mid - center).length() - r).abs();
                assert!(
                    deviation <= tol + 1e-10,
                    "case {i}: deviation {deviation} > tolerance {tol}"
                );
            }
        }
    }

    #[test]
    fn t_2d_point_with_derivs_matches_point_and_tangent() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        let len = arc.length();
        let mut results = [DVec([0.0, 0.0]); 3];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            arc.point_with_derivs(t, &mut results).unwrap();
            let err_p = (results[0] - arc.point(t).unwrap()).length();
            let err_t = (results[1] - arc.tangent(t).unwrap()).length();
            assert!(err_p < 1e-12, "point mismatch at t={t}: {err_p}");
            assert!(err_t < 1e-12, "tangent mismatch at t={t}: {err_t}");
        }
    }

    #[test]
    fn t_2d_point_with_derivs_curvature() {
        // d²p/ds² has magnitude 1/r and points toward center.
        let arc = Arc2d::from_three_points(DVec([3.0, 0.0]), DVec([0.0, 3.0]), DVec([-3.0, 0.0]))
            .unwrap();
        let r = arc.radius();
        let len = arc.length();
        let mut results = [DVec([0.0, 0.0]); 3];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            arc.point_with_derivs(t, &mut results).unwrap();
            let mag = results[2].length();
            assert!(
                (mag - 1.0 / r).abs() < 1e-6,
                "curvature magnitude at t={t}: {mag} != {}",
                1.0 / r
            );
            let inward = (arc.center() - results[0]).normalize();
            let dot = inward.dot(results[2].normalize());
            assert!(
                (dot - 1.0).abs() < 1e-6,
                "curvature not pointing inward at t={t}: dot={dot}"
            );
        }
    }

    #[test]
    fn t_2d_point_with_derivs_out_of_domain() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        let mut results = [DVec([0.0, 0.0]); 2];
        assert!(arc.point_with_derivs(-1.0, &mut results).is_err());
        assert!(
            arc.point_with_derivs(arc.length() + 1.0, &mut results)
                .is_err()
        );
        assert!(arc.point_with_derivs(-1.0, &mut []).is_ok());
    }

    #[test]
    fn t_2d_point_with_derivs_point_only() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        let mut results = [DVec([0.0, 0.0]); 1];
        arc.point_with_derivs(0.0, &mut results).unwrap();
        assert!((results[0] - arc.point(0.0).unwrap()).length() < 1e-12);
    }

    #[test]
    fn t_2d_point_with_derivs_higher_derivs_zero() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        let mut results = [DVec([1.0, 1.0]); 5];
        arc.point_with_derivs(arc.length() / 2.0, &mut results)
            .unwrap();
        for (i, r) in results[3..].iter().enumerate() {
            assert_eq!(*r, DVec([0.0, 0.0]), "results[{}] not zero", i + 3);
        }
    }

    #[test]
    fn t_2d_start_tangent_end_semicircle() {
        let arc =
            Arc2d::from_start_tangent_end(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
                .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6);
        assert!((arc.length() - PI).abs() < 1e-6);
        let mid = arc.point(arc.length() / 2.0).unwrap();
        assert!(mid[0].abs() < 1e-6);
        assert!((mid[1] - 1.0).abs() < 1e-6);
    }

    // ======================================================================
    // uniform_samples tests
    // ======================================================================

    // Helper: collect uniform_samples and assert each point matches arc.point(start + i*step).
    fn check_uniform_samples(arc: &Arc3d, start: f64, step: f64, expected_count: usize) {
        let pts: Vec<_> = arc.uniform_samples(start, step, 1e-6).collect();
        assert_eq!(
            pts.len(),
            expected_count,
            "expected {expected_count} points, got {}",
            pts.len()
        );
        for (i, &pt) in pts.iter().enumerate() {
            let u = start + i as f64 * step;
            let expected = arc.point(u).unwrap();
            assert!(
                (pt - expected).length() < 1e-9,
                "pt[{i}] at u={u:.6}: got {pt:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn t_uniform_samples_minor_arc() {
        // Quarter circle (1,0,0) → (√2/2,√2/2,0) → (0,1,0); radius=1, angle=π/2, len=π/2.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let len = arc.length();
        // step = len/4 → 5 points at u = 0, len/4, len/2, 3len/4, len.
        check_uniform_samples(&arc, 0.0, len / 4.0, 5);
        // Endpoints match arc.start() and arc.end() exactly.
        let pts: Vec<_> = arc.uniform_samples(0.0, len / 4.0, 1e-6).collect();
        assert!(
            (pts[0] - arc.start()).length() < 1e-9,
            "first point not at start"
        );
        assert!(
            (pts[4] - arc.end()).length() < 1e-9,
            "last point not at end"
        );
        // Non-zero start offset: start=len/4, step=len/4 → 4 points.
        check_uniform_samples(&arc, len / 4.0, len / 4.0, 4);
    }

    #[test]
    fn t_uniform_samples_major_arc() {
        // Major arc (5π/3 ≈ 300°): (1,0,0) → (0,-1,0) → (0.5, √3/2, 0); radius=1.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.5, 3f64.sqrt() / 2.0, 0.0]),
        )
        .unwrap();
        assert!(arc.angle() > PI, "expected major arc");
        let len = arc.length();
        // step = len/6 → 7 points (6 equal steps covering the full arc).
        check_uniform_samples(&arc, 0.0, len / 6.0, 7);
        // Endpoints.
        let pts: Vec<_> = arc.uniform_samples(0.0, len / 6.0, 1e-6).collect();
        assert!(
            (pts[0] - arc.start()).length() < 1e-9,
            "first point not at start"
        );
        assert!(
            (pts[6] - arc.end()).length() < 1e-9,
            "last point not at end"
        );
        // Start offset splits the arc differently.
        check_uniform_samples(&arc, len / 6.0, len / 6.0, 6);
    }

    #[test]
    fn t_uniform_samples_semicircle() {
        // Semicircle (angle = π, boundary between minor and major).
        let arc = Arc3d::from_three_points(
            DVec([-1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let len = arc.length(); // π
        // step = len/4 → 5 points; midpoint is at u=len/2.
        check_uniform_samples(&arc, 0.0, len / 4.0, 5);
        let pts: Vec<_> = arc.uniform_samples(0.0, len / 4.0, 1e-6).collect();
        let mid = pts[2];
        assert!(
            (mid - DVec([0.0, 1.0, 0.0])).length() < 1e-9,
            "midpoint wrong: {mid:?}"
        );
    }

    #[test]
    fn t_uniform_samples_edge_cases() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let len = arc.length();
        // start > len → 0 points.
        assert_eq!(arc.uniform_samples(len + 0.1, 0.1, 1e-6).count(), 0);
        // start = len → 1 point at the endpoint.
        let at_end: Vec<_> = arc.uniform_samples(len, 0.1, 1e-6).collect();
        assert_eq!(at_end.len(), 1);
        assert!((at_end[0] - arc.end()).length() < 1e-9);
        // step > len → 1 point at the start.
        let big_step: Vec<_> = arc.uniform_samples(0.0, len * 2.0, 1e-6).collect();
        assert_eq!(big_step.len(), 1);
        assert!((big_step[0] - arc.start()).length() < 1e-9);
        // step = len → 2 points: start and end.
        let full_step: Vec<_> = arc.uniform_samples(0.0, len, 1e-6).collect();
        assert_eq!(full_step.len(), 2);
        assert!((full_step[0] - arc.start()).length() < 1e-9);
        assert!((full_step[1] - arc.end()).length() < 1e-9);
        // Negative start → clamped to 0, same result as start=0.
        let from_zero: Vec<_> = arc.uniform_samples(0.0, len / 4.0, 1e-6).collect();
        let from_neg: Vec<_> = arc.uniform_samples(-1.0, len / 4.0, 1e-6).collect();
        assert_eq!(from_neg.len(), from_zero.len());
        for (a, b) in from_zero.iter().zip(from_neg.iter()) {
            assert!((*a - *b).length() < 1e-9);
        }
    }

    #[test]
    fn t_2d_start_tangent_end_collinear_fails() {
        assert!(
            Arc2d::from_start_tangent_end(DVec([0.0, 0.0]), DVec([1.0, 0.0]), DVec([2.0, 0.0]),)
                .is_err()
        );
    }

    #[test]
    fn t_2d_start_tangent_end_coincident_fails() {
        assert!(
            Arc2d::from_start_tangent_end(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([1.0, 0.0]),)
                .is_err()
        );
    }

    #[test]
    fn t_2d_start_tangent_end_matches_from_three_points() {
        // Build from three points, extract start tangent, reconstruct, compare.
        let arc1 = Arc2d::from_three_points(DVec([2.0, 0.0]), DVec([0.0, 2.0]), DVec([-2.0, 0.0]))
            .unwrap();
        let tan = arc1.tangent(0.0).unwrap();
        let arc2 = Arc2d::from_start_tangent_end(DVec([2.0, 0.0]), tan, DVec([-2.0, 0.0])).unwrap();
        assert!((arc1.radius() - arc2.radius()).abs() < 1e-6);
        assert!((arc1.length() - arc2.length()).abs() < 1e-6);
        assert!((arc1.center() - arc2.center()).length() < 1e-6);
    }

    // ── curvature tests ───────────────────────────────────────────────────

    #[test]
    fn t_curvature_magnitude_and_direction() {
        // Off-origin center: arc of radius 3 centered at (1, 1, 0).
        // This specifically exercises the (center - p) / r² formula —
        // with center at origin the old bug (-p/r) would give wrong magnitude
        // but not wrong direction, so an off-origin center catches both.
        let arc = Arc3d::from_three_points(
            DVec([4.0, 1.0, 0.0]),
            DVec([1.0, 4.0, 0.0]),
            DVec([-2.0, 1.0, 0.0]),
        )
        .unwrap();
        let r = arc.radius();
        let len = arc.length();
        for i in 0..=6 {
            let t = len * i as f64 / 6.0;
            let curv = arc.curvature(t).unwrap();
            let p = arc.point(t).unwrap();
            // Magnitude must be 1/r.
            assert!(
                (curv.length() - 1.0 / r).abs() < 1e-10,
                "magnitude at t={t}: got {}, expected {}",
                curv.length(),
                1.0 / r
            );
            // Must point from the arc point toward the center.
            let inward = (arc.center() - p).normalize();
            assert!(
                (inward.dot(curv.normalize()) - 1.0).abs() < 1e-10,
                "direction at t={t}"
            );
        }
    }

    #[test]
    fn t_curvature_out_of_domain() {
        let arc = Arc2d::from_three_points(DVec([1.0, 0.0]), DVec([0.0, 1.0]), DVec([-1.0, 0.0]))
            .unwrap();
        assert!(arc.curvature(-0.001).is_none());
        assert!(arc.curvature(arc.length() + 0.001).is_none());
    }

    // ── is_closed tests ───────────────────────────────────────────────────

    #[test]
    fn t_is_closed() {
        // Partial arcs are never closed — the constructor rejects full circles.
        let semicircle = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        assert!(!semicircle.is_closed());

        let quarter = Arc2d::from_three_points(
            DVec([1.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
            DVec([0.0, 1.0]),
        )
        .unwrap();
        assert!(!quarter.is_closed());
    }
}
