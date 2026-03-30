use crate::{Vec3, error::Error};
use std::f64::{
    self,
    consts::{PI, TAU},
};

const EPS: f64 = f64::EPSILON;
const ANG_EPS: f64 = 1e-10;

#[derive(Clone, Debug)]
pub struct Arc {
    center: Vec3,
    start_dir: Vec3, // Unit vector from center toward start.
    mid_dir: Vec3,   // Unit vector from center toward midpoint (used for antipodal SLERP).
    end_dir: Vec3,   // Unit vector from center toward end.
    radius: f64,
    angle: f64, // Signed sweep angle from start to end through mid. Positive = CCW around normal.
}

impl Arc {
    /// Construct an arc from three points: start, a point along the arc, and end.
    pub fn from_three_points(start: Vec3, middle: Vec3, end: Vec3) -> Result<Self, Error> {
        let (center, radius, normal) =
            circumcircle(start, middle, end).ok_or(Error::PointsCollinear)?;
        let start_dir = (start - center).normalize();
        let end_dir = (end - center).normalize();
        let angle = oriented_angle(start_dir, end_dir, normal);
        let mid_dir = rotate_around(start_dir, normal, angle * 0.5);
        check_radius_and_angle(radius, angle)?;
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
    pub fn from_start_tangent_end(start: Vec3, tangent: Vec3, end: Vec3) -> Result<Self, Error> {
        let tangent = tangent.normalize();
        let chord = end - start;
        let normal = tangent.cross(chord).normalize();
        if normal.length_sq() < 0.5 {
            return Err(Error::PointsCollinear);
        }
        let chord_dir = chord.normalize();
        let dot = chord_dir.dot(tangent).clamp(-1.0, 1.0);
        let angle = dot.acos();
        if angle <= ANG_EPS || angle >= PI - ANG_EPS {
            return Err(Error::PointsCollinear);
        }
        let halfchord = chord.length() * 0.5;
        let radius = halfchord / angle.sin();
        let sweep = 2.0 * angle;
        check_radius_and_angle(radius, sweep)?;
        let shift = if (angle - PI * 0.5).abs() < ANG_EPS {
            0.0
        } else {
            halfchord / angle.tan()
        };
        let center = (start + end) * 0.5 + normal.cross(chord).normalize() * shift;
        let start_dir = (start - center).normalize();
        let end_dir = (end - center).normalize();
        let mid_dir = rotate_around(start_dir, normal, sweep * 0.5);
        Ok(Arc {
            center,
            radius,
            start_dir,
            mid_dir,
            end_dir,
            angle: sweep,
        })
    }

    /// Construct an arc from center, normal, start point on the circle, and sweep angle.
    /// Angle is signed: positive = CCW around normal.
    pub fn from_center_normal_start_angle(
        center: Vec3,
        mut normal: Vec3,
        start: Vec3,
        mut angle: f64,
    ) -> Result<Self, Error> {
        if angle < 0.0 {
            normal = -normal;
            angle = -angle;
        }
        let radius = (start - center).length();
        check_radius_and_angle(radius, angle)?;
        let normal = normal.normalize();
        if normal.length_sq() < 0.5 {
            return Err(Error::InvalidParameter);
        }
        if (start - center).dot(normal).abs() > EPS * radius {
            return Err(Error::InvalidParameter);
        }
        let start_dir = (start - center).normalize();
        let end_dir = rotate_around(start_dir, normal, angle);
        let mid_dir = rotate_around(start_dir, normal, angle * 0.5);
        Ok(Arc {
            center,
            radius,
            start_dir,
            mid_dir,
            end_dir,
            angle,
        })
    }

    /// Parameter domain: `[0, arc_length]`.
    pub fn domain(&self) -> (f64, f64) {
        (0.0, self.length())
    }

    /// Total arc length.
    pub fn length(&self) -> f64 {
        self.radius * self.angle
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// Sweep angle (signed).
    pub fn angle(&self) -> f64 {
        self.angle
    }

    pub fn start(&self) -> Vec3 {
        self.center + self.start_dir * self.radius
    }

    pub fn end(&self) -> Vec3 {
        self.center + self.end_dir * self.radius
    }

    /// Evaluate a point on the arc at arc-length parameter `t` in `[0, length]`.
    pub fn point(&self, t: f64) -> Option<Vec3> {
        let len = self.length();
        if t < 0.0 || t > len {
            return None;
        }
        let coeff = slerp(self.angle, t / len);
        Some(
            self.center
                + self.start_dir * coeff[0] * self.radius
                + self.mid_dir * coeff[1] * self.radius
                + self.end_dir * coeff[2] * self.radius,
        )
    }

    /// Evaluate the unit tangent at arc-length parameter `t` in `[0, length]`.
    pub fn tangent(&self, t: f64) -> Option<Vec3> {
        let len = self.length();
        if t < 0.0 || t > len || self.radius < f64::EPSILON {
            return None;
        }
        let coeff = slerp_deriv(self.angle, t / len);
        // Arc-length parameterization: scale by ds/dt = 1/length to get unit tangent.
        Some(
            (self.start_dir * coeff[0] + self.mid_dir * coeff[1] + self.end_dir * coeff[2])
                / self.angle,
        )
    }

    pub fn point_with_derivs(&self, t: f64, results: &mut [Vec3]) -> Result<(), Error> {
        if results.is_empty() {
            return Ok(());
        }
        let len = self.length();
        if t < 0.0 || t > len {
            return Err(Error::InvalidParameter);
        }
        let frac = t / len;
        if results.len() == 1 {
            let coeff = slerp(self.angle, frac);
            results[0] = self.center
                + self.start_dir * coeff[0] * self.radius
                + self.mid_dir * coeff[1] * self.radius
                + self.end_dir * coeff[2] * self.radius;
            Ok(())
        } else {
            let [pos_coeff, deriv_coeff] = slerp_with_deriv(self.angle, frac);
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
                results[3..].fill(Vec3(0.0, 0.0, 0.0));
            }
            Ok(())
        }
    }

    /// Uniformly sample the arc with the given tolerance (max chord deviation).
    /// For a circle, the chord error at step angle α is `r(1 - cos(α/2))`.
    /// Solving for α: `α = 2 * acos(1 - tolerance/r)`.
    pub fn adaptive_samples(&self, tolerance: f64) -> impl Iterator<Item = Vec3> {
        let ang_step = if self.radius > EPS && tolerance > 0.0 {
            2.0 * (1.0 - (tolerance / self.radius).min(1.0)).acos()
        } else {
            0.1 // fallback
        };
        let half_angle = 0.5 * self.angle;
        let n = if self.angle < ANG_EPS {
            1
        } else {
            ((half_angle / ang_step).ceil() as usize).max(1)
        };
        let nf = n as f64;
        let inv_sin = 1.0 / half_angle.sin();
        std::iter::once([1.0, 0.0, 0.0])
            .chain((1..=n).map(move |i| {
                let coeff = slerp_raw_no_adjust(half_angle, (i as f64) / nf);
                [inv_sin * coeff[0], inv_sin * coeff[1], 0.0]
            }))
            .chain((1..n).map(move |i| {
                let coeff = slerp_raw_no_adjust(half_angle, (i as f64) / nf);
                [0.0, inv_sin * coeff[0], inv_sin * coeff[1]]
            }))
            .chain(std::iter::once([0.0, 0.0, 1.0]))
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
    pub fn bounds(&self) -> (Vec3, Vec3) {
        arc_bounds(
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
}

fn arc_bounds(
    center: Vec3,
    radius: f64,
    start_dir: Vec3,
    mid_dir: Vec3,
    end_dir: Vec3,
    angle: f64,
) -> (Vec3, Vec3) {
    let (min, max) = {
        let start = center + start_dir * radius;
        let end = center + end_dir * radius;
        (
            Vec3(start.0.min(end.0), start.1.min(end.1), start.2.min(end.2)),
            Vec3(start.0.max(end.0), start.1.max(end.1), start.2.max(end.2)),
        )
    };
    if angle < ANG_EPS {
        // Near degenerate arc.
        return (min, max);
    }
    if (angle - PI).abs() < ANG_EPS {
        // Near semicircle.
        let half = angle * 0.5;
        let new_mid = (start_dir + mid_dir).normalize();
        let (lmin, lmax) = arc_bounds(center, radius, start_dir, new_mid, mid_dir, half);
        let new_mid = (mid_dir + end_dir).normalize();
        let (rmin, rmax) = arc_bounds(center, radius, mid_dir, new_mid, end_dir, half);
        return (
            Vec3(lmin.0.min(rmin.0), lmin.1.min(rmin.1), lmin.2.min(rmin.2)),
            Vec3(lmax.0.max(rmax.0), lmax.1.max(rmax.1), lmax.2.max(rmax.2)),
        );
    }
    // Find the roots (zeros of the derivative) and check if they lie within
    // the arc. At the extremum angle u for component i, the direction vector
    // satisfies (cos u, sin u) ∝ (denom_i, numer_i), and the extremal slerp
    // value is ±h_i / sin(θ) where h_i = sqrt(numer_i² + denom_i²).
    // Instead of computing u = atan2(n, d) and checking u ∈ [0, θ), we test
    // sector membership with cross products: sin(u) > 0 ↔ numer_i > 0, and
    // sin(u−θ) < 0 ↔ numer_i·cosθ − denom_i·sinθ < 0.
    let (sin, cos) = angle.sin_cos();
    let numer = end_dir - start_dir * cos;
    let denom = start_dir * sin;
    let inv_sin = 1.0 / sin;
    let numer = [numer.0, numer.1, numer.2];
    let denom = [denom.0, denom.1, denom.2];
    let center = [center.0, center.1, center.2];
    let mut min = [min.0, min.1, min.2];
    let mut max = [max.0, max.1, max.2];
    let short = angle < PI;
    for ci in 0..3 {
        let n = numer[ci];
        let d = denom[ci];
        let cross_end = n * cos - d * sin; // ∝ sin(u - θ)
        let ext = radius * (n * n + d * d).sqrt() * inv_sin;
        // u root: (cos u, sin u) ∝ (d, n).
        let u_in = if short {
            n > 0.0 && cross_end < 0.0
        } else {
            n > 0.0 || cross_end < 0.0
        };
        // u + π root: (cos u, sin u) ∝ (-d, -n). Flips both signs.
        let u_pi_in = if short {
            n < 0.0 && cross_end > 0.0
        } else {
            n < 0.0 || cross_end > 0.0
        };
        if u_in {
            let val = center[ci] + ext;
            min[ci] = min[ci].min(val);
            max[ci] = max[ci].max(val);
        }
        if u_pi_in {
            let val = center[ci] - ext;
            min[ci] = min[ci].min(val);
            max[ci] = max[ci].max(val);
        }
    }
    (Vec3(min[0], min[1], min[2]), Vec3(max[0], max[1], max[2]))
}

/// Check radius and angle invariants. Call early, as soon as both are known.
fn check_radius_and_angle(radius: f64, angle: f64) -> Result<(), Error> {
    assert!(
        angle >= 0. && angle <= TAU,
        "INTERNAL ERROR: angle {} is outside bounds.",
        angle
    );
    if radius < EPS {
        Err(Error::RadiusTooSmall)
    } else if angle.abs() >= TAU - ANG_EPS {
        Err(Error::ArcCannotBeCircle)
    } else {
        Ok(())
    }
}

/// Rotate unit vector `v` around unit axis `axis` by `angle` radians (Rodrigues' formula).
fn rotate_around(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

/// Circumcircle of three 3D points. Returns (center, radius, normal).
fn circumcircle(a: Vec3, b: Vec3, c: Vec3) -> Option<(Vec3, f64, Vec3)> {
    let ca = c - a;
    let ba = b - a;
    let crs = ba.cross(ca);
    let crs_len2 = crs.length_sq();
    if crs_len2 < 1e-20 {
        return None;
    }
    let ca_len2 = ca.length_sq();
    let ba_len2 = ba.length_sq();
    let rvec = (crs.cross(ba) * ca_len2 + ca.cross(crs) * ba_len2) / (2.0 * crs_len2);
    let center = a + rvec;
    let radius = rvec.length();
    let normal = crs.normalize();
    Some((center, radius, normal))
}

/// Oriented angle from `from` to `to` around `normal`, in [0, 2π).
fn oriented_angle(from: Vec3, to: Vec3, normal: Vec3) -> f64 {
    let f = from.normalize();
    let t = to.normalize();
    let dot = f.dot(t).clamp(-1.0, 1.0);
    let cross = f.cross(t);
    let angle = dot.acos().copysign(cross.dot(normal));
    if angle < 0.0 { angle + TAU } else { angle }
}

#[inline(always)]
fn slerp(angle: f64, t: f64) -> [f64; 3] {
    if angle < ANG_EPS {
        return [1.0 - t, 0.0, t];
    }
    let half = angle * 0.5;
    if t <= 0.5 {
        let coeff = slerp_raw(half, t * 2.0);
        [coeff[0], coeff[1], 0.0]
    } else {
        let coeff = slerp_raw(half, (t - 0.5) * 2.0);
        [0.0, coeff[0], coeff[1]]
    }
}

#[inline(always)]
fn slerp_raw(angle: f64, t: f64) -> [f64; 2] {
    let inv_sin = 1.0 / angle.sin();
    slerp_raw_no_adjust(angle, t).map(|c| c * inv_sin)
}

#[inline(always)]
fn slerp_raw_no_adjust(angle: f64, t: f64) -> [f64; 2] {
    [((1.0 - t) * angle).sin(), (t * angle).sin()]
}

/// Derivative of SLERP with respect to the fractional parameter `t` in [0, 1].
/// d/dt [sin((1-t)θ)/sin(θ) · a + sin(tθ)/sin(θ) · b]
///    = θ/sin(θ) · [-cos((1-t)θ) · a + cos(tθ) · b]
#[inline(always)]
fn slerp_deriv(angle: f64, t: f64) -> [f64; 3] {
    if angle < ANG_EPS {
        return [-1.0, 0.0, 1.0];
    }
    let half = angle / 2.0;
    // Chain rule: d/dt f(2t) = 2 * f'(2t).
    if t <= 0.5 {
        let coeff = slerp_deriv_unchecked(half, t * 2.0).map(|c| c * 2.0);
        [coeff[0], coeff[1], 0.0]
    } else {
        let coeff = slerp_deriv_unchecked(half, (t - 0.5) * 2.0).map(|c| c * 2.0);
        [0.0, coeff[0], coeff[1]]
    }
}

fn slerp_deriv_unchecked(angle: f64, t: f64) -> [f64; 2] {
    let theta_over_sin = angle / angle.sin();
    let w0 = -((1.0 - t) * angle).cos() * theta_over_sin;
    let w1 = (t * angle).cos() * theta_over_sin;
    [w0, w1]
}

/// Combined SLERP value and derivative, sharing branching and trig.
#[inline(always)]
fn slerp_with_deriv(angle: f64, t: f64) -> [[f64; 3]; 2] {
    if angle < ANG_EPS {
        return [[1.0 - t, 0.0, t], [-1.0, 0.0, 1.0]];
    }
    let half = angle / 2.0;
    if t <= 0.5 {
        let [val_coeff, deriv_coeff] = slerp_with_deriv_unchecked(half, t * 2.0);
        [
            [val_coeff[0], val_coeff[1], 0.0],
            [deriv_coeff[0] * 2.0, deriv_coeff[1] * 2.0, 0.0], // Chain rule.
        ]
    } else {
        let [val_coeff, deriv_coeff] = slerp_with_deriv_unchecked(half, (t - 0.5) * 2.0);
        [
            [0.0, val_coeff[0], val_coeff[1]],
            [0.0, deriv_coeff[0] * 2.0, deriv_coeff[1] * 2.0], // Chain rule.
        ]
    }
}

fn slerp_with_deriv_unchecked(angle: f64, t: f64) -> [[f64; 2]; 2] {
    let inv_sin = 1.0 / angle.sin();
    let arg0 = (1.0 - t) * angle;
    let arg1 = t * angle;
    let (sin0, cos0) = arg0.sin_cos();
    let (sin1, cos1) = arg1.sin_cos();
    [
        [(sin0 * inv_sin), (sin1 * inv_sin)],
        [(-cos0 * angle * inv_sin), (cos1 * angle * inv_sin)],
    ]
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn t_semicircle() {
        // Semicircle from (-1,0,0) through (0,1,0) to (1,0,0).
        let arc = Arc::from_three_points(
            Vec3(-1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        )
        .unwrap();
        let expected_len = PI; // radius=1, angle=π
        assert!((arc.length() - expected_len).abs() < 1e-10);
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        // Midpoint should be at (0, 1, 0).
        let mid = arc.point(expected_len / 2.0).unwrap();
        assert!((mid.0).abs() < 1e-6);
        assert!((mid.1 - 1.0).abs() < 1e-6);
        // Endpoints exact.
        assert!((arc.start().0 + 1.0).abs() < 1e-10);
        assert!((arc.end().0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_quarter_circle() {
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ),
            Vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let expected_len = PI / 2.0;
        assert!(
            (arc.length() - expected_len).abs() < 1e-6,
            "length {} != {}",
            arc.length(),
            expected_len
        );
    }

    #[test]
    fn t_major_arc() {
        // Arc going the long way (> π) from (1,0,0) through (0,-1,0) to (-1,0,0).
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, -1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
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
        assert!((mid.0).abs() < 1e-6);
        assert!((mid.1 + 1.0).abs() < 1e-6, "mid.y = {}", mid.1);
    }

    #[test]
    fn t_reversed() {
        let arc = Arc::from_three_points(
            Vec3(-1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        )
        .unwrap();
        let rev = arc.clone().reversed();
        let len = arc.length();
        assert!((len - rev.length()).abs() < 1e-12);
        assert!((arc.start().0 - rev.end().0).abs() < 1e-10);
        assert!((arc.end().0 - rev.start().0).abs() < 1e-10);
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
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
            // Arc-length parameterization ⇒ unit tangent.
            let mag = tan.length();
            assert!(
                (mag - 1.0).abs() < 1e-6,
                "tangent not unit length at t={t}: |tan|={mag}"
            );
        }
    }

    #[test]
    fn t_unit_speed_parameterization() {
        // Arc-length parameterization means |dp/dt| = 1 everywhere.
        let arc = Arc::from_three_points(
            Vec3(2.0, 0.0, 0.0),
            Vec3(0.0, 2.0, 0.0),
            Vec3(-2.0, 0.0, 0.0),
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
            Arc::from_three_points(
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(2.0, 0.0, 0.0),
            )
            .is_err()
        );
    }

    #[test]
    fn t_3d_arc() {
        // Arc not in a coordinate plane.
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 1.0),
            Vec3(-1.0, 0.0, 0.0),
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
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
        let cases: &[(Vec3, Vec3, Vec3, f64)] = &[
            // Semicircle, r=3, XY plane
            (
                Vec3(3.0, 0.0, 0.0),
                Vec3(0.0, 3.0, 0.0),
                Vec3(-3.0, 0.0, 0.0),
                0.01,
            ),
            // Small quarter-circle, r=1, XY plane
            (
                Vec3(1.0, 0.0, 0.0),
                Vec3(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
                Vec3(0.0, 1.0, 0.0),
                0.001,
            ),
            // Large radius, r=100, tight tolerance
            (
                Vec3(100.0, 0.0, 0.0),
                Vec3(0.0, 100.0, 0.0),
                Vec3(-100.0, 0.0, 0.0),
                0.1,
            ),
            // Tiny radius, r≈0.1
            (
                Vec3(0.1, 0.0, 0.0),
                Vec3(0.0, 0.1, 0.0),
                Vec3(-0.1, 0.0, 0.0),
                0.001,
            ),
            // 3D arc, not in a coordinate plane
            (
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.0, 1.0, 1.0),
                Vec3(-1.0, 0.0, 0.0),
                0.01,
            ),
            // Major arc (going the long way)
            (
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.0, -1.0, 0.0),
                Vec3(-1.0, 0.0, 0.0),
                0.01,
            ),
            // Offset center
            (
                Vec3(10.0, 5.0, 0.0),
                Vec3(7.0, 8.0, 0.0),
                Vec3(4.0, 5.0, 0.0),
                0.005,
            ),
        ];
        for (i, &(p0, p1, p2, tol)) in cases.iter().enumerate() {
            let arc = Arc::from_three_points(p0, p1, p2).unwrap();
            let r = arc.radius();
            let center = arc.center();
            let samples: Vec<Vec3> = arc.adaptive_samples(tol).collect();
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let len = arc.length();
        let mut results = [Vec3(0.0, 0.0, 0.0); 3];
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
        let arc = Arc::from_three_points(
            Vec3(3.0, 0.0, 0.0),
            Vec3(0.0, 3.0, 0.0),
            Vec3(-3.0, 0.0, 0.0),
        )
        .unwrap();
        let r = arc.radius();
        let len = arc.length();
        let mut results = [Vec3(0.0, 0.0, 0.0); 3];
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let mut results = [Vec3(0.0, 0.0, 0.0); 2];
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let mut results = [Vec3(0.0, 0.0, 0.0); 1];
        arc.point_with_derivs(0.0, &mut results).unwrap();
        let p = arc.point(0.0).unwrap();
        assert!((results[0] - p).length() < 1e-12);
    }

    #[test]
    fn t_point_with_derivs_higher_derivs_zero() {
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let mut results = [Vec3(1.0, 1.0, 1.0); 5]; // pre-fill with non-zero
        arc.point_with_derivs(arc.length() / 2.0, &mut results)
            .unwrap();
        for (i, r) in results[3..].iter().enumerate() {
            assert_eq!(*r, Vec3(0.0, 0.0, 0.0), "results[{}] not zero", i + 3);
        }
    }

    #[test]
    fn t_point_with_derivs_3d_arc() {
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 1.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let r = arc.radius();
        let len = arc.length();
        let mut results = [Vec3(0.0, 0.0, 0.0); 3];
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
    fn assert_arc_valid(arc: &Arc, expected_start: Vec3, expected_end: Vec3, tol: f64) {
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
        let arc = Arc::from_start_tangent_end(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6);
        assert!((arc.length() - PI).abs() < 1e-6);
        assert_arc_valid(&arc, Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 1e-6);
    }

    #[test]
    fn t_start_tangent_end_quarter_circle() {
        // Tangent at (1,0,0) pointing up, end at (0,1,0) → quarter circle.
        let arc = Arc::from_start_tangent_end(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-6, "r={}", arc.radius());
        assert!(
            (arc.length() - PI / 2.0).abs() < 1e-6,
            "len={}",
            arc.length()
        );
        assert_arc_valid(&arc, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1e-6);
    }

    #[test]
    fn t_start_tangent_end_matches_from_three_points() {
        // Build an arc from three points, get its start tangent, then reconstruct.
        let arc1 = Arc::from_three_points(
            Vec3(2.0, 0.0, 0.0),
            Vec3(0.0, 2.0, 0.0),
            Vec3(-2.0, 0.0, 0.0),
        )
        .unwrap();
        let tan = arc1.tangent(0.0).unwrap();
        let arc2 =
            Arc::from_start_tangent_end(Vec3(2.0, 0.0, 0.0), tan, Vec3(-2.0, 0.0, 0.0)).unwrap();
        assert!((arc1.radius() - arc2.radius()).abs() < 1e-6);
        assert!((arc1.length() - arc2.length()).abs() < 1e-6);
        assert!((arc1.center() - arc2.center()).length() < 1e-6);
    }

    #[test]
    fn t_start_tangent_end_collinear_fails() {
        // Tangent parallel to chord.
        assert!(
            Arc::from_start_tangent_end(
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(2.0, 0.0, 0.0),
            )
            .is_err()
        );
    }

    #[test]
    fn t_start_tangent_end_coincident_fails() {
        // Start == end.
        assert!(
            Arc::from_start_tangent_end(
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.0, 1.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
            )
            .is_err()
        );
    }

    // ===================== from_center_normal_start_angle =====================

    #[test]
    fn t_center_normal_start_angle_quarter() {
        let arc = Arc::from_center_normal_start_angle(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 0.0),
            PI / 2.0,
        )
        .unwrap();
        assert!((arc.radius() - 1.0).abs() < 1e-10);
        assert!((arc.length() - PI / 2.0).abs() < 1e-10);
        assert_arc_valid(&arc, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1e-6);
    }

    #[test]
    fn t_center_normal_start_angle_negative() {
        // Negative angle = CW.
        let arc = Arc::from_center_normal_start_angle(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(1.0, 0.0, 0.0),
            -PI / 2.0,
        )
        .unwrap();
        assert!((arc.length() - PI / 2.0).abs() < 1e-10);
        // End should be at (0,-1,0) for CW quarter.
        assert!((arc.end() - Vec3(0.0, -1.0, 0.0)).length() < 1e-6);
        assert_arc_valid(&arc, Vec3(1.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0), 1e-6);
    }

    #[test]
    fn t_center_normal_start_angle_large_arc() {
        // 270 degrees.
        let arc = Arc::from_center_normal_start_angle(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(5.0, 0.0, 0.0),
            3.0 * PI / 2.0,
        )
        .unwrap();
        assert!((arc.radius() - 5.0).abs() < 1e-10);
        assert!((arc.length() - 5.0 * 3.0 * PI / 2.0).abs() < 1e-6);
        assert_arc_valid(&arc, Vec3(5.0, 0.0, 0.0), Vec3(0.0, -5.0, 0.0), 1e-6);
    }

    #[test]
    fn t_center_normal_start_angle_full_circle_fails() {
        assert!(
            Arc::from_center_normal_start_angle(
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(1.0, 0.0, 0.0),
                TAU,
            )
            .is_err()
        );
    }

    #[test]
    fn t_center_normal_start_angle_zero_radius_fails() {
        assert!(
            Arc::from_center_normal_start_angle(
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 0.0), // start == center
                PI / 2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn t_center_normal_start_angle_zero_normal_fails() {
        assert!(
            Arc::from_center_normal_start_angle(
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                PI / 2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn t_center_normal_start_angle_not_in_plane_fails() {
        // Start not perpendicular to normal.
        assert!(
            Arc::from_center_normal_start_angle(
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(1.0, 0.0, 1.0), // has Z component
                PI / 2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn t_center_normal_start_angle_3d() {
        // Arc in a tilted plane.
        let normal = Vec3(1.0, 1.0, 1.0).normalize();
        // Start must be perpendicular to normal from center. Pick a point in the plane.
        let start_dir = Vec3(1.0, -1.0, 0.0).normalize();
        let center = Vec3(5.0, 5.0, 5.0);
        let radius = 3.0;
        let start = center + start_dir * radius;
        let arc = Arc::from_center_normal_start_angle(center, normal, start, PI / 3.0).unwrap();
        assert!((arc.radius() - radius).abs() < 1e-6);
        assert_arc_valid(&arc, start, arc.end(), 1e-6);
    }

    // ── bounds tests ─────────────────────────────────────────────────

    /// Verify bounds by sampling: every sampled point must lie within bounds,
    /// and at least one sample must be near each face.
    fn verify_bounds(arc: &Arc, n_samples: usize) {
        let (lo, hi) = arc.bounds();
        let lo = [lo.0, lo.1, lo.2];
        let hi = [hi.0, hi.1, hi.2];
        let len = arc.length();
        let mut observed_min = [f64::INFINITY; 3];
        let mut observed_max = [f64::NEG_INFINITY; 3];
        for i in 0..=n_samples {
            let t = len * i as f64 / n_samples as f64;
            let p = arc.point(t).unwrap();
            let coords = [p.0, p.1, p.2];
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
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        assert!((lo.0 - 0.0).abs() < 1e-10);
        assert!((lo.1 - 0.0).abs() < 1e-10);
        assert!((hi.0 - 1.0).abs() < 1e-10);
        assert!((hi.1 - 1.0).abs() < 1e-10);
        assert!((lo.2).abs() < 1e-10);
        assert!((hi.2).abs() < 1e-10);
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_semicircle() {
        // Semicircle from (-1,0,0) through (0,1,0) to (1,0,0).
        let arc = Arc::from_three_points(
            Vec3(-1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        assert!((lo.0 - -1.0).abs() < 1e-10);
        assert!((hi.0 - 1.0).abs() < 1e-10);
        assert!((lo.1 - 0.0).abs() < 1e-10);
        assert!((hi.1 - 1.0).abs() < 1e-10);
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_major_arc() {
        // 270-degree arc: from (1,0,0) through (0,-1,0) to (-1,0,0)... actually
        // through (0,-1,0) to get the "long way around".
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, -1.0, 0.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        .unwrap();
        let (lo, hi) = arc.bounds();
        // This arc goes through (0,-1,0) and (1,0,0) to (-1,0,0), sweeping CW.
        // It should include y=-1 and x=1, x=-1.
        assert!(lo.1 <= -1.0 + 1e-10);
        assert!(hi.0 >= 1.0 - 1e-10);
        verify_bounds(&arc, 2000);
    }

    #[test]
    fn t_bounds_3d_arc() {
        // Arc in 3D: not axis-aligned.
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 1.0).normalize(),
            Vec3(0.0, 0.0, 1.0),
        )
        .unwrap();
        verify_bounds(&arc, 2000);
    }

    #[test]
    fn t_bounds_offset_center() {
        // Arc with center not at origin.
        let arc = Arc::from_three_points(
            Vec3(10.0, 5.0, 0.0),
            Vec3(10.5, 5.5, 0.0),
            Vec3(11.0, 5.0, 0.0),
        )
        .unwrap();
        verify_bounds(&arc, 1000);
    }

    #[test]
    fn t_bounds_reversed_matches() {
        let arc = Arc::from_three_points(
            Vec3(1.0, 0.0, 0.0),
            Vec3(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let (lo1, hi1) = arc.bounds();
        let (lo2, hi2) = arc.reversed().bounds();
        assert!((lo1.0 - lo2.0).abs() < 1e-10);
        assert!((lo1.1 - lo2.1).abs() < 1e-10);
        assert!((lo1.2 - lo2.2).abs() < 1e-10);
        assert!((hi1.0 - hi2.0).abs() < 1e-10);
        assert!((hi1.1 - hi2.1).abs() < 1e-10);
        assert!((hi1.2 - hi2.2).abs() < 1e-10);
    }

    #[test]
    fn arc_bounds_failing_special_case() {
        // This test case came up during manual testing and exposed a bug.
        let arc = Arc::from_three_points(
            Vec3(-4.3480377197265625, 2.220446049250313e-16, 0.0),
            Vec3(-2.3053700923919678, 1.0424094200134277, 0.35949984192848206),
            Vec3(-2.5, 2.220446049250313e-16, 0.0),
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
            min_x = min_x.min(pt.0);
            assert!(
                pt.0 >= lo.0 - f64::EPSILON && pt.0 <= hi.0 + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt.0,
                lo.0,
                hi.0
            );
            assert!(
                pt.1 >= lo.1 - f64::EPSILON && pt.1 <= hi.1 + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt.1,
                lo.1,
                hi.1
            );
            assert!(
                pt.2 >= lo.2 - f64::EPSILON && pt.2 <= hi.2 + f64::EPSILON,
                "Point at t={t} has x={}; Which is outside [{}, {}]",
                pt.2,
                lo.2,
                hi.2
            );
        }
        eprintln!("bounds lo.x = {}, actual min_x = {}", lo.0, min_x);
        eprintln!("angle = {}, radius = {}", arc.angle(), arc.radius());
        eprintln!("center = {:?}", arc.center());
    }
}
