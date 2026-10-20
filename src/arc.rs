use crate::{
    Adaptor, ScalarAdaptor, SerialAdaptor, TrigonometryAdaptor,
    error::Error,
    vec::{F32Adaptor, F64Adaptor},
};
use core::f64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};

pub type Arc2d = Arc<2, F64Adaptor>;
pub type Arc3d = Arc<3, F64Adaptor>;
pub type Arc2f = Arc<2, F32Adaptor>;
pub type Arc3f = Arc<3, F32Adaptor>;

pub type EllipticArc2d = EllipticArc<2, F64Adaptor>;
pub type EllipticArc3d = EllipticArc<3, F64Adaptor>;
pub type EllipticArc2f = EllipticArc<2, F32Adaptor>;
pub type EllipticArc3f = EllipticArc<3, F32Adaptor>;

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
    angle: A::Scalar, // Sweep angle from start to end through mid.
}

#[derive(Clone, Debug)]
pub struct EllipticArc<const DIM: usize, A>
where
    A: Adaptor<DIM>,
{
    center: A::Vector,
    start_vec: A::Vector, // Vector from center toward start.
    mid_vec: A::Vector,   // Vector from center toward midpoint (used for antipodal SLERP).
    end_vec: A::Vector,   // Vector from center toward end.
    /// Sweep of the *parameter* angle, not the visible angle between `start_vec` and `end_vec`
    /// (the two agree only for circles). See ELLIPSE ARC MATH §1 above the `impl`.
    angle: A::Scalar,
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
            start_dir: self.end_dir,
            end_dir: self.start_dir,
            mid_dir: self.mid_dir,
            radius: self.radius,
            angle: self.angle,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.start() == self.end()
    }

    pub fn serialize(&self, mut w: impl std::io::Write) -> Result<(), std::io::Error>
    where
        A: SerialAdaptor<Value = A::Scalar>,
    {
        // Write out the scalars of center, start_dir, mid_dir, end_dir, radius, angle, in that order.
        for v in [self.center, self.start_dir, self.mid_dir, self.end_dir] {
            for coord in A::coord_arr(v) {
                A::write(coord, &mut w)?;
            }
        }
        A::write(self.radius, &mut w)?;
        A::write(self.angle, &mut w)?;
        Ok(())
    }

    pub fn deserialize(mut src: impl std::io::Read) -> Result<Self, std::io::Error>
    where
        A: SerialAdaptor<Value = A::Scalar>,
    {
        // Deserialize the same order what was written out by serialize.
        let mut vecs = [A::zero_vector(); 4];
        for v in vecs.iter_mut() {
            let mut coords = [A::scalar(0.0); DIM];
            for c in coords.iter_mut() {
                *c = A::read(&mut src)?;
            }
            *v = A::vector(coords);
        }
        let [center, start_dir, mid_dir, end_dir] = vecs;
        let radius = A::read(&mut src)?;
        let angle = A::read(&mut src)?;
        Ok(Self {
            center,
            start_dir,
            mid_dir,
            end_dir,
            radius,
            angle,
        })
    }
}

impl<const DIM: usize, A> Default for Arc<DIM, A>
where
    A: Adaptor<DIM>,
{
    /// A unit-radius quarter arc at the origin, swept from the first axis
    /// toward the second (all higher axes zero).
    fn default() -> Self {
        // An arc needs a 2-plane to sweep in: below that, `end_dir` can't be
        // a unit vector distinct from `start_dir` (this is also why every
        // fallible constructor, e.g. `from_three_points`, can only ever
        // return `Error::PointsCollinear` for `DIM < 2`).
        const {
            assert!(DIM >= 2, "Arc<DIM, A>::default() requires DIM >= 2");
        }
        Arc {
            center: A::zero_vector(),
            start_dir: A::vector(std::array::from_fn(|i| {
                A::scalar(if i == 0 { 1.0 } else { 0.0 })
            })),
            mid_dir: A::vector(std::array::from_fn(|i| {
                A::scalar(if i < 2 { FRAC_1_SQRT_2 } else { 0.0 })
            })),
            end_dir: A::vector(std::array::from_fn(|i| {
                A::scalar(if i == 1 { 1.0 } else { 0.0 })
            })),
            radius: A::scalar(1.0),
            angle: A::scalar(FRAC_PI_2),
        }
    }
}

/*
ELLIPSE ARC MATH

Derivations for `EllipticArc`. Code comments refer to the sections below as "§N".

Notation: c = center, S = start_vec, T = mid_vec, θ = angle, h = θ/2.
"·" is the dot product and |x|² = x·x.

§1 The model

   Every point on the (full) ellipse is
       P(φ) = c + cos φ · U + sin φ · V,
   for two vectors U, V and a parameter angle φ. U and V need not be perpendicular, so
   they are generally NOT the axes of the ellipse.

   The arc is the part with φ in [0, θ]. It starts at φ = 0, passes T at φ = h, and ends
   at φ = θ, so
       U = S,    T = cos h · U + sin h · V,    end_vec = cos θ · U + sin θ · V.

   `point` evaluates exactly this curve, using slerp weights on each half of the sweep.

   `angle` is this parameter sweep θ. It is generally NOT the visible angle between S and
   end_vec; the two agree only for circles. For example, squash a unit circle to half its
   height and take the arc from 0° to 60°: θ is still 60°, but the visible angle between
   the squashed start and end is about 40.9°. Many different ellipses share the same
   center, start and end; θ is what picks one of them. Projecting a circular arc onto a
   plane is linear, so it keeps θ unchanged (see `from_projected_arc`).

§2 Recovering V

   Subtract cos h · U from both sides of the equation for T in §1 and divide by sin h:
       V = (T - cos h · S) / sin h.
   (Using T and not end_vec keeps the divisor sin h, which is only zero for an empty arc.
   end_vec would need sin θ, which is zero at a semicircle.)

§3 Distance from the center, and the axes

   3.1 Squared distance as a function of φ.
       Let a = U·U, b = V·V, d = U·V. Expanding P - c = cos φ U + sin φ V:
           |P - c|² = cos²φ · a + 2 sin φ cos φ · d + sin²φ · b.
       Use the double-angle identities
           cos²φ = (1 + cos 2φ)/2,   sin²φ = (1 - cos 2φ)/2,   2 sin φ cos φ = sin 2φ:
           |P - c|² = (a + b)/2 + ((a - b)/2) · cos 2φ + d · sin 2φ.

   3.2 Collapse the cos/sin pair into one cosine.
       Let r = sqrt((a - b)² + 4d²). Then ((a - b)/r, 2d/r) is a point on the unit circle,
       so there is an angle φ0 with
           cos 2φ0 = (a - b)/r,   sin 2φ0 = 2d/r,   i.e.   φ0 = atan2(2d, a - b) / 2.
       Substitute (a - b) = r cos 2φ0 and 2d = r sin 2φ0 into 3.1:
           ((a - b)/2) cos 2φ + d sin 2φ
               = (r/2) (cos 2φ0 cos 2φ + sin 2φ0 sin 2φ)
               = (r/2) cos(2φ - 2φ0)          [cos(x - y) = cos x cos y + sin x sin y]
       so
           |P - c|² = (a + b)/2 + (r/2) · cos(2(φ - φ0)).
       A cosine ranges over [-1, 1], hence
           farthest point:  φ = φ0,        |P - c|² = A² = (a + b + r)/2,
           nearest point:   φ = φ0 + π/2,  |P - c|² = B² = (a + b - r)/2.
       The farthest point is the end of the semi-major axis (length A), the nearest the
       end of the semi-minor axis (length B). Note A² - B² = r, and A² B² = ab - d².

   3.3 The axis frame.
       Let K = P(φ0) - c (semi-major vector, |K| = A) and L = P(φ0 + π/2) - c (semi-minor
       vector, |L| = B). With ψ = φ - φ0 and the angle-addition formulas
           cos(φ0 + ψ) = cos φ0 cos ψ - sin φ0 sin ψ
           sin(φ0 + ψ) = sin φ0 cos ψ + cos φ0 sin ψ
       we get
           P - c = cos(φ0 + ψ) U + sin(φ0 + ψ) V
                 = cos ψ · (cos φ0 U + sin φ0 V) + sin ψ · (-sin φ0 U + cos φ0 V)
                 = cos ψ · K + sin ψ · L.
       K and L are perpendicular: P'(φ) = -sin φ U + cos φ V is exactly P(φ + π/2) - c.
       |P - c|² has derivative 2 (P - c)·P', which is zero at the extremum φ0, so
       K · P'(φ0) = K · L = 0.

§4 Foci

   4.1 Distance of the foci from the center.
       The foci lie on the major axis, on either side of the center, at distance
           f = sqrt(A² - B²) = sqrt(r)          (A² - B² = r, see 3.2).
       For a circle a = b and d = 0, so r = 0 and both foci are the center.

   4.2 The major axis direction without any trig.
       We need the vector W = K = x·U + y·V, where (x, y) = (cos φ0, sin φ0).
       Note |x·U + y·V|² = x²a + 2xy·d + y²b, the quadratic form of the table
           G = [ a  d ]
               [ d  b ]
       evaluated at (x, y). By 3.2, (x, y) maximizes it among unit (x, y), and the maximum
       is A². This is the "largest eigenvalue" problem: (x, y) satisfies
       G·(x, y) = λ·(x, y) with λ = A². We check this directly:
         * Eigenvalues solve det(G - λI) = 0, i.e. (a - λ)(b - λ) - d² = 0, i.e.
               λ² - (a + b)λ + (ab - d²) = 0
               λ = ((a + b) ± sqrt((a + b)² - 4(ab - d²))) / 2
                 = ((a + b) ± sqrt((a - b)² + 4d²)) / 2 = (a + b ± r)/2,
           which are exactly A² and B² from 3.2.
         * For λ = A², the two rows of G·(x, y) = λ·(x, y) read
               (a - λ)x + d·y = 0   =>   (x, y) ∝ (d, λ - a)
               d·x + (b - λ)y = 0   =>   (x, y) ∝ (λ - b, d).
           Both describe the same direction because det(G - λI) = 0.
       With λ - b = (a - b + r)/2 and λ - a = (b - a + r)/2, one of the two candidates
       always has a component of at least r/2 (the first if a >= b, the second otherwise),
       which avoids subtracting two nearly equal numbers. We pick that one. No
       normalization of (x, y) is needed, see 4.3.

   4.3 Normalizing W without computing its length separately.
       Because G·(x, y) = λ·(x, y),
           |W|² = (x, y)·G·(x, y) = λ·(x² + y²).
       The unit major direction is W / sqrt(λ(x² + y²)), so the focal offset is
           f · W / |W| = W · sqrt(r) / sqrt(λ(x² + y²)) = W · sqrt(r / (λ(x² + y²))),
       and the foci are c + offset and c - offset.

§5 Bounding box

   Work one coordinate at a time. Let ui = U[i], vi = V[i]. That coordinate is
       P_i(φ) = c_i + ui · cos φ + vi · sin φ.

   5.1 Extrema of a coordinate are where its derivative is zero:
           dP_i/dφ = -ui · sin φ + vi · cos φ = 0   =>   ui · sin φ = vi · cos φ.
       This says the vector (cos φ, sin φ) is parallel to (ui, vi), so there are two
       solutions, half a turn apart:
           (cos φ, sin φ) = +(ui, vi)/n   and   (cos φ, sin φ) = -(ui, vi)/n,
       where n = sqrt(ui² + vi²).

   5.2 The value of the coordinate at each solution.
           +: P_i = c_i + ui·(ui/n) + vi·(vi/n) = c_i + (ui² + vi²)/n = c_i + n   (maximum)
           -: P_i = c_i - n                                                       (minimum)
       If n = 0 (the arc is flat along this coordinate) there is no extremum to find, and
       the endpoints already give the right answer.

   5.3 Is a solution inside the arc, i.e. is φ in [0, θ]?
       We avoid computing φ by looking at signs. For an angle φ:
           φ is "past the start" when sin φ >= 0 (φ in [0, π]),
           φ is "before the end" when sin(θ - φ) >= 0.
       Expand the second one with sin(x - y) = sin x cos y - cos x sin y:
           sin(θ - φ) = sin θ · cos φ - cos θ · sin φ.
       For the + solution, sin φ = vi/n and cos φ = ui/n. Dropping the positive factor
       1/n, which does not change a sign:
           sign(sin φ)       = sign(vi)
           sign(sin(θ - φ))  = sign(sin θ · ui - cos θ · vi)
       The - solution has both signs flipped.
       If θ <= π, φ is in [0, θ] exactly when BOTH conditions hold: the arc is no longer
       than half a turn, so being past the start and before the end pins it down.
       If θ > π, the two conditions each cover a part of the arc and overlap, so EITHER
       one is enough.
       Using strict inequalities is fine: an extremum exactly at an endpoint is already
       counted by the endpoints.

   5.4 Start from the box of the two endpoints, and widen it by each extremum that lies
       inside the arc. The sqrt in 5.2 is only evaluated when needed.

§6 Length

   6.1 Speed along the curve.
       Differentiate the axis frame form in 3.3 with respect to ψ (equivalently φ):
           P' = -sin ψ · K + cos ψ · L.
       Because K ⟂ L, the squared length has no cross term:
           |P'|² = sin²ψ |K|² + cos²ψ |L|² = A² sin²ψ + B² cos²ψ.

   6.2 Arc length is the integral of speed.
       ψ runs from -φ0 (at φ = 0) to θ - φ0 (at φ = θ), so
           length = ∫ sqrt(A² sin²ψ + B² cos²ψ) dψ   over ψ in [-φ0, θ - φ0].

   6.3 Put it in the standard form of an elliptic integral.
       Substitute t = ψ + π/2. Then sin ψ = -cos t and cos ψ = sin t, so
           A² sin²ψ + B² cos²ψ = A² cos²t + B² sin²t.
       Let q = B²/A² (in [0, 1]) and m = 1 - q. Using cos²t = 1 - sin²t:
           A² cos²t + B² sin²t = A² (1 - sin²t + q sin²t) = A² (1 - m sin²t).
       Define E(t) = ∫ from 0 to t of sqrt(1 - m sin²w) dw. Then
           length = A · (E(t1) - E(t0)),
           t0 = π/2 - φ0,   t1 = t0 + θ.
       When the ellipse is a circle, m = 0, E(t) = t, and the length is A · θ.

   6.4 Reduce t to the range where the formulas below are valid.
       The integrand sqrt(1 - m sin²w) has period π (sin² has period π), and is symmetric
       about w = π/2 (sin²(π - w) = sin²w). Hence the integral over one full period
       [0, π] is 2 E_c, where E_c = E(π/2) is the "complete" integral. Write
           t = n π + y,  n = round(t / π),  y in [-π/2, π/2],
       then E(t) = 2 n E_c + E(y). Here t is always positive: φ0 is in (-π/2, π/2], so t0
       is in [0, π), and t1 = t0 + θ < 3π, so n >= 0 and n is small.

   6.5 Evaluate E(y) with Carlson's integrals.
       Carlson's symmetric integrals R_F and R_D (see `carlson_rf`, `carlson_rd`) satisfy
       the standard identity (DLMF 19.25.9, not rederived here), for |y| <= π/2 with
       s = sin y, c = cos y:
           E(y) = s · R_F(c², Δ, 1) - (m/3) · s³ · R_D(c², Δ, 1),
           Δ = 1 - m s² = c² + q s²    (the second form has no cancellation).
       The complete integral is the same formula at y = π/2 (s = 1, c = 0):
           E_c = R_F(0, q, 1) - (m/3) · R_D(0, q, 1).
       R_F needs at most one zero argument, so q must be positive. A degenerate ellipse
       (B = 0, a back-and-forth line) is handled by clamping q to ε², which changes the
       length by far less than the rounding error.

   6.6 Why no tolerance is needed.
       Both R_F and R_D repeat a step that shrinks the relative spread of their arguments
       by a factor of 4. Once the spread is below a fixed threshold, a short Taylor series
       finishes the job, and its truncation error is below 1e-16 (see `carlson_rf`). The
       stopping test is the error bound itself.
*/

impl<const DIM: usize, A: Adaptor<DIM>> EllipticArc<DIM, A> {
    pub fn from_projected_arc(
        arc: &Arc<DIM, A>,
        plane_pt: A::Vector,
        plane_normal: A::Vector,
    ) -> Result<Self, Error>
    where
        A: TrigonometryAdaptor,
    {
        let plane_normal = A::normalize(plane_normal);
        let center =
            arc.center - plane_normal * A::dot_product(plane_normal, arc.center - plane_pt);
        let [start_vec, mid_vec, end_vec] = [
            arc.start_dir * arc.radius,
            arc.mid_dir * arc.radius,
            arc.end_dir * arc.radius,
        ]
        .map(|v| v - plane_normal * A::dot_product(plane_normal, v));
        if [start_vec, mid_vec, end_vec]
            .iter()
            .any(|v| A::vector_length(*v) < A::epsilon())
        {
            return Err(Error::DegenerateValue);
        }
        // Projection is linear, so projecting the circle's slerp combination of
        // start/mid/end equals the same slerp combination of the projected vectors.
        // The slerp weights must therefore use the original circle's sweep angle, not
        // the (distorted) angle between the projected vectors.
        Ok(Self {
            center,
            start_vec,
            mid_vec,
            end_vec,
            angle: arc.angle,
        })
    }

    pub fn from_center_start_end(
        center: A::Vector,
        start: A::Vector,
        end: A::Vector,
        flip_dir: bool,
    ) -> Result<Self, Error>
    where
        A: TrigonometryAdaptor,
    {
        let start_vec = start - center;
        if A::vector_length(start - end) < A::epsilon() {
            return Err(Error::DegenerateValue);
        }
        let end_vec = end - center;
        if A::vector_length(start_vec) < A::epsilon() || A::vector_length(end_vec) < A::epsilon() {
            return Err(Error::DegenerateValue);
        }
        let mut angle = A::acos(A::clamp(
            A::dot_product(A::normalize(start_vec), A::normalize(end_vec)),
            A::scalar(-1.0),
            A::scalar(1.0),
        ));
        if angle < A::epsilon() || angle >= (A::scalar(PI) - A::epsilon()) {
            return Err(Error::DegenerateValue);
        }
        let [tstart, tend] = slerp_raw::<A>(angle, A::scalar(0.5));
        let mut mid_vec = tstart * start_vec + tend * end_vec;
        if A::vector_length(mid_vec) < A::epsilon() {
            return Err(Error::DegenerateValue);
        }
        // Flip the direction if necessary.
        if flip_dir {
            angle = A::scalar(TAU) - angle;
            mid_vec = -mid_vec;
        }
        Ok(Self {
            center,
            start_vec,
            mid_vec,
            end_vec,
            angle,
        })
    }

    pub fn domain(&self) -> (A::Scalar, A::Scalar) {
        (A::scalar(0.0), A::scalar(1.0))
    }

    /// Arc length. There is no elementary closed form for an ellipse; this evaluates the
    /// incomplete elliptic integral of the second kind with Carlson's algorithm, which
    /// converges to the full precision of the scalar type with no tolerance parameter.
    pub fn length(&self) -> A::Scalar
    where
        A: TrigonometryAdaptor,
    {
        // See ELLIPSE ARC MATH §6.
        let (sin_h, cos_h) = A::sin_cos(self.angle * A::scalar(0.5));
        if sin_h < A::epsilon() {
            // Degenerate arc. The chord is the length, to within rounding.
            return A::vector_length(self.end_vec - self.start_vec);
        }
        // U and V (§1, §2), then a, b, d, r (§3).
        let u = self.start_vec;
        let v = (self.mid_vec - u * cos_h) / sin_h;
        let a = A::dot_product(u, u);
        let b = A::dot_product(v, v);
        let d = A::dot_product(u, v);
        let diff = a - b;
        let half = A::scalar(0.5);
        let r = A::sqrt(diff * diff + A::scalar(4.0) * d * d);
        if r <= A::epsilon() * (a + b) {
            // Circle (to within precision). Constant speed.
            return A::sqrt((a + b) * half) * self.angle;
        }
        let lambda = (a + b + r) * half; // A², §3.2
        // q = B²/A² = (a b - d²)/A⁴ by §3.2. Clamped away from zero, see §6.5.
        let det = A::max(a * b - d * d, A::scalar(0.0));
        let q = A::clamp(
            det / (lambda * lambda),
            A::epsilon() * A::epsilon(),
            A::scalar(1.0),
        );
        let m = A::scalar(1.0) - q;
        // Integration limits (§6.3), each split as t = n π + y with |y| <= π/2 (§6.4).
        let t0 = A::scalar(PI * 0.5) - half * A::atan2(A::scalar(2.0) * d, diff);
        let t1 = t0 + self.angle;
        let inv_pi = A::scalar(1.0 / PI);
        let reduce = |t: A::Scalar| {
            let n = A::to_usize(t * inv_pi + half);
            (n, t - A::scalar(n as f64 * PI))
        };
        let (n0, y0) = reduce(t0);
        let (n1, y1) = reduce(t1);
        let third_m = m * A::scalar(1.0 / 3.0);
        // E(y) in terms of Carlson's integrals (§6.5).
        let incomplete = |y: A::Scalar| {
            let (s, c) = A::sin_cos(y);
            let c2 = c * c;
            let delta = c2 + q * s * s;
            let one = A::scalar(1.0);
            s * carlson_rf::<A>(c2, delta, one)
                - third_m * s * s * s * carlson_rd::<A>(c2, delta, one)
        };
        let mut total = incomplete(y1) - incomplete(y0);
        if n1 > n0 {
            // Every whole period between the limits adds 2 E_c (§6.4, §6.5).
            let zero = A::scalar(0.0);
            let one = A::scalar(1.0);
            let complete = carlson_rf::<A>(zero, q, one) - third_m * carlson_rd::<A>(zero, q, one);
            total += A::scalar(2.0 * (n1 - n0) as f64) * complete;
        }
        // length = A · (E(t1) - E(t0)), §6.3.
        A::sqrt(lambda) * total
    }

    pub fn center(&self) -> A::Vector {
        self.center
    }

    /// The two foci of the ellipse this arc lies on. For a circle both foci are the center.
    pub fn focii(&self) -> (A::Vector, A::Vector)
    where
        A: TrigonometryAdaptor,
    {
        // See ELLIPSE ARC MATH §4.
        let (sin_h, cos_h) = A::sin_cos(self.angle * A::scalar(0.5));
        if sin_h < A::epsilon() {
            return (self.center, self.center);
        }
        // U and V (§1, §2), then a, b, d, r (§3).
        let u = self.start_vec;
        let v = (self.mid_vec - u * cos_h) / sin_h;
        let a = A::dot_product(u, u);
        let b = A::dot_product(v, v);
        let d = A::dot_product(u, v);
        let diff = a - b;
        let r = A::sqrt(diff * diff + A::scalar(4.0) * d * d);
        if r <= A::epsilon() * (a + b) {
            // Circle (to within precision). The direction of the major axis is undefined.
            return (self.center, self.center);
        }
        let half = A::scalar(0.5);
        let lambda = (a + b + r) * half; // A², §3.2
        // Major axis direction as x·U + y·V, using the better-conditioned eigenvector (§4.2).
        let (x, y) = if diff >= A::scalar(0.0) {
            ((diff + r) * half, d)
        } else {
            (d, (r - diff) * half)
        };
        // Distance sqrt(r) along the unit major direction (§4.1, §4.3).
        let offset = (u * x + v * y) * A::sqrt(r / (lambda * (x * x + y * y)));
        (self.center + offset, self.center - offset)
    }

    pub fn angle(&self) -> A::Scalar {
        self.angle
    }

    pub fn start(&self) -> A::Vector {
        self.center + self.start_vec
    }

    pub fn end(&self) -> A::Vector {
        self.center + self.end_vec
    }

    pub fn point(&self, t: A::Scalar) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        if t < A::scalar(0.0) || t > A::scalar(1.0) {
            return None;
        }
        let coeff = slerp::<A>(self.angle, t);
        Some(
            self.center
                + self.start_vec * coeff[0]
                + self.mid_vec * coeff[1]
                + self.end_vec * coeff[2],
        )
    }

    pub fn tangent(&self, t: A::Scalar) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        if t < A::scalar(0.0) || t > A::scalar(1.0) {
            return None;
        }
        let coeff = slerp_deriv::<A>(self.angle, t);
        Some(self.start_vec * coeff[0] + self.mid_vec * coeff[1] + self.end_vec * coeff[2])
    }

    pub fn curvature(&self) -> Option<A::Vector>
    where
        A: TrigonometryAdaptor,
    {
        todo!()
    }

    pub fn point_with_deriv(&self) -> Option<A::Scalar> {
        todo!()
    }

    /// Compute the axis-aligned bounding box of the arc.
    ///
    /// Returns `(min_corner, max_corner)`.
    pub fn bounds(&self) -> (A::Vector, A::Vector)
    where
        A: TrigonometryAdaptor,
    {
        // See ELLIPSE ARC MATH §5. Start from the endpoints (§5.4), then widen by the
        // extrema that lie inside the arc.
        let start = A::coord_arr(self.center + self.start_vec);
        let end = A::coord_arr(self.center + self.end_vec);
        let mut min: [A::Scalar; DIM] = std::array::from_fn(|i| A::min(start[i], end[i]));
        let mut max: [A::Scalar; DIM] = std::array::from_fn(|i| A::max(start[i], end[i]));
        let (sin_h, cos_h) = A::sin_cos(self.angle * A::scalar(0.5));
        if sin_h < A::epsilon() {
            // Degenerate arc. The endpoints are the whole story.
            return (A::vector(min), A::vector(max));
        }
        // U and V (§1, §2).
        let u = self.start_vec;
        let v = (self.mid_vec - u * cos_h) / sin_h;
        let (sin_t, cos_t) = A::sin_cos(self.angle);
        let short = self.angle <= A::scalar(PI);
        let zero = A::scalar(0.0);
        let center = A::coord_arr(self.center);
        let u = A::coord_arr(u);
        let v = A::coord_arr(v);
        for i in 0..DIM {
            let (ui, vi) = (u[i], v[i]);
            // Sign tests for "extremum is inside the arc" (§5.3).
            let past_start = vi;
            let before_end = sin_t * ui - cos_t * vi;
            let (max_in, min_in) = if short {
                (
                    past_start > zero && before_end > zero,
                    past_start < zero && before_end < zero,
                )
            } else {
                (
                    past_start > zero || before_end > zero,
                    past_start < zero || before_end < zero,
                )
            };
            if max_in || min_in {
                // Extremal values are c_i ± n (§5.2).
                let n = A::sqrt(ui * ui + vi * vi);
                if max_in {
                    max[i] = A::max(max[i], center[i] + n);
                }
                if min_in {
                    min[i] = A::min(min[i], center[i] - n);
                }
            }
        }
        (A::vector(min), A::vector(max))
    }

    pub fn reversed(&self) -> Self {
        Self {
            center: self.center,
            start_vec: self.end_vec,
            mid_vec: self.mid_vec,
            end_vec: self.start_vec,
            angle: self.angle,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.start() == self.end()
    }
}

impl<const DIM: usize, A: Adaptor<DIM>> From<Arc<DIM, A>> for EllipticArc<DIM, A> {
    fn from(arc: Arc<DIM, A>) -> Self {
        Self {
            center: arc.center,
            start_vec: arc.start_dir * arc.radius,
            mid_vec: arc.mid_dir * arc.radius,
            end_vec: arc.end_dir * arc.radius,
            angle: arc.angle,
        }
    }
}

impl<const DIM: usize, A: Adaptor<DIM>> From<&Arc<DIM, A>> for EllipticArc<DIM, A> {
    fn from(arc: &Arc<DIM, A>) -> Self {
        Self {
            center: arc.center,
            start_vec: arc.start_dir * arc.radius,
            mid_vec: arc.mid_dir * arc.radius,
            end_vec: arc.end_dir * arc.radius,
            angle: arc.angle,
        }
    }
}

/// 3d arcs have some special features that don't exist in other dimensions.
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

/// Maximum iterations for the Carlson duplication loops. Ordinary ellipses need 5-7 steps and
/// the flattest representable one (q clamped to ε², f64) needs 10, so this is only reached
/// for non-finite input. It guarantees termination.
const CARLSON_MAX_ITER: usize = 16;

/// Carlson's symmetric elliptic integral of the first kind,
/// R_F(x, y, z) = (1/2) ∫ from 0 to ∞ of dt / sqrt((t + x)(t + y)(t + z)).
/// Requires x, y, z >= 0 with at most one of them zero.
///
/// Duplication theorem: R_F(x, y, z) = R_F(x', y', z') where, with
/// λ = sqrt(x)sqrt(y) + sqrt(y)sqrt(z) + sqrt(z)sqrt(x), each argument becomes (x + λ)/4.
/// Each step shrinks the relative spread of the arguments around their mean by about 4.
/// Once the spread is below 0.0025, a Taylor series in the spread, truncated after the
/// fifth order, has error below 0.0025⁶ / 4 ≈ 6e-17.
fn carlson_rf<A: ScalarAdaptor>(x: A::Float, y: A::Float, z: A::Float) -> A::Float {
    const ERRTOL: f64 = 0.0025;
    let (mut x, mut y, mut z) = (x, y, z);
    let third = A::scalar(1.0 / 3.0);
    let quarter = A::scalar(0.25);
    let tol = A::scalar(ERRTOL);
    let mut iter = 0;
    loop {
        let ave = (x + y + z) * third;
        let (dx, dy, dz) = ((ave - x) / ave, (ave - y) / ave, (ave - z) / ave);
        let spread = A::max(A::abs(dx), A::max(A::abs(dy), A::abs(dz)));
        if spread < tol || iter >= CARLSON_MAX_ITER {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            let series = A::scalar(1.0)
                + (e2 * A::scalar(1.0 / 24.0) - A::scalar(0.1) - e3 * A::scalar(3.0 / 44.0)) * e2
                + e3 * A::scalar(1.0 / 14.0);
            return series / A::sqrt(ave);
        }
        let (sx, sy, sz) = (A::sqrt(x), A::sqrt(y), A::sqrt(z));
        let lambda = sx * (sy + sz) + sy * sz;
        x = (x + lambda) * quarter;
        y = (y + lambda) * quarter;
        z = (z + lambda) * quarter;
        iter += 1;
    }
}

/// Carlson's symmetric elliptic integral of the second kind,
/// R_D(x, y, z) = (3/2) ∫ from 0 to ∞ of dt / ((t + z) sqrt((t + x)(t + y)(t + z))).
/// Requires x, y >= 0 with at most one of them zero, and z > 0.
///
/// Uses the same duplication step as `carlson_rf`, accumulating a running sum. The
/// termination threshold is 0.0015 and the truncation error of the series is below
/// 0.0015⁶ ≈ 1e-17.
fn carlson_rd<A: ScalarAdaptor>(x: A::Float, y: A::Float, z: A::Float) -> A::Float {
    const ERRTOL: f64 = 0.0015;
    let (mut x, mut y, mut z) = (x, y, z);
    let quarter = A::scalar(0.25);
    let tol = A::scalar(ERRTOL);
    let mut sum = A::scalar(0.0);
    let mut fac = A::scalar(1.0);
    let mut iter = 0;
    loop {
        let ave = (x + y + z * A::scalar(3.0)) * A::scalar(0.2);
        let (dx, dy, dz) = ((ave - x) / ave, (ave - y) / ave, (ave - z) / ave);
        let spread = A::max(A::abs(dx), A::max(A::abs(dy), A::abs(dz)));
        if spread < tol || iter >= CARLSON_MAX_ITER {
            let c1 = A::scalar(3.0 / 14.0);
            let c2 = A::scalar(1.0 / 6.0);
            let c3 = A::scalar(9.0 / 22.0);
            let c4 = A::scalar(3.0 / 26.0);
            let c5 = c3 * quarter;
            let c6 = c4 * A::scalar(1.5);
            let ea = dx * dy;
            let eb = dz * dz;
            let ec = ea - eb;
            let ed = ea - eb * A::scalar(6.0);
            let ee = ed + ec + ec;
            let series = A::scalar(1.0)
                + ed * (-c1 + c5 * ed - c6 * dz * ee)
                + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea));
            return sum * A::scalar(3.0) + fac * series / (ave * A::sqrt(ave));
        }
        let (sx, sy, sz) = (A::sqrt(x), A::sqrt(y), A::sqrt(z));
        let lambda = sx * (sy + sz) + sy * sz;
        sum += fac / (sz * (z + lambda));
        fac *= quarter;
        x = (x + lambda) * quarter;
        y = (y + lambda) * quarter;
        z = (z + lambda) * quarter;
        iter += 1;
    }
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

    // ===================== EllipticArc::from_center_start_end =====================

    #[test]
    fn t_center_start_end_minor_arc() {
        // Quarter arc: center at origin, start on +X, end on +Y.
        let earc = EllipticArc3d::from_center_start_end(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            false,
        )
        .unwrap();
        assert!(
            (earc.angle - PI / 2.0).abs() < 1e-10,
            "angle = {}",
            earc.angle
        );
        // start_vec and end_vec should match the input vectors from center.
        assert!((earc.start_vec - DVec([1.0, 0.0, 0.0])).length() < 1e-10);
        assert!((earc.end_vec - DVec([0.0, 1.0, 0.0])).length() < 1e-10);
        // mid_vec should point toward (1,1,0) normalized * radius = (√2/2, √2/2, 0).
        let mid_dir = earc.mid_vec.normalize();
        let expected_mid_dir = DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]);
        assert!(
            (mid_dir - expected_mid_dir).length() < 1e-10,
            "mid_dir = {:?}",
            mid_dir
        );
        // mid_vec magnitude should equal the radius (circular case, equal magnitudes).
        assert!((earc.mid_vec.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_center_start_end_flip_dir() {
        let center = DVec([0.0, 0.0, 0.0]);
        let start = DVec([1.0, 0.0, 0.0]);
        let end = DVec([0.0, 1.0, 0.0]);
        let minor = EllipticArc3d::from_center_start_end(center, start, end, false).unwrap();
        let major = EllipticArc3d::from_center_start_end(center, start, end, true).unwrap();
        // Angles should sum to TAU.
        assert!(
            (minor.angle + major.angle - TAU).abs() < 1e-10,
            "minor={} major={}",
            minor.angle,
            major.angle
        );
        // mid_vecs should be negations of each other.
        assert!(
            (minor.mid_vec + major.mid_vec).length() < 1e-10,
            "minor.mid={:?} major.mid={:?}",
            minor.mid_vec,
            major.mid_vec
        );
        // start_vec and end_vec are the same for both.
        assert!((minor.start_vec - major.start_vec).length() < 1e-10);
        assert!((minor.end_vec - major.end_vec).length() < 1e-10);
    }

    #[test]
    fn t_center_start_end_degenerate() {
        // Coincident start and end (angle ≈ 0).
        assert!(
            EllipticArc3d::from_center_start_end(
                DVec([0.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                false,
            )
            .is_err()
        );
        // Antipodal start and end (angle ≈ π).
        assert!(
            EllipticArc3d::from_center_start_end(
                DVec([0.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([-1.0, 0.0, 0.0]),
                false,
            )
            .is_err()
        );
        // Center coincides with start.
        assert!(
            EllipticArc3d::from_center_start_end(
                DVec([1.0, 0.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([0.0, 1.0, 0.0]),
                false,
            )
            .is_err()
        );
        // Center coincides with end.
        assert!(
            EllipticArc3d::from_center_start_end(
                DVec([0.0, 1.0, 0.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([0.0, 1.0, 0.0]),
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn t_center_start_end_matches_arc_from_three_points() {
        // Build a circular arc via Arc::from_three_points, convert to EllipticArc,
        // and compare with from_center_start_end using the same center/start/end.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let from_arc: EllipticArc3d = (&arc).into();
        let from_cse =
            EllipticArc3d::from_center_start_end(arc.center(), arc.start(), arc.end(), false)
                .unwrap();
        assert!((from_arc.angle - from_cse.angle).abs() < 1e-10);
        assert!((from_arc.start_vec - from_cse.start_vec).length() < 1e-10);
        assert!((from_arc.end_vec - from_cse.end_vec).length() < 1e-10);
        // mid_vec directions should match (magnitudes are equal for circular case).
        assert!((from_arc.mid_vec.normalize() - from_cse.mid_vec.normalize()).length() < 1e-10);
    }

    #[test]
    fn t_center_start_end_3d_off_axis() {
        // Non-axis-aligned arc in 3D with offset center.
        let center = DVec([1.0, 2.0, 3.0]);
        let start = DVec([2.0, 2.0, 3.0]); // +X from center
        let end = DVec([1.0, 3.0, 3.0]); // +Y from center
        let earc = EllipticArc3d::from_center_start_end(center, start, end, false).unwrap();
        assert!((earc.angle - PI / 2.0).abs() < 1e-10);
        assert!((earc.center - center).length() < 1e-10);
        assert!((earc.start_vec - DVec([1.0, 0.0, 0.0])).length() < 1e-10);
        assert!((earc.end_vec - DVec([0.0, 1.0, 0.0])).length() < 1e-10);
    }

    // ===================== EllipticArc::from_projected_arc =====================

    #[test]
    fn t_projected_arc_identity() {
        // Projecting an arc in the XY plane onto the XY plane should give the same arc.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let earc =
            EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0]))
                .unwrap();
        let from_arc: EllipticArc3d = (&arc).into();
        assert!((earc.angle - from_arc.angle).abs() < 1e-10);
        assert!((earc.start_vec - from_arc.start_vec).length() < 1e-10);
        assert!((earc.mid_vec - from_arc.mid_vec).length() < 1e-10);
        assert!((earc.end_vec - from_arc.end_vec).length() < 1e-10);
        assert!((earc.center - from_arc.center).length() < 1e-10);
    }

    #[test]
    fn t_projected_arc_oblique() {
        // Arc in XY plane projected onto a tilted plane.
        // Semicircle r=1 in XY from (1,0,0) through (0,1,0) to (-1,0,0).
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        // Project onto a plane tilted 45° around X axis (normal = (0, -sin45, cos45)).
        let normal = DVec([0.0, -FRAC_1_SQRT_2, FRAC_1_SQRT_2]);
        let earc = EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 0.0]), normal).unwrap();
        // Start and end are on the X axis, so they're unaffected by Y/Z projection.
        assert!((earc.start_vec - DVec([1.0, 0.0, 0.0])).length() < 1e-10);
        assert!((earc.end_vec - DVec([-1.0, 0.0, 0.0])).length() < 1e-10);
        // The projected mid_vec should have reduced Y component.
        // Original mid_vec = (0, 1, 0). Projection removes component along normal.
        // proj = (0,1,0) - normal * dot(normal, (0,1,0))
        //      = (0,1,0) - (0,-√2/2,√2/2) * (-√2/2) = (0,1,0) - (0,0.5,-0.5) = (0,0.5,0.5)
        let expected_mid = DVec([0.0, 0.5, 0.5]);
        assert!(
            (earc.mid_vec - expected_mid).length() < 1e-10,
            "mid_vec = {:?}, expected {:?}",
            earc.mid_vec,
            expected_mid
        );
        // Projected angle: start_vec=(1,0,0), end_vec=(-1,0,0) → angle=π (semicircle preserved).
        assert!((earc.angle - PI).abs() < 1e-10);
    }

    #[test]
    fn t_projected_arc_major_arc_preserved() {
        // A major arc should remain a major arc after projection.
        // Arc in XY plane going the long way: start=(1,0,0), through (0,-1,0), to (0,1,0).
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        assert!(arc.angle() > PI, "precondition: arc should be major");
        // Project onto a plane tilted slightly from XY (normal nearly +Z).
        // This shrinks Z components but preserves all directions in XY.
        let earc =
            EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 0.0]), DVec([0.0, 0.0, 1.0]))
                .unwrap();
        assert!(
            earc.angle > PI,
            "projected major arc should remain major, got angle={}",
            earc.angle
        );
    }

    #[test]
    fn t_projected_arc_degenerate() {
        // Arc in XY plane projected along Y axis → all vectors collapse to X axis.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        // Project along the arc's plane normal: plane normal = (0,0,1) projected with
        // projection normal = (0,1,0) means removing Y. mid_dir=(0,1,0) becomes zero.
        let result =
            EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 0.0]), DVec([0.0, 1.0, 0.0]));
        assert!(result.is_err());
    }

    #[test]
    fn t_projected_arc_offset_plane() {
        // Verify projection onto an offset plane shifts the center correctly.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        // Project onto Z=5 plane (normal=(0,0,1), point=(0,0,5)).
        let earc =
            EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 5.0]), DVec([0.0, 0.0, 1.0]))
                .unwrap();
        // Arc center is at (0,0,0), projected to Z=5 → center should be (0,0,5).
        assert!(
            (earc.center - DVec([0.0, 0.0, 5.0])).length() < 1e-10,
            "center = {:?}",
            earc.center
        );
        // Vectors from center are projections of direction vectors (Z component removed),
        // so they should be unchanged since the arc is already in the XY plane.
        assert!((earc.start_vec - DVec([1.0, 0.0, 0.0])).length() < 1e-10);
    }

    #[test]
    fn t_projected_arc_matches_projected_points() {
        // Every point on the elliptic arc must be the projection of the corresponding
        // point on the original circular arc. The plane is tilted about an axis that is
        // not aligned with the arc, so projected angles differ from the original angles.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let plane_pt = DVec([0.0, 0.0, 0.0]);
        let normal = DVec([1.0, 1.0, 1.0]).normalize();
        let earc = EllipticArc3d::from_projected_arc(&arc, plane_pt, normal).unwrap();
        let project = |p: DVec3| p - normal * normal.dot(p - plane_pt);
        let len = arc.length();
        for i in 0..=16 {
            let t = i as f64 / 16.0;
            let expected = project(arc.point(t * len).unwrap());
            let actual = earc.point(t).unwrap();
            let err = (expected - actual).length();
            assert!(err < 1e-10, "t={t}: err={err}");
        }
    }

    // ===================== EllipticArc::focii =====================

    /// Build an elliptic arc from the swept-loop form `c + cos φ · u + sin φ · v`,
    /// covering `φ` in `[0, theta]`.
    fn swept_arc(c: DVec3, u: DVec3, v: DVec3, theta: f64) -> EllipticArc3d {
        let at = |phi: f64| u * phi.cos() + v * phi.sin();
        EllipticArc3d {
            center: c,
            start_vec: at(0.0),
            mid_vec: at(theta / 2.0),
            end_vec: at(theta),
            angle: theta,
        }
    }

    /// Foci are defined by: the sum of distances to them is the same for every point on
    /// the ellipse, and equals twice the semi-major length.
    fn assert_focal_sum_constant(earc: &EllipticArc3d, semi_major: f64) {
        let (f1, f2) = earc.focii();
        for i in 0..=32 {
            let p = earc.point(i as f64 / 32.0).unwrap();
            let sum = (p - f1).length() + (p - f2).length();
            assert!(
                (sum - 2.0 * semi_major).abs() < 1e-9,
                "i={i}: focal sum {sum} != {}",
                2.0 * semi_major
            );
        }
    }

    #[test]
    fn t_focii_circle() {
        let arc = Arc3d::from_three_points(
            DVec([3.0, 1.0, 0.0]),
            DVec([1.0, 3.0, 0.0]),
            DVec([-1.0, 1.0, 0.0]),
        )
        .unwrap();
        let earc: EllipticArc3d = (&arc).into();
        let (f1, f2) = earc.focii();
        assert!((f1 - arc.center()).length() < 1e-10);
        assert!((f2 - arc.center()).length() < 1e-10);
    }

    #[test]
    fn t_focii_axis_aligned() {
        // Semi-axes 5 and 3: foci at (+-4, 0, 0) from the center.
        let c = DVec([1.0, 2.0, 3.0]);
        let earc = swept_arc(c, DVec([5.0, 0.0, 0.0]), DVec([0.0, 3.0, 0.0]), 1.0);
        let (f1, f2) = earc.focii();
        let (lo, hi) = if f1[0] < f2[0] { (f1, f2) } else { (f2, f1) };
        assert!((lo - DVec([-3.0, 2.0, 3.0])).length() < 1e-10, "{lo:?}");
        assert!((hi - DVec([5.0, 2.0, 3.0])).length() < 1e-10, "{hi:?}");
        // Same ellipse, but the stored start lies on the minor axis instead.
        let earc = swept_arc(c, DVec([0.0, 3.0, 0.0]), DVec([-5.0, 0.0, 0.0]), 1.0);
        let (f1, f2) = earc.focii();
        let (lo, hi) = if f1[0] < f2[0] { (f1, f2) } else { (f2, f1) };
        assert!((lo - DVec([-3.0, 2.0, 3.0])).length() < 1e-10, "{lo:?}");
        assert!((hi - DVec([5.0, 2.0, 3.0])).length() < 1e-10, "{hi:?}");
    }

    #[test]
    fn t_focii_conjugate_pair() {
        // u = (1, 0), v = (1, 1): semi-axes are sqrt((3 +- sqrt5)/2) ~ 1.618, 0.618.
        let semi_major = ((3.0 + 5f64.sqrt()) / 2.0).sqrt();
        for theta in [0.5, 2.0, PI, 4.0, 5.5] {
            let earc = swept_arc(
                DVec([0.5, -1.0, 2.0]),
                DVec([1.0, 0.0, 0.0]),
                DVec([1.0, 1.0, 0.0]),
                theta,
            );
            assert_focal_sum_constant(&earc, semi_major);
        }
    }

    #[test]
    fn t_focii_projected_arc() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        // Projecting a unit circle onto a plane at angle t to its own plane gives semi-axes
        // 1 and |cos t|. The normal (1,1,1)/sqrt3 gives cos t = 1/sqrt3.
        let normal = DVec([1.0, 1.0, 1.0]).normalize();
        let earc = EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 0.0]), normal).unwrap();
        assert_focal_sum_constant(&earc, 1.0);
        let (f1, f2) = earc.focii();
        let expected_f = (1.0 - 1.0 / 3.0f64).sqrt();
        assert!(((f1 - earc.center()).length() - expected_f).abs() < 1e-9);
        assert!(((f2 - earc.center()).length() - expected_f).abs() < 1e-9);
    }

    #[test]
    fn t_focii_reversed() {
        let semi_major = ((3.0 + 5f64.sqrt()) / 2.0).sqrt();
        let earc = swept_arc(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([1.0, 1.0, 0.0]),
            2.5,
        );
        let rev = earc.reversed();
        assert_focal_sum_constant(&rev, semi_major);
        let (a1, a2) = earc.focii();
        let (b1, b2) = rev.focii();
        let same = (a1 - b1).length() + (a2 - b2).length();
        let swapped = (a1 - b2).length() + (a2 - b1).length();
        assert!(same.min(swapped) < 1e-9);
    }

    // ===================== EllipticArc::bounds =====================

    /// Bounding box of `n + 1` samples along the arc.
    fn sampled_bounds<const DIM: usize>(
        earc: &EllipticArc<DIM, F64Adaptor>,
        n: usize,
    ) -> ([f64; DIM], [f64; DIM]) {
        let mut min = [<f64>::INFINITY; DIM];
        let mut max = [<f64>::NEG_INFINITY; DIM];
        for i in 0..=n {
            let p = earc.point(i as f64 / n as f64).unwrap();
            for k in 0..DIM {
                min[k] = min[k].min(p.0[k]);
                max[k] = max[k].max(p.0[k]);
            }
        }
        (min, max)
    }

    /// The exact bounds must contain every sample, and be within sampling error of the
    /// sampled bounds.
    fn assert_bounds_match_sampling<const DIM: usize>(
        earc: &EllipticArc<DIM, F64Adaptor>,
        label: &str,
    ) {
        let (lo, hi) = earc.bounds();
        let (slo, shi) = sampled_bounds(earc, 20000);
        for k in 0..DIM {
            assert!(
                lo.0[k] <= slo[k] + 1e-12 && slo[k] - lo.0[k] < 1e-6,
                "{label}: min[{k}] = {}, sampled {}",
                lo.0[k],
                slo[k]
            );
            assert!(
                hi.0[k] >= shi[k] - 1e-12 && hi.0[k] - shi[k] < 1e-6,
                "{label}: max[{k}] = {}, sampled {}",
                hi.0[k],
                shi[k]
            );
        }
    }

    #[test]
    fn t_elliptic_bounds_axis_aligned() {
        // Full-ish ellipse, semi-axes 5 and 3 around (1, 2, 3).
        let c = DVec([1.0, 2.0, 3.0]);
        let earc = swept_arc(c, DVec([5.0, 0.0, 0.0]), DVec([0.0, 3.0, 0.0]), 6.0);
        let (lo, hi) = earc.bounds();
        // The sweep (6.0 of 2π) covers nearly the whole ellipse, including both ends of
        // each axis: the start itself is the +x extreme.
        assert!((lo[0] + 4.0).abs() < 1e-10, "{lo:?}");
        assert!((hi[0] - 6.0).abs() < 1e-10, "{hi:?}");
        assert!((lo[1] + 1.0).abs() < 1e-10, "{lo:?}");
        assert!((hi[1] - 5.0).abs() < 1e-10, "{hi:?}");
        assert!((lo[2] - 3.0).abs() < 1e-10 && (hi[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn t_elliptic_bounds_tilted_pairs() {
        let c = DVec([0.5, -1.0, 2.0]);
        let u = DVec([1.0, 0.0, 0.0]);
        let v = DVec([1.0, 1.0, 0.0]);
        // Includes a semicircle sweep and sweeps of every size.
        for theta in [0.3, 1.0, 1.5, PI / 2.0, 2.0, PI, 3.5, 4.0, 5.0, 5.5, 6.2] {
            assert_bounds_match_sampling(&swept_arc(c, u, v, theta), &format!("theta={theta}"));
        }
    }

    #[test]
    fn t_elliptic_bounds_out_of_plane_vectors() {
        // All three coordinates vary, and the pair is far from perpendicular.
        let c = DVec([-3.0, 4.0, 1.0]);
        let u = DVec([2.0, -1.0, 0.5]);
        let v = DVec([0.3, 1.7, -2.2]);
        for theta in [0.4, 1.2, PI, 2.8, 4.4, 5.9] {
            assert_bounds_match_sampling(&swept_arc(c, u, v, theta), &format!("theta={theta}"));
        }
    }

    #[test]
    fn t_elliptic_bounds_every_start_phase() {
        // Rotating the starting point around the ellipse moves the extrema in and out of
        // the arc. Sweep through all phases, for short, half, and long arcs.
        let c = DVec([0.0, 0.0, 0.0]);
        let u0 = DVec([3.0, 0.0, 0.0]);
        let v0 = DVec([0.0, 1.0, 0.0]);
        for theta in [1.0, PI, 4.5] {
            for k in 0..24 {
                let phase = TAU * k as f64 / 24.0;
                let (s, co) = phase.sin_cos();
                let u = u0 * co + v0 * s;
                let v = v0 * co - u0 * s;
                assert_bounds_match_sampling(
                    &swept_arc(c, u, v, theta),
                    &format!("theta={theta} phase={phase}"),
                );
            }
        }
    }

    #[test]
    fn t_elliptic_bounds_projected_arc() {
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let normal = DVec([1.0, 1.0, 1.0]).normalize();
        let earc = EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 1.0]), normal).unwrap();
        assert_bounds_match_sampling(&earc, "projected minor");
        let major = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let earc = EllipticArc3d::from_projected_arc(&major, DVec([0.0, 0.0, 0.0]), normal).unwrap();
        assert_bounds_match_sampling(&earc, "projected major");
    }

    #[test]
    fn t_elliptic_bounds_matches_circular_arc() {
        let cases = [
            // Minor, semicircle, major, and 3D tilted.
            ([1.0, 0.0, 0.0], [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0], [0.0, 1.0, 0.0]),
            ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]),
            ([3.0, 1.0, 2.0], [0.0, 2.0, 1.0], [-2.0, 1.0, 4.0]),
            ([10.0, 5.0, 0.0], [7.0, 8.0, 0.0], [4.0, 5.0, 0.0]),
        ];
        for (i, (a, b, c)) in cases.into_iter().enumerate() {
            let arc = Arc3d::from_three_points(DVec(a), DVec(b), DVec(c)).unwrap();
            let earc: EllipticArc3d = (&arc).into();
            let (alo, ahi) = arc.bounds();
            let (elo, ehi) = earc.bounds();
            assert!((alo - elo).length() < 1e-9, "case {i}: {alo:?} vs {elo:?}");
            assert!((ahi - ehi).length() < 1e-9, "case {i}: {ahi:?} vs {ehi:?}");
        }
    }

    #[test]
    fn t_elliptic_bounds_2d() {
        let at = |phi: f64| DVec([2.0 * phi.cos() + phi.sin(), 0.5 * phi.sin()]);
        let theta = 4.0;
        let earc = EllipticArc2d {
            center: DVec([1.0, 1.0]),
            start_vec: at(0.0),
            mid_vec: at(theta / 2.0),
            end_vec: at(theta),
            angle: theta,
        };
        assert_bounds_match_sampling(&earc, "2d");
    }

    #[test]
    fn t_elliptic_bounds_reversed() {
        let earc = swept_arc(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([1.0, 1.0, 0.0]),
            2.5,
        );
        let (lo, hi) = earc.bounds();
        let (rlo, rhi) = earc.reversed().bounds();
        assert!((lo - rlo).length() < 1e-9 && (hi - rhi).length() < 1e-9);
    }

    // ===================== EllipticArc::length =====================

    /// Reference length: Simpson's rule on |dP/dt| over t in [0, 1] with `n` (even) steps.
    fn simpson_length(earc: &EllipticArc3d, n: usize) -> f64 {
        let speed = |i: usize| earc.tangent(i as f64 / n as f64).unwrap().length();
        let mut sum = speed(0) + speed(n);
        for i in 1..n {
            sum += speed(i) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum / (3.0 * n as f64)
    }

    fn assert_length_matches_integration(earc: &EllipticArc3d, n: usize, tol: f64, label: &str) {
        let expected = simpson_length(earc, n);
        let actual = earc.length();
        assert!(
            (actual - expected).abs() <= tol * expected,
            "{label}: length {actual} != {expected} (rel err {})",
            (actual - expected).abs() / expected
        );
    }

    #[test]
    fn t_elliptic_length_circle() {
        let c = DVec([1.0, 2.0, 3.0]);
        for theta in [0.1, 1.0, PI, 4.0, 6.0] {
            let earc = swept_arc(c, DVec([2.0, 0.0, 0.0]), DVec([0.0, 2.0, 0.0]), theta);
            assert!(
                (earc.length() - 2.0 * theta).abs() < 1e-12,
                "theta={theta}: {}",
                earc.length()
            );
        }
        // From a circular Arc, including a major arc.
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let earc: EllipticArc3d = (&arc).into();
        assert!((earc.length() - arc.length()).abs() < 1e-12);
    }

    #[test]
    fn t_elliptic_length_tilted_pairs() {
        let c = DVec([0.5, -1.0, 2.0]);
        let u = DVec([1.0, 0.0, 0.0]);
        let v = DVec([1.0, 1.0, 0.0]);
        for theta in [0.01, 0.3, 1.0, PI / 2.0, 2.0, PI, 3.5, 4.0, 5.0, 5.5, 6.2] {
            assert_length_matches_integration(
                &swept_arc(c, u, v, theta),
                200_000,
                1e-11,
                &format!("theta={theta}"),
            );
        }
    }

    #[test]
    fn t_elliptic_length_out_of_plane_vectors() {
        let c = DVec([-3.0, 4.0, 1.0]);
        let u = DVec([2.0, -1.0, 0.5]);
        let v = DVec([0.3, 1.7, -2.2]);
        for theta in [0.4, 1.2, PI, 2.8, 4.4, 5.9] {
            assert_length_matches_integration(
                &swept_arc(c, u, v, theta),
                200_000,
                1e-11,
                &format!("theta={theta}"),
            );
        }
    }

    #[test]
    fn t_elliptic_length_every_start_phase() {
        // Rotating the start around the ellipse moves the arc across the axes, which
        // exercises every branch of the range reduction.
        let c = DVec([0.0, 0.0, 0.0]);
        let u0 = DVec([3.0, 0.0, 0.0]);
        let v0 = DVec([0.0, 1.0, 0.0]);
        for theta in [1.0, PI, 4.5, 6.0] {
            for k in 0..24 {
                let phase = TAU * k as f64 / 24.0;
                let (s, co) = phase.sin_cos();
                let u = u0 * co + v0 * s;
                let v = v0 * co - u0 * s;
                assert_length_matches_integration(
                    &swept_arc(c, u, v, theta),
                    50_000,
                    1e-11,
                    &format!("theta={theta} phase={phase}"),
                );
            }
        }
    }

    #[test]
    fn t_elliptic_length_projected_arc() {
        let normal = DVec([1.0, 1.0, 1.0]).normalize();
        let minor = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        let major = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
        )
        .unwrap();
        for (arc, label) in [(minor, "minor"), (major, "major")] {
            let earc =
                EllipticArc3d::from_projected_arc(&arc, DVec([0.0, 0.0, 1.0]), normal).unwrap();
            assert_length_matches_integration(&earc, 200_000, 1e-11, label);
        }
    }

    #[test]
    fn t_elliptic_length_thin_ellipses() {
        // Semi-axes 1 and ratio 1e-2, 1e-4, 1e-6: the speed has a sharp dip that the
        // reference integration only resolves for the milder ratios.
        for (ratio, n, tol) in [(1e-2, 200_000, 1e-10), (1e-4, 2_000_000, 1e-7)] {
            for theta in [1.0, 4.0] {
                let earc = swept_arc(
                    DVec([0.0, 0.0, 0.0]),
                    DVec([1.0, 0.0, 0.0]),
                    DVec([0.0, ratio, 0.0]),
                    theta,
                );
                assert_length_matches_integration(&earc, n, tol, &format!("ratio={ratio}"));
            }
        }
        // Extremely thin: bounded by the line length 4A, and by the semi-major sweep.
        let earc = swept_arc(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1e-9, 0.0]),
            3.0,
        );
        let expected = 1.0 - 3f64.cos(); // ∫ |sin φ| dφ over [0, 3]
        assert!((earc.length() - expected).abs() < 1e-8, "{}", earc.length());
    }

    #[test]
    fn t_elliptic_length_degenerate_line() {
        // V = 0: the "ellipse" is a segment of length 5 traversed back and forth.
        // P = (5 cos φ, 0, 0), speed 5|sin φ|, so length = 5 (1 - cos 3) for φ in [0, 3].
        let earc = swept_arc(
            DVec([0.0, 0.0, 0.0]),
            DVec([5.0, 0.0, 0.0]),
            DVec([0.0, 0.0, 0.0]),
            3.0,
        );
        let expected = 5.0 * (1.0 - 3f64.cos());
        assert!(
            (earc.length() - expected).abs() < 1e-9,
            "{} != {expected}",
            earc.length()
        );
    }

    #[test]
    fn t_elliptic_length_additive() {
        // Splitting the sweep at its middle: the two halves add up to the whole.
        let c = DVec([0.0, 1.0, 0.0]);
        let u = DVec([2.0, 0.5, 0.0]);
        let v = DVec([-0.3, 1.1, 0.7]);
        for theta in [1.0f64, 3.0, 5.0] {
            let h = theta / 2.0;
            let (s, co) = h.sin_cos();
            let first = swept_arc(c, u, v, h);
            let second = swept_arc(c, u * co + v * s, v * co - u * s, h);
            let sum = first.length() + second.length();
            let whole = swept_arc(c, u, v, theta).length();
            assert!((sum - whole).abs() < 1e-12 * whole, "theta={theta}: {sum} vs {whole}");
        }
    }

    #[test]
    fn t_elliptic_length_f32() {
        // Same arc in single precision: the Carlson loops must terminate and agree with f64.
        let at = |phi: f64| (DVec([phi.cos() + phi.sin(), 0.4 * phi.sin(), 0.0]));
        let theta = 4.0;
        let wide = swept_arc(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([1.0, 0.4, 0.0]),
            theta,
        );
        let narrow = |v: DVec3| crate::Vec::<3>([v.0[0] as f32, v.0[1] as f32, v.0[2] as f32]);
        let earc = EllipticArc3f {
            center: narrow(DVec([0.0, 0.0, 0.0])),
            start_vec: narrow(at(0.0)),
            mid_vec: narrow(at(theta / 2.0)),
            end_vec: narrow(at(theta)),
            angle: theta as f32,
        };
        let expected = wide.length();
        let actual = earc.length() as f64;
        assert!(
            (actual - expected).abs() < 1e-4 * expected,
            "f32 length {actual} vs f64 {expected}"
        );
    }

    // ===================== EllipticArc::point and tangent =====================

    #[test]
    fn t_elliptic_arc_point() {
        // Unit quarter circle: center=origin, start=(1,0,0), end=(0,1,0).
        let earc = EllipticArc3d::from_center_start_end(
            DVec([0.0, 0.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            false,
        )
        .unwrap();
        // Endpoints.
        assert!((earc.point(0.0).unwrap() - DVec([1.0, 0.0, 0.0])).length() < 1e-10);
        assert!((earc.point(1.0).unwrap() - DVec([0.0, 1.0, 0.0])).length() < 1e-10);
        // Midpoint of a unit quarter circle lies on (√2/2, √2/2, 0).
        let pm = earc.point(0.5).unwrap();
        assert!(
            (pm - DVec([FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0])).length() < 1e-10,
            "pm={pm:?}"
        );
        // Out of domain.
        assert!(earc.point(-0.001).is_none());
        assert!(earc.point(1.001).is_none());
    }

    #[test]
    fn t_elliptic_arc_tangent() {
        // Circular arc (radius 3, center offset): tangent must be perpendicular to radius
        // at every sampled point, and out-of-domain must return None.
        let earc = EllipticArc3d::from_center_start_end(
            DVec([1.0, 2.0, 0.0]),
            DVec([4.0, 2.0, 0.0]), // start = center + (3,0,0)
            DVec([1.0, 5.0, 0.0]), // end   = center + (0,3,0)
            false,
        )
        .unwrap();
        for i in 0..=8 {
            let t = i as f64 / 8.0;
            let p = earc.point(t).unwrap();
            let tan = earc.tangent(t).unwrap();
            let dot = (p - earc.center()).dot(tan).abs();
            assert!(dot < 1e-10, "t={t}: dot={dot}");
        }
        assert!(earc.tangent(-0.001).is_none());
        assert!(earc.tangent(1.001).is_none());
    }

    #[test]
    fn t_serialize_deserialize_roundtrip() {
        let arc = Arc3d::from_three_points(
            DVec([-1.0, 0.0, 0.0]),
            DVec([0.0, 1.0, 0.0]),
            DVec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut bytes = Vec::new();
        arc.serialize(&mut bytes).unwrap();
        let restored = Arc3d::deserialize(&bytes[..]).unwrap();
        assert_eq!(restored.center, arc.center);
        assert_eq!(restored.start_dir, arc.start_dir);
        assert_eq!(restored.mid_dir, arc.mid_dir);
        assert_eq!(restored.end_dir, arc.end_dir);
        assert_eq!(restored.radius, arc.radius);
        assert_eq!(restored.angle, arc.angle);
    }

    #[test]
    fn t_serialize_deserialize_2d() {
        let arc = Arc2d::default();
        let mut bytes = Vec::new();
        arc.serialize(&mut bytes).unwrap();
        let restored = Arc2d::deserialize(&bytes[..]).unwrap();
        assert_eq!(restored.center, arc.center);
        assert_eq!(restored.start_dir, arc.start_dir);
        assert_eq!(restored.mid_dir, arc.mid_dir);
        assert_eq!(restored.end_dir, arc.end_dir);
        assert_eq!(restored.radius, arc.radius);
        assert_eq!(restored.angle, arc.angle);
    }

    #[test]
    fn t_serialize_deserialize_major_arc() {
        // Arc3d going the long way (> π) from (1,0,0) through (0,-1,0) to (-1,0,0).
        let arc = Arc3d::from_three_points(
            DVec([1.0, 0.0, 0.0]),
            DVec([0.0, -1.0, 0.0]),
            DVec([-1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut bytes = Vec::new();
        arc.serialize(&mut bytes).unwrap();
        let restored = Arc3d::deserialize(&bytes[..]).unwrap();
        assert_eq!(restored.center, arc.center);
        assert_eq!(restored.start_dir, arc.start_dir);
        assert_eq!(restored.mid_dir, arc.mid_dir);
        assert_eq!(restored.end_dir, arc.end_dir);
        assert_eq!(restored.radius, arc.radius);
        assert_eq!(restored.angle, arc.angle);
    }

    #[test]
    fn t_serialize_deserialize_f32() {
        let arc = Arc3f::from_three_points(
            crate::Vec([-1.0, 0.0, 0.0]),
            crate::Vec([0.0, 1.0, 0.0]),
            crate::Vec([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let mut bytes = Vec::new();
        arc.serialize(&mut bytes).unwrap();
        let restored = Arc3f::deserialize(&bytes[..]).unwrap();
        assert_eq!(restored.center, arc.center);
        assert_eq!(restored.start_dir, arc.start_dir);
        assert_eq!(restored.mid_dir, arc.mid_dir);
        assert_eq!(restored.end_dir, arc.end_dir);
        assert_eq!(restored.radius, arc.radius);
        assert_eq!(restored.angle, arc.angle);
    }

    #[test]
    fn t_default_evaluates_without_nan() {
        // The default must be a genuinely usable arc, not a degenerate
        // placeholder: every method should produce finite, sensible values.
        let arc = Arc3d::default();
        let len = arc.length();
        assert!(len.is_finite() && len > 0.0);
        for i in 0..=4 {
            let t = len * i as f64 / 4.0;
            let p = arc.point(t).unwrap();
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
            let tan = arc.tangent(t).unwrap();
            assert!((tan.length() - 1.0).abs() < 1e-10);
        }
        assert_eq!(arc.start(), DVec([1.0, 0.0, 0.0]));
        assert_eq!(arc.end(), DVec([0.0, 1.0, 0.0]));
    }
}
