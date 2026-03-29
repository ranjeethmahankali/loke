use crate::{Vec3, error::Error, polynomial};
use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub struct Spline {
    knots: Vec<f64>,
    control_points: Vec<Vec3>,
    power_basis_coeff: Vec<f64>,
    unique_knots: Vec<f64>,
}

impl Spline {
    /// Create a spline from explicit knots and control points.
    /// Validates that `control_points.len() + degree + 1 == knots.len()`.
    pub fn create(
        control_points: Vec<Vec3>,
        knots: Vec<f64>,
        degree: usize,
    ) -> Result<Self, Error> {
        if control_points.len() + degree + 1 != knots.len() {
            return Err(Error::IncorrectKnotCount);
        }
        let power_basis_coeff = compute_polynomial_coeff(&knots, &control_points);
        let mut unique_knots = knots.clone();
        unique_knots.dedup();
        assert_eq!(
            power_basis_coeff.len(),
            (degree + 1) * (unique_knots.len() - 1) * 3
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
    pub fn create_clamped(control_points: Vec<Vec3>, degree: usize) -> Result<Self, Error> {
        let nclamp = degree + 1;
        if control_points.len() < nclamp {
            return Err(Error::InsufficientControlPoints);
        }
        let ntotal = control_points.len() + nclamp;
        let nmiddle = ntotal - 2 * nclamp;
        let mut knots = Vec::with_capacity(ntotal);
        knots.extend(
            std::iter::repeat_n(0.0, nclamp)
                .chain((0..nmiddle).map(|i| (i + 1) as f64))
                .chain(std::iter::repeat_n((nmiddle + 1) as f64, nclamp)),
        );
        debug_assert_eq!(knots.len(), ntotal);
        let power_basis_coeff = compute_polynomial_coeff(&knots, &control_points);
        let mut unique_knots = knots.clone();
        unique_knots.dedup();
        assert_eq!(
            power_basis_coeff.len(),
            (degree + 1) * (unique_knots.len() - 1) * 3
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

    pub fn control_points(&self) -> &[Vec3] {
        &self.control_points
    }

    pub fn domain(&self) -> (f64, f64) {
        let degree = self.degree();
        (
            self.knots[degree],
            self.knots[self.knots.len() - degree - 1],
        )
    }

    fn valid_param(&self, u: f64) -> bool {
        let (lo, hi) = self.domain();
        u >= lo && u <= hi
    }

    pub fn point(&self, u: f64) -> Option<Vec3> {
        // Check if point is in domain.
        if !self.valid_param(u) {
            return None;
        }
        // Compute points with borrowed buffers.
        let degree = self.degree();
        let span = find_span(&self.knots, degree, u);
        Some(BUFFERS.with_borrow_mut(|buf| {
            calc_basis(span, u, degree, &self.knots, buf);
            (0..=degree)
                .map(|i| self.control_points[span - degree + i] * buf.basis[i])
                .reduce(|acc, v| acc + v)
                .unwrap_or(Vec3(0., 0., 0.))
        }))
    }

    pub fn tangent(&self, u: f64) -> Option<Vec3> {
        let mut results = [Vec3(0., 0., 0.); 2];
        self.point_with_derivs(u, &mut results)
            .ok()
            .map(|()| results[1])
    }

    pub fn point_with_derivs(&self, u: f64, results: &mut [Vec3]) -> Result<(), Error> {
        if results.is_empty() {
            return Ok(()); // Nothing to evaluate.
        }
        if !self.valid_param(u) {
            return Err(Error::InvalidParameter);
        }
        let n_derivs = results.len() - 1;
        results.fill(Vec3(0., 0., 0.));
        let degree = self.degree();
        let n_derivs = n_derivs.min(degree); // Because higher order derivatives over degree-th are all zero.
        let span = find_span(&self.knots, degree, u);
        BUFFERS.with_borrow_mut(|buf| {
            calc_ders_basis(span, u, degree, n_derivs, &self.knots, buf);
            assert_eq!(buf.basis.len(), (n_derivs + 1) * (degree + 1));
            let nders = View2D::create(&buf.basis, n_derivs + 1, degree + 1);
            let span = span - degree;
            for k in 0..=n_derivs {
                for j in 0..=degree {
                    results[k] += self.control_points[span + j] * nders[k][j];
                }
            }
        });
        Ok(())
    }

    pub fn start(&self) -> Vec3 {
        self.point(self.domain().0)
            .expect("Internal error, can never fail")
    }

    pub fn end(&self) -> Vec3 {
        self.point(self.domain().1)
            .expect("Internal error, can never fail")
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let degree = self.degree();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut deriv = vec![0.0; degree];
        let mut roots = vec![0.0; degree.saturating_sub(1)];
        let n_segs = self.unique_knots.len() - 1;
        for ci in 0..3 {
            for si in 0..n_segs {
                let poly = self.power_basis_polynomial(si, ci);
                if degree > 1 {
                    polynomial::differentiate(poly, &mut deriv);
                    let n_roots =
                        polynomial::find_roots_in_range(&deriv, &mut roots, 0.0, 1.0, f64::EPSILON)
                            .expect("This is an internal error. This should never happen");
                    for r in &roots[0..n_roots] {
                        let val = polynomial::eval(poly, *r);
                        lo[ci] = lo[ci].min(val);
                        hi[ci] = hi[ci].max(val);
                    }
                }
                let (start, end) = (polynomial::eval(poly, 0.0), polynomial::eval(poly, 1.0));
                lo[ci] = lo[ci].min(start).min(end);
                hi[ci] = hi[ci].max(start).max(end);
            }
        }
        (Vec3(lo[0], lo[1], lo[2]), Vec3(hi[0], hi[1], hi[2]))
    }

    pub fn adaptive_samples(&self, tolerance: f64) -> impl Iterator<Item = Vec3> {
        SplineAdaptiveSamples::new(self, tolerance).map(|SplineSample { point, .. }| point)
    }

    pub fn length(&self, tolerance: f64) -> f64 {
        let mut samples = SplineAdaptiveSamples::new(self, tolerance);
        let first = match samples.next() {
            Some(first) => first,
            None => return 0.0,
        };
        samples
            .fold((first, 0.0), |(prev, total), s| {
                let chord = (s.point - prev.point).length();
                // Sagitta (midpoint deviation) from the curvature at the previous sample:
                // h ≈ |C''(t)| * Δt² / 8
                let h =
                    prev.curvature_magnitude_sq.sqrt() * prev.param_step * prev.param_step / 8.0;
                // Taylor expansion of circular arc length given chord c and sagitta h:
                // arc ≈ c + 8h²/(3c)
                let arc = if chord > 0.0 {
                    chord + 8.0 * h * h / (3.0 * chord)
                } else {
                    0.0
                };
                (s, total + arc)
            })
            .1
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
        self
    }

    pub fn with_control_points<F>(mut self, func: F) -> Self
    where
        F: Fn(&Vec3) -> Vec3,
    {
        for p in self.control_points.iter_mut() {
            *p = func(p);
        }
        self
    }

    pub fn power_basis_polynomial(&self, segment: usize, coord: usize) -> &[f64] {
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
         */
        let n_segs = self.unique_knots.len() - 1;
        let n_coeff = self.degree() + 1;
        let offset = (n_coeff * n_segs * coord) + (n_coeff * segment);
        &self.power_basis_coeff[offset..(offset + n_coeff)]
    }
}

struct SplineAdaptiveSamples<'a> {
    curve: &'a Spline,
    n_coeff: usize,
    n_segments: usize,
    curvature_magnitude_sq: Box<[f64]>,
    tolerance: f64,
    segment_index: usize,
    param: f64,
}

impl<'a> SplineAdaptiveSamples<'a> {
    fn new(curve: &'a Spline, tolerance: f64) -> Self {
        let degree = curve.degree();
        let n_segments = curve.unique_knots.len() - 1;
        // Compute the polynomials for the curvature magnitude squared.
        // Compute second derivative polynomials with a temporary buffer for the first derivative.
        let n_coeff_cmag2 = degree.saturating_sub(2) * 2 + 1;
        let mut curvature_magnitude_sq = vec![0.0; n_coeff_cmag2 * n_segments].into_boxed_slice();
        let mut deriv_buf = vec![0.0; degree].into_boxed_slice(); // temp buffer.
        let mut curv_buf = vec![0.0; degree.saturating_sub(1)]; // temp buffer.
        for ci in 0..3 {
            for (si, dst) in curvature_magnitude_sq
                .chunks_exact_mut(n_coeff_cmag2)
                .enumerate()
            {
                deriv_buf.fill(0.0);
                polynomial::differentiate(curve.power_basis_polynomial(si, ci), &mut deriv_buf);
                curv_buf.fill(0.0);
                polynomial::differentiate(&deriv_buf, &mut curv_buf);
                polynomial::mul_add(&curv_buf, &curv_buf, dst);
            }
        }
        SplineAdaptiveSamples {
            curve,
            n_coeff: n_coeff_cmag2,
            n_segments,
            curvature_magnitude_sq,
            tolerance: tolerance.abs() / (degree as f64),
            segment_index: 0,
            param: 0.0,
        }
    }
}

struct SplineSample {
    point: Vec3,
    param_step: f64,
    curvature_magnitude_sq: f64,
}

impl<'a> Iterator for SplineAdaptiveSamples<'a> {
    type Item = SplineSample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segment_index >= self.n_segments {
            return None;
        }
        // We assume the previous iteration left us in a clean state, and attempt to evaluate a point.
        let [x, y, z] = [0, 1, 2].map(|ci| {
            polynomial::eval(
                self.curve.power_basis_polynomial(self.segment_index, ci),
                self.param,
            )
        });
        // Try to advance.
        if self.param == 1.0 {
            self.segment_index += 1;
            self.param = 0.0;
        }
        let (cmag2, step) = if self.segment_index < self.n_segments {
            let offset = self.n_coeff * self.segment_index;
            let cmag2 = polynomial::eval(
                &self.curvature_magnitude_sq[offset..(offset + self.n_coeff)],
                self.param,
            )
            .abs();
            let step = if cmag2 == 0.0 {
                2.0 // Just a large enough value to snap to the end of this segment.
            } else {
                (8.0 * self.tolerance / cmag2.sqrt()).sqrt()
            };
            self.param = (self.param + step).max(self.param.next_up()); // Always increment no matter how small the step.
            (cmag2, step)
        } else {
            (0.0, 0.0)
        };
        if self.param > 1.0 {
            self.param = 1.0;
        }
        Some(SplineSample {
            point: Vec3(x, y, z),
            param_step: step,
            curvature_magnitude_sq: cmag2,
        })
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
fn find_span(knots: &[f64], degree: usize, u: f64) -> usize {
    let n = knots.len() - degree - 2;
    if u == knots[n + 1] {
        // Special case at the end of the domain
        return n;
    }
    let low = degree;
    let hi = n + 1;
    // The book manually implements the binary search, here we use the std function that does the same.
    low + &knots[low..hi].partition_point(|k| k <= &u) - 1
}

/// Temporary storage used during evaluations. The naming tries to
/// closely follow the names in the book.
#[derive(Default)]
struct EvalBuffers {
    basis: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
    ndu: Vec<f64>,
    alt_coeff: Vec<f64>, // This corresponds to the temporary storage 'a' in algorithm A2.3.
}

// These static thread local instances of EvalBuffers will be used for all
// spline evaluations.
thread_local! {
    static BUFFERS: RefCell<EvalBuffers> = RefCell::new(Default::default());
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

/// This is based on algorithm A2.2 from the book.
fn calc_basis(i: usize, u: f64, degree: usize, knots: &[f64], buf: &mut EvalBuffers) {
    let basis = &mut buf.basis;
    let left = &mut buf.left;
    let right = &mut buf.right;
    basis.clear();
    basis.resize(degree + 1, 1.0);
    left.resize(degree + 1, 0.0);
    right.resize(degree + 1, 0.0);
    for j in 1..=degree {
        left[j] = u - knots[i + 1 - j];
        right[j] = knots[i + j] - u;
        let mut saved = 0.0f64;
        for r in 0..j {
            let temp = basis[r] / (right[r + 1] + left[j - r]);
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
}

/// This is algorithm A2.3 from the book.
fn calc_ders_basis(
    span: usize,
    u: f64,
    degree: usize,
    n_derivs: usize,
    knots: &[f64],
    buf: &mut EvalBuffers,
) {
    // Prep the temporary storage.
    buf.basis.clear();
    buf.basis.resize((n_derivs + 1) * (degree + 1), 0.);
    let mut ders = View2DMut::create(&mut buf.basis, n_derivs + 1, degree + 1);
    buf.ndu.clear();
    buf.ndu.resize((degree + 1).pow(2), 0.);
    let mut ndu = View2DMut::create(&mut buf.ndu, degree + 1, degree + 1);
    buf.alt_coeff.clear();
    buf.alt_coeff.resize((degree + 1) * 2, 0.);
    let mut a = View2DMut::create(&mut buf.alt_coeff, 2, degree + 1);
    buf.left.clear();
    buf.left.resize(degree + 1, 0.);
    let left = &mut buf.left;
    buf.right.clear();
    buf.right.resize(degree + 1, 0.);
    let right = &mut buf.right;
    // Implement the algo.
    ndu[0][0] = 1.0_f64;
    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0_f64;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
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
        a[0][0] = 1.;
        // Loop to compute the kth derivative.
        for k in 1..=n_derivs {
            let mut d = 0.0_f64;
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
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r as usize];
                d += a[s2][k] * ndu[r][pk as usize];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2); // Switch rows.
        }
    }
    // Multiply through by the correct factors.
    let mut r = degree as f64;
    for k in 1..=n_derivs {
        for j in 0..=degree {
            ders[k][j] *= r;
        }
        r *= degree as f64 - k as f64;
    }
}

fn piecewise_bezier(knots: &[f64], control_points: &[Vec3], dst: &mut Vec<Vec3>) {
    thread_local! {
        static ALPHAS: RefCell<Vec<f64>> = RefCell::new(Default::default());
    }
    let degree = knots.len() - control_points.len() - 1;
    let n_segments = knots
        .iter()
        .zip(knots.iter().skip(1))
        .filter(|&(a, b)| a != b)
        .count();
    dst.clear();
    dst.resize(n_segments * (degree + 1), Vec3(0.0, 0.0, 0.0));
    ALPHAS.with_borrow_mut(|alphas| {
        // Algorithm A5.6 from the NURBS book. Matching the variable names
        // within reason for legibility.
        let n = control_points.len() - 1;
        let p = degree;
        let u_vec: &[f64] = knots;
        let p_vec: &[Vec3] = control_points;
        let mut q_mat = View2DMut::create(dst, n_segments, degree + 1);
        let m = n + p + 1;
        let mut a = p;
        let mut b = p + 1;
        let mut nb = 0;
        for i in 0..=p {
            q_mat[nb][i] = p_vec[i];
        }
        while b < m {
            let i = b;
            while b < m && u_vec[b + 1] == u_vec[b] {
                b += 1;
            }
            let mult = b - i + 1;
            if mult < p {
                let numer = u_vec[b] - u_vec[a]; // Numerator of alpha.
                // Compute and store alphas.
                alphas.resize(p - mult, 0.);
                for j in ((mult + 1)..=p).rev() {
                    alphas[j - mult - 1] = numer / (u_vec[a + j] - u_vec[a]);
                }
                let r = p - mult; // Insert the knot r times.
                for j in 1..=r {
                    let save = r - j;
                    let s = mult + j; // This many new points.
                    for k in (s..=p).rev() {
                        let alpha = alphas[k - s];
                        q_mat[nb][k] = alpha * q_mat[nb][k] + (1.0 - alpha) * q_mat[nb][k - 1];
                    }
                    if b < m {
                        // Control point of.
                        q_mat[nb + 1][save] = q_mat[nb][p]; // Next segment.
                    }
                }
            }
            nb = nb + 1; // Bezier segment completed.
            if b < m {
                // Initialize for next segment.
                for i in (p.saturating_sub(mult))..=p {
                    q_mat[nb][i] = p_vec[b - p + i];
                }
                a = b;
                b = b + 1;
            }
        }
    });
}

/// Compute the power-basis polynomial coefficients from the piecewise Bézier
/// decomposition. The output is a flat buffer of dimensions
/// `(degree+1) × n_segments × 3`, packed so that the first `degree+1` values
/// are the x polynomial of segment 0, followed by segment 1, etc., then all y
/// polynomials, then all z polynomials.
fn compute_polynomial_coeff(knots: &[f64], control_points: &[Vec3]) -> Vec<f64> {
    let mut bezier_cps = Vec::new();
    piecewise_bezier(knots, control_points, &mut bezier_cps);
    let p = knots.len() - control_points.len() - 1;
    let msize = p + 1;
    let n_seg = bezier_cps.len() / msize;
    debug_assert_eq!(bezier_cps.len() % msize, 0);
    // Get the Bézier-to-power-basis conversion matrix.
    let mut bezier_mat = vec![0.0; msize * msize];
    get_bezier_matrix(p, &mut bezier_mat);
    let bmat = View2D::create(&bezier_mat, msize, msize);
    // For each segment, multiply bezierMat by the column of scalar Bézier CP
    // values for each component. The C++ transposes the segment matrix to make
    // columns be segments; we skip the transpose and just read from rows.
    let mut coeff = vec![0.0; 3 * n_seg * msize];
    for ci in 0..3usize {
        let comp_offset = ci * n_seg * msize;
        for seg in 0..n_seg {
            let cps = &bezier_cps[seg * msize..(seg + 1) * msize];
            let out_base = comp_offset + seg * msize;
            for i in 0..msize {
                let mut val = 0.0;
                // bezierMat is lower triangular, so k only goes up to i.
                for k in 0..=i {
                    let v = match ci {
                        0 => cps[k].0,
                        1 => cps[k].1,
                        _ => cps[k].2,
                    };
                    val += bmat[i][k] * v;
                }
                coeff[out_base + i] = val;
            }
        }
    }
    coeff
}

pub(crate) fn get_bezier_matrix(degree: usize, dst: &mut [f64]) {
    thread_local! {
        static BEZIER_MATS: RefCell<Vec<Vec<f64>>> = RefCell::new(Default::default());
    }
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
        dst.copy_from_slice(mat); // Panics if the lengths don't match, so we don't need to check.
    });
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::polynomial;

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
        // Symmetry
        for n in 0..=8 {
            for k in 0..=n {
                assert_eq!(binomial_coeff(n, k), binomial_coeff(n, n - k));
            }
        }
    }

    #[test]
    fn t_find_span_clamped_cubic() {
        // Clamped cubic (p=3) with 7 control points.
        // n = 6, m = n + p + 1 = 10, so 11 knots.
        let degree = 3;
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        // Domain is [knots[3], knots[7]] = [0.0, 4.0]
        // Start of domain.
        assert_eq!(find_span(&knots, degree, 0.0), 3);
        // End of domain (special case).
        assert_eq!(find_span(&knots, degree, 4.0), 6);
        // At each internal knot, t lands in [u_i, u_{i+1}).
        assert_eq!(find_span(&knots, degree, 1.0), 4);
        assert_eq!(find_span(&knots, degree, 2.0), 5);
        assert_eq!(find_span(&knots, degree, 3.0), 6);
        // Mid-span values.
        assert_eq!(find_span(&knots, degree, 0.5), 3);
        assert_eq!(find_span(&knots, degree, 1.5), 4);
        assert_eq!(find_span(&knots, degree, 2.5), 5);
        assert_eq!(find_span(&knots, degree, 3.5), 6);
        // Values very close to knot boundaries.
        let eps = 1e-14;
        assert_eq!(find_span(&knots, degree, 0.0 + eps), 3);
        assert_eq!(find_span(&knots, degree, 1.0 - eps), 3);
        assert_eq!(find_span(&knots, degree, 1.0 + eps), 4);
        assert_eq!(find_span(&knots, degree, 4.0 - eps), 6);
        // Clamped linear (p=1) with 4 control points.
        // n = 3, m = 5, so 6 knots.
        let degree = 1;
        let knots = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        assert_eq!(find_span(&knots, degree, 0.0), 1);
        assert_eq!(find_span(&knots, degree, 0.5), 1);
        assert_eq!(find_span(&knots, degree, 1.0), 2);
        assert_eq!(find_span(&knots, degree, 2.0), 3);
        assert_eq!(find_span(&knots, degree, 2.5), 3);
        assert_eq!(find_span(&knots, degree, 3.0), 3); // special case
        // Clamped quadratic (p=2) with 3 control points (minimum).
        // n = 2, m = 5, so 6 knots. Single span.
        let degree = 2;
        let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(find_span(&knots, degree, 0.0), 2);
        assert_eq!(find_span(&knots, degree, 0.5), 2);
        assert_eq!(find_span(&knots, degree, 1.0), 2); // special case
        // Clamped cubic with non-uniform interior knots.
        let degree = 3;
        let knots = [0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(find_span(&knots, degree, 0.0), 3);
        assert_eq!(find_span(&knots, degree, 0.15), 3);
        assert_eq!(find_span(&knots, degree, 0.3), 4);
        assert_eq!(find_span(&knots, degree, 0.5), 4);
        assert_eq!(find_span(&knots, degree, 0.7), 5);
        assert_eq!(find_span(&knots, degree, 0.85), 5);
        assert_eq!(find_span(&knots, degree, 1.0), 5); // special case
    }

    #[test]
    fn t_create_validates_knot_count() {
        let cps = vec![Vec3(0., 0., 0.), Vec3(1., 0., 0.), Vec3(2., 0., 0.)];
        // Correct: 3 cps + degree 2 + 1 = 6 knots.
        assert!(Spline::create(cps.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2,).is_ok());
        // Too few knots.
        assert!(matches!(
            Spline::create(cps.clone(), vec![0.0, 0.0, 0.0, 1.0, 1.0], 2),
            Err(Error::IncorrectKnotCount)
        ));
        // Too many knots.
        assert!(matches!(
            Spline::create(cps, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 2),
            Err(Error::IncorrectKnotCount)
        ));
    }

    #[test]
    fn t_create_clamped_validates_control_points() {
        // Degree 2 requires at least 3 control points.
        assert!(matches!(
            Spline::create_clamped(vec![Vec3(0., 0., 0.), Vec3(1., 0., 0.)], 2),
            Err(Error::InsufficientControlPoints)
        ));
        // Degree 0 with empty control points.
        assert!(matches!(
            Spline::create_clamped(vec![], 0),
            Err(Error::InsufficientControlPoints)
        ));
        // Exactly enough.
        let spline = Spline::create_clamped(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            2,
        )
        .unwrap();
        assert_eq!(spline.degree(), 2);
        assert_eq!(spline.domain(), (0.0, 1.0));
    }

    #[test]
    fn t_create_clamped_knot_generation() {
        // 7 control points, degree 3: nclamp=4, ntotal=11, nmiddle=3
        // Expected knots: [0,0,0,0, 1,2,3, 4,4,4,4]
        let cps: Vec<Vec3> = (0..7).map(|i| Vec3(i as f64, 0., 0.)).collect();
        let spline = Spline::create_clamped(cps, 3).unwrap();
        assert_eq!(spline.degree(), 3);
        assert_eq!(spline.domain(), (0.0, 4.0));
        // 4 control points, degree 1: nclamp=2, ntotal=6, nmiddle=2
        // Expected knots: [0,0, 1,2, 3,3]
        let cps: Vec<Vec3> = (0..4).map(|i| Vec3(i as f64, 0., 0.)).collect();
        let spline = Spline::create_clamped(cps, 1).unwrap();
        assert_eq!(spline.degree(), 1);
        assert_eq!(spline.domain(), (0.0, 3.0));
    }

    #[test]
    fn t_start_end_linear() {
        // Degree 1: start/end should be the first/last control points.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(3., 1., 0.),
                Vec3(4., 4., 0.),
            ],
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        )
        .unwrap();
        assert_eq!(spline.start(), Vec3(0., 0., 0.));
        assert_eq!(spline.end(), Vec3(4., 4., 0.));
    }

    #[test]
    fn t_start_end_quadratic_bezier() {
        // Single-span quadratic Bézier.
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        assert_eq!(spline.start(), Vec3(0., 0., 0.));
        assert_eq!(spline.end(), Vec3(2., 0., 0.));
    }

    #[test]
    fn t_start_end_cubic_bezier() {
        // Single-span cubic Bézier.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        assert_eq!(spline.start(), Vec3(0., 0., 0.));
        assert_eq!(spline.end(), Vec3(1., 0., 0.));
    }

    #[test]
    fn t_start_end_clamped_cubic_multi_segment() {
        // Clamped cubic with multiple segments and 3D control points.
        let spline = Spline::create(
            vec![
                Vec3(1., 2., 3.),
                Vec3(4., 5., 6.),
                Vec3(7., 8., 9.),
                Vec3(10., 11., 12.),
                Vec3(13., 14., 15.),
                Vec3(16., 17., 18.),
                Vec3(19., 20., 21.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        assert_eq!(spline.start(), Vec3(1., 2., 3.));
        assert_eq!(spline.end(), Vec3(19., 20., 21.));
    }

    #[test]
    fn t_start_end_matches_eval_point() {
        // Verify start()/end() agree with eval_point at domain boundaries
        // for several splines created with create_clamped.
        for (cps, degree) in [
            (vec![Vec3(0., 0., 0.), Vec3(1., 1., 1.)], 1),
            (
                vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
                2,
            ),
            (
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(1., 5., -2.),
                    Vec3(2., -3., 4.),
                    Vec3(3., 5., -2.),
                    Vec3(4., 0., 0.),
                ],
                3,
            ),
        ] {
            let spline = Spline::create_clamped(cps, degree).unwrap();
            let (dom_start, dom_end) = spline.domain();
            assert_eq!(spline.start(), spline.point(dom_start).unwrap());
            assert_eq!(spline.end(), spline.point(dom_end).unwrap());
        }
    }

    #[test]
    fn t_eval_point_out_of_domain() {
        // Clamped cubic, domain [0.0, 4.0]
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., 0.),
                Vec3(2., 0., 0.),
                Vec3(3., 2., 0.),
                Vec3(4., 0., 0.),
                Vec3(5., 2., 0.),
                Vec3(6., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        assert_eq!(spline.domain(), (0.0, 4.0));
        // Out of domain returns None.
        assert!(spline.point(-1.0).is_none());
        assert!(spline.point(-f64::EPSILON).is_none());
        assert!(spline.point(4.0_f64.next_up()).is_none());
        assert!(spline.point(100.0).is_none());
        assert!(spline.point(f64::NEG_INFINITY).is_none());
        assert!(spline.point(f64::INFINITY).is_none());
        assert!(spline.point(f64::NAN).is_none());
        // In domain returns Some.
        assert!(spline.point(0.0).is_some());
        assert!(spline.point(2.0).is_some());
        assert!(spline.point(4.0).is_some());
    }

    #[test]
    fn t_eval_point_linear() {
        // Degree 1: the curve interpolates control points at knot values
        // and linearly interpolates between them.
        // knots [0,0,1,2,3,3], 4 control points, domain [0,3].
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(3., 1., 0.),
                Vec3(4., 4., 0.),
            ],
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        )
        .unwrap();
        // At knot values, passes through control points exactly.
        assert_eq!(spline.point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), Vec3(1., 3., 0.));
        assert_eq!(spline.point(2.0).unwrap(), Vec3(3., 1., 0.));
        assert_eq!(spline.point(3.0).unwrap(), Vec3(4., 4., 0.));
        // Midpoints: linear interpolation between control points.
        assert_eq!(spline.point(0.5).unwrap(), Vec3(0.5, 1.5, 0.));
        assert_eq!(spline.point(1.5).unwrap(), Vec3(2., 2., 0.));
        assert_eq!(spline.point(2.5).unwrap(), Vec3(3.5, 2.5, 0.));
    }

    #[test]
    fn t_eval_point_quadratic_bezier() {
        // Single-span quadratic Bezier: C(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        // knots [0,0,0,1,1,1], 3 control points, domain [0,1].
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        // Endpoints match first and last control points.
        assert_eq!(spline.point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), Vec3(2., 0., 0.));
        // t=0.5: 0.25*(0,0,0) + 0.5*(1,2,0) + 0.25*(2,0,0) = (1,1,0)
        assert_eq!(spline.point(0.5).unwrap(), Vec3(1., 1., 0.));
        // t=0.25: 0.5625*(0,0,0) + 0.375*(1,2,0) + 0.0625*(2,0,0) = (0.5, 0.75, 0)
        assert_eq!(spline.point(0.25).unwrap(), Vec3(0.5, 0.75, 0.));
        // t=0.75: 0.0625*(0,0,0) + 0.375*(1,2,0) + 0.5625*(2,0,0) = (1.5, 0.75, 0)
        assert_eq!(spline.point(0.75).unwrap(), Vec3(1.5, 0.75, 0.));
    }

    #[test]
    fn t_eval_point_cubic_bezier() {
        // Single-span cubic Bezier: C(t) = (1-t)^3 P0 + 3t(1-t)^2 P1 + 3t^2(1-t) P2 + t^3 P3
        // knots [0,0,0,0,1,1,1,1], 4 control points, domain [0,1].
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        // Endpoints.
        assert_eq!(spline.point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.point(1.0).unwrap(), Vec3(1., 0., 0.));
        // t=0.5: 0.125*(0,0,0) + 0.375*(0,1,0) + 0.375*(1,1,0) + 0.125*(1,0,0)
        //       = (0,0,0) + (0, 0.375, 0) + (0.375, 0.375, 0) + (0.125, 0, 0)
        //       = (0.5, 0.75, 0)
        assert_eq!(spline.point(0.5).unwrap(), Vec3(0.5, 0.75, 0.));
    }

    #[test]
    fn t_eval_derivs_out_of_domain() {
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., 0.),
                Vec3(2., 0., 0.),
                Vec3(3., 2., 0.),
                Vec3(4., 0., 0.),
                Vec3(5., 2., 0.),
                Vec3(6., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        let mut results = [Vec3(0., 0., 0.); 2];
        assert!(spline.point_with_derivs(-1.0, &mut results).is_err());
        assert!(spline.point_with_derivs(5.0, &mut results).is_err());
        assert!(spline.point_with_derivs(f64::NAN, &mut results).is_err());
        // Empty results slice is a no-op, even out of domain.
        assert!(spline.point_with_derivs(-1.0, &mut []).is_ok());
    }

    #[test]
    fn t_eval_derivs_linear() {
        // Degree 1: piecewise linear. The derivative is constant within each span.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(3., 1., 0.),
                Vec3(4., 4., 0.),
            ],
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        )
        .unwrap();
        // At t=0.5 (first span): point should match eval_point, derivative = P1 - P0 = (1,3,0).
        let mut results = [Vec3(0., 0., 0.); 2];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], Vec3(1., 3., 0.));
        // At t=1.5 (second span): derivative = P2 - P1 = (2,-2,0).
        spline.point_with_derivs(1.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(1.5).unwrap());
        assert_eq!(results[1], Vec3(2., -2., 0.));
        // At t=2.5 (third span): derivative = P3 - P2 = (1,3,0).
        spline.point_with_derivs(2.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(2.5).unwrap());
        assert_eq!(results[1], Vec3(1., 3., 0.));
        // Requesting only the point (no derivatives).
        let mut results = [Vec3(0., 0., 0.); 1];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
    }

    #[test]
    fn t_eval_derivs_quadratic_bezier() {
        // Single-span quadratic Bezier: C(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        // C'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
        // C''(t) = 2(P2 - 2P1 + P0)
        let p0 = Vec3(0., 0., 0.);
        let p1 = Vec3(1., 2., 0.);
        let p2 = Vec3(2., 0., 0.);
        let spline =
            Spline::create(vec![p0, p1, p2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2).unwrap();
        let mut results = [Vec3(0., 0., 0.); 3];
        // At t=0: C'(0) = 2(P1-P0) = (2,4,0), C''(0) = 2(P2-2P1+P0) = (0,-8,0)
        spline.point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(0., 0., 0.));
        assert_eq!(results[1], Vec3(2., 4., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // At t=1: C'(1) = 2(P2-P1) = (2,-4,0)
        spline.point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(2., 0., 0.));
        assert_eq!(results[1], Vec3(2., -4., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // At t=0.5: C'(0.5) = 2[0.5*(1,2,0) + 0.5*(1,-2,0)] = 2*(1,0,0) = (2,0,0)
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], Vec3(2., 0., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // Requesting more derivatives than degree: 3rd derivative should be zero.
        let mut results = [Vec3(0., 0., 0.); 4];
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[3], Vec3(0., 0., 0.));
    }

    #[test]
    fn t_eval_derivs_cubic_bezier() {
        // Single-span cubic Bezier with known control points.
        // C(t) = (1-t)^3 P0 + 3t(1-t)^2 P1 + 3t^2(1-t) P2 + t^3 P3
        // C'(t) = 3[(1-t)^2(P1-P0) + 2t(1-t)(P2-P1) + t^2(P3-P2)]
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        let mut results = [Vec3(0., 0., 0.); 3];
        // At t=0: C'(0) = 3(P1-P0) = 3*(0,1,0) = (0,3,0)
        spline.point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(0., 0., 0.));
        assert_eq!(results[1], Vec3(0., 3., 0.));
        // At t=1: C'(1) = 3(P3-P2) = 3*(0,-1,0) = (0,-3,0)
        spline.point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(1., 0., 0.));
        assert_eq!(results[1], Vec3(0., -3., 0.));
        // At t=0.5: point should match eval_point.
        // C'(0.5) = 3[0.25*(0,1,0) + 0.5*(1,0,0) + 0.25*(0,-1,0)]
        //         = 3*(0.5, 0, 0) = (1.5, 0, 0)
        spline.point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.point(0.5).unwrap());
        assert_eq!(results[1], Vec3(1.5, 0., 0.));
    }

    #[test]
    fn t_eval_tangent_out_of_domain() {
        // Degree 1, domain [0, 2].
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 1., 0.), Vec3(3., 0., 0.)],
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        )
        .unwrap();
        assert!(spline.tangent(-0.1).is_none());
        assert!(spline.tangent(2.1).is_none());
        assert!(spline.tangent(f64::NAN).is_none());
    }

    #[test]
    fn t_eval_tangent_linear() {
        // Degree 1: tangent within each span equals the difference of adjacent control points.
        // P0=(0,0,0), P1=(2,6,0), P2=(5,3,0)  knots [0,0,1,2,2]
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(2., 6., 0.), Vec3(5., 3., 0.)],
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        )
        .unwrap();
        // First span tangent = P1 - P0 = (2,6,0)
        assert_eq!(spline.tangent(0.0).unwrap(), Vec3(2., 6., 0.));
        assert_eq!(spline.tangent(0.5).unwrap(), Vec3(2., 6., 0.));
        // Second span tangent = P2 - P1 = (3,-3,0)
        assert_eq!(spline.tangent(1.5).unwrap(), Vec3(3., -3., 0.));
        assert_eq!(spline.tangent(2.0).unwrap(), Vec3(3., -3., 0.));
    }

    #[test]
    fn t_eval_tangent_quadratic_bezier() {
        // Single-span quadratic Bezier: P0=(1,0,0), P1=(1,1,0), P2=(0,1,0)
        // C'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
        //       = 2[(1-t)(0,1,0) + t(-1,0,0)]
        let spline = Spline::create(
            vec![Vec3(1., 0., 0.), Vec3(1., 1., 0.), Vec3(0., 1., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        // C'(0) = 2(0,1,0) = (0,2,0)
        assert_eq!(spline.tangent(0.0).unwrap(), Vec3(0., 2., 0.));
        // C'(1) = 2(-1,0,0) = (-2,0,0)
        assert_eq!(spline.tangent(1.0).unwrap(), Vec3(-2., 0., 0.));
        // C'(0.5) = 2[0.5*(0,1,0) + 0.5*(-1,0,0)] = (-1,1,0)
        assert_eq!(spline.tangent(0.5).unwrap(), Vec3(-1., 1., 0.));
        // C'(0.25) = 2[0.75*(0,1,0) + 0.25*(-1,0,0)] = (-0.5, 1.5, 0)
        assert_eq!(spline.tangent(0.25).unwrap(), Vec3(-0.5, 1.5, 0.));
    }

    #[test]
    fn t_eval_tangent_cubic_bezier() {
        // Single-span cubic Bezier: P0=(0,0,0), P1=(1,0,0), P2=(1,1,0), P3=(0,1,0)
        // C'(t) = 3[(1-t)^2(P1-P0) + 2t(1-t)(P2-P1) + t^2(P3-P2)]
        //       = 3[(1-t)^2(1,0,0) + 2t(1-t)(0,1,0) + t^2(-1,0,0)]
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 0., 0.),
                Vec3(1., 1., 0.),
                Vec3(0., 1., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        // C'(0) = 3(1,0,0) = (3,0,0)
        assert_eq!(spline.tangent(0.0).unwrap(), Vec3(3., 0., 0.));
        // C'(1) = 3(-1,0,0) = (-3,0,0)
        assert_eq!(spline.tangent(1.0).unwrap(), Vec3(-3., 0., 0.));
        // C'(0.5) = 3[0.25*(1,0,0) + 0.5*(0,1,0) + 0.25*(-1,0,0)]
        //         = 3*(0, 0.5, 0) = (0, 1.5, 0)
        assert_eq!(spline.tangent(0.5).unwrap(), Vec3(0., 1.5, 0.));
    }

    #[test]
    fn t_eval_point_clamped_cubic_endpoints() {
        // Clamped cubic always passes through first and last control points.
        let spline = Spline::create(
            vec![
                Vec3(1., 2., 3.),
                Vec3(4., 5., 6.),
                Vec3(7., 8., 9.),
                Vec3(10., 11., 12.),
                Vec3(13., 14., 15.),
                Vec3(16., 17., 18.),
                Vec3(19., 20., 21.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        assert_eq!(spline.point(0.0).unwrap(), Vec3(1., 2., 3.));
        assert_eq!(spline.point(4.0).unwrap(), Vec3(19., 20., 21.));
    }

    /// De Casteljau evaluation of a Bézier curve defined by `cps` at parameter `t` in [0, 1].
    fn de_casteljau(cps: &[Vec3], t: f64) -> Vec3 {
        let mut work: Vec<Vec3> = cps.to_vec();
        for level in 1..cps.len() {
            for i in 0..cps.len() - level {
                work[i] = (1.0 - t) * work[i] + t * work[i + 1];
            }
        }
        work[0]
    }

    /// Verify piecewise_bezier by evaluating the original spline and the decomposed
    /// Bézier segments at many parameter values and checking they agree.
    fn check_bezier_decomposition(spline: &Spline) {
        let mut bezier_cps = Vec::new();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut bezier_cps);
        let p = spline.degree();
        let n_segments = bezier_cps.len() / (p + 1);
        assert_eq!(bezier_cps.len() % (p + 1), 0);
        // Collect the distinct knot spans.
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
            // Check at many parameter values within this span.
            let n_samples = 17;
            for si in 0..=n_samples {
                let t = si as f64 / n_samples as f64;
                let u = u_lo + t * (u_hi - u_lo);
                let from_spline = spline.point(u).unwrap();
                let from_bezier = de_casteljau(&view[seg], t);
                let diff = from_spline - from_bezier;
                let err = diff.0.abs().max(diff.1.abs()).max(diff.2.abs());
                assert!(
                    err < 1e-12,
                    "seg={seg}, t={t}, u={u}: spline={from_spline:?} bezier={from_bezier:?} err={err}"
                );
            }
        }
    }

    #[test]
    fn t_piecewise_bezier_all_cases() {
        // --- Single segment cases: Bézier CPs should equal the original CPs ---
        // Degree 1, single segment (2 CPs).
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(3., 4., 0.)],
            vec![0.0, 0.0, 1.0, 1.0],
            1,
        )
        .unwrap();
        let mut dst = Vec::new();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 2);
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[1], Vec3(3., 4., 0.));
        check_bezier_decomposition(&spline);
        // Degree 2, single segment (3 CPs). Bézier CPs == original CPs.
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3);
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[1], Vec3(1., 2., 0.));
        assert_eq!(dst[2], Vec3(2., 0., 0.));
        check_bezier_decomposition(&spline);
        // Degree 3, single segment (4 CPs). Bézier CPs == original CPs.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 4);
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[1], Vec3(0., 1., 0.));
        assert_eq!(dst[2], Vec3(1., 1., 0.));
        assert_eq!(dst[3], Vec3(1., 0., 0.));
        check_bezier_decomposition(&spline);
        // --- Multi-segment cases ---
        // Degree 1, 4 CPs, 3 segments. Each segment is a line between consecutive CPs.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(3., 1., 0.),
                Vec3(4., 4., 0.),
            ],
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 6); // 3 segments * 2 CPs
        // Segment 0.
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[1], Vec3(1., 3., 0.));
        // Segment 1.
        assert_eq!(dst[2], Vec3(1., 3., 0.));
        assert_eq!(dst[3], Vec3(3., 1., 0.));
        // Segment 2.
        assert_eq!(dst[4], Vec3(3., 1., 0.));
        assert_eq!(dst[5], Vec3(4., 4., 0.));
        check_bezier_decomposition(&spline);
        // Degree 3, 7 CPs, 4 segments, uniform clamped.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., 0.),
                Vec3(2., -1., 3.),
                Vec3(3., 2., 1.),
                Vec3(4., 0., -1.),
                Vec3(5., 3., 2.),
                Vec3(6., 1., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 4 * 4); // 4 segments * 4 CPs
        // First segment starts at first CP, last segment ends at last CP.
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[15], Vec3(6., 1., 0.));
        // Adjacent segments must share endpoints (C0 continuity).
        for seg in 0..3 {
            let end = &dst[seg * 4 + 3];
            let start = &dst[(seg + 1) * 4];
            assert_eq!(
                end,
                start,
                "segments {seg} and {} don't share endpoint",
                seg + 1
            );
        }
        check_bezier_decomposition(&spline);
        // Degree 2, 5 CPs, 3 segments, uniform clamped.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 4., 0.),
                Vec3(3., -2., 1.),
                Vec3(5., 1., 3.),
                Vec3(7., 0., 0.),
            ],
            2,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3 * 3); // 3 segments * 3 CPs
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[8], Vec3(7., 0., 0.));
        for seg in 0..2 {
            let end = &dst[seg * 3 + 2];
            let start = &dst[(seg + 1) * 3];
            assert_eq!(
                end,
                start,
                "segments {seg} and {} don't share endpoint",
                seg + 1
            );
        }
        check_bezier_decomposition(&spline);
        // --- Interior knot with multiplicity > 1 ---
        // Degree 3, interior knot at u=1 with multiplicity 2.
        // 6 CPs, knots [0,0,0,0, 1,1, 2,2,2,2] => 2 segments.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(2., 0., 1.),
                Vec3(3., 2., -1.),
                Vec3(4., -1., 2.),
                Vec3(5., 1., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            3,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 2 * 4); // 2 segments * 4 CPs
        assert_eq!(dst[0], Vec3(0., 0., 0.));
        assert_eq!(dst[7], Vec3(5., 1., 0.));
        assert_eq!(dst[3], dst[4]); // C0 at the join.
        check_bezier_decomposition(&spline);
        // --- 3D control points with all nonzero components ---
        let spline = Spline::create(
            vec![
                Vec3(1., 2., 3.),
                Vec3(4., 5., 6.),
                Vec3(7., 8., 9.),
                Vec3(10., 11., 12.),
                Vec3(13., 14., 15.),
                Vec3(16., 17., 18.),
                Vec3(19., 20., 21.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        check_bezier_decomposition(&spline);
        // --- Non-uniform interior knots ---
        // Degree 3, 6 CPs, knots [0,0,0,0, 0.3, 0.7, 1,1,1,1] => 3 segments.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0.5, 2., 1.),
                Vec3(1.5, -1., 2.),
                Vec3(2.5, 3., -1.),
                Vec3(3.5, 0., 1.),
                Vec3(4., 1., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        piecewise_bezier(&spline.knots, &spline.control_points, &mut dst);
        assert_eq!(dst.len(), 3 * 4);
        check_bezier_decomposition(&spline);
        // --- Reuse of dst vector (ensure clear works) ---
        let spline_small = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 1., 1.)],
            vec![0.0, 0.0, 1.0, 1.0],
            1,
        )
        .unwrap();
        piecewise_bezier(&spline_small.knots, &spline_small.control_points, &mut dst);
        assert_eq!(dst.len(), 2);
    }

    /// Evaluate a point on the spline using the power basis polynomial coefficients,
    /// and compare against the standard B-spline evaluation.
    fn check_power_basis(spline: &Spline) {
        let unique = &spline.unique_knots;
        let n_segments = unique.len() - 1;
        let p = spline.degree();
        let (domain_lo, domain_hi) = spline.domain();
        // Precompute derivative polynomials for all segments and coordinates.
        // Flat buffer: deriv_polys[(ci * n_segments + seg) * p .. +p]
        let mut deriv_polys = vec![0.0; 3 * n_segments * p];
        for ci in 0..3 {
            for seg in 0..n_segments {
                let offset = (ci * n_segments + seg) * p;
                polynomial::differentiate(
                    spline.power_basis_polynomial(seg, ci),
                    &mut deriv_polys[offset..offset + p],
                );
            }
        }
        let n_samples = 51;
        for si in 0..=n_samples {
            let t = si as f64 / n_samples as f64;
            let u = domain_lo + t * (domain_hi - domain_lo);
            // Find the power basis segment: binary search in unique knots.
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
            // Remap u to local parameter [0, 1] within this segment.
            let u_lo = unique[seg];
            let u_hi = unique[seg + 1];
            let t_local = (u - u_lo) / (u_hi - u_lo);
            // Evaluate the power basis polynomial for each coordinate.
            let x = polynomial::eval(spline.power_basis_polynomial(seg, 0), t_local);
            let y = polynomial::eval(spline.power_basis_polynomial(seg, 1), t_local);
            let z = polynomial::eval(spline.power_basis_polynomial(seg, 2), t_local);
            let from_poly = Vec3(x, y, z);
            // Evaluate point and tangent from spline in one call.
            let mut results = [Vec3(0., 0., 0.); 2];
            spline.point_with_derivs(u, &mut results).unwrap();
            let from_spline = results[0];
            let tangent_spline = results[1];
            let diff = from_spline - from_poly;
            let err = diff.0.abs().max(diff.1.abs()).max(diff.2.abs());
            assert!(
                err < 1e-10,
                "point: seg={seg}, u={u}, t_local={t_local}: spline={from_spline:?} poly={from_poly:?} err={err}"
            );
            // Compare derivatives. The power basis polynomial is parameterized
            // by t_local in [0, 1], so by chain rule:
            //   d/du = (d/dt_local) / (u_hi - u_lo)
            let span = u_hi - u_lo;
            let dpoly = |ci: usize| {
                let offset = (ci * n_segments + seg) * p;
                &deriv_polys[offset..offset + p]
            };
            let dx = polynomial::eval(dpoly(0), t_local) / span;
            let dy = polynomial::eval(dpoly(1), t_local) / span;
            let dz = polynomial::eval(dpoly(2), t_local) / span;
            let tangent_poly = Vec3(dx, dy, dz);
            let diff = tangent_spline - tangent_poly;
            let err = diff.0.abs().max(diff.1.abs()).max(diff.2.abs());
            assert!(
                err < 1e-8,
                "tangent: seg={seg}, u={u}, t_local={t_local}: spline={tangent_spline:?} poly={tangent_poly:?} err={err}"
            );
        }
    }

    #[test]
    fn t_power_basis_matches_eval() {
        // Single segment, degree 1.
        check_power_basis(
            &Spline::create(
                vec![Vec3(0., 0., 0.), Vec3(3., 4., 5.)],
                vec![0.0, 0.0, 1.0, 1.0],
                1,
            )
            .unwrap(),
        );
        // Single segment, degree 2.
        check_power_basis(
            &Spline::create(
                vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                2,
            )
            .unwrap(),
        );
        // Single segment, degree 3.
        check_power_basis(
            &Spline::create(
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(0., 1., 0.),
                    Vec3(1., 1., 0.),
                    Vec3(1., 0., 0.),
                ],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                3,
            )
            .unwrap(),
        );
        // Multi-segment cubic, uniform clamped.
        check_power_basis(
            &Spline::create(
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(1., 2., 0.),
                    Vec3(2., -1., 3.),
                    Vec3(3., 2., 1.),
                    Vec3(4., 0., -1.),
                    Vec3(5., 3., 2.),
                    Vec3(6., 1., 0.),
                ],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                3,
            )
            .unwrap(),
        );
        // Multi-segment quadratic, clamped.
        check_power_basis(
            &Spline::create_clamped(
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(1., 4., 0.),
                    Vec3(3., -2., 1.),
                    Vec3(5., 1., 3.),
                    Vec3(7., 0., 0.),
                ],
                2,
            )
            .unwrap(),
        );
        // Non-uniform interior knots, degree 3.
        check_power_basis(
            &Spline::create(
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(0.5, 2., 1.),
                    Vec3(1.5, -1., 2.),
                    Vec3(2.5, 3., -1.),
                    Vec3(3.5, 0., 1.),
                    Vec3(4., 1., 0.),
                ],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0],
                3,
            )
            .unwrap(),
        );
        // Interior knot with multiplicity 2, degree 3.
        check_power_basis(
            &Spline::create(
                vec![
                    Vec3(0., 0., 0.),
                    Vec3(1., 3., 0.),
                    Vec3(2., 0., 1.),
                    Vec3(3., 2., -1.),
                    Vec3(4., -1., 2.),
                    Vec3(5., 1., 0.),
                ],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                3,
            )
            .unwrap(),
        );
        // 3D control points with all nonzero components.
        check_power_basis(
            &Spline::create(
                vec![
                    Vec3(1., 2., 3.),
                    Vec3(4., 5., 6.),
                    Vec3(7., 8., 9.),
                    Vec3(10., 11., 12.),
                    Vec3(13., 14., 15.),
                    Vec3(16., 17., 18.),
                    Vec3(19., 20., 21.),
                ],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                3,
            )
            .unwrap(),
        );
    }

    /// Verify bounds are valid by sampling the spline densely and checking
    /// that all sampled points lie within the bounds, and that the bounds are tight
    /// (i.e. at least one sampled point is close to each bound face).
    fn verify_bounds(spline: &Spline, n_samples: usize) {
        let (bmin, bmax) = spline.bounds();
        let b_lo = [bmin.0, bmin.1, bmin.2];
        let b_hi = [bmax.0, bmax.1, bmax.2];
        let (lo, hi) = spline.domain();
        let mut observed_min = [f64::INFINITY; 3];
        let mut observed_max = [f64::NEG_INFINITY; 3];
        for i in 0..=n_samples {
            let t = lo + (hi - lo) * (i as f64) / (n_samples as f64);
            let pt = spline.point(t).unwrap();
            let coords = [pt.0, pt.1, pt.2];
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
        // Check tightness: observed extremes should be close to computed bounds.
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
        // Three constant segments with values 1, 5, 2 in x; y and z are 0.
        let spline = Spline::create(
            vec![Vec3(1., 0., 0.), Vec3(5., 0., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 1.0, 2.0, 3.0],
            0,
        )
        .unwrap();
        let (lo, hi) = spline.bounds();
        assert_eq!(lo.0, 1.0); // min x
        assert_eq!(hi.0, 5.0); // max x
        // Single control point: degenerate case.
        let spline = Spline::create(vec![Vec3(3., 7., -2.)], vec![0.0, 1.0], 0).unwrap();
        let (lo, hi) = spline.bounds();
        assert!((lo.0 - 3.0).abs() < 1e-12);
        assert!((lo.1 - 7.0).abs() < 1e-12);
        assert!((lo.2 - -2.0).abs() < 1e-12);
        assert!((hi.0 - 3.0).abs() < 1e-12);
        assert!((hi.1 - 7.0).abs() < 1e-12);
        assert!((hi.2 - -2.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_degree_1() {
        // Tent shape: peak at interior knot, NOT at domain endpoints.
        let spline = Spline::create_clamped(
            vec![Vec3(0., 0., 0.), Vec3(1., 3., 0.), Vec3(2., 0., 0.)],
            1,
        )
        .unwrap();
        let (lo, hi) = spline.bounds();
        assert!((lo.0 - 0.0).abs() < 1e-12); // min x
        assert!((hi.0 - 2.0).abs() < 1e-12); // max x
        assert!((lo.1 - 0.0).abs() < 1e-12); // min y
        assert!((hi.1 - 3.0).abs() < 1e-12); // max y (interior knot)
        verify_bounds(&spline, 1000);
        // V-shape: minimum at interior knot.
        let spline = Spline::create_clamped(
            vec![Vec3(0., 2., 0.), Vec3(1., -1., 0.), Vec3(2., 2., 0.)],
            1,
        )
        .unwrap();
        let (lo, hi) = spline.bounds();
        assert!((lo.1 - -1.0).abs() < 1e-12); // min y at interior knot
        assert!((hi.1 - 2.0).abs() < 1e-12); // max y at endpoints
        verify_bounds(&spline, 1000);
        // Straight line: bounds should be tight to endpoints.
        let spline = Spline::create_clamped(vec![Vec3(1., 2., 3.), Vec3(4., 5., 6.)], 1).unwrap();
        let (lo, hi) = spline.bounds();
        assert!((lo.0 - 1.0).abs() < 1e-12);
        assert!((lo.1 - 2.0).abs() < 1e-12);
        assert!((lo.2 - 3.0).abs() < 1e-12);
        assert!((hi.0 - 4.0).abs() < 1e-12);
        assert!((hi.1 - 5.0).abs() < 1e-12);
        assert!((hi.2 - 6.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_degree_2() {
        // Single segment (Bézier): extremum inside segment.
        // Quadratic Bézier: P0=(0,0,0), P1=(0.5,2,0), P2=(1,0,0)
        // y(t) = 2*2*t*(1-t) = 4t - 4t^2, max at t=0.5 => y=1.0
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(0.5, 2., 0.), Vec3(1., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        let (lo, hi) = spline.bounds();
        assert!((hi.1 - 1.0).abs() < 1e-10); // max y = 1.0 found by root-finding
        assert!((lo.1 - 0.0).abs() < 1e-10); // min y at endpoints
        verify_bounds(&spline, 1000);
        // Multi-segment: extrema in both segments.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(2., -1., 0.),
                Vec3(3., 0., 0.),
            ],
            2,
        )
        .unwrap();
        let (lo, hi) = spline.bounds();
        assert!(lo.1 < 0.0); // min y should be negative
        assert!(hi.1 > 0.0); // max y should be positive
        verify_bounds(&spline, 10000);
    }

    #[test]
    fn bounds_degree_3() {
        // Cubic with overshoot in all 3 coordinates.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 5., -2.),
                Vec3(2., -3., 4.),
                Vec3(3., 5., -2.),
                Vec3(4., 0., 0.),
            ],
            3,
        )
        .unwrap();
        verify_bounds(&spline, 10000);
        // Many segments.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., -1.),
                Vec3(2., -1., 3.),
                Vec3(3., 4., -2.),
                Vec3(4., -3., 1.),
                Vec3(5., 1., -1.),
                Vec3(6., 0., 0.),
            ],
            3,
        )
        .unwrap();
        verify_bounds(&spline, 10000);
    }

    /// Helper: verify that reversing a spline produces a curve that traces
    /// the same path in the opposite direction.
    fn verify_reversed(spline: &Spline, n_samples: usize) {
        let (lo, hi) = spline.domain();
        let ksum = lo + hi;
        let rev = spline.clone().reversed();
        // Domain and degree are preserved.
        assert_eq!(rev.domain(), spline.domain());
        assert_eq!(rev.degree(), spline.degree());
        // Endpoints swap.
        assert_eq!(rev.start(), spline.end());
        assert_eq!(rev.end(), spline.start());
        // reversed.eval_point(u) == original.eval_point(ksum - u)
        for i in 0..=n_samples {
            let u = lo + (hi - lo) * (i as f64) / (n_samples as f64);
            let pt_orig = spline.point(ksum - u).unwrap();
            let pt_rev = rev.point(u).unwrap();
            let diff = (pt_orig.0 - pt_rev.0).abs()
                + (pt_orig.1 - pt_rev.1).abs()
                + (pt_orig.2 - pt_rev.2).abs();
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
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 3., 0.),
                Vec3(3., 1., 0.),
                Vec3(4., 4., 0.),
            ],
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            1,
        )
        .unwrap();
        verify_reversed(&spline, 1000);
    }

    #[test]
    fn t_reversed_quadratic_bezier() {
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        verify_reversed(&spline, 1000);
    }

    #[test]
    fn t_reversed_cubic_bezier() {
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        verify_reversed(&spline, 1000);
    }

    #[test]
    fn t_reversed_cubic_multi_segment() {
        let spline = Spline::create(
            vec![
                Vec3(1., 2., 3.),
                Vec3(4., 5., 6.),
                Vec3(7., 8., 9.),
                Vec3(10., 11., 12.),
                Vec3(13., 14., 15.),
                Vec3(16., 17., 18.),
                Vec3(19., 20., 21.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            3,
        )
        .unwrap();
        verify_reversed(&spline, 1000);
    }

    #[test]
    fn t_reversed_clamped_3d() {
        // Multi-segment with all 3 coordinates varying.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 5., -2.),
                Vec3(2., -3., 4.),
                Vec3(3., 5., -2.),
                Vec3(4., 0., 0.),
            ],
            3,
        )
        .unwrap();
        verify_reversed(&spline, 1000);
    }

    #[test]
    fn t_reversed_twice_is_identity() {
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., -1.),
                Vec3(2., -1., 3.),
                Vec3(3., 4., -2.),
                Vec3(4., -3., 1.),
                Vec3(5., 1., -1.),
                Vec3(6., 0., 0.),
            ],
            3,
        )
        .unwrap();
        let (lo, hi) = spline.domain();
        let roundtrip = spline.clone().reversed().reversed();
        for i in 0..=1000 {
            let u = lo + (hi - lo) * (i as f64) / 1000.0;
            let pt_orig = spline.point(u).unwrap();
            let pt_rt = roundtrip.point(u).unwrap();
            let diff = (pt_orig.0 - pt_rt.0).abs()
                + (pt_orig.1 - pt_rt.1).abs()
                + (pt_orig.2 - pt_rt.2).abs();
            assert!(diff < 1e-10, "Round-trip mismatch at u={u}");
        }
    }

    #[test]
    fn t_reversed_single_segment() {
        // Minimal: degree 1, two control points.
        let spline = Spline::create_clamped(vec![Vec3(1., 2., 3.), Vec3(4., 5., 6.)], 1).unwrap();
        verify_reversed(&spline, 100);
    }

    #[test]
    fn t_length_approx_degree_1_straight_line() {
        // Straight line from (0,0,0) to (3,4,0). Length = 5.0 exactly.
        let spline = Spline::create_clamped(vec![Vec3(0., 0., 0.), Vec3(3., 4., 0.)], 1).unwrap();
        let len = spline.length(1e-6);
        assert!((len - 5.0).abs() < 1e-6, "Expected 5.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_1_polyline_3d() {
        // Piecewise linear 3D path: (0,0,0)→(1,0,0)→(1,1,0)→(1,1,1). Length = 3.0.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 0., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 1., 1.),
            ],
            1,
        )
        .unwrap();
        let len = spline.length(1e-6);
        assert!((len - 3.0).abs() < 1e-6, "Expected 3.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_1_zero_length() {
        // Two identical control points. Length = 0.
        let spline = Spline::create_clamped(vec![Vec3(1., 2., 3.), Vec3(1., 2., 3.)], 1).unwrap();
        let len = spline.length(1e-6);
        assert!(len.abs() < 1e-12, "Expected 0.0, got {len}");
    }

    #[test]
    fn t_length_approx_degree_2_bezier() {
        // Quadratic Bézier: P0=(0,0,0), P1=(0.5,1,0), P2=(1,0,0).
        // B'(t) = (1, 2-4t, 0), |B'| = sqrt(1 + (2-4t)^2).
        // Exact length = (1/4) * [u*sqrt(1+u^2) + asinh(u)] from -2 to 2
        //              = 1.4789428575445974...
        let expected = {
            let f = |u: f64| (u * (1.0 + u * u).sqrt() + u.asinh()) / 2.0;
            (f(2.0) - f(-2.0)) / 4.0
        };
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(0.5, 1., 0.), Vec3(1., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-6);
        assert!(
            (fine - expected).abs() < 1e-4,
            "Expected {expected}, got {fine}"
        );
        // Finer tolerance should be closer to truth than coarser.
        assert!(
            (fine - expected).abs() <= (coarse - expected).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - expected).abs(),
            (coarse - expected).abs()
        );
    }

    #[test]
    fn t_length_approx_degree_3_bezier_exact() {
        // Cubic Bézier: P0=(0,0,0), P1=(0,1,0), P2=(1,1,0), P3=(1,0,0).
        // |B'(t)| = 3*(2t^2 - 2t + 1). Exact length = 2.0.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(0., 1., 0.),
                Vec3(1., 1., 0.),
                Vec3(1., 0., 0.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-6);
        assert!((fine - 2.0).abs() < 1e-4, "Expected 2.0, got {fine}");
        // Finer tolerance should be closer to truth than coarser.
        assert!(
            (fine - 2.0).abs() <= (coarse - 2.0).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - 2.0).abs(),
            (coarse - 2.0).abs()
        );
    }

    #[test]
    fn t_length_approx_cubic_multi_segment() {
        // Multi-segment cubic. Verify convergence toward a reference value.
        let spline = Spline::create_clamped(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., -1.),
                Vec3(2., -1., 3.),
                Vec3(3., 4., -2.),
                Vec3(4., 0., 0.),
            ],
            3,
        )
        .unwrap();
        let reference = spline.length(1e-6);
        let coarse = spline.length(1e-2);
        let fine = spline.length(1e-4);
        // Both should be close to the reference.
        assert!(
            (fine - reference).abs() < 1e-4,
            "fine {fine} not close to reference {reference}"
        );
        // Finer tolerance should be closer to truth than coarser.
        assert!(
            (fine - reference).abs() <= (coarse - reference).abs() + 1e-12,
            "fine err {} > coarse err {}",
            (fine - reference).abs(),
            (coarse - reference).abs()
        );
        // Both should be positive.
        assert!(fine > 0.0);
        // Sanity: length should be less than the control polygon length (upper bound).
        let polygon_len: f64 = spline
            .control_points()
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum();
        assert!(fine <= polygon_len + 1e-12);
    }

    #[test]
    fn t_length_approx_3d_cubic() {
        // Cubic Bézier with all 3 coordinates active.
        // Verify convergence: finer tolerance → closer to truth.
        let spline = Spline::create(
            vec![
                Vec3(0., 0., 0.),
                Vec3(1., 2., 3.),
                Vec3(3., 1., -1.),
                Vec3(4., 0., 2.),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            3,
        )
        .unwrap();
        let reference = spline.length(1e-6);
        let l1 = spline.length(1e-1);
        let l2 = spline.length(1e-3);
        let l3 = spline.length(1e-5);
        // Each finer tolerance should be closer to truth than the coarser one.
        let e1 = (l1 - reference).abs();
        let e2 = (l2 - reference).abs();
        let e3 = (l3 - reference).abs();
        assert!(e3 <= e2 + 1e-12, "l3 err {e3} > l2 err {e2}");
        assert!(e2 <= e1 + 1e-12, "l2 err {e2} > l1 err {e1}");
        // Finest should be close to reference.
        assert!(e3 < 1e-4, "l3 err {e3} too large");
        // Chord length is a lower bound, polygon is upper bound.
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
        // Length should be the same for the reversed curve.
        let cps = vec![
            Vec3(0., 0., 0.),
            Vec3(1., 5., -2.),
            Vec3(2., -3., 4.),
            Vec3(3., 5., -2.),
            Vec3(4., 0., 0.),
        ];
        let spline = Spline::create_clamped(cps, 3).unwrap();
        let len_fwd = spline.length(1e-6);
        let len_rev = spline.reversed().length(1e-6);
        assert!(
            (len_fwd - len_rev).abs() < 1e-6,
            "Forward {len_fwd} != reversed {len_rev}"
        );
    }
}
