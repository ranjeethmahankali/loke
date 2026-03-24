use crate::{Vec3, error::Error};
use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
};

struct PascalTriangle {
    entries: Vec<usize>,
    n_rows: usize,
}

// Thread local cache to store lazily computed binomial coefficients.
thread_local! {
    static PASCAL_TRIANGLE: RefCell<PascalTriangle> = RefCell::new(PascalTriangle { entries: vec![1], n_rows: 1 });
}

pub(crate) fn binomial_coeff(n: usize, k: usize) -> usize {
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
    rows: usize,
    cols: usize,
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
    rows: usize,
    cols: usize,
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
    let mut ders = View2DMut {
        data: &mut buf.basis,
        rows: n_derivs + 1,
        cols: degree + 1,
    };
    buf.ndu.clear();
    buf.ndu.resize((degree + 1).pow(2), 0.);
    let mut ndu = View2DMut::<f64> {
        data: &mut buf.ndu,
        rows: degree + 1,
        cols: degree + 1,
    };
    buf.alt_coeff.clear();
    buf.alt_coeff.resize((degree + 1) * 2, 0.);
    let mut a = View2DMut::<f64> {
        data: &mut buf.alt_coeff,
        rows: 2,
        cols: degree + 1,
    };
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
            let rk: isize = r as isize - k as isize;
            let pk: isize = degree as isize - k as isize;
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1: isize = if rk >= -1 { 1 } else { -rk };
            let j2: isize = if r as isize - 1 <= pk {
                k as isize - 1
            } else {
                degree as isize - r as isize
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

pub struct Spline {
    knots: Vec<f64>,
    control_points: Vec<Vec3>,
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
        Ok(Self {
            knots,
            control_points,
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
        Ok(Self {
            knots,
            control_points,
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

    pub fn eval_point(&self, u: f64) -> Option<Vec3> {
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

    pub fn eval_tangent(&self, u: f64) -> Option<Vec3> {
        let mut results = [Vec3(0., 0., 0.); 2];
        self.eval_point_with_derivs(u, &mut results)
            .ok()
            .map(|()| results[1])
    }

    pub fn eval_point_with_derivs(&self, u: f64, results: &mut [Vec3]) -> Result<(), Error> {
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
            let nders = View2D {
                data: &buf.basis,
                rows: n_derivs + 1,
                cols: degree + 1,
            };
            let span = span - degree;
            for k in 0..=n_derivs {
                for j in 0..=degree {
                    results[k] += self.control_points[span + j] * nders[k][j];
                }
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn binomial_coefficients() {
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
    fn find_span_clamped_cubic() {
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
    fn create_validates_knot_count() {
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
    fn create_clamped_validates_control_points() {
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
    fn create_clamped_knot_generation() {
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
    fn eval_point_out_of_domain() {
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
        assert!(spline.eval_point(-1.0).is_none());
        assert!(spline.eval_point(-f64::EPSILON).is_none());
        assert!(spline.eval_point(4.0_f64.next_up()).is_none());
        assert!(spline.eval_point(100.0).is_none());
        assert!(spline.eval_point(f64::NEG_INFINITY).is_none());
        assert!(spline.eval_point(f64::INFINITY).is_none());
        assert!(spline.eval_point(f64::NAN).is_none());
        // In domain returns Some.
        assert!(spline.eval_point(0.0).is_some());
        assert!(spline.eval_point(2.0).is_some());
        assert!(spline.eval_point(4.0).is_some());
    }

    #[test]
    fn eval_point_linear() {
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
        assert_eq!(spline.eval_point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.eval_point(1.0).unwrap(), Vec3(1., 3., 0.));
        assert_eq!(spline.eval_point(2.0).unwrap(), Vec3(3., 1., 0.));
        assert_eq!(spline.eval_point(3.0).unwrap(), Vec3(4., 4., 0.));
        // Midpoints: linear interpolation between control points.
        assert_eq!(spline.eval_point(0.5).unwrap(), Vec3(0.5, 1.5, 0.));
        assert_eq!(spline.eval_point(1.5).unwrap(), Vec3(2., 2., 0.));
        assert_eq!(spline.eval_point(2.5).unwrap(), Vec3(3.5, 2.5, 0.));
    }

    #[test]
    fn eval_point_quadratic_bezier() {
        // Single-span quadratic Bezier: C(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        // knots [0,0,0,1,1,1], 3 control points, domain [0,1].
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 2., 0.), Vec3(2., 0., 0.)],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
        )
        .unwrap();
        // Endpoints match first and last control points.
        assert_eq!(spline.eval_point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.eval_point(1.0).unwrap(), Vec3(2., 0., 0.));
        // t=0.5: 0.25*(0,0,0) + 0.5*(1,2,0) + 0.25*(2,0,0) = (1,1,0)
        assert_eq!(spline.eval_point(0.5).unwrap(), Vec3(1., 1., 0.));
        // t=0.25: 0.5625*(0,0,0) + 0.375*(1,2,0) + 0.0625*(2,0,0) = (0.5, 0.75, 0)
        assert_eq!(spline.eval_point(0.25).unwrap(), Vec3(0.5, 0.75, 0.));
        // t=0.75: 0.0625*(0,0,0) + 0.375*(1,2,0) + 0.5625*(2,0,0) = (1.5, 0.75, 0)
        assert_eq!(spline.eval_point(0.75).unwrap(), Vec3(1.5, 0.75, 0.));
    }

    #[test]
    fn eval_point_cubic_bezier() {
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
        assert_eq!(spline.eval_point(0.0).unwrap(), Vec3(0., 0., 0.));
        assert_eq!(spline.eval_point(1.0).unwrap(), Vec3(1., 0., 0.));
        // t=0.5: 0.125*(0,0,0) + 0.375*(0,1,0) + 0.375*(1,1,0) + 0.125*(1,0,0)
        //       = (0,0,0) + (0, 0.375, 0) + (0.375, 0.375, 0) + (0.125, 0, 0)
        //       = (0.5, 0.75, 0)
        assert_eq!(spline.eval_point(0.5).unwrap(), Vec3(0.5, 0.75, 0.));
    }

    #[test]
    fn eval_derivs_out_of_domain() {
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
        assert!(spline.eval_point_with_derivs(-1.0, &mut results).is_err());
        assert!(spline.eval_point_with_derivs(5.0, &mut results).is_err());
        assert!(
            spline
                .eval_point_with_derivs(f64::NAN, &mut results)
                .is_err()
        );
        // Empty results slice is a no-op, even out of domain.
        assert!(spline.eval_point_with_derivs(-1.0, &mut []).is_ok());
    }

    #[test]
    fn eval_derivs_linear() {
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
        spline.eval_point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(0.5).unwrap());
        assert_eq!(results[1], Vec3(1., 3., 0.));
        // At t=1.5 (second span): derivative = P2 - P1 = (2,-2,0).
        spline.eval_point_with_derivs(1.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(1.5).unwrap());
        assert_eq!(results[1], Vec3(2., -2., 0.));
        // At t=2.5 (third span): derivative = P3 - P2 = (1,3,0).
        spline.eval_point_with_derivs(2.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(2.5).unwrap());
        assert_eq!(results[1], Vec3(1., 3., 0.));
        // Requesting only the point (no derivatives).
        let mut results = [Vec3(0., 0., 0.); 1];
        spline.eval_point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(0.5).unwrap());
    }

    #[test]
    fn eval_derivs_quadratic_bezier() {
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
        spline.eval_point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(0., 0., 0.));
        assert_eq!(results[1], Vec3(2., 4., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // At t=1: C'(1) = 2(P2-P1) = (2,-4,0)
        spline.eval_point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(2., 0., 0.));
        assert_eq!(results[1], Vec3(2., -4., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // At t=0.5: C'(0.5) = 2[0.5*(1,2,0) + 0.5*(1,-2,0)] = 2*(1,0,0) = (2,0,0)
        spline.eval_point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(0.5).unwrap());
        assert_eq!(results[1], Vec3(2., 0., 0.));
        assert_eq!(results[2], Vec3(0., -8., 0.));
        // Requesting more derivatives than degree: 3rd derivative should be zero.
        let mut results = [Vec3(0., 0., 0.); 4];
        spline.eval_point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[3], Vec3(0., 0., 0.));
    }

    #[test]
    fn eval_derivs_cubic_bezier() {
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
        spline.eval_point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(0., 0., 0.));
        assert_eq!(results[1], Vec3(0., 3., 0.));
        // At t=1: C'(1) = 3(P3-P2) = 3*(0,-1,0) = (0,-3,0)
        spline.eval_point_with_derivs(1.0, &mut results).unwrap();
        assert_eq!(results[0], Vec3(1., 0., 0.));
        assert_eq!(results[1], Vec3(0., -3., 0.));
        // At t=0.5: point should match eval_point.
        // C'(0.5) = 3[0.25*(0,1,0) + 0.5*(1,0,0) + 0.25*(0,-1,0)]
        //         = 3*(0.5, 0, 0) = (1.5, 0, 0)
        spline.eval_point_with_derivs(0.5, &mut results).unwrap();
        assert_eq!(results[0], spline.eval_point(0.5).unwrap());
        assert_eq!(results[1], Vec3(1.5, 0., 0.));
    }

    #[test]
    fn eval_tangent_out_of_domain() {
        // Degree 1, domain [0, 2].
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(1., 1., 0.), Vec3(3., 0., 0.)],
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        )
        .unwrap();
        assert!(spline.eval_tangent(-0.1).is_none());
        assert!(spline.eval_tangent(2.1).is_none());
        assert!(spline.eval_tangent(f64::NAN).is_none());
    }

    #[test]
    fn eval_tangent_linear() {
        // Degree 1: tangent within each span equals the difference of adjacent control points.
        // P0=(0,0,0), P1=(2,6,0), P2=(5,3,0)  knots [0,0,1,2,2]
        let spline = Spline::create(
            vec![Vec3(0., 0., 0.), Vec3(2., 6., 0.), Vec3(5., 3., 0.)],
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
            1,
        )
        .unwrap();
        // First span tangent = P1 - P0 = (2,6,0)
        assert_eq!(spline.eval_tangent(0.0).unwrap(), Vec3(2., 6., 0.));
        assert_eq!(spline.eval_tangent(0.5).unwrap(), Vec3(2., 6., 0.));
        // Second span tangent = P2 - P1 = (3,-3,0)
        assert_eq!(spline.eval_tangent(1.5).unwrap(), Vec3(3., -3., 0.));
        assert_eq!(spline.eval_tangent(2.0).unwrap(), Vec3(3., -3., 0.));
    }

    #[test]
    fn eval_tangent_quadratic_bezier() {
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
        assert_eq!(spline.eval_tangent(0.0).unwrap(), Vec3(0., 2., 0.));
        // C'(1) = 2(-1,0,0) = (-2,0,0)
        assert_eq!(spline.eval_tangent(1.0).unwrap(), Vec3(-2., 0., 0.));
        // C'(0.5) = 2[0.5*(0,1,0) + 0.5*(-1,0,0)] = (-1,1,0)
        assert_eq!(spline.eval_tangent(0.5).unwrap(), Vec3(-1., 1., 0.));
        // C'(0.25) = 2[0.75*(0,1,0) + 0.25*(-1,0,0)] = (-0.5, 1.5, 0)
        assert_eq!(spline.eval_tangent(0.25).unwrap(), Vec3(-0.5, 1.5, 0.));
    }

    #[test]
    fn eval_tangent_cubic_bezier() {
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
        assert_eq!(spline.eval_tangent(0.0).unwrap(), Vec3(3., 0., 0.));
        // C'(1) = 3(-1,0,0) = (-3,0,0)
        assert_eq!(spline.eval_tangent(1.0).unwrap(), Vec3(-3., 0., 0.));
        // C'(0.5) = 3[0.25*(1,0,0) + 0.5*(0,1,0) + 0.25*(-1,0,0)]
        //         = 3*(0, 0.5, 0) = (0, 1.5, 0)
        assert_eq!(spline.eval_tangent(0.5).unwrap(), Vec3(0., 1.5, 0.));
    }

    #[test]
    fn eval_point_clamped_cubic_endpoints() {
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
        assert_eq!(spline.eval_point(0.0).unwrap(), Vec3(1., 2., 3.));
        assert_eq!(spline.eval_point(4.0).unwrap(), Vec3(19., 20., 21.));
    }
}
