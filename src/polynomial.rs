// Rust port of cyPolynomial.h by Cem Yuksel
// High-Performance Polynomial Root Finding for Graphics (2022)
//
// Coefficients are in ascending degree order: coef[0] + coef[1]*x + ... + coef[N]*x^N
//
// Supports polynomials up to degree 16.

use crate::{ScalarAdaptor, error::Error};

// ---- Helpers ----

#[inline(always)]
fn mult_sign<P: ScalarAdaptor>(v: P::Float, sign: P::Float) -> P::Float {
    if sign < P::scalar(0.0) { -v } else { v }
}

#[inline(always)]
fn is_different_sign<P: ScalarAdaptor>(a: P::Float, b: P::Float) -> bool {
    (a < P::scalar(0.0)) != (b < P::scalar(0.0))
}

/// Evaluate polynomial using Horner's method.
#[inline(always)]
pub fn eval<P: ScalarAdaptor>(coef: &[P::Float], x: P::Float) -> P::Float {
    coef.iter()
        .rev()
        .fold(P::scalar(0.0), |acc, c| P::mul_add(acc, x, *c))
}

/// Evaluate polynomial and its derivative simultaneously using Horner's method.
/// Returns (f(x), f'(x)). Saves N-1 multiplications vs two separate evaluations.
#[inline(always)]
pub fn eval_with_deriv<P: ScalarAdaptor>(coef: &[P::Float], x: P::Float) -> (P::Float, P::Float) {
    let n = coef.len() - 1;
    let mut r = coef[n];
    let mut d = P::scalar(0.0);
    for i in (0..n).rev() {
        d = P::mul_add(d, x, r);
        r = P::mul_add(r, x, coef[i]);
    }
    (r, d)
}

/// Differentiate a polynomial and write the coefficients of the derivative
/// polynomial into `deriv`.
#[inline(always)]
pub(crate) fn differentiate<P: ScalarAdaptor>(coef: &[P::Float], deriv: &mut [P::Float]) {
    let n = coef.len() - 1;
    debug_assert_eq!(deriv.len() + 1, coef.len());
    for i in 0..n {
        deriv[i] = (P::scalar(i as f64 + 1.0)) * coef[i + 1];
    }
}

/// Multiply two polynomials and add the result to `sum`.
/// `sum` must have length >= `a.len() + b.len() - 1`.
#[inline(always)]
pub(crate) fn mul_add<P: ScalarAdaptor>(a: &[P::Float], b: &[P::Float], dst: &mut [P::Float]) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    debug_assert!(dst.len() >= a.len() + b.len() - 1);
    for i in 0..a.len() {
        for j in 0..b.len() {
            dst[i + j] += a[i] * b[j];
        }
    }
}

/// Given a polynomial and one of its roots, this deflates the polynomial to one
/// lower degree. This is the equivalent of dividing the original polynomial
/// with `(x - R)` where `R` is the provided root. The coefficients of the
/// resulting polynomial are written into `def_poly`.
#[inline(always)]
fn deflate<P: ScalarAdaptor>(coef: &[P::Float], root: P::Float, def_poly: &mut [P::Float]) {
    let n = coef.len() - 1;
    debug_assert_eq!(def_poly.len(), n);
    def_poly[n - 1] = coef[n];
    for i in (0..n - 1).rev() {
        def_poly[i] = P::mul_add(root, def_poly[i + 1], coef[i + 1]);
    }
}

/// Finds roots in a closed interval, using a combination of Newton method and bisection.
fn find_closed<P: ScalarAdaptor>(
    coef: &[P::Float],
    x0: P::Float,
    x1: P::Float,
    y0: P::Float,
    x_error: P::Float,
) -> P::Float {
    let n = coef.len() - 1;
    let ep2 = P::scalar(2.0) * x_error;
    let mut xr = (x0 + x1) / P::scalar(2.0);
    if x1 - x0 <= ep2 {
        return xr;
    }
    // Fast Newton path for low degree
    if n == 2 || n == 3 {
        let xr0 = xr;
        for _ in 0..16 {
            let (fx, dy) = eval_with_deriv::<P>(coef, xr);
            let xn = P::clamp(xr - fx / dy, x0, x1);
            if P::abs(xr - xn) <= x_error {
                return xn;
            }
            xr = xn;
        }
        if !P::is_finite(xr) {
            xr = xr0;
        }
    }
    let (mut yr, mut dy) = eval_with_deriv::<P>(coef, xr);
    let mut xb0 = x0;
    let mut xb1 = x1;
    loop {
        let side = is_different_sign::<P>(y0, yr);
        if side {
            xb1 = xr;
        } else {
            xb0 = xr;
        }
        let dx = yr / dy;
        let xn = xr - dx;
        if xn > xb0 && xn < xb1 {
            let stepsize = P::abs(xr - xn);
            xr = xn;
            if stepsize > x_error {
                (yr, dy) = eval_with_deriv::<P>(coef, xr);
            } else {
                break;
            }
        } else {
            xr = (xb0 + xb1) / P::scalar(2.0);
            if xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2 {
                break;
            }
            (yr, dy) = eval_with_deriv::<P>(coef, xr);
        }
    }
    xr
}

fn find_open_impl<P: ScalarAdaptor>(
    coef: &[P::Float],
    deriv: &[P::Float],
    mut xm: P::Float,
    mut ym: P::Float,
    mut xr: P::Float,
    x_error: P::Float,
    open_min: bool,
) -> P::Float {
    let mut delta = P::scalar(1.0);
    let mut yr = eval::<P>(coef, xr);
    let mut otherside = is_different_sign::<P>(ym, yr);
    while yr != P::scalar(0.0) {
        if otherside {
            return if open_min {
                find_closed::<P>(coef, xr, xm, yr, x_error)
            } else {
                find_closed::<P>(coef, xm, xr, ym, x_error)
            };
        }
        // Search the open interval
        loop {
            xm = xr;
            ym = yr;
            let dy = eval::<P>(deriv, xr);
            let dx = yr / dy;
            let xn = xr - dx;
            let dif = if open_min { xr - xn } else { xn - xr };
            if dif <= P::scalar(0.0) && P::is_finite(xn) {
                xr = xn;
                if dif <= x_error {
                    if xr == xm {
                        return xr;
                    }
                    let xs = xn
                        - mult_sign::<P>(
                            x_error,
                            if open_min {
                                P::scalar(-1.0)
                            } else {
                                P::scalar(0.0)
                            },
                        );
                    let ys = eval::<P>(coef, xs);
                    let s = is_different_sign::<P>(ym, ys);
                    if s {
                        return xr;
                    }
                    xr = xs;
                    yr = ys;
                    continue; // goto open_interval
                }
            } else {
                xr = if open_min { xr - delta } else { xr + delta };
                delta *= P::scalar(2.0);
            }
            yr = eval::<P>(coef, xr);
            otherside = is_different_sign::<P>(ym, yr);
            break;
        }
    }
    xr
}

#[inline(always)]
fn find_open_min<P: ScalarAdaptor>(
    coef: &[P::Float],
    deriv: &[P::Float],
    x1: P::Float,
    y1: P::Float,
    x_error: P::Float,
) -> P::Float {
    find_open_impl::<P>(coef, deriv, x1, y1, x1 - P::scalar(1.0), x_error, true)
}

#[inline(always)]
fn find_open_max<P: ScalarAdaptor>(
    coef: &[P::Float],
    deriv: &[P::Float],
    x0: P::Float,
    y0: P::Float,
    x_error: P::Float,
) -> P::Float {
    find_open_impl::<P>(coef, deriv, x0, y0, x0 + P::scalar(1.0), x_error, false)
}

fn find_open<P: ScalarAdaptor>(
    coef: &[P::Float],
    deriv: &[P::Float],
    x_error: P::Float,
) -> P::Float {
    let n = coef.len() - 1;
    debug_assert!(n & 1 == 1, "FindOpen only works for odd degree polynomials");
    let xr = P::scalar(0.0);
    let yr = coef[0];
    if is_different_sign::<P>(coef[n], yr) {
        find_open_max::<P>(coef, deriv, xr, yr, x_error)
    } else {
        find_open_min::<P>(coef, deriv, xr, yr, x_error)
    }
}

// ---- Linear root ----

fn linear_root_bounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    x0: P::Float,
    x1: P::Float,
) -> (P::Float, usize) {
    if coef[1] != P::scalar(0.0) {
        let r = -coef[0] / coef[1];
        (r, if r >= x0 && r <= x1 { 1 } else { 0 })
    } else {
        (
            (x0 + x1) / P::scalar(2.0),
            if coef[0] == P::scalar(0.0) { 1 } else { 0 },
        )
    }
}

fn linear_root_unbounded<P: ScalarAdaptor>(coef: &[P::Float]) -> (P::Float, usize) {
    (
        -coef[0] / coef[1],
        if coef[1] != P::scalar(0.0) { 1 } else { 0 },
    )
}

// ---- Quadratic roots ----

fn quadratic_roots_unbounded<P: ScalarAdaptor>(coef: &[P::Float], roots: &mut [P::Float]) -> usize {
    let c = coef[0];
    let b = coef[1];
    let a = coef[2];
    let delta = b * b - P::scalar(4.0) * a * c;
    if delta > P::scalar(0.0) {
        let d = P::sqrt(delta);
        let q = P::scalar(-0.5) * (b + mult_sign::<P>(d, b));
        let rv0 = q / a;
        let rv1 = c / q;
        roots[0] = P::min(rv0, rv1);
        roots[1] = P::max(rv1, rv0);
        2
    } else if delta < P::scalar(0.0) {
        0
    } else {
        roots[0] = P::scalar(-0.5) * b / a;
        if a != P::scalar(0.0) { 1 } else { 0 }
    }
}

fn quadratic_roots_bounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    roots: &mut [P::Float],
    x0: P::Float,
    x1: P::Float,
) -> usize {
    let c = coef[0];
    let b = coef[1];
    let a = coef[2];
    let delta = b * b - P::scalar(4.0) * a * c;
    if delta > P::scalar(0.0) {
        let d = P::sqrt(delta);
        let q = P::scalar(-0.5) * (b + mult_sign::<P>(d, b));
        let rv0 = q / a;
        let rv1 = c / q;
        let r0 = P::min(rv0, rv1);
        let r1 = P::max(rv1, rv0);
        let r0i = (r0 >= x0 && r0 <= x1) as usize;
        let r1i = (r1 >= x0 && r1 <= x1) as usize;
        roots[0] = r0;
        roots[r0i as usize] = r1;
        r0i + r1i
    } else if delta < P::scalar(0.0) {
        0
    } else {
        let r0 = P::scalar(-0.5) * b / a;
        roots[0] = r0;
        (r0 >= x0 && r0 <= x1) as usize
    }
}

// ---- Cubic roots ----

fn cubic_roots_bounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    roots: &mut [P::Float],
    x0: P::Float,
    x1: P::Float,
    x_error: P::Float,
) -> usize {
    let y0 = eval::<P>(coef, x0);
    let y1 = eval::<P>(coef, x1);
    let a = coef[3] * P::scalar(3.0);
    let b_2 = coef[2];
    let c = coef[1];
    let delta_4 = b_2 * b_2 - a * c;
    if delta_4 > P::scalar(0.0) {
        let d_2 = P::sqrt(delta_4);
        let q = -(b_2 + mult_sign::<P>(d_2, b_2));
        let rv0 = q / a;
        let rv1 = c / q;
        let xa = P::min(rv0, rv1);
        let xb = P::max(rv0, rv1);
        if is_different_sign::<P>(y0, y1) {
            if xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1) {
                roots[0] = find_closed::<P>(coef, x0, x1, y0, x_error);
                return 1;
            }
        } else if (xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1) {
            return 0;
        }
        let num_roots = 0usize;
        if xa > x0 {
            let ya = eval::<P>(coef, xa);
            if is_different_sign::<P>(y0, ya) {
                roots[0] = find_closed::<P>(coef, x0, xa, y0, x_error);
                if is_different_sign::<P>(ya, y1)
                    || (xb < x1 && is_different_sign::<P>(ya, eval::<P>(coef, xb)))
                {
                    let mut def_poly = [P::scalar(0.0); 4];
                    deflate::<P>(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_bounded::<P>(&def_poly[..3], &mut roots[1..], xa, x1)
                        + 1;
                } else {
                    return 1;
                }
            }
            if xb < x1 {
                let yb = eval::<P>(coef, xb);
                if is_different_sign::<P>(ya, yb) {
                    roots[0] = find_closed::<P>(coef, xa, xb, ya, x_error);
                    if is_different_sign::<P>(yb, y1) {
                        let mut def_poly = [P::scalar(0.0); 4];
                        deflate::<P>(coef, roots[0], &mut def_poly[..3]);
                        return quadratic_roots_bounded::<P>(
                            &def_poly[..3],
                            &mut roots[1..],
                            xb,
                            x1,
                        ) + 1;
                    } else {
                        return 1;
                    }
                }
                if is_different_sign::<P>(yb, y1) {
                    roots[0] = find_closed::<P>(coef, xb, x1, yb, x_error);
                    return 1;
                }
            } else if is_different_sign::<P>(ya, y1) {
                roots[0] = find_closed::<P>(coef, xa, x1, ya, x_error);
                return 1;
            }
        } else {
            let yb = eval::<P>(coef, xb);
            if is_different_sign::<P>(y0, yb) {
                roots[0] = find_closed::<P>(coef, x0, xb, y0, x_error);
                if is_different_sign::<P>(yb, y1) {
                    let mut def_poly = [P::scalar(0.0); 4];
                    deflate::<P>(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_bounded::<P>(&def_poly[..3], &mut roots[1..], xb, x1)
                        + 1;
                } else {
                    return 1;
                }
            }
            if is_different_sign::<P>(yb, y1) {
                roots[0] = find_closed::<P>(coef, xb, x1, yb, x_error);
                return 1;
            }
        }
        num_roots
    } else {
        if is_different_sign::<P>(y0, y1) {
            roots[0] = find_closed::<P>(coef, x0, x1, y0, x_error);
            1
        } else {
            0
        }
    }
}

fn cubic_roots_unbounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    roots: &mut [P::Float],
    x_error: P::Float,
) -> usize {
    if coef[3] != P::scalar(0.0) {
        let a = coef[3] * P::scalar(3.0);
        let b_2 = coef[2];
        let c = coef[1];
        let deriv = [c, P::scalar(2.0) * b_2, a, P::scalar(0.0)];
        let delta_4 = b_2 * b_2 - a * c;
        if delta_4 > P::scalar(0.0) {
            let d_2 = P::sqrt(delta_4);
            let q = -(b_2 + mult_sign::<P>(d_2, b_2));
            let rv0 = q / a;
            let rv1 = c / q;
            let xa = P::min(rv0, rv1);
            let xb = P::max(rv0, rv1);
            let ya = eval::<P>(coef, xa);
            let yb = eval::<P>(coef, xb);
            if !is_different_sign::<P>(coef[3], ya) {
                roots[0] = find_open_min::<P>(coef, &deriv, xa, ya, x_error);
                if is_different_sign::<P>(ya, yb) {
                    let mut def_poly = [P::scalar(0.0); 4];
                    deflate::<P>(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_unbounded::<P>(&def_poly[..3], &mut roots[1..]) + 1;
                }
            } else {
                roots[0] = find_open_max::<P>(coef, &deriv, xb, yb, x_error);
            }
            1
        } else {
            let x_inf = -b_2 / a;
            let y_inf = eval::<P>(coef, x_inf);
            if is_different_sign::<P>(coef[3], y_inf) {
                roots[0] = find_open_max::<P>(coef, &deriv, x_inf, y_inf, x_error);
            } else {
                roots[0] = find_open_min::<P>(coef, &deriv, x_inf, y_inf, x_error);
            }
            1
        }
    } else {
        quadratic_roots_unbounded::<P>(coef, roots)
    }
}

// ---- General polynomial roots (degree N >= 4) ----

fn polynomial_roots_bounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    roots: &mut [P::Float],
    x0: P::Float,
    x1: P::Float,
    x_error: P::Float,
) -> usize {
    let n = coef.len() - 1; // degree
    match n {
        1 => {
            let (r, count) = linear_root_bounded::<P>(coef, x0, x1);
            roots[0] = r;
            count
        }
        2 => quadratic_roots_bounded::<P>(coef, roots, x0, x1),
        3 => cubic_roots_bounded::<P>(coef, roots, x0, x1, x_error),
        _ => {
            if coef[n] == P::scalar(0.0) {
                return polynomial_roots_bounded::<P>(&coef[..n], roots, x0, x1, x_error);
            }
            let y0 = eval::<P>(coef, x0);
            let mut deriv = [P::scalar(0.0); 16]; // max degree 16 -> 16 derivative coeffs
            differentiate::<P>(coef, &mut deriv[..n]);
            let mut deriv_roots = [P::scalar(0.0); 15];
            let nd = polynomial_roots_bounded::<P>(&deriv[..n], &mut deriv_roots, x0, x1, x_error);
            let mut x = [P::scalar(0.0); 17]; // max 16+1 = 17 entries
            let mut y = [P::scalar(0.0); 17];
            x[0] = x0;
            y[0] = y0;
            for i in 0..nd {
                x[i + 1] = deriv_roots[i];
                y[i + 1] = eval::<P>(coef, deriv_roots[i]);
            }
            x[nd + 1] = x1;
            y[nd + 1] = eval::<P>(coef, x1);
            let mut nr = 0usize;
            for i in 0..=nd {
                if is_different_sign::<P>(y[i], y[i + 1]) {
                    roots[nr] = find_closed::<P>(coef, x[i], x[i + 1], y[i], x_error);
                    nr += 1;
                }
            }
            nr
        }
    }
}

fn polynomial_roots_unbounded<P: ScalarAdaptor>(
    coef: &[P::Float],
    roots: &mut [P::Float],
    x_error: P::Float,
) -> usize {
    let n = coef.len() - 1; // degree
    match n {
        1 => {
            let (r, count) = linear_root_unbounded::<P>(coef);
            roots[0] = r;
            count
        }
        2 => quadratic_roots_unbounded::<P>(coef, roots),
        3 => cubic_roots_unbounded::<P>(coef, roots, x_error),
        _ => {
            if coef[n] == P::scalar(0.0) {
                return polynomial_roots_unbounded::<P>(&coef[..n], roots, x_error);
            }
            let mut deriv = [P::scalar(0.0); 16];
            differentiate::<P>(coef, &mut deriv[..n]);
            let mut deriv_roots = [P::scalar(0.0); 15];
            let nd = polynomial_roots_unbounded::<P>(&deriv[..n], &mut deriv_roots, x_error);
            if (n & 1 == 1) || (n & 1 == 0 && nd > 0) {
                let mut nr = 0usize;
                let mut xa = deriv_roots[0];
                let mut ya = eval::<P>(coef, xa);
                if is_different_sign::<P>(coef[n], ya) != (n & 1 == 1) {
                    roots[0] = find_open_min::<P>(coef, &deriv[..n], xa, ya, x_error);
                    nr = 1;
                }
                for &xb in deriv_roots.iter().take(nd).skip(1) {
                    let yb = eval::<P>(coef, xb);
                    if is_different_sign::<P>(ya, yb) {
                        roots[nr] = find_closed::<P>(coef, xa, xb, ya, x_error);
                        nr += 1;
                    }
                    xa = xb;
                    ya = yb;
                }
                if is_different_sign::<P>(coef[n], ya) {
                    roots[nr] = find_open_max::<P>(coef, &deriv[..n], xa, ya, x_error);
                    nr += 1;
                }
                nr
            } else if n & 1 == 1 {
                roots[0] = find_open::<P>(coef, &deriv[..n], x_error);
                1
            } else {
                0 // should not happen
            }
        }
    }
}

#[inline(always)]
fn handle_trivial_cases<P: ScalarAdaptor>(
    coeff: &[P::Float],
    roots: &mut [P::Float],
) -> Result<Option<usize>, Error> {
    if coeff.len() < 2 {
        return Ok(Some(0usize));
    }
    if coeff.len() > 17 {
        return Err(Error::OutOfBounds(
            "polynomial degree exceeds maximum supported degree of 16",
        ));
    }
    if roots.len() < coeff.len() - 1 {
        return Err(Error::OutOfBounds(
            "roots buffer too small; must have length >= degree",
        ));
    }
    Ok(None)
}

// ---- Public API ----

/// Finds all real roots of a polynomial with coefficients in ascending degree order.
/// Returns the number of roots found. Roots are written to `roots`.
///
/// `coef`: slice of length degree+1, coef[0] + coef[1]*x + ... + coef[N]*x^N
/// `roots`: output slice, must have length >= degree
/// `x_error`: positional error tolerance (use `DEFAULT_ERROR` for the default)
pub fn polynomial_roots<P: ScalarAdaptor>(
    coeff: &[P::Float],
    roots: &mut [P::Float],
    x_error: P::Float,
) -> Result<usize, Error> {
    match handle_trivial_cases::<P>(coeff, roots) {
        Ok(Some(n)) => Ok(n),
        Ok(None) => Ok(polynomial_roots_unbounded::<P>(coeff, roots, x_error)),
        Err(e) => Err(e),
    }
}

/// Finds all real roots of a polynomial within [x_min, x_max].
/// Returns the number of roots found.
pub fn polynomial_roots_in_range<P: ScalarAdaptor>(
    coeff: &[P::Float],
    roots: &mut [P::Float],
    x_min: P::Float,
    x_max: P::Float,
    x_error: P::Float,
) -> Result<usize, Error> {
    match handle_trivial_cases::<P>(coeff, roots) {
        Ok(Some(n)) => Ok(n),
        Ok(None) => Ok(polynomial_roots_bounded::<P>(
            coeff, roots, x_min, x_max, x_error,
        )),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::vec::F64Adaptor;
    pub const DEFAULT_ERROR: f64 = 6e-7;
    const TOL: f64 = 1e-6;

    /// Verify that each reported root actually evaluates close to zero.
    fn verify_roots(coef: &[f64], roots: &[f64], n: usize) {
        // Scale tolerance by the polynomial's coefficient magnitude so that
        // polynomials with large coefficients (or roots near zero where high-
        // degree terms dominate) don't produce false negatives.
        let coef_scale: f64 = coef
            .iter()
            .map(|c| c.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        for (i, &r) in roots.iter().enumerate().take(n) {
            let val = eval::<F64Adaptor>(coef, r);
            let scale = coef_scale * (1.0 + r.abs().powi(coef.len() as i32 - 1));
            assert!(
                val.abs() < TOL * scale,
                "root {} = {} does not evaluate to ~0: f(x) = {}, coef = {:?}",
                i,
                r,
                val,
                coef
            );
        }
    }

    /// Verify roots are sorted in ascending order.
    fn verify_sorted(roots: &[f64], n: usize) {
        for i in 1..n {
            assert!(
                roots[i] >= roots[i - 1],
                "roots not sorted: roots[{}] = {} < roots[{}] = {}",
                i - 1,
                roots[i - 1],
                i,
                roots[i]
            );
        }
    }

    /// Build polynomial from known roots: (x - r0)(x - r1)...(x - r_{n-1})
    fn poly_from_roots(known_roots: &[f64], coef: &mut [f64]) {
        let degree = known_roots.len();
        assert_eq!(coef.len(), degree + 1);
        for c in coef.iter_mut() {
            *c = 0.0;
        }
        coef[0] = 1.0;
        for (k, &r) in known_roots.iter().enumerate().take(degree) {
            for i in (1..=k + 1).rev() {
                coef[i] = coef[i - 1] - r * coef[i];
            }
            coef[0] *= -r;
        }
    }

    /// Check that every expected root appears in the found roots (order-independent).
    fn verify_expected_roots(expected: &[f64], found: &[f64], n: usize) {
        let found = &found[..n];
        for &e in expected {
            let closest = found
                .iter()
                .copied()
                .min_by(|a, b| (a - e).abs().partial_cmp(&(b - e).abs()).unwrap());
            match closest {
                Some(f) => assert!(
                    (f - e).abs() < TOL * (1.0 + e.abs()),
                    "expected root {} not found in {:?}",
                    e,
                    found
                ),
                None => panic!("expected root {} but no roots found", e),
            }
        }
    }

    /// Simple xorshift PRNG returning values in [lo, hi).
    fn xor_rng(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn stress_test_degree(degree: usize, iters: usize, seed: u64, lo: f64, hi: f64) {
        let mut state = seed;
        let range = hi - lo;
        for _ in 0..iters {
            let mut known = vec![0.0; degree];
            for k in known.iter_mut() {
                *k = lo + xor_rng(&mut state) * range;
            }
            let mut coef = vec![0.0; degree + 1];
            poly_from_roots(&known, &mut coef);
            let mut roots = vec![0.0; degree];
            let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
            assert_eq!(
                n as usize, degree,
                "expected {} roots for known roots {:?}, got {}",
                degree, known, n
            );
            verify_roots(&coef, &roots, n);
        }
    }

    #[test]
    fn t_eval() {
        assert_eq!(eval::<F64Adaptor>(&[42.0], 999.0), 42.0); // constant
        assert_eq!(eval::<F64Adaptor>(&[3.0, 2.0], 5.0), 13.0); // linear: 3 + 2x at x=5 -> 13
        assert_eq!(eval::<F64Adaptor>(&[1.0, -3.0, 2.0], 2.0), 3.0); // quadratic: 1 - 3x + 2x^2 at x=2 -> 1 - 6 + 8 = 3
        assert_eq!(eval::<F64Adaptor>(&[7.0, 1.0, 2.0, 3.0], 0.0), 7.0); // at zero: constant term dominates
    }

    #[test]
    fn t_eval_with_deriv_basic() {
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[7.0], 3.0); // constant: f(x) = 7, f'(x) = 0
        assert_eq!(f, 7.0);
        assert_eq!(d, 0.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[3.0, 2.0], 5.0); // linear: f(x) = 3 + 2x, f'(x) = 2 at x=5
        assert_eq!(f, 13.0);
        assert_eq!(d, 2.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[1.0, -3.0, 2.0], 2.0); // quadratic: f(x) = 1 - 3x + 2x^2, f'(x) = -3 + 4x at x=2
        assert_eq!(f, 3.0);
        assert_eq!(d, 5.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[2.0, 0.0, 0.0, 1.0], 3.0); // cubic: f(x) = 2 + x^3, f'(x) = 3x^2 at x=3
        assert_eq!(f, 29.0);
        assert_eq!(d, 27.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[5.0, 3.0, 2.0, 1.0], 0.0); // at zero: f(x) = 5 + 3x + 2x^2 + x^3, f'(0) = 3
        assert_eq!(f, 5.0);
        assert_eq!(d, 3.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[1.0, 1.0, 1.0], -3.0); // negative x: f(x) = 1 + x + x^2, f'(x) = 1 + 2x at x=-3
        assert_eq!(f, 7.0);
        assert_eq!(d, -5.0);
        let coef = [2.0, -3.0, 1.0]; // at roots: f(x) = (x-1)(x-2) = 2 - 3x + x^2
        let (f, d) = eval_with_deriv::<F64Adaptor>(&coef, 1.0);
        assert_eq!(f, 0.0);
        assert_eq!(d, -1.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&coef, 2.0);
        assert_eq!(f, 0.0);
        assert_eq!(d, 1.0);
    }

    #[test]
    fn t_eval_with_deriv_edge_cases() {
        let mut coef = [0.0; 11]; // high degree: x^10 at x=2
        coef[10] = 1.0;
        let (f, d) = eval_with_deriv::<F64Adaptor>(&coef, 2.0);
        assert_eq!(f, 1024.0);
        assert_eq!(d, 5120.0);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[1e12, 1e12], 1e6); // large coefficients: f(x) = 1e12 + 1e12*x at x=1e6
        assert_eq!(f, 1e12 + 1e18);
        assert_eq!(d, 1e12);
        let (f, d) = eval_with_deriv::<F64Adaptor>(&[1.0, 1.0, 1.0], 1e-15); // tiny x: f(x) = 1 + x + x^2 at x=1e-15
        assert!((f - 1.0).abs() < 1e-10);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_eval_with_deriv_agrees_with_separate_eval() {
        let cases: &[(&[f64], &[f64])] = &[
            (&[1.0, -2.0, 3.0], &[-4.0, 0.5, 7.0, -1.0, 2.5]),
            (&[0.0, 0.0, 0.0, 1.0], &[-10.0, -1.0, 0.0, 1.0, 10.0]),
            (&[3.0, -1.0, 4.0, -1.0, 5.0], &[-2.0, 0.0, 0.1, 1.0, 3.0]),
            (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[-1.5, 0.0, 1.5]),
        ];
        for &(coef, xs) in cases {
            let mut deriv_coef = vec![0.0; coef.len() - 1];
            differentiate::<F64Adaptor>(coef, &mut deriv_coef);
            for &x in xs {
                let (f, d) = eval_with_deriv::<F64Adaptor>(coef, x);
                let f_expected = eval::<F64Adaptor>(coef, x);
                let d_expected = eval::<F64Adaptor>(&deriv_coef, x);
                assert!(
                    (f - f_expected).abs() < 1e-10,
                    "f mismatch at x={}: got {}, expected {}, coef={:?}",
                    x,
                    f,
                    f_expected,
                    coef
                );
                assert!(
                    (d - d_expected).abs() < 1e-10,
                    "f' mismatch at x={}: got {}, expected {}, coef={:?}",
                    x,
                    d,
                    d_expected,
                    coef
                );
            }
        }
    }

    #[test]
    fn t_differentiate() {
        let mut d = [0.0; 2]; // quadratic: 1 + 2x + 3x^2 -> 2 + 6x
        differentiate::<F64Adaptor>(&[1.0, 2.0, 3.0], &mut d);
        assert_eq!(d, [2.0, 6.0]);
        let mut d = [0.0; 3]; // cubic: 5 + x^3 -> 3x^2
        differentiate::<F64Adaptor>(&[5.0, 0.0, 0.0, 1.0], &mut d);
        assert_eq!(d, [0.0, 0.0, 3.0]);
    }

    #[test]
    fn t_deflate_known_root() {
        let mut def = [0.0; 2]; // (x-1)(x-2) = 2 - 3x + x^2, deflate by root=1 -> (x-2) = -2 + x
        deflate::<F64Adaptor>(&[2.0, -3.0, 1.0], 1.0, &mut def);
        assert!((def[0] - (-2.0)).abs() < 1e-12);
        assert!((def[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t_linear_roots() {
        // 6 + 2x = 0 -> x = -3
        let mut roots = [0.0; 1];
        let n = polynomial_roots::<F64Adaptor>(&[6.0, 2.0], &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 1);
        assert!((roots[0] - (-3.0)).abs() < TOL);
        // zero slope: 5 + 0x -> no root
        let n = polynomial_roots::<F64Adaptor>(&[5.0, 0.0], &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 0);
        // bounded inside: -2 + x = 0 -> x = 2, within [0, 10]
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[-2.0, 1.0],
            &mut roots,
            0.0,
            10.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 1);
        assert!((roots[0] - 2.0).abs() < TOL);
        // bounded outside: -20 + x = 0 -> x = 20, outside [0, 10]
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[-20.0, 1.0],
            &mut roots,
            0.0,
            10.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn t_quadratic_roots() {
        let coef = [3.0, -4.0, 1.0]; // two distinct roots: (x-1)(x-3)
        let mut roots = [0.0; 2];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 3.0).abs() < TOL);
        let n =
            polynomial_roots::<F64Adaptor>(&[4.0, -4.0, 1.0], &mut roots, DEFAULT_ERROR).unwrap(); // double root: (x-2)^2
        assert_eq!(n, 1);
        assert!((roots[0] - 2.0).abs() < TOL);
        let n =
            polynomial_roots::<F64Adaptor>(&[1.0, 0.0, 1.0], &mut roots, DEFAULT_ERROR).unwrap(); // no real roots: x^2 + 1
        assert_eq!(n, 0);
        let coef = [-5.0, 6.0, -1.0]; // negative leading: -(x-1)(x-5) = -x^2 + 6x - 5
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 5.0).abs() < TOL);
        let coef = [-1_000_000.0, 0.0, 1.0]; // large roots: (x-1000)(x+1000) = x^2 - 1e6
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        assert!((roots[0] - (-1000.0)).abs() < 0.01);
        assert!((roots[1] - 1000.0).abs() < 0.01);
        let coef = [-1e-16, 0.0, 1.0]; // small roots: (x-1e-8)(x+1e-8) = x^2 - 1e-16
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
    }

    #[test]
    fn t_quadratic_bounded() {
        let mut roots = [0.0; 2];
        // (x-1)(x-10), range [0,5] -> only root at 1
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[10.0, -11.0, 1.0],
            &mut roots,
            0.0,
            5.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 1);
        assert!((roots[0] - 1.0).abs() < TOL);
        // (x-1)(x-3), range [5,10] -> no roots
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[3.0, -4.0, 1.0],
            &mut roots,
            5.0,
            10.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn t_cubic_roots() {
        let mut roots = [0.0; 3];
        let coef = [6.0, -5.0, -2.0, 1.0]; // three distinct roots: (x+2)(x-1)(x-3) = x^3 - 2x^2 - 5x + 6
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&[-2.0, 1.0, 3.0], &roots, n);
        let coef = [2.0, 1.0, 0.0, 1.0]; // one real root: x^3 + x + 2
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 1);
        verify_roots(&coef, &roots, n);
        let coef = [0.0, 0.0, 0.0, 1.0]; // triple root at zero: x^3 = 0
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert!(n >= 1);
        assert!(roots[0].abs() < 2e-6, "root = {} too far from 0", roots[0]);
        let coef = [-1.0, 0.0, 1.0, 0.0]; // degenerates to quadratic: 0*x^3 + x^2 - 1 = 0
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef[..3], &roots, n);
        let coef = [-6.0, 5.0, 2.0, -1.0]; // negative leading: -(x+2)(x-1)(x-3)
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-2.0, 1.0, 3.0], &roots, n);
        let known = [0.1, 0.2, 0.3]; // clustered roots: (x-0.1)(x-0.2)(x-0.3)
        let mut coef4 = [0.0; 4];
        poly_from_roots(&known, &mut coef4);
        let n = polynomial_roots::<F64Adaptor>(&coef4, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef4, &roots, n);
        verify_expected_roots(&known, &roots, n);
        let coef = [0.0, -10000.0, 0.0, 1.0]; // widely separated: (x+100)(x)(x-100) = x^3 - 10000x
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-100.0, 0.0, 100.0], &roots, n);
        let coef = [6.0, -5.0, -2.0, 1.0]; // bounded: (x+2)(x-1)(x-3), range [0,2] -> only root at 1
        let n = polynomial_roots_in_range::<F64Adaptor>(&coef, &mut roots, 0.0, 2.0, DEFAULT_ERROR)
            .unwrap();
        assert_eq!(n, 1);
        assert!((roots[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn t_quartic_roots() {
        let mut roots = [0.0; 4];
        // four roots: (x+3)(x+1)(x-1)(x-3) = x^4 - 10x^2 + 9
        let coef = [9.0, 0.0, -10.0, 0.0, 1.0];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 4);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&[-3.0, -1.0, 1.0, 3.0], &roots, n);
        // two real roots: (x^2+1)(x^2-4) = x^4 - 3x^2 - 4
        let coef = [-4.0, 0.0, -3.0, 0.0, 1.0];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-2.0, 2.0], &roots, n);
        // no real roots: (x^2+1)(x^2+4) = x^4 + 5x^2 + 4
        let n =
            polynomial_roots::<F64Adaptor>(&[4.0, 0.0, 5.0, 0.0, 1.0], &mut roots, DEFAULT_ERROR)
                .unwrap();
        assert_eq!(n, 0);
        // degenerates to cubic: 0*x^4 + (x-1)(x-2)(x-3)
        let n = polynomial_roots::<F64Adaptor>(
            &[-6.0, 11.0, -6.0, 1.0, 0.0],
            &mut roots,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 3);
        verify_expected_roots(&[1.0, 2.0, 3.0], &roots, n);
        // bounded: (x+3)(x+1)(x-1)(x-3), range [-2,2] -> roots at -1, 1
        let coef = [9.0, 0.0, -10.0, 0.0, 1.0];
        let n =
            polynomial_roots_in_range::<F64Adaptor>(&coef, &mut roots, -2.0, 2.0, DEFAULT_ERROR)
                .unwrap();
        assert_eq!(n, 2);
        verify_expected_roots(&[-1.0, 1.0], &roots, n);
    }

    #[test]
    fn t_quintic_roots() {
        let known = [-2.0, -1.0, 0.0, 1.0, 2.0]; // five roots: (x+2)(x+1)(x)(x-1)(x-2)
        let mut coef = [0.0; 6];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 5];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 5);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
        let coef = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // one root: x^5 + 1 = 0 -> x = -1
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 1);
        verify_roots(&coef, &roots, n);
        assert!((roots[0] - (-1.0)).abs() < TOL);
        let known_real = [1.0, 2.0, 3.0]; // three real roots: (x^2+1)(x-1)(x-2)(x-3)
        let mut cubic = [0.0; 4];
        poly_from_roots(&known_real, &mut cubic);
        let mut coef = [0.0; 6];
        for i in 0..4 {
            coef[i] += cubic[i];
            coef[i + 2] += cubic[i];
        }
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&known_real, &roots, n);
    }

    #[test]
    fn t_higher_degree_all_roots() {
        // Helper to test a set of known roots at a given degree
        let test_known_roots = |known: &[f64]| {
            let deg = known.len();
            let mut coef = vec![0.0; deg + 1];
            poly_from_roots(known, &mut coef);
            let mut roots = vec![0.0; deg];
            let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
            assert_eq!(
                n as usize, deg,
                "degree {} expected {} roots, got {}",
                deg, deg, n
            );
            verify_roots(&coef, &roots, n);
            verify_sorted(&roots, n);
            verify_expected_roots(known, &roots, n);
        };
        test_known_roots(&[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]); // degree 6
        test_known_roots(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]); // degree 7
        test_known_roots(&[-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]); // degree 8
        test_known_roots(&[-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // degree 10
    }

    #[test]
    fn t_higher_degree_edge_cases() {
        // degree 8, no real roots: (x^2+1)^4 = x^8 + 4x^6 + 6x^4 + 4x^2 + 1
        let coef = [1.0, 0.0, 4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 1.0];
        let mut roots = [0.0; 8];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 0);
        // degree 6, bounded: roots at ±1, ±2, ±3 filtered to [-1.5, 1.5]
        let known = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let mut coef = [0.0; 7];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 6];
        let n =
            polynomial_roots_in_range::<F64Adaptor>(&coef, &mut roots, -1.5, 1.5, DEFAULT_ERROR)
                .unwrap();
        assert_eq!(n, 2);
        verify_expected_roots(&[-1.0, 1.0], &roots, n);
    }

    #[test]
    fn t_edge_cases() {
        // degree reduction: nominal degree 5 with zero leading coeffs -> x^2 - 4
        let mut roots5 = [0.0; 5];
        let n = polynomial_roots::<F64Adaptor>(
            &[-4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            &mut roots5,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 2);
        verify_expected_roots(&[-2.0, 2.0], &roots5, n);
        // root at zero: x(x-1)(x-2) = x^3 - 3x^2 + 2x
        let mut roots3 = [0.0; 3];
        let coef = [0.0, 2.0, -3.0, 1.0];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots3, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 3);
        verify_roots(&coef, &roots3, n);
        verify_expected_roots(&[0.0, 1.0, 2.0], &roots3, n);
        // symmetric roots: (x+a)(x-a) = x^2 - a^2 for several scales
        let mut roots2 = [0.0; 2];
        for &a in &[0.001, 1.0, 100.0, 10000.0] {
            let coef = [-(a * a), 0.0, 1.0];
            let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots2, DEFAULT_ERROR).unwrap();
            assert_eq!(n, 2, "failed for a={}", a);
            assert!(
                (roots2[0] - (-a)).abs() < TOL * (1.0 + a),
                "failed for a={}",
                a
            );
            assert!(
                (roots2[1] - a).abs() < TOL * (1.0 + a),
                "failed for a={}",
                a
            );
        }
        // large & tiny coefficient scales: s * (x-1)(x-2) for s = 1e12 and 1e-12
        for &s in &[1e12, 1e-12] {
            let coef = [2.0 * s, -3.0 * s, 1.0 * s];
            let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots2, DEFAULT_ERROR).unwrap();
            assert_eq!(n, 2, "failed for scale={}", s);
            assert!((roots2[0] - 1.0).abs() < TOL, "failed for scale={}", s);
            assert!((roots2[1] - 2.0).abs() < TOL, "failed for scale={}", s);
        }
        // bounded: root exactly at boundary
        let mut roots1 = [0.0; 1];
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[-5.0, 1.0],
            &mut roots1,
            5.0,
            10.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 1);
        assert!((roots1[0] - 5.0).abs() < TOL);
        // bounded: (x-5)(x-10), range [6,9] -> no roots
        let n = polynomial_roots_in_range::<F64Adaptor>(
            &[50.0, -15.0, 1.0],
            &mut roots2,
            6.0,
            9.0,
            DEFAULT_ERROR,
        )
        .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn t_stress_random_roots() {
        stress_test_degree(3, 10000, 0xCAFEBABE, -10.0, 10.0);
        stress_test_degree(4, 10000, 0xDEADBEEF, -10.0, 10.0);
        stress_test_degree(7, 1000, 0x12345678, -5.0, 5.0);
        stress_test_degree(10, 500, 0xABCD1234, -5.0, 5.0);
    }

    #[test]
    fn t_accuracy() {
        // irrational cubic: x^3 - 2 = 0 -> x = 2^(1/3)
        let mut roots3 = [0.0; 3];
        let n = polynomial_roots::<F64Adaptor>(&[-2.0, 0.0, 0.0, 1.0], &mut roots3, DEFAULT_ERROR)
            .unwrap();
        assert_eq!(n, 1);
        let expected = 2.0_f64.cbrt();
        assert!(
            (roots3[0] - expected).abs() < 1e-9,
            "got {} expected {}",
            roots3[0],
            expected
        );
        // Wilkinson-style quartic: (x-1)(x-2)(x-3)(x-4)
        let known = [1.0, 2.0, 3.0, 4.0];
        let mut coef = [0.0; 5];
        poly_from_roots(&known, &mut coef);
        let mut roots4 = [0.0; 4];
        let n = polynomial_roots::<F64Adaptor>(&coef, &mut roots4, DEFAULT_ERROR).unwrap();
        assert_eq!(n, 4);
        for i in 0..4 {
            assert!(
                (roots4[i] - known[i]).abs() < 1e-9,
                "root {} = {} expected {}",
                i,
                roots4[i],
                known[i]
            );
        }
    }

    #[test]
    fn t_stress_bounded_cubics() {
        let mut state: u64 = 0xFEEDFACE;
        let mut xor_next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            -10.0 + f * 20.0
        };
        for _ in 0..5000 {
            let known = [xor_next(), xor_next(), xor_next()];
            let mut coef = [0.0; 4];
            poly_from_roots(&known, &mut coef);
            let x0 = -5.0;
            let x1 = 5.0;
            let expected_count = known.iter().filter(|&&r| r >= x0 && r <= x1).count();
            let mut roots = [0.0; 3];
            let n =
                polynomial_roots_in_range::<F64Adaptor>(&coef, &mut roots, x0, x1, DEFAULT_ERROR)
                    .unwrap();
            assert_eq!(
                n as usize, expected_count,
                "known roots {:?}, range [{}, {}], expected {} got {}",
                known, x0, x1, expected_count, n
            );
            verify_roots(&coef, &roots, n);
            // Verify all found roots are within bounds
            for (i, &r) in roots.iter().enumerate().take(n) {
                assert!(
                    r >= x0 - TOL && r <= x1 + TOL,
                    "root {} = {} outside [{}, {}]",
                    i,
                    r,
                    x0,
                    x1
                );
            }
        }
    }
}
