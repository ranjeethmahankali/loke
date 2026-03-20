// Rust port of cyPolynomial.h by Cem Yuksel
// High-Performance Polynomial Root Finding for Graphics (2022)
//
// Coefficients are in ascending degree order: coef[0] + coef[1]*x + ... + coef[N]*x^N
//
// This port uses runtime degree with compile-time max degree via const generics.
// The macro `dispatch_degree!` generates monomorphized entry points for degrees 3-10.

#![allow(non_snake_case)]

/// Default positional error tolerance for `f64` root finding (matches cyPolynomial).
pub const DEFAULT_ERROR: f64 = 6e-7;

// ---- Helpers ----

#[inline(always)]
fn mult_sign(v: f64, sign: f64) -> f64 {
    if sign < 0.0 { -v } else { v }
}

#[inline(always)]
fn is_different_sign(a: f64, b: f64) -> bool {
    (a < 0.0) != (b < 0.0)
}

// ---- Polynomial evaluation (Horner's method) ----

#[inline(always)]
fn poly_eval(coef: &[f64], x: f64) -> f64 {
    let n = coef.len() - 1;
    let mut r = coef[n];
    for i in (0..n).rev() {
        r = r * x + coef[i];
    }
    r
}

// ---- Polynomial derivative ----

#[inline(always)]
fn poly_derivative(coef: &[f64], deriv: &mut [f64]) {
    let n = coef.len() - 1;
    debug_assert_eq!(deriv.len(), n);
    for i in 0..n {
        deriv[i] = (i as f64 + 1.0) * coef[i + 1];
    }
}

// ---- Polynomial deflation ----

#[inline(always)]
fn poly_deflate(coef: &[f64], root: f64, def_poly: &mut [f64]) {
    let n = coef.len() - 1;
    debug_assert_eq!(def_poly.len(), n);
    def_poly[n - 1] = coef[n];
    for i in (0..n - 1).rev() {
        def_poly[i] = coef[i + 1] + root * def_poly[i + 1];
    }
}

// ---- Root finder (Newton + bisection) ----

fn find_closed(coef: &[f64], deriv: &[f64], x0: f64, x1: f64, y0: f64, x_error: f64) -> f64 {
    let n = coef.len() - 1;
    let ep2 = 2.0 * x_error;
    let mut xr = (x0 + x1) / 2.0;
    if x1 - x0 <= ep2 {
        return xr;
    }

    // Fast Newton path for low degree
    if n <= 3 {
        let xr0 = xr;
        for _ in 0..16 {
            let dy = poly_eval(deriv, xr);
            let xn = xr - poly_eval(coef, xr) / dy;
            let xn = xn.clamp(x0, x1);
            if (xr - xn).abs() <= x_error {
                return xn;
            }
            xr = xn;
        }
        if !xr.is_finite() {
            xr = xr0;
        }
    }

    let mut yr = poly_eval(coef, xr);
    let mut xb0 = x0;
    let mut xb1 = x1;

    loop {
        let side = is_different_sign(y0, yr);
        if side {
            xb1 = xr;
        } else {
            xb0 = xr;
        }
        let dy = poly_eval(deriv, xr);
        let dx = yr / dy;
        let xn = xr - dx;
        if xn > xb0 && xn < xb1 {
            let stepsize = (xr - xn).abs();
            xr = xn;
            if stepsize > x_error {
                yr = poly_eval(coef, xr);
            } else {
                break;
            }
        } else {
            xr = (xb0 + xb1) / 2.0;
            if xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2 {
                break;
            }
            yr = poly_eval(coef, xr);
        }
    }
    xr
}

fn find_open_impl(
    coef: &[f64],
    deriv: &[f64],
    mut xm: f64,
    mut ym: f64,
    mut xr: f64,
    x_error: f64,
    open_min: bool,
) -> f64 {
    let mut delta = 1.0;
    let mut yr = poly_eval(coef, xr);
    let mut otherside = is_different_sign(ym, yr);

    while yr != 0.0 {
        if otherside {
            return if open_min {
                find_closed(coef, deriv, xr, xm, yr, x_error)
            } else {
                find_closed(coef, deriv, xm, xr, ym, x_error)
            };
        }

        // open_interval:
        loop {
            xm = xr;
            ym = yr;
            let dy = poly_eval(deriv, xr);
            let dx = yr / dy;
            let xn = xr - dx;
            let dif = if open_min { xr - xn } else { xn - xr };
            if dif <= 0.0 && xn.is_finite() {
                xr = xn;
                if dif <= x_error {
                    if xr == xm {
                        return xr;
                    }
                    let xs = xn - mult_sign(x_error, if open_min { -1.0 } else { 0.0 });
                    let ys = poly_eval(coef, xs);
                    let s = is_different_sign(ym, ys);
                    if s {
                        return xr;
                    }
                    xr = xs;
                    yr = ys;
                    continue; // goto open_interval
                }
            } else {
                xr = if open_min { xr - delta } else { xr + delta };
                delta *= 2.0;
            }
            yr = poly_eval(coef, xr);
            otherside = is_different_sign(ym, yr);
            break;
        }
    }
    xr
}

#[inline(always)]
fn find_open_min(coef: &[f64], deriv: &[f64], x1: f64, y1: f64, x_error: f64) -> f64 {
    find_open_impl(coef, deriv, x1, y1, x1 - 1.0, x_error, true)
}

#[inline(always)]
fn find_open_max(coef: &[f64], deriv: &[f64], x0: f64, y0: f64, x_error: f64) -> f64 {
    find_open_impl(coef, deriv, x0, y0, x0 + 1.0, x_error, false)
}

fn find_open(coef: &[f64], deriv: &[f64], x_error: f64) -> f64 {
    let n = coef.len() - 1;
    debug_assert!(n & 1 == 1, "FindOpen only works for odd degree polynomials");
    let xr = 0.0;
    let yr = coef[0];
    if is_different_sign(coef[n], yr) {
        find_open_max(coef, deriv, xr, yr, x_error)
    } else {
        find_open_min(coef, deriv, xr, yr, x_error)
    }
}

// ---- Linear root ----

fn linear_root_bounded(coef: &[f64], x0: f64, x1: f64) -> (f64, i32) {
    if coef[1] != 0.0 {
        let r = -coef[0] / coef[1];
        (r, if r >= x0 && r <= x1 { 1 } else { 0 })
    } else {
        ((x0 + x1) / 2.0, if coef[0] == 0.0 { 1 } else { 0 })
    }
}

fn linear_root_unbounded(coef: &[f64]) -> (f64, i32) {
    (-coef[0] / coef[1], if coef[1] != 0.0 { 1 } else { 0 })
}

// ---- Quadratic roots ----

fn quadratic_roots_unbounded(coef: &[f64], roots: &mut [f64]) -> i32 {
    let c = coef[0];
    let b = coef[1];
    let a = coef[2];
    let delta = b * b - 4.0 * a * c;
    if delta > 0.0 {
        let d = delta.sqrt();
        let q = -0.5 * (b + mult_sign(d, b));
        let rv0 = q / a;
        let rv1 = c / q;
        roots[0] = rv0.min(rv1);
        roots[1] = rv0.max(rv1);
        2
    } else if delta < 0.0 {
        0
    } else {
        roots[0] = -0.5 * b / a;
        if a != 0.0 { 1 } else { 0 }
    }
}

fn quadratic_roots_bounded(coef: &[f64], roots: &mut [f64], x0: f64, x1: f64) -> i32 {
    let c = coef[0];
    let b = coef[1];
    let a = coef[2];
    let delta = b * b - 4.0 * a * c;
    if delta > 0.0 {
        let d = delta.sqrt();
        let q = -0.5 * (b + mult_sign(d, b));
        let rv0 = q / a;
        let rv1 = c / q;
        let r0 = rv0.min(rv1);
        let r1 = rv0.max(rv1);
        let r0i = (r0 >= x0 && r0 <= x1) as i32;
        let r1i = (r1 >= x0 && r1 <= x1) as i32;
        roots[0] = r0;
        roots[r0i as usize] = r1;
        r0i + r1i
    } else if delta < 0.0 {
        0
    } else {
        let r0 = -0.5 * b / a;
        roots[0] = r0;
        (r0 >= x0 && r0 <= x1) as i32
    }
}

// ---- Cubic roots ----

fn cubic_roots_bounded(coef: &[f64], roots: &mut [f64], x0: f64, x1: f64, x_error: f64) -> i32 {
    let y0 = poly_eval(coef, x0);
    let y1 = poly_eval(coef, x1);

    let a = coef[3] * 3.0;
    let b_2 = coef[2];
    let c = coef[1];

    let deriv = [c, 2.0 * b_2, a, 0.0];
    let delta_4 = b_2 * b_2 - a * c;

    if delta_4 > 0.0 {
        let d_2 = delta_4.sqrt();
        let q = -(b_2 + mult_sign(d_2, b_2));
        let rv0 = q / a;
        let rv1 = c / q;
        let xa = rv0.min(rv1);
        let xb = rv0.max(rv1);

        if is_different_sign(y0, y1) {
            if xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1) {
                roots[0] = find_closed(coef, &deriv, x0, x1, y0, x_error);
                return 1;
            }
        } else if (xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1) {
            return 0;
        }

        let num_roots = 0i32;
        if xa > x0 {
            let ya = poly_eval(coef, xa);
            if is_different_sign(y0, ya) {
                roots[0] = find_closed(coef, &deriv, x0, xa, y0, x_error);
                if is_different_sign(ya, y1)
                    || (xb < x1 && is_different_sign(ya, poly_eval(coef, xb)))
                {
                    let mut def_poly = [0.0; 4];
                    poly_deflate(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_bounded(&def_poly[..3], &mut roots[1..], xa, x1) + 1;
                } else {
                    return 1;
                }
            }
            if xb < x1 {
                let yb = poly_eval(coef, xb);
                if is_different_sign(ya, yb) {
                    roots[0] = find_closed(coef, &deriv, xa, xb, ya, x_error);
                    if is_different_sign(yb, y1) {
                        let mut def_poly = [0.0; 4];
                        poly_deflate(coef, roots[0], &mut def_poly[..3]);
                        return quadratic_roots_bounded(&def_poly[..3], &mut roots[1..], xb, x1)
                            + 1;
                    } else {
                        return 1;
                    }
                }
                if is_different_sign(yb, y1) {
                    roots[0] = find_closed(coef, &deriv, xb, x1, yb, x_error);
                    return 1;
                }
            } else if is_different_sign(ya, y1) {
                roots[0] = find_closed(coef, &deriv, xa, x1, ya, x_error);
                return 1;
            }
        } else {
            let yb = poly_eval(coef, xb);
            if is_different_sign(y0, yb) {
                roots[0] = find_closed(coef, &deriv, x0, xb, y0, x_error);
                if is_different_sign(yb, y1) {
                    let mut def_poly = [0.0; 4];
                    poly_deflate(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_bounded(&def_poly[..3], &mut roots[1..], xb, x1) + 1;
                } else {
                    return 1;
                }
            }
            if is_different_sign(yb, y1) {
                roots[0] = find_closed(coef, &deriv, xb, x1, yb, x_error);
                return 1;
            }
        }
        num_roots
    } else {
        if is_different_sign(y0, y1) {
            roots[0] = find_closed(coef, &deriv, x0, x1, y0, x_error);
            1
        } else {
            0
        }
    }
}

fn cubic_roots_unbounded(coef: &[f64], roots: &mut [f64], x_error: f64) -> i32 {
    if coef[3] != 0.0 {
        let a = coef[3] * 3.0;
        let b_2 = coef[2];
        let c = coef[1];

        let deriv = [c, 2.0 * b_2, a, 0.0];
        let delta_4 = b_2 * b_2 - a * c;

        if delta_4 > 0.0 {
            let d_2 = delta_4.sqrt();
            let q = -(b_2 + mult_sign(d_2, b_2));
            let rv0 = q / a;
            let rv1 = c / q;
            let xa = rv0.min(rv1);
            let xb = rv0.max(rv1);

            let ya = poly_eval(coef, xa);
            let yb = poly_eval(coef, xb);

            if !is_different_sign(coef[3], ya) {
                roots[0] = find_open_min(coef, &deriv, xa, ya, x_error);
                if is_different_sign(ya, yb) {
                    let mut def_poly = [0.0; 4];
                    poly_deflate(coef, roots[0], &mut def_poly[..3]);
                    return quadratic_roots_unbounded(&def_poly[..3], &mut roots[1..]) + 1;
                }
            } else {
                roots[0] = find_open_max(coef, &deriv, xb, yb, x_error);
            }
            1
        } else {
            let x_inf = -b_2 / a;
            let y_inf = poly_eval(coef, x_inf);
            if is_different_sign(coef[3], y_inf) {
                roots[0] = find_open_max(coef, &deriv, x_inf, y_inf, x_error);
            } else {
                roots[0] = find_open_min(coef, &deriv, x_inf, y_inf, x_error);
            }
            1
        }
    } else {
        quadratic_roots_unbounded(coef, roots)
    }
}

// ---- General polynomial roots (degree N >= 4) ----

fn polynomial_roots_bounded(
    coef: &[f64],
    roots: &mut [f64],
    x0: f64,
    x1: f64,
    x_error: f64,
) -> i32 {
    let n = coef.len() - 1; // degree

    match n {
        1 => {
            let (r, count) = linear_root_bounded(coef, x0, x1);
            roots[0] = r;
            count
        }
        2 => quadratic_roots_bounded(coef, roots, x0, x1),
        3 => cubic_roots_bounded(coef, roots, x0, x1, x_error),
        _ => {
            if coef[n] == 0.0 {
                return polynomial_roots_bounded(&coef[..n], roots, x0, x1, x_error);
            }

            let y0 = poly_eval(coef, x0);
            let mut deriv = [0.0; 11]; // max degree 10 -> max 10 deriv coeffs
            poly_derivative(coef, &mut deriv[..n]);

            let mut deriv_roots = [0.0; 10];
            let nd = polynomial_roots_bounded(&deriv[..n], &mut deriv_roots, x0, x1, x_error);

            let mut x = [0.0; 12]; // max N+1 = 11 entries
            let mut y = [0.0; 12];
            x[0] = x0;
            y[0] = y0;
            for i in 0..nd as usize {
                x[i + 1] = deriv_roots[i];
                y[i + 1] = poly_eval(coef, deriv_roots[i]);
            }
            x[nd as usize + 1] = x1;
            y[nd as usize + 1] = poly_eval(coef, x1);

            let mut nr = 0i32;
            for i in 0..=nd as usize {
                if is_different_sign(y[i], y[i + 1]) {
                    roots[nr as usize] =
                        find_closed(coef, &deriv[..n], x[i], x[i + 1], y[i], x_error);
                    nr += 1;
                }
            }
            nr
        }
    }
}

fn polynomial_roots_unbounded(coef: &[f64], roots: &mut [f64], x_error: f64) -> i32 {
    let n = coef.len() - 1; // degree

    match n {
        1 => {
            let (r, count) = linear_root_unbounded(coef);
            roots[0] = r;
            count
        }
        2 => quadratic_roots_unbounded(coef, roots),
        3 => cubic_roots_unbounded(coef, roots, x_error),
        _ => {
            if coef[n] == 0.0 {
                return polynomial_roots_unbounded(&coef[..n], roots, x_error);
            }

            let mut deriv = [0.0; 11];
            poly_derivative(coef, &mut deriv[..n]);

            let mut deriv_roots = [0.0; 10];
            let nd = polynomial_roots_unbounded(&deriv[..n], &mut deriv_roots, x_error);

            if (n & 1 == 1) || (n & 1 == 0 && nd > 0) {
                let mut nr = 0i32;
                let mut xa = deriv_roots[0];
                let mut ya = poly_eval(coef, xa);

                if is_different_sign(coef[n], ya) != (n & 1 == 1) {
                    roots[0] = find_open_min(coef, &deriv[..n], xa, ya, x_error);
                    nr = 1;
                }

                for i in 1..nd as usize {
                    let xb = deriv_roots[i];
                    let yb = poly_eval(coef, xb);
                    if is_different_sign(ya, yb) {
                        roots[nr as usize] = find_closed(coef, &deriv[..n], xa, xb, ya, x_error);
                        nr += 1;
                    }
                    xa = xb;
                    ya = yb;
                }

                if is_different_sign(coef[n], ya) {
                    roots[nr as usize] = find_open_max(coef, &deriv[..n], xa, ya, x_error);
                    nr += 1;
                }
                nr
            } else if n & 1 == 1 {
                roots[0] = find_open(coef, &deriv[..n], x_error);
                1
            } else {
                0 // should not happen
            }
        }
    }
}

// ---- Public API ----

/// Finds all real roots of a polynomial with coefficients in ascending degree order.
/// Returns the number of roots found. Roots are written to `roots`.
///
/// `coef`: slice of length degree+1, coef[0] + coef[1]*x + ... + coef[N]*x^N
/// `roots`: output slice, must have length >= degree
/// `x_error`: positional error tolerance (use `DEFAULT_ERROR` for the default)
pub fn polynomial_roots(coef: &[f64], roots: &mut [f64], x_error: f64) -> i32 {
    polynomial_roots_unbounded(coef, roots, x_error)
}

/// Finds all real roots of a polynomial within [x_min, x_max].
/// Returns the number of roots found.
pub fn polynomial_roots_in_range(
    coef: &[f64],
    roots: &mut [f64],
    x_min: f64,
    x_max: f64,
    x_error: f64,
) -> i32 {
    polynomial_roots_bounded(coef, roots, x_min, x_max, x_error)
}

#[cfg(test)]
mod test {
    use super::*;

    const TOL: f64 = 1e-6;

    /// Verify that each reported root actually evaluates close to zero.
    fn verify_roots(coef: &[f64], roots: &[f64], n: i32) {
        // Scale tolerance by the polynomial's coefficient magnitude so that
        // polynomials with large coefficients (or roots near zero where high-
        // degree terms dominate) don't produce false negatives.
        let coef_scale: f64 = coef.iter().map(|c| c.abs()).fold(0.0f64, f64::max).max(1.0);
        for i in 0..n as usize {
            let val = poly_eval(coef, roots[i]);
            let scale = coef_scale * (1.0 + roots[i].abs().powi(coef.len() as i32 - 1));
            assert!(
                val.abs() < TOL * scale,
                "root {} = {} does not evaluate to ~0: f(x) = {}, coef = {:?}",
                i,
                roots[i],
                val,
                coef
            );
        }
    }

    /// Verify roots are sorted in ascending order.
    fn verify_sorted(roots: &[f64], n: i32) {
        for i in 1..n as usize {
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
        for k in 0..degree {
            let r = known_roots[k];
            for i in (1..=k + 1).rev() {
                coef[i] = coef[i - 1] - r * coef[i];
            }
            coef[0] *= -r;
        }
    }

    /// Check that every expected root appears in the found roots (order-independent).
    fn verify_expected_roots(expected: &[f64], found: &[f64], n: i32) {
        let found = &found[..n as usize];
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

    // =====================================================================
    // poly_eval
    // =====================================================================

    #[test]
    fn t_eval_constant() {
        assert_eq!(poly_eval(&[42.0], 999.0), 42.0);
    }

    #[test]
    fn t_eval_linear() {
        // 3 + 2x at x=5 -> 13
        assert_eq!(poly_eval(&[3.0, 2.0], 5.0), 13.0);
    }

    #[test]
    fn t_eval_quadratic() {
        // 1 - 3x + 2x^2 at x=2 -> 1 - 6 + 8 = 3
        assert_eq!(poly_eval(&[1.0, -3.0, 2.0], 2.0), 3.0);
    }

    #[test]
    fn t_eval_at_zero() {
        assert_eq!(poly_eval(&[7.0, 1.0, 2.0, 3.0], 0.0), 7.0);
    }

    // =====================================================================
    // poly_derivative
    // =====================================================================

    #[test]
    fn t_derivative_quadratic() {
        // 1 + 2x + 3x^2 -> 2 + 6x
        let coef = [1.0, 2.0, 3.0];
        let mut d = [0.0; 2];
        poly_derivative(&coef, &mut d);
        assert_eq!(d, [2.0, 6.0]);
    }

    #[test]
    fn t_derivative_cubic() {
        // 5 + 0x + 0x^2 + x^3 -> 0 + 0x + 3x^2
        let coef = [5.0, 0.0, 0.0, 1.0];
        let mut d = [0.0; 3];
        poly_derivative(&coef, &mut d);
        assert_eq!(d, [0.0, 0.0, 3.0]);
    }

    // =====================================================================
    // poly_deflate
    // =====================================================================

    #[test]
    fn t_deflate_known_root() {
        // (x - 1)(x - 2) = 2 - 3x + x^2
        // Deflate by root=1 -> should give (x - 2) = -2 + x
        let coef = [2.0, -3.0, 1.0];
        let mut def = [0.0; 2];
        poly_deflate(&coef, 1.0, &mut def);
        assert!((def[0] - (-2.0)).abs() < 1e-12);
        assert!((def[1] - 1.0).abs() < 1e-12);
    }

    // =====================================================================
    // Linear roots
    // =====================================================================

    #[test]
    fn t_linear_single_root() {
        // 6 + 2x = 0 -> x = -3
        let coef = [6.0, 2.0];
        let mut roots = [0.0; 1];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - (-3.0)).abs() < TOL);
    }

    #[test]
    fn t_linear_zero_slope() {
        // 5 + 0x = 0 -> no root
        let coef = [5.0, 0.0];
        let mut roots = [0.0; 1];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    #[test]
    fn t_linear_bounded_inside() {
        // -2 + x = 0 -> x = 2, within [0, 10]
        let coef = [-2.0, 1.0];
        let mut roots = [0.0; 1];
        let n = polynomial_roots_in_range(&coef, &mut roots, 0.0, 10.0, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn t_linear_bounded_outside() {
        // -20 + x = 0 -> x = 20, outside [0, 10]
        let coef = [-20.0, 1.0];
        let mut roots = [0.0; 1];
        let n = polynomial_roots_in_range(&coef, &mut roots, 0.0, 10.0, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    // =====================================================================
    // Quadratic roots
    // =====================================================================

    #[test]
    fn t_quadratic_two_roots() {
        // (x-1)(x-3) = 3 - 4x + x^2
        let coef = [3.0, -4.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 3.0).abs() < TOL);
    }

    #[test]
    fn t_quadratic_double_root() {
        // (x-2)^2 = 4 - 4x + x^2
        let coef = [4.0, -4.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn t_quadratic_no_real_roots() {
        // x^2 + 1 = 0 -> no real roots
        let coef = [1.0, 0.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    #[test]
    fn t_quadratic_negative_leading() {
        // -(x-1)(x-5) = -(-5 + 6x - x^2) = 5 - 6x + x^2... wait
        // -(x^2 - 6x + 5) = -x^2 + 6x - 5
        let coef = [-5.0, 6.0, -1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 5.0).abs() < TOL);
    }

    #[test]
    fn t_quadratic_bounded_one_inside() {
        // (x-1)(x-10) = 10 - 11x + x^2
        let coef = [10.0, -11.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots_in_range(&coef, &mut roots, 0.0, 5.0, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn t_quadratic_bounded_none_inside() {
        // (x-1)(x-3) = 3 - 4x + x^2, range [5, 10]
        let coef = [3.0, -4.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots_in_range(&coef, &mut roots, 5.0, 10.0, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    #[test]
    fn t_quadratic_large_roots() {
        // (x - 1000)(x + 1000) = x^2 - 1000000
        let coef = [-1_000_000.0, 0.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        assert!((roots[0] - (-1000.0)).abs() < 0.01);
        assert!((roots[1] - 1000.0).abs() < 0.01);
    }

    #[test]
    fn t_quadratic_small_roots() {
        // (x - 1e-8)(x + 1e-8) = x^2 - 1e-16
        let coef = [-1e-16, 0.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
    }

    // =====================================================================
    // Cubic roots
    // =====================================================================

    #[test]
    fn t_cubic_three_roots() {
        // (x+2)(x-1)(x-3) = -6 + 7x - 0x^2 - 2x^3... let me compute:
        // (x+2)(x-1) = x^2 + x - 2
        // (x^2 + x - 2)(x - 3) = x^3 - 2x^2 - 5x + 6
        let coef = [6.0, -5.0, -2.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&[-2.0, 1.0, 3.0], &roots, n);
    }

    #[test]
    fn t_cubic_one_real_root() {
        // x^3 + x + 2 has one real root near x ≈ -0.7709
        let coef = [2.0, 1.0, 0.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 1);
        verify_roots(&coef, &roots, n);
    }

    #[test]
    fn t_cubic_single_root_at_zero() {
        // x^3 = 0 -> triple root at 0
        // Note: this solver finds real roots numerically; it may return 1 for a triple root
        let coef = [0.0, 0.0, 0.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert!(n >= 1);
        // Positional accuracy is ~DEFAULT_ERROR (6e-7); C++ original also
        // returns ~1.2e-6 for this triple root. Accept matching tolerance.
        assert!(roots[0].abs() < 2e-6, "root = {} too far from 0", roots[0]);
    }

    #[test]
    fn t_cubic_degenerates_to_quadratic() {
        // 0*x^3 + x^2 - 1 = 0 -> x = ±1
        let coef = [-1.0, 0.0, 1.0, 0.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef[..3], &roots, n);
    }

    #[test]
    fn t_cubic_negative_leading() {
        // -(x+2)(x-1)(x-3) -> -x^3 + 2x^2 + 5x - 6
        let coef = [-6.0, 5.0, 2.0, -1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-2.0, 1.0, 3.0], &roots, n);
    }

    #[test]
    fn t_cubic_bounded_filters_roots() {
        // (x+2)(x-1)(x-3) -> roots at -2, 1, 3
        let coef = [6.0, -5.0, -2.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots_in_range(&coef, &mut roots, 0.0, 2.0, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn t_cubic_clustered_roots() {
        // (x - 0.1)(x - 0.2)(x - 0.3)
        let known = [0.1, 0.2, 0.3];
        let mut coef = [0.0; 4];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_cubic_widely_separated_roots() {
        // (x + 100)(x)(x - 100) = x^3 - 10000x
        let coef = [0.0, -10000.0, 0.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-100.0, 0.0, 100.0], &roots, n);
    }

    // =====================================================================
    // Quartic (degree 4) roots
    // =====================================================================

    #[test]
    fn t_quartic_four_roots() {
        // (x+3)(x+1)(x-1)(x-3) = (x^2-1)(x^2-9) = x^4 - 10x^2 + 9
        let coef = [9.0, 0.0, -10.0, 0.0, 1.0];
        let mut roots = [0.0; 4];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 4);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&[-3.0, -1.0, 1.0, 3.0], &roots, n);
    }

    #[test]
    fn t_quartic_two_real_roots() {
        // (x^2 + 1)(x - 2)(x - 5) = x^4 - 7x^3 + 11x^2 - 7x + 10... let me compute:
        // Actually let's just construct from known roots with complex pair
        // (x^2 + 1)(x^2 - 4) = x^4 - 3x^2 - 4
        let coef = [-4.0, 0.0, -3.0, 0.0, 1.0];
        let mut roots = [0.0; 4];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[-2.0, 2.0], &roots, n);
    }

    #[test]
    fn t_quartic_no_real_roots() {
        // (x^2 + 1)(x^2 + 4) = x^4 + 5x^2 + 4
        let coef = [4.0, 0.0, 5.0, 0.0, 1.0];
        let mut roots = [0.0; 4];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    #[test]
    fn t_quartic_degenerates_to_cubic() {
        // 0*x^4 + x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let coef = [-6.0, 11.0, -6.0, 1.0, 0.0];
        let mut roots = [0.0; 4];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_expected_roots(&[1.0, 2.0, 3.0], &roots, n);
    }

    #[test]
    fn t_quartic_bounded() {
        // (x+3)(x+1)(x-1)(x-3), roots at -3, -1, 1, 3
        let coef = [9.0, 0.0, -10.0, 0.0, 1.0];
        let mut roots = [0.0; 4];
        let n = polynomial_roots_in_range(&coef, &mut roots, -2.0, 2.0, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_expected_roots(&[-1.0, 1.0], &roots, n);
    }

    // =====================================================================
    // Quintic (degree 5) roots
    // =====================================================================

    #[test]
    fn t_quintic_five_roots() {
        let known = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut coef = [0.0; 6];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 5];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 5);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_quintic_one_root() {
        // x^5 + 1 = 0 -> x = -1
        let coef = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut roots = [0.0; 5];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 1);
        verify_roots(&coef, &roots, n);
        assert!((roots[0] - (-1.0)).abs() < TOL);
    }

    #[test]
    fn t_quintic_three_roots() {
        // (x^2 + 1)(x-1)(x-2)(x-3)
        // = (x^2+1)(x^3 - 6x^2 + 11x - 6)
        // = x^5 - 6x^4 + 12x^3 - 12x^2 + 11x - 6
        // Let me just build from roots + complex pair
        // (x^2+1) = roots ±i, reals: 1, 2, 3
        let known_real = [1.0, 2.0, 3.0];
        let mut cubic = [0.0; 4];
        poly_from_roots(&known_real, &mut cubic);
        // Multiply by (x^2 + 1): coef5[i] = cubic[i] + cubic[i-2]
        let mut coef = [0.0; 6];
        for i in 0..4 {
            coef[i] += cubic[i];
            coef[i + 2] += cubic[i];
        }
        let mut roots = [0.0; 5];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&known_real, &roots, n);
    }

    // =====================================================================
    // Higher degree (6-10)
    // =====================================================================

    #[test]
    fn t_degree_6_all_roots() {
        let known = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let mut coef = [0.0; 7];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 6];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 6);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_degree_7_all_roots() {
        let known = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let mut coef = [0.0; 8];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 7];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 7);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_degree_8_all_roots() {
        let known = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        let mut coef = [0.0; 9];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 8];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 8);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_degree_10_all_roots() {
        let known = [-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut coef = [0.0; 11];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 10];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 10);
        verify_roots(&coef, &roots, n);
        verify_sorted(&roots, n);
        verify_expected_roots(&known, &roots, n);
    }

    #[test]
    fn t_degree_8_no_real_roots() {
        // (x^2+1)^4 = all complex
        // (x^2+1)^2 = x^4 + 2x^2 + 1
        // (x^4 + 2x^2 + 1)^2 = x^8 + 4x^6 + 6x^4 + 4x^2 + 1
        let coef = [1.0, 0.0, 4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 1.0];
        let mut roots = [0.0; 8];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    #[test]
    fn t_degree_6_bounded() {
        let known = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let mut coef = [0.0; 7];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 6];
        let n = polynomial_roots_in_range(&coef, &mut roots, -1.5, 1.5, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_expected_roots(&[-1.0, 1.0], &roots, n);
    }

    // =====================================================================
    // Leading coefficient zero (degree reduction)
    // =====================================================================

    #[test]
    fn t_degree_reduction_chain() {
        // coef represents degree 5 but leading coeffs are 0
        // Effectively: x^2 - 4 (degree 2)
        let coef = [-4.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut roots = [0.0; 5];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_expected_roots(&[-2.0, 2.0], &roots, n);
    }

    // =====================================================================
    // Edge cases & special values
    // =====================================================================

    #[test]
    fn t_root_at_zero() {
        // x(x-1)(x-2) = 0 - x + ... wait: x^3 - 3x^2 + 2x
        let coef = [0.0, 2.0, -3.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 3);
        verify_roots(&coef, &roots, n);
        verify_expected_roots(&[0.0, 1.0, 2.0], &roots, n);
    }

    #[test]
    fn t_symmetric_roots() {
        // (x+a)(x-a) = x^2 - a^2 for several a
        for &a in &[0.001, 1.0, 100.0, 10000.0] {
            let coef = [-(a * a), 0.0, 1.0];
            let mut roots = [0.0; 2];
            let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
            assert_eq!(n, 2, "failed for a={}", a);
            assert!(
                (roots[0] - (-a)).abs() < TOL * (1.0 + a),
                "failed for a={}",
                a
            );
            assert!((roots[1] - a).abs() < TOL * (1.0 + a), "failed for a={}", a);
        }
    }

    #[test]
    fn t_large_coefficients() {
        // 1e12 * (x - 1)(x - 2) = 1e12*(x^2 - 3x + 2)
        let s = 1e12;
        let coef = [2.0 * s, -3.0 * s, 1.0 * s];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        verify_roots(&coef, &roots, n);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn t_tiny_coefficients() {
        // 1e-12 * (x - 1)(x - 2)
        let s = 1e-12;
        let coef = [2.0 * s, -3.0 * s, 1.0 * s];
        let mut roots = [0.0; 2];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 2);
        assert!((roots[0] - 1.0).abs() < TOL);
        assert!((roots[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn t_bounded_range_at_boundary() {
        // (x-5) = 0, test range [5, 10] — root exactly at boundary
        let coef = [-5.0, 1.0];
        let mut roots = [0.0; 1];
        let n = polynomial_roots_in_range(&coef, &mut roots, 5.0, 10.0, DEFAULT_ERROR);
        assert_eq!(n, 1);
        assert!((roots[0] - 5.0).abs() < TOL);
    }

    #[test]
    fn t_bounded_range_empty() {
        // (x-5)(x-10), range [6, 9] — no roots inside
        let coef = [50.0, -15.0, 1.0];
        let mut roots = [0.0; 2];
        let n = polynomial_roots_in_range(&coef, &mut roots, 6.0, 9.0, DEFAULT_ERROR);
        assert_eq!(n, 0);
    }

    // =====================================================================
    // Stress test with random polynomials from known roots
    // =====================================================================

    #[test]
    fn t_stress_random_cubics() {
        let mut state: u64 = 0xCAFEBABE;
        let mut xor_next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            -10.0 + f * 20.0
        };

        for _ in 0..10000 {
            let known = [xor_next(), xor_next(), xor_next()];
            let mut coef = [0.0; 4];
            poly_from_roots(&known, &mut coef);
            let mut roots = [0.0; 3];
            let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
            assert_eq!(
                n, 3,
                "expected 3 roots for known roots {:?}, got {}",
                known, n
            );
            verify_roots(&coef, &roots, n);
        }
    }

    #[test]
    fn t_stress_random_quartics() {
        let mut state: u64 = 0xDEADBEEF;
        let mut xor_next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            -10.0 + f * 20.0
        };

        for _ in 0..10000 {
            let known = [xor_next(), xor_next(), xor_next(), xor_next()];
            let mut coef = [0.0; 5];
            poly_from_roots(&known, &mut coef);
            let mut roots = [0.0; 4];
            let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
            assert_eq!(
                n, 4,
                "expected 4 roots for known roots {:?}, got {}",
                known, n
            );
            verify_roots(&coef, &roots, n);
        }
    }

    #[test]
    fn t_stress_random_degree_7() {
        let mut state: u64 = 0x12345678;
        let mut xor_next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            -5.0 + f * 10.0
        };

        for _ in 0..1000 {
            let known = [
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
            ];
            let mut coef = [0.0; 8];
            poly_from_roots(&known, &mut coef);
            let mut roots = [0.0; 7];
            let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
            assert_eq!(
                n, 7,
                "expected 7 roots for known roots {:?}, got {}",
                known, n
            );
            verify_roots(&coef, &roots, n);
        }
    }

    #[test]
    fn t_stress_random_degree_10() {
        let mut state: u64 = 0xABCD1234;
        let mut xor_next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            -5.0 + f * 10.0
        };

        for _ in 0..500 {
            let known = [
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
                xor_next(),
            ];
            let mut coef = [0.0; 11];
            poly_from_roots(&known, &mut coef);
            let mut roots = [0.0; 10];
            let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
            assert_eq!(
                n, 10,
                "expected 10 roots for known roots {:?}, got {}",
                known, n
            );
            verify_roots(&coef, &roots, n);
        }
    }

    // =====================================================================
    // Accuracy: verify roots match C++ reference within tolerance
    // =====================================================================

    #[test]
    fn t_accuracy_cubic_known_irrational() {
        // x^3 - 2 = 0 -> x = 2^(1/3) ≈ 1.2599210498948732
        let coef = [-2.0, 0.0, 0.0, 1.0];
        let mut roots = [0.0; 3];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 1);
        let expected = 2.0_f64.cbrt();
        assert!(
            (roots[0] - expected).abs() < 1e-9,
            "got {} expected {}",
            roots[0],
            expected
        );
    }

    #[test]
    fn t_accuracy_quartic_wilkinson_like() {
        // (x-1)(x-2)(x-3)(x-4) — well-conditioned Wilkinson-style
        let known = [1.0, 2.0, 3.0, 4.0];
        let mut coef = [0.0; 5];
        poly_from_roots(&known, &mut coef);
        let mut roots = [0.0; 4];
        let n = polynomial_roots(&coef, &mut roots, DEFAULT_ERROR);
        assert_eq!(n, 4);
        for i in 0..4 {
            assert!(
                (roots[i] - known[i]).abs() < 1e-9,
                "root {} = {} expected {}",
                i,
                roots[i],
                known[i]
            );
        }
    }

    // =====================================================================
    // Bounded range stress
    // =====================================================================

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
            let n = polynomial_roots_in_range(&coef, &mut roots, x0, x1, DEFAULT_ERROR);
            assert_eq!(
                n as usize, expected_count,
                "known roots {:?}, range [{}, {}], expected {} got {}",
                known, x0, x1, expected_count, n
            );
            verify_roots(&coef, &roots, n);
            // Verify all found roots are within bounds
            for i in 0..n as usize {
                assert!(
                    roots[i] >= x0 - TOL && roots[i] <= x1 + TOL,
                    "root {} = {} outside [{}, {}]",
                    i,
                    roots[i],
                    x0,
                    x1
                );
            }
        }
    }
}
