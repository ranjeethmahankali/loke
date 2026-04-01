use criterion::{Criterion, black_box, criterion_group, criterion_main};
use loke::{Arc3d, DVec, F64Adaptor, Spline3d, polynomial::find_roots};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

// ---------------------------------------------------------------------------
// Expand monic polynomial from its roots (ascending coefficient order)
// ---------------------------------------------------------------------------

fn expand_from_roots<const MDP1: usize>(roots: &[f64], coef: &mut [f64; MDP1]) {
    let degree = MDP1 - 1;
    coef.fill(0.0);
    coef[0] = 1.0;

    for k in 0..degree {
        let r = roots[k];
        for i in (1..=k + 1).rev() {
            coef[i] = coef[i - 1] - r * coef[i];
        }
        coef[0] *= -r;
    }
}

// ---------------------------------------------------------------------------
// Generate test polynomials (must match bench.cpp)
//
// Polynomial mix (per degree):
//   [0, 500)    — from N well-separated real roots in [-10, 10]
//   [500, 750)  — from N clustered real roots (center ± 0.5)
//   [750, 1000) — random coefficients in [-5, 5] (may have complex roots)
// ---------------------------------------------------------------------------

const NUM_POLYS: usize = 1000;
const SEED: u64 = 0xDEADBEEF12345678;

fn generate_polys<const MDP1: usize>() -> Vec<[f64; MDP1]> {
    let degree = MDP1 - 1;
    let mut rng = SmallRng::seed_from_u64(SEED + degree as u64);
    let mut polys = Vec::with_capacity(NUM_POLYS);
    for i in 0..NUM_POLYS {
        let mut coef = [0.0f64; MDP1];
        if i < 500 {
            let roots: Vec<f64> = (0..degree).map(|_| rng.random_range(-10.0..10.0)).collect();
            expand_from_roots(&roots, &mut coef);
        } else if i < 750 {
            let center: f64 = rng.random_range(-2.0..2.0);
            let roots: Vec<f64> = (0..degree)
                .map(|_| center + rng.random_range(-0.5..0.5))
                .collect();
            expand_from_roots(&roots, &mut coef);
        } else {
            for j in 0..MDP1 {
                coef[j] = rng.random_range(-5.0..5.0);
            }
            if coef[degree].abs() < 0.1 {
                coef[degree] = if coef[degree] >= 0.0 { 1.0 } else { -1.0 };
            }
        }
        polys.push(coef);
    }
    polys
}

pub const DEFAULT_ERROR: f64 = 6e-7;

fn b_root_finding<const MDP1: usize>(c: &mut Criterion, label: &str) {
    let polys = generate_polys::<MDP1>();
    c.bench_function(label, |b| {
        let mut i = 0usize;
        let mut roots = [0.0f64; MDP1];
        b.iter(|| {
            let n = find_roots::<F64Adaptor>(black_box(&polys[i]), &mut roots, DEFAULT_ERROR)
                .expect("Cannot fail");
            black_box(n);
            i += 1;
            if i >= NUM_POLYS {
                i = 0;
            }
        });
    });
}

fn make_spline() -> Spline3d {
    Spline3d::create_clamped(
        vec![
            DVec([-2.0, 0.0, 0.0]),
            DVec([-0.5, 2.0, 1.0]),
            DVec([0.5, -2.0, 1.0]),
            DVec([2.0, 2.0, 0.0]),
            DVec([3.5, 0.0, 0.0]),
        ],
        3,
    )
    .unwrap()
}

fn b_spline_adaptive_samples(c: &mut Criterion) {
    let spline = make_spline();
    let mut buf = Vec::new();
    c.bench_function("degree_03_spline_adaptive_samples", |b| {
        buf.clear();
        b.iter(|| {
            buf.extend(black_box(spline.adaptive_samples(0.00001)));
        });
    });
}

fn b_spline_length(c: &mut Criterion) {
    let spline = make_spline();
    c.bench_function("degree_03_spline_length", |b| {
        let mut lsum = 0.0_f64;
        b.iter(|| {
            lsum += black_box(spline.length(1e-5));
        });
    });
}

fn b_spline_bounds(c: &mut Criterion) {
    let spline = make_spline();
    c.bench_function("degree_03_spline_bounds", |b| {
        let mut bounds = (DVec([0.0; 3]), DVec([0.0; 3]));
        b.iter(|| {
            bounds = black_box(spline.bounds());
        });
    });
}

fn b_spline_eval(c: &mut Criterion) {
    let spline = make_spline();
    const N_SAMPLES: usize = 1000;
    let (dom_start, dom_end) = spline.domain();
    let params: Vec<f64> = (0..=N_SAMPLES)
        .map(|i| {
            let t = (i as f64) / (N_SAMPLES as f64);
            dom_start * (1.0 - t) + dom_end * t
        })
        .collect();
    c.bench_function("degree_03_spline_eval_point", move |b| {
        let mut psum = DVec([0.0; 3]);
        b.iter(|| {
            for t in params.iter() {
                psum += black_box(spline.point(*t).unwrap());
            }
        });
    });
}

fn b_spline_eval_with_deriv(c: &mut Criterion) {
    let spline = make_spline();
    const N_SAMPLES: usize = 1000;
    let (dom_start, dom_end) = spline.domain();
    let params: Vec<f64> = (0..=N_SAMPLES)
        .map(|i| {
            let t = (i as f64) / (N_SAMPLES as f64);
            dom_start * (1.0 - t) + dom_end * t
        })
        .collect();
    c.bench_function("degree_03_spline_eval_point_with_deriv", move |b| {
        let mut results = [DVec([0.0; 3]); 3];
        b.iter(|| {
            for t in params.iter() {
                black_box(
                    spline
                        .point_with_derivs(*t, black_box(&mut results))
                        .unwrap(),
                );
            }
        });
    });
}

fn b_arc_adaptive_samples(c: &mut Criterion) {
    let arc = Arc3d::from_three_points(
        DVec([-3.5, 0.0, 0.0]), // start
        DVec([-3.0, 1.5, 0.0]), // middle
        DVec([-2.5, 0.0, 0.0]), // end
    )
    .unwrap();
    let mut buf = Vec::new();
    c.bench_function("arc_adaptive_samples", |b| {
        buf.clear();
        b.iter(|| {
            buf.extend(black_box(arc.adaptive_samples(0.00001)));
        });
    });
}

fn benchmarks(c: &mut Criterion) {
    b_root_finding::<4>(c, "degree_03_root_finding");
    b_root_finding::<5>(c, "degree_04_root_finding");
    b_root_finding::<6>(c, "degree_05_root_finding");
    b_root_finding::<7>(c, "degree_06_root_finding");
    b_root_finding::<8>(c, "degree_07_root_finding");
    b_root_finding::<9>(c, "degree_08_root_finding");
    b_root_finding::<10>(c, "degree_09_root_finding");
    b_root_finding::<11>(c, "degree_10_root_finding");
    b_spline_adaptive_samples(c);
    b_arc_adaptive_samples(c);
    b_spline_length(c);
    b_spline_bounds(c);
    b_spline_eval(c);
    b_spline_eval_with_deriv(c);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
