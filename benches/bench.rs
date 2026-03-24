use criterion::{Criterion, black_box, criterion_group, criterion_main};
use loke::polynomial::{DEFAULT_ERROR, find_roots};

// ---------------------------------------------------------------------------
// Deterministic PRNG — identical implementation in bench.cpp
// ---------------------------------------------------------------------------

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let f = (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        lo + f * (hi - lo)
    }
}

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
    let mut rng = Xorshift64::new(SEED + degree as u64);
    let mut polys = Vec::with_capacity(NUM_POLYS);

    for i in 0..NUM_POLYS {
        let mut coef = [0.0f64; MDP1];

        if i < 500 {
            let roots: Vec<f64> = (0..degree).map(|_| rng.uniform(-10.0, 10.0)).collect();
            expand_from_roots(&roots, &mut coef);
        } else if i < 750 {
            let center = rng.uniform(-2.0, 2.0);
            let roots: Vec<f64> = (0..degree)
                .map(|_| center + rng.uniform(-0.5, 0.5))
                .collect();
            expand_from_roots(&roots, &mut coef);
        } else {
            for j in 0..MDP1 {
                coef[j] = rng.uniform(-5.0, 5.0);
            }
            if coef[degree].abs() < 0.1 {
                coef[degree] = if coef[degree] >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        polys.push(coef);
    }

    polys
}

fn bench_degree<const MDP1: usize>(c: &mut Criterion, label: &str) {
    let polys = generate_polys::<MDP1>();

    c.bench_function(label, |b| {
        let mut i = 0usize;
        let mut roots = [0.0f64; MDP1];
        b.iter(|| {
            let n = find_roots(black_box(&polys[i]), &mut roots, DEFAULT_ERROR);
            black_box(n);
            i += 1;
            if i >= NUM_POLYS {
                i = 0;
            }
        });
    });
}

fn benchmarks(c: &mut Criterion) {
    bench_degree::<4>(c, "degree_03");
    bench_degree::<5>(c, "degree_04");
    bench_degree::<6>(c, "degree_05");
    bench_degree::<7>(c, "degree_06");
    bench_degree::<8>(c, "degree_07");
    bench_degree::<9>(c, "degree_08");
    bench_degree::<10>(c, "degree_09");
    bench_degree::<11>(c, "degree_10");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
