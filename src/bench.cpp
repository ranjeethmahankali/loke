// Benchmark for cyPolynomial root finding (degrees 3-10)
//
// Compile: g++ -std=c++17 -O3 -o bench src/bench.cpp
// Run:     ./bench

#include "cyPolynomial.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Deterministic PRNG — identical implementation in main.rs
// ---------------------------------------------------------------------------

struct Xorshift64 {
    uint64_t state;

    Xorshift64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    double uniform(double lo, double hi) {
        double f = (next() >> 11) * (1.0 / (double)(1ull << 53));
        return lo + f * (hi - lo);
    }
};

// ---------------------------------------------------------------------------
// Expand monic polynomial from its roots (ascending coefficient order)
//
//   (x - r0)(x - r1)...(x - r_{N-1})
//   = coef[0] + coef[1]*x + ... + coef[N]*x^N   (coef[N] == 1)
// ---------------------------------------------------------------------------

template <int N>
void expand_from_roots(double coef[N + 1], const double roots[N]) {
    for (int i = 0; i <= N; i++)
        coef[i] = 0.0;
    coef[0] = 1.0;

    for (int k = 0; k < N; k++) {
        double r = roots[k];
        for (int i = k + 1; i >= 1; i--)
            coef[i] = coef[i - 1] - r * coef[i];
        coef[0] = -r * coef[0];
    }
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------

static constexpr int      NUM_POLYS = 1000;
static constexpr int      NUM_ITERS = 1000;
static constexpr uint64_t SEED      = 0xDEADBEEF12345678ull;

// ---------------------------------------------------------------------------
// Per-degree benchmark
//
// Polynomial mix (per degree):
//   [0, 500)    — from N well-separated real roots in [-10, 10]
//   [500, 750)  — from N clustered real roots (center ± 0.5)
//   [750, 1000) — random coefficients in [-5, 5] (may have complex roots)
// ---------------------------------------------------------------------------

template <int N>
void bench_degree() {
    Xorshift64 rng(SEED + N);

    // --- generate ---
    std::vector<std::array<double, N + 1>> polys(NUM_POLYS);

    for (int i = 0; i < NUM_POLYS; i++) {
        if (i < 500) {
            double roots[N];
            for (int j = 0; j < N; j++)
                roots[j] = rng.uniform(-10.0, 10.0);
            expand_from_roots<N>(polys[i].data(), roots);
        } else if (i < 750) {
            double roots[N];
            double center = rng.uniform(-2.0, 2.0);
            for (int j = 0; j < N; j++)
                roots[j] = center + rng.uniform(-0.5, 0.5);
            expand_from_roots<N>(polys[i].data(), roots);
        } else {
            for (int j = 0; j <= N; j++)
                polys[i][j] = rng.uniform(-5.0, 5.0);
            if (std::abs(polys[i][N]) < 0.1)
                polys[i][N] = polys[i][N] >= 0 ? 1.0 : -1.0;
        }
    }

    // --- warmup ---
    volatile double sink = 0;
    for (int i = 0; i < NUM_POLYS; i++) {
        double roots[N];
        int    n = cy::PolynomialRoots<N>(roots, polys[i].data());
        for (int j = 0; j < n; j++)
            sink = sink + roots[j];
    }

    // --- timed run ---
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (int i = 0; i < NUM_POLYS; i++) {
            double roots[N];
            int    n = cy::PolynomialRoots<N>(roots, polys[i].data());
            for (int j = 0; j < n; j++)
                sink = sink + roots[j];
        }
    }

    auto   end        = std::chrono::high_resolution_clock::now();
    double total_ns   = std::chrono::duration<double, std::nano>(end - start).count();
    double ns_per     = total_ns / ((double)NUM_POLYS * NUM_ITERS);
    int    total      = NUM_POLYS * NUM_ITERS;

    printf("Degree %2d: %8.1f ns/poly  (%d solves)\n", N, ns_per, total);
}

int main() {
    printf("=== cyPolynomial Benchmark ===\n");
    printf("Polynomials per degree: %d\n", NUM_POLYS);
    printf("  [0,500)   well-separated real roots in [-10,10]\n");
    printf("  [500,750) clustered real roots (center +/- 0.5)\n");
    printf("  [750,1000) random coefficients in [-5,5]\n");
    printf("Iterations: %d\n\n", NUM_ITERS);

    bench_degree<3>();
    bench_degree<4>();
    bench_degree<5>();
    bench_degree<6>();
    bench_degree<7>();
    bench_degree<8>();
    bench_degree<9>();
    bench_degree<10>();

    return 0;
}
