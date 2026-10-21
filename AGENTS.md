# Working on loke

Loke is a Rust curve library: lines, circular arcs, elliptic arcs and B-splines, plus a closest
point `Projector`. Everything is generic over an `Adaptor` (f32 or f64, any dimension), and the
code is dimension agnostic by design.

## Tasks and notes

- Tasks live in `PROJECT.org` (org-mode). Each has an id (e.g. `tn8bmt`) and a state: `TODO`,
  `NEXT` (in progress) or `DONE`. Finished tasks are archived to `archived/<year>.org`.
- One branch per task, named `<id>-<slug>` (e.g. `tn8bmt-closest-point`). PRs are titled
  `<id> - <Title>`.
- Progress notes go in the task body, prefixed with a timestamp like `<2026-10-20 Tue 19:38>`.
- When writing up findings: write only what we should do, skip the alternatives, and briefly
  mention the tests or measurements that led to the decision.
- The maintainer edits `PROJECT.org` in Emacs. If `.#PROJECT.org` (a lock file) exists, the open
  buffer may have unsaved changes: ask them to save before editing the file.
- Comments in the code from the maintainer often sketch a direction. Once understood, they may
  be replaced with real code and documentation.

## Code

- Keep code terse. Inline small computations; don't add helpers for two-line calculations.
- Math derivations live in comment blocks (e.g. `ELLIPSE ARC MATH` in `src/arc.rs`), split into
  numbered sections. Code refers to them with short comments like `// ELLIPSE ARC MATH §8.2.`
- The public API never panics on bad input: return an `Error` instead. Reuse existing variants
  where they fit (e.g. `InvalidArgument` for slice length mismatches), and add one only when
  none does. Internal invariants may still use
  `expect("INTERNAL ERROR: ...")`.
- No heap allocation in hot paths. Sample iterators are built with `std::iter::from_fn`.
- Emit curve end points exactly from `start()` and `end()`, never from a recurrence.
- No 2D or 3D special cases unless a measurement shows they pay off.
- Tests: check against a brute-force reference where possible. Use random-looking numbers for
  arbitrary values, and `PI` and friends only where the exact value matters.

## Before finishing

These must all pass (CI runs them on Linux, Windows and macOS):

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps
```

Don't commit or push unless asked. The maintainer usually stages and commits themselves; leave
the index alone.

## Performance work

- Back every performance claim with a measurement, and say plainly what was not measured. Give
  the with/without numbers. Predictions about codegen and inlining have been wrong several
  times here.
- Benchmarks are in `benches/bench.rs` (criterion): `cargo bench --bench bench -- <filter>`.
- `.cargo/config.toml` sets `target-cpu=native`, but only for builds inside this repo. Crates that
  depend on loke get baseline x86-64: no FMA, AVX2 is not guaranteed. Keep that in mind for FMA
  and SIMD decisions.
- For paths of a few ns, criterion numbers swing ~1 ns with the surrounding code. Cross-check
  with a plain loop before acting on them.
- Profile with perf: `perf stat -e cycles:u,instructions:u ...` and `perf record -e cycles:u ...`
  on the bench binary with `--bench <name> --profile-time 3`. This needs
  `kernel.perf_event_paranoid <= 1`.
- Experiments use throwaway code, deleted afterwards. Record the findings in `PROJECT.org`.
  Restore any file you edit temporarily.
- Lesson from the projector: call boundaries pass vectors through memory, and the resulting
  store-to-load forwarding chains made a 4 ns line projection take 17 ns. Inlining the hot path
  into the caller fixed it.

## The closest point projector

`src/projector.rs`. The long-term use is querying millions of points, so optimize for bulk
throughput, and SIMD means across queries (not across coordinates). A rewrite inspired by P2M
(closest point on mesh via a KD tree and an interception table), adapted to tessellated curves,
is on the table. See the tn8bmt task in `PROJECT.org` for the math, the decisions and the
measurements behind them. The current exact projector serves as the accuracy reference for any
faster approximate method.
