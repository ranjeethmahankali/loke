use crate::{Adaptor, Error, F32Adaptor, F64Adaptor, SerialAdaptor};

pub struct LineSeg<const DIM: usize, A: Adaptor<DIM>> {
    from: A::Vector,
    to: A::Vector,
}

pub type LineSeg3d = LineSeg<3, F64Adaptor>;
pub type LineSeg3f = LineSeg<3, F32Adaptor>;
pub type LineSeg2d = LineSeg<2, F64Adaptor>;
pub type LineSeg2f = LineSeg<2, F32Adaptor>;

impl<const DIM: usize, A: Adaptor<DIM>> LineSeg<DIM, A> {
    pub fn create(start: A::Vector, end: A::Vector) -> Self {
        LineSeg {
            from: start,
            to: end,
        }
    }

    #[inline(always)]
    pub fn domain(&self) -> (A::Scalar, A::Scalar) {
        (A::scalar(0.0), self.length())
    }

    #[inline(always)]
    pub fn length(&self) -> A::Scalar {
        A::vector_length(self.to - self.from)
    }

    pub fn point(&self, u: A::Scalar) -> Option<A::Vector> {
        let len = self.length();
        if len == A::scalar(0.0) {
            Some(self.from)
        } else if u >= A::scalar(0.0) && u <= len {
            let u = u / len;
            Some(self.from * (A::scalar(1.0) - u) + self.to * u)
        } else {
            None
        }
    }

    pub fn tangent(&self, u: A::Scalar) -> Option<A::Vector> {
        let len = self.length();
        if u >= A::scalar(0.0) && u <= len && len > A::scalar(0.0) {
            Some((self.to - self.from) / len)
        } else {
            None
        }
    }

    pub fn curvature(&self, u: A::Scalar) -> Option<A::Vector> {
        let len = self.length();
        if u >= A::scalar(0.0) && u <= len {
            Some(A::zero_vector())
        } else {
            None
        }
    }

    pub fn point_with_derivs(&self, u: A::Scalar, results: &mut [A::Vector]) -> Result<(), Error> {
        if results.is_empty() {
            return Ok(()); // Nothing to evaluate.
        }
        let len = self.length();
        if len == A::scalar(0.0) {
            results[0] = self.from;
            results[1..].fill(A::zero_vector());
            Ok(())
        } else if u >= A::scalar(0.0) && u <= len {
            let u = u / len;
            results[0] = self.from * (A::scalar(1.0) - u) + self.to * u;
            if results.len() > 1 {
                results[1] = (self.to - self.from) / len;
                results[2..].fill(A::zero_vector());
            }
            Ok(())
        } else {
            Err(Error::InvalidParameter)
        }
    }

    pub fn start(&self) -> A::Vector {
        self.from
    }

    pub fn end(&self) -> A::Vector {
        self.to
    }

    pub fn bounds(&self) -> (A::Vector, A::Vector) {
        (
            A::vector(std::array::from_fn(|ci| {
                A::min(A::vector_coord(self.from, ci), A::vector_coord(self.to, ci))
            })),
            A::vector(std::array::from_fn(|ci| {
                A::max(A::vector_coord(self.from, ci), A::vector_coord(self.to, ci))
            })),
        )
    }

    pub fn adaptive_samples(&self) -> impl Iterator<Item = A::Vector> {
        [self.from, self.to].into_iter()
    }

    pub fn uniform_samples(
        &self,
        start: A::Scalar,
        step: A::Scalar,
        _tolerance: A::Scalar,
    ) -> impl Iterator<Item = A::Vector> {
        let len = self.length();
        let one = A::scalar(1.0);
        let zero = A::scalar(0.0);
        let step = if len == zero { zero } else { step / len };
        let vstep = (self.to - self.from) * step;
        let mut param = if len == zero {
            zero
        } else {
            A::max(start / len, zero)
        };
        let mut pt = param * self.to + (one - param) * self.from;
        std::iter::from_fn(move || {
            if param <= one && step > zero {
                param += step;
                let out = pt;
                pt += vstep;
                Some(out)
            } else {
                None
            }
        })
    }

    pub fn reversed(&self) -> Self {
        LineSeg {
            from: self.to,
            to: self.from,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.from == self.to
    }

    pub fn serialize(&self, w: impl std::io::Write) -> Result<(), std::io::Error>
    where
        A: SerialAdaptor,
    {
        // TODO: Write out the scalars of start and end points.
        todo!()
    }

    pub fn deserialize(src: impl std::io::Read) -> Result<Self, std::io::Error>
    where
        A: SerialAdaptor,
    {
        // Deserialize the same order what was written out by serialize.
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DVec;

    #[test]
    fn length_and_domain() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        assert!((line.length() - 3.0).abs() < 1e-12);
        let (lo, hi) = line.domain();
        assert!(lo.abs() < 1e-12);
        assert!((hi - 3.0).abs() < 1e-12);
    }

    #[test]
    fn endpoints() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        assert_eq!(line.start(), DVec([1.0, 0.0, 0.0]));
        assert_eq!(line.end(), DVec([4.0, 0.0, 0.0]));
    }

    #[test]
    fn point_at_endpoints_and_midpoint() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        let len = line.length();
        let p0 = line.point(0.0).unwrap();
        let p1 = line.point(len).unwrap();
        let pm = line.point(len / 2.0).unwrap();
        assert!((p0 - line.start()).length() < 1e-12);
        assert!((p1 - line.end()).length() < 1e-12);
        assert!((pm - DVec([2.5, 0.0, 0.0])).length() < 1e-12);
    }

    #[test]
    fn point_out_of_range() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        assert!(line.point(-0.001).is_none());
        assert!(line.point(line.length() + 0.001).is_none());
    }

    #[test]
    fn tangent_is_unit_and_constant() {
        let line = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([3.0, 4.0, 0.0]));
        let len = line.length();
        assert!((len - 5.0).abs() < 1e-12);
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            let tan = line.tangent(t).unwrap();
            assert!((tan - DVec([0.6, 0.8, 0.0])).length() < 1e-12);
        }
    }

    #[test]
    fn tangent_out_of_range() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        assert!(line.tangent(-0.001).is_none());
        assert!(line.tangent(line.length() + 0.001).is_none());
    }

    #[test]
    fn point_with_derivs_matches_point_and_tangent() {
        let line = LineSeg3d::create(DVec([1.0, 2.0, 3.0]), DVec([4.0, 6.0, 3.0]));
        let len = line.length();
        let mut results = [DVec([0.0; 3]); 4];
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            line.point_with_derivs(t, &mut results).unwrap();
            let pt = line.point(t).unwrap();
            let tan = line.tangent(t).unwrap();
            assert!((results[0] - pt).length() < 1e-12);
            assert!((results[1] - tan).length() < 1e-12);
            // Higher derivatives of a line are zero.
            assert!(results[2].length() < 1e-12);
            assert!(results[3].length() < 1e-12);
        }
    }

    #[test]
    fn point_with_derivs_out_of_range() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        let mut results = [DVec([0.0; 3]); 2];
        assert!(line.point_with_derivs(-0.001, &mut results).is_err());
        assert!(
            line.point_with_derivs(line.length() + 0.001, &mut results)
                .is_err()
        );
    }

    #[test]
    fn reversed() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        let rev = line.reversed();
        assert_eq!(rev.start(), line.end());
        assert_eq!(rev.end(), line.start());
        let len = line.length();
        assert!((rev.length() - len).abs() < 1e-12);
        for i in 0..=10 {
            let t = len * i as f64 / 10.0;
            let fwd = line.point(t).unwrap();
            let bwd = rev.point(len - t).unwrap();
            assert!((fwd - bwd).length() < 1e-12);
        }
    }

    #[test]
    fn bounds() {
        let line = LineSeg3d::create(DVec([3.0, -1.0, 5.0]), DVec([1.0, 2.0, -3.0]));
        let (lo, hi) = line.bounds();
        assert_eq!(lo, DVec([1.0, -1.0, -3.0]));
        assert_eq!(hi, DVec([3.0, 2.0, 5.0]));
    }

    #[test]
    fn uniform_samples_positions() {
        let line = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([10.0, 0.0, 0.0]));
        // start=0, step=2 → 6 points at x = 0, 2, 4, 6, 8, 10.
        let pts: Vec<_> = line.uniform_samples(0.0, 2.0, 1e-6).collect();
        assert_eq!(pts.len(), 6);
        for (i, pt) in pts.iter().enumerate() {
            assert!(
                (*pt - DVec([2.0 * i as f64, 0.0, 0.0])).length() < 1e-12,
                "pt[{i}] = {pt:?}"
            );
        }
        // start=1, step=2 → 5 points at x = 1, 3, 5, 7, 9.
        let pts2: Vec<_> = line.uniform_samples(1.0, 2.0, 1e-6).collect();
        assert_eq!(pts2.len(), 5);
        for (i, pt) in pts2.iter().enumerate() {
            assert!(
                (*pt - DVec([1.0 + 2.0 * i as f64, 0.0, 0.0])).length() < 1e-12,
                "pt2[{i}] = {pt:?}"
            );
        }
        // Diagonal (0,0,0)→(3,4,0), length=5, step=2.5 → 3 points at start, midpoint, end.
        let diag = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([3.0, 4.0, 0.0]));
        let pts3: Vec<_> = diag.uniform_samples(0.0, 2.5, 1e-6).collect();
        assert_eq!(pts3.len(), 3);
        assert!((pts3[0] - DVec([0.0, 0.0, 0.0])).length() < 1e-12);
        assert!((pts3[1] - DVec([1.5, 2.0, 0.0])).length() < 1e-12);
        assert!((pts3[2] - DVec([3.0, 4.0, 0.0])).length() < 1e-12);
    }

    #[test]
    fn uniform_samples_edge_cases() {
        let line = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([5.0, 0.0, 0.0]));
        // start > len → 0 points.
        assert_eq!(line.uniform_samples(6.0, 1.0, 1e-6).count(), 0);
        // start = len → 1 point at the endpoint.
        let at_end: Vec<_> = line.uniform_samples(5.0, 1.0, 1e-6).collect();
        assert_eq!(at_end.len(), 1);
        assert!((at_end[0] - DVec([5.0, 0.0, 0.0])).length() < 1e-12);
        // step > len → 1 point at the start.
        let big_step: Vec<_> = line.uniform_samples(0.0, 100.0, 1e-6).collect();
        assert_eq!(big_step.len(), 1);
        assert!((big_step[0] - DVec([0.0, 0.0, 0.0])).length() < 1e-12);
        // step = len → 2 points: start and end.
        let full_step: Vec<_> = line.uniform_samples(0.0, 5.0, 1e-6).collect();
        assert_eq!(full_step.len(), 2);
        assert!((full_step[0] - DVec([0.0, 0.0, 0.0])).length() < 1e-12);
        assert!((full_step[1] - DVec([5.0, 0.0, 0.0])).length() < 1e-12);
        // Zero-length line → 0 points regardless of step.
        let zero = LineSeg3d::create(DVec([3.0, 3.0, 3.0]), DVec([3.0, 3.0, 3.0]));
        assert_eq!(zero.uniform_samples(0.0, 1.0, 1e-6).count(), 0);
        // Negative start → clamped to 0, same as start=0.
        let neg: Vec<_> = line.uniform_samples(-2.0, 5.0, 1e-6).collect();
        assert_eq!(neg.len(), 2);
        assert!((neg[0] - DVec([0.0, 0.0, 0.0])).length() < 1e-12);
    }

    #[test]
    fn adaptive_samples() {
        let line = LineSeg3d::create(DVec([1.0, 0.0, 0.0]), DVec([4.0, 0.0, 0.0]));
        let samples: Vec<_> = line.adaptive_samples().collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], line.start());
        assert_eq!(samples[1], line.end());
    }

    #[test]
    fn degenerate_zero_length() {
        let line = LineSeg3d::create(DVec([5.0, 5.0, 5.0]), DVec([5.0, 5.0, 5.0]));
        assert!(line.length() < 1e-12);
        let p = line.point(0.0).unwrap();
        assert_eq!(p, DVec([5.0, 5.0, 5.0]));
        assert!(line.tangent(0.0).is_none());
        let mut results = [DVec([0.0; 3]); 3];
        line.point_with_derivs(0.0, &mut results).unwrap();
        assert_eq!(results[0], DVec([5.0, 5.0, 5.0]));
        assert!(results[1].length() < 1e-12);
        assert!(results[2].length() < 1e-12);
    }

    #[test]
    fn diagonal_3d() {
        let line = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([1.0, 1.0, 1.0]));
        let expected_len = 3.0_f64.sqrt();
        assert!((line.length() - expected_len).abs() < 1e-12);
        let mid = line.point(expected_len / 2.0).unwrap();
        assert!((mid - DVec([0.5, 0.5, 0.5])).length() < 1e-12);
    }

    #[test]
    fn curvature_zero_everywhere_none_outside() {
        let line = LineSeg3d::create(DVec([1.0, 2.0, 3.0]), DVec([4.0, 6.0, 3.0]));
        let len = line.length();
        assert_eq!(line.curvature(0.0), Some(DVec([0.0; 3])));
        assert_eq!(line.curvature(len / 2.0), Some(DVec([0.0; 3])));
        assert_eq!(line.curvature(len), Some(DVec([0.0; 3])));
        assert!(line.curvature(-0.001).is_none());
        assert!(line.curvature(len + 0.001).is_none());
    }

    #[test]
    fn is_closed() {
        let open = LineSeg3d::create(DVec([0.0, 0.0, 0.0]), DVec([1.0, 0.0, 0.0]));
        assert!(!open.is_closed());
        // A zero-length line (same point for both endpoints) is closed.
        let closed = LineSeg3d::create(DVec([1.0, 2.0, 3.0]), DVec([1.0, 2.0, 3.0]));
        assert!(closed.is_closed());
    }
}
