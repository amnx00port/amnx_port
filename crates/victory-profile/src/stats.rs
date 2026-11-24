//! The arithmetic behind the summaries: percentiles of frame intervals,
//! the slowest and biggest items of a period, a bounded history. Plain
//! data, no clocks, so it is tested without a GPU or a title.

use std::collections::VecDeque;

/// Frame intervals (milliseconds) of a period, summed up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    /// The mean of the slowest 1% (at least one sample): "1% low" is
    /// [`fps`] of this.
    pub low_1: f64,
}

impl Summary {
    /// `None` without a finite sample.
    pub fn of(samples: &[f64]) -> Option<Summary> {
        let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let worst = (n / 100).max(1);
        Some(Summary {
            n,
            avg: sorted.iter().sum::<f64>() / n as f64,
            p50: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
            max: sorted[n - 1],
            low_1: sorted[n - worst..].iter().sum::<f64>() / worst as f64,
        })
    }
}

/// The nearest-rank percentile `p` (0-100) of ascending `sorted`; 0 when
/// empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p.clamp(0.0, 100.0) / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Frames per second of an interval in milliseconds (0 for none).
pub fn fps(ms: f64) -> f64 {
    if ms > 0.0 { 1000.0 / ms } else { 0.0 }
}

/// How many intervals exceed `factor` times the period's median: frames
/// that stand out from their neighbours whatever the frame rate.
pub fn stutters(samples: &[f64], factor: f64) -> usize {
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f64::total_cmp);
    let median = percentile(&sorted, 50.0);
    if median <= 0.0 {
        return 0;
    }
    sorted.iter().filter(|&&v| v > median * factor).count()
}

/// The `n` items with the largest keys seen; ties keep the earlier item.
#[derive(Debug, Clone)]
pub struct Top<T> {
    n: usize,
    /// Largest key first.
    items: Vec<(u64, T)>,
}

impl<T> Top<T> {
    pub fn new(n: usize) -> Self {
        Top {
            n,
            items: Vec::with_capacity(n + 1),
        }
    }

    pub fn push(&mut self, key: u64, item: T) {
        if self.n == 0 || (self.items.len() == self.n && self.items[self.n - 1].0 >= key) {
            return;
        }
        let at = self.items.partition_point(|(k, _)| *k >= key);
        self.items.insert(at, (key, item));
        self.items.truncate(self.n);
    }

    /// Largest key first.
    pub fn items(&self) -> &[(u64, T)] {
        &self.items
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

/// The last `cap` items.
#[derive(Debug, Clone)]
pub struct Ring<T> {
    cap: usize,
    items: VecDeque<T>,
}

impl<T> Ring<T> {
    pub fn new(cap: usize) -> Self {
        Ring {
            cap,
            items: VecDeque::with_capacity(cap),
        }
    }

    pub fn push(&mut self, item: T) {
        if self.cap == 0 {
            return;
        }
        if self.items.len() == self.cap {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    /// Oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let s: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&s, 50.0), 50.0);
        assert_eq!(percentile(&s, 95.0), 95.0);
        assert_eq!(percentile(&s, 99.0), 99.0);
        assert_eq!(percentile(&s, 100.0), 100.0);
        assert_eq!(percentile(&s, 0.0), 1.0);
        assert_eq!(percentile(&[7.0], 99.0), 7.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn summary_of_unsorted_intervals() {
        let mut s: Vec<f64> = (1..=200).map(f64::from).collect();
        s.reverse();
        s.push(f64::NAN);
        let sum = Summary::of(&s).unwrap();
        assert_eq!(sum.n, 200);
        assert_eq!(sum.avg, 100.5);
        assert_eq!(sum.p50, 100.0);
        assert_eq!(sum.p95, 190.0);
        assert_eq!(sum.p99, 198.0);
        assert_eq!(sum.max, 200.0);
        // The slowest 1% of 200 is two frames: 199 and 200.
        assert_eq!(sum.low_1, 199.5);
        assert!(Summary::of(&[]).is_none());
        assert!(Summary::of(&[f64::INFINITY]).is_none());
    }

    #[test]
    fn one_percent_low_of_few_frames_is_the_slowest() {
        let sum = Summary::of(&[33.0, 34.0, 80.0]).unwrap();
        assert_eq!(sum.low_1, 80.0);
        assert_eq!(fps(sum.low_1), 12.5);
        assert_eq!(fps(0.0), 0.0);
    }

    #[test]
    fn stutters_stand_out_from_the_median() {
        let mut s = vec![33.3; 30];
        s.extend([70.0, 100.0, 60.0]);
        assert_eq!(stutters(&s, 2.0), 2);
        assert_eq!(stutters(&[], 2.0), 0);
        assert_eq!(stutters(&[0.0, 0.0, 5.0], 2.0), 0);
    }

    #[test]
    fn top_keeps_the_largest() {
        let mut top = Top::new(3);
        for (k, v) in [(5, "a"), (1, "b"), (9, "c"), (5, "d"), (7, "e"), (2, "f")] {
            top.push(k, v);
        }
        let got: Vec<_> = top.items().iter().map(|&(k, v)| (k, v)).collect();
        assert_eq!(got, [(9, "c"), (7, "e"), (5, "a")]);
        top.clear();
        assert!(top.items().is_empty());
        let mut none = Top::new(0);
        none.push(1, ());
        assert!(none.items().is_empty());
    }

    #[test]
    fn ring_keeps_the_last() {
        let mut ring = Ring::new(3);
        assert!(ring.is_empty());
        for i in 0..5 {
            ring.push(i);
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), [2, 3, 4]);
    }
}
