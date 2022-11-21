use numpy::{IntoPyArray, ToPyArray};
use ordered_float::NotNan;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::{mem, ops, ptr};

// 0 1 2 3 4 5 6 7 8 9 A B C D E F
// -   -   -   -   -   -   -   -
// ---     ---     ---     ---
// -------         -------
// ---------------
// -------------------------------
//
// x = 20, i = 0, middle = 16: x < data[15]
// x = 10.5, i = 0, middle = 8: data[7] < x
// x = 3.5, i = 8, middle = 4: x < data[11]
// x = 3.5, i = 8, middle = 2: data[9] < x
// x = 1.5, i = 10, middle = 1: data[10] < x
// x = .5, i = 11, middle = 0:

// #[inline]
// const fn lsb(n: usize) -> usize {
//     n & n.wrapping_neg()
// }

const fn msb(n: usize) -> usize {
    if n < 2 {
        n
    } else {
        1 << usize::BITS - n.leading_zeros()
    }
}

#[inline]
fn map_ref<T, F: FnOnce(T) -> T>(x: &mut T, f: F) {
    unsafe {
        ptr::write(x, f(ptr::read(x)));
    }
}

#[inline]
const fn up(n: usize) -> usize {
    // up(n) = n | lsb(!n)
    //       = n | !n & n + 1
    //       = n | !(n | !(n + 1))
    //       = !(!n & (n | !(n + 1)))
    //       = !(!n & !(n + 1))
    //       = n | n + 1
    n | n + 1
}

#[inline]
const fn start(n: usize) -> usize {
    // start(n - 1) = n & !lsb(n)
    //              = n & !(n & !(n - 1))
    //              = n & (!n | n - 1)
    //              = n & n - 1
    // start(n) = n & n + 1
    n & n + 1
}

// #[inline]
// const fn left(n: usize) -> usize {
//     start(n) - 1
// }

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct MaxF64(NotNan<f64>);

impl ops::Add for MaxF64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }
}

impl num::Zero for MaxF64 {
    fn zero() -> Self {
        Self(NotNan::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(Clone, Debug)]
pub struct FenwickTree<T> {
    data: Vec<T>,
}

impl<T: num::Zero + Clone> FenwickTree<T> {
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Constructs a new, empty `FenwickTree<G>` with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn new_zeroed(len: usize) -> Self {
        Self {
            data: vec![T::zero(); len],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// O(log n)
    pub fn update_add(&mut self, mut i: usize, dx: T) {
        loop {
            map_ref(&mut self.data[i], |x| x + dx.clone());
            i = up(i);
            if self.len() <= i {
                break;
            }
        }
    }

    /// O(log n)
    pub fn prefix_sum(&self, mut i: usize) -> T {
        let mut ps = T::zero();
        while 0 < i {
            i -= 1;
            ps = ps + self.data[i].clone();
            i = start(i);
        }
        ps
    }

    /// Simular to `get`, therefore has similar performance.
    ///
    /// Avg: O(1), Worst case: O(log n)
    pub fn push(&mut self, mut x: T) {
        let first_start = start(self.len());
        let mut i = self.len();
        while first_start < i {
            i -= 1;
            x = x + self.data[i].clone();
            i = start(i);
        }

        self.data.push(x);
    }

    /// O(1)
    #[inline]
    pub fn pop(&mut self) {
        self.data.pop();
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `FenwickTree<G>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// O(n)
    pub fn resize(&mut self, size: usize, x: T) {
        while self.len() < size {
            self.push(x.clone());
        }
        self.data.resize(size, T::zero());
    }

    /// O(n)
    pub fn resize_with<F: FnMut() -> T>(&mut self, size: usize, f: &mut F) {
        while self.len() < size {
            self.push(f());
        }
        self.data.resize(size, T::zero());
    }

    /// `update_add` should be perferred over this, because it's faster (it doesn't call `get`).
    ///
    /// O(log n)
    pub fn update_set(&mut self, i: usize, x: T)
    where
        T: ops::Sub<Output = T>,
    {
        self.update_add(i, x - self.get(i));
    }

    /// Avg: O(1), Worst case: O(log i)
    pub fn get(&self, mut i: usize) -> T
    where
        T: ops::Sub<Output = T>,
    {
        let mut x = self.data[i].clone();
        let first_start = start(i);
        while first_start < i {
            i -= 1;
            x = x - self.data[i].clone();
            i = start(i);
        }
        x
    }

    /// O(log n)
    pub fn sample(&self, mut x: T) -> (usize, Option<&T>)
    where
        T: PartialOrd + ops::Sub<Output = T>,
    {
        // 0 1 2 3 4
        // -   -   -
        // ---
        // -------
        let mut idx = 0;
        let mut middle = msb(self.len());
        while 0 < middle && idx < self.len() {
            if let Some(data) = self.data.get(idx + middle - 1) {
                if *data < x {
                    idx += middle;
                    x = x - data.clone();
                }
            }
            middle /= 2;
        }
        (idx, self.data.get(idx))
    }
}

impl<T: Clone + num::Zero> Extend<T> for FenwickTree<T> {
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        let iter = iter.into_iter();
        self.reserve(match iter.size_hint() {
            (_, Some(len)) => len,
            (len, None) => len,
        });
        for x in iter {
            self.push(x);
        }
    }
}

//        0
//    1       2
//  3   4   5   6
// 7 8 9 A B C D E

#[derive(Clone, Debug)]
pub struct SegmentTree<T> {
    data: Vec<T>,
}

impl<T> SegmentTree<T> {
    #[inline]
    pub fn len(&self) -> usize {
        (self.data.len() + 1) / 2
    }

    #[inline]
    fn offset(&self) -> usize {
        self.data.len() / 2
    }

    /// O(n)
    pub fn new(len: usize) -> Self
    where
        T: Clone + num::Zero,
    {
        // let mut data = Vec::new();
        // data.resize_with(2 * len - 1, T::zero);
        // Self { data }
        Self {
            data: vec![T::zero(); 2 * len - 1],
        }
    }

    #[inline]
    pub fn total(&self) -> &T {
        &self.data[0]
    }

    fn build(&mut self)
    where
        T: Clone + ops::Add<Output = T>,
    {
        for i in (0..self.offset()).rev() {
            let l = 2 * i + 1;
            let r = l + 1;
            self.data[i] = self.data[l].clone() + self.data[r].clone();
        }
    }

    /// O(n)
    pub fn clone_from_slice(&mut self, slice: &[T])
    where
        T: Clone + ops::Add<Output = T>,
    {
        let offset = self.offset();
        self.data[offset..].clone_from_slice(slice);
        self.build();
    }

    /// O(n)
    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy + ops::Add<Output = T>,
    {
        let offset = self.offset();
        self.data[offset..].copy_from_slice(slice);
        self.build();
    }

    /// O(n)
    pub fn build_from_iter<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter)
    where
        T: Clone + ops::Add<Output = T>,
    {
        let offset = self.offset();
        for (data, x) in self.data[offset..].iter_mut().zip(iter.into_iter()) {
            *data = x;
        }
        self.build();
    }

    pub fn sum(&self, mut l: usize, mut r: usize) -> T
    where
        T: Clone + num::Zero,
    {
        let offset = self.offset();
        l += offset;
        r += offset;

        let mut result = T::zero();
        while l < r {
            if l & 1 == 0 {
                result = result + self.data[l].clone();
                l += 1;
            }
            if r & 1 == 0 {
                r -= 1;
                result = result + self.data[l].clone();
            }
            l /= 2;
            r /= 2;
        }
        result
    }

    pub fn update(&mut self, mut i: usize, x: T)
    where
        T: Clone + ops::Add<Output = T> + std::fmt::Debug,
    {
        i += self.offset();
        self.data[i] = x;
        while 0 < i {
            // r = (i - 1) / 2 * 2 + 2
            //   = (i - 1 & !1) + 2
            //   = i + 1 & !1
            //
            // l = (i - 1) / 2 * 2 + 1
            //   = (i - 1 & !1) + 1
            //   = i - 1 | 1
            let l = i - 1 | 1;
            let r = l + 1;
            i = l / 2;
            unsafe {
                *self.data.get_unchecked_mut(i) =
                    self.data.get_unchecked(l).clone() + self.data.get_unchecked(r).clone();
            }
        }
    }
}

impl<T> ops::Deref for SegmentTree<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { self.data.get_unchecked(self.offset()..) }
    }
}

pub struct Priorities {
    max_tree: SegmentTree<MaxF64>,
    sample_tree: FenwickTree<NotNan<f64>>,
}

impl Priorities {
    #[inline]
    pub fn new(len: usize) -> Self {
        Self {
            max_tree: SegmentTree::new(len),
            sample_tree: FenwickTree::new_zeroed(len),
        }
    }

    #[inline]
    pub fn build(array: &[NotNan<f64>]) -> Self {
        let mut sample_tree = FenwickTree::with_capacity(array.len());
        sample_tree.extend(array.iter().copied());

        let mut max_tree = SegmentTree::new(array.len());
        max_tree.copy_from_slice(unsafe { mem::transmute::<&[NotNan<f64>], &[MaxF64]>(array) });

        Self {
            max_tree,
            sample_tree,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.sample_tree.len()
    }

    #[inline]
    pub fn sum(&self) -> NotNan<f64> {
        self.sample_tree.prefix_sum(self.len())
    }

    #[inline]
    pub fn max(&self) -> NotNan<f64> {
        self.max_tree.total().0
    }

    #[inline]
    pub fn update(&mut self, i: usize, x: NotNan<f64>) {
        self.max_tree.update(i, MaxF64(x));
        self.sample_tree.update_set(i, x);
    }

    #[inline]
    pub fn sample(&self, x: NotNan<f64>) -> (usize, Option<NotNan<f64>>) {
        let (i, x) = self.sample_tree.sample(x);
        (i, x.copied())
    }

    #[inline]
    pub fn random_sample(&self) -> (usize, NotNan<f64>) {
        let (i, x) =
            self.sample(NotNan::new(thread_rng().gen_range(0.0..self.sum().into())).unwrap());
        (i, x.unwrap())
    }

    #[inline]
    pub fn multi_sample(&self, x: &[NotNan<f64>]) -> (Vec<usize>, Vec<NotNan<f64>>) {
        x.par_iter()
            .map(|&x| {
                let (i, x) = self.sample(x);
                (i, x.unwrap_or(num::zero()))
            })
            .unzip()
    }

    #[inline]
    pub fn random_multi_sample(&self, num_samples: usize) -> (Vec<usize>, Vec<NotNan<f64>>) {
        (0..num_samples)
            .into_par_iter()
            .map(|_| self.random_sample())
            .unzip()
    }

    #[inline]
    pub fn as_ndarray_f64(&self) -> ndarray::ArrayView1<f64> {
        unsafe { mem::transmute::<&[NotNan<f64>], &[f64]>(&**self) }.into()
    }

    #[inline]
    pub fn as_ndarray(&self) -> ndarray::ArrayView1<NotNan<f64>> {
        (&**self).into()
    }
}

impl ops::Deref for Priorities {
    type Target = [NotNan<f64>];

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute::<&[MaxF64], &[NotNan<f64>]>(&*self.max_tree) }
    }
}

#[pyclass(name = "Priorities")]
#[pyo3(text_signature = "(len)")]
#[repr(transparent)]
pub struct PrioritiesPy(Priorities);

#[pymethods]
impl PrioritiesPy {
    #[classattr]
    const __contains__: Option<PyObject> = None;

    #[new]
    #[inline]
    pub fn new(len: usize) -> Self {
        Self(Priorities::new(len))
    }

    #[pyo3(text_signature = "(array)")]
    #[staticmethod]
    #[inline]
    pub fn build(array: numpy::PyReadonlyArray1<f64>) -> PyResult<Self> {
        let array = array.as_slice()?;
        for &x in array {
            NotNan::new(x).map_err(|_| {
                pyo3::exceptions::PyFloatingPointError::new_err(
                    "NaN is not allowed in Priorities.sample",
                )
            })?;
        }
        Ok(Self(Priorities::build(unsafe {
            mem::transmute::<&[f64], &[NotNan<f64>]>(array)
        })))
    }

    #[pyo3(text_signature = "()")]
    #[inline]
    pub fn sum(&self) -> f64 {
        self.0.sum().into()
    }

    #[pyo3(text_signature = "()")]
    #[inline]
    pub fn max(&self) -> f64 {
        self.0.max().into()
    }

    #[pyo3(text_signature = "(i)")]
    #[cfg(debug_assertions)]
    #[inline]
    pub fn prefix_sum(&self, i: usize) -> f64 {
        self.0.sample_tree.prefix_sum(i).into()
    }

    #[pyo3(text_signature = "(left, right)")]
    #[cfg(debug_assertions)]
    #[inline]
    pub fn seg_max(&self, left: usize, right: usize) -> f64 {
        self.0.max_tree.sum(left, right).0.into()
    }

    #[inline]
    pub fn __len__(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn __getitem__(&self, mut i: isize) -> PyResult<f64> {
        if i < 0 {
            i += self.0.len() as isize;
        }
        if i < 0 {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Priorities index out of range (negative)",
            ))
        } else {
            if let Some(&x) = self.0.get(i as usize) {
                Ok(x.into())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "Priorities index out of range (positive)",
                ))
            }
        }
    }

    #[pyo3(text_signature = "()")]
    pub fn into_numpy<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<f64> {
        self.0.as_ndarray_f64().to_pyarray(py)
    }

    #[pyo3(text_signature = "(i, value)")]
    #[inline]
    pub fn update(&mut self, i: usize, x: f64) -> PyResult<()> {
        let x = NotNan::new(x).map_err(|_| {
            pyo3::exceptions::PyFloatingPointError::new_err(
                "NaN is not allowed in Priorities.update",
            )
        })?;
        self.0.update(i, x);
        Ok(())
    }

    #[pyo3(text_signature = "(sample)")]
    #[inline]
    pub fn sample(&self, x: f64) -> PyResult<(usize, Option<f64>)> {
        let x = NotNan::new(x).map_err(|_| {
            pyo3::exceptions::PyFloatingPointError::new_err(
                "NaN is not allowed in Priorities.sample",
            )
        })?;
        let (i, x) = self.0.sample(x);
        Ok((i, x.map(Into::into)))
    }

    #[pyo3(text_signature = "()")]
    #[inline]
    pub fn random_sample(&self) -> PyResult<(usize, f64)> {
        let (i, x) = self.0.random_sample();
        Ok((i, x.into()))
    }

    #[pyo3(text_signature = "(samples)")]
    #[inline]
    pub fn multi_sample<'py>(
        &self,
        py: Python<'py>,
        x: numpy::PyReadonlyArray1<f64>,
    ) -> PyResult<(&'py numpy::PyArray1<usize>, &'py numpy::PyArray1<f64>)> {
        let samples = x.as_slice()?;
        for &x in samples {
            NotNan::new(x).map_err(|_| {
                pyo3::exceptions::PyFloatingPointError::new_err(
                    "NaN is not allowed in Priorities.sample",
                )
            })?;
        }
        let samples = unsafe { mem::transmute::<&[f64], &[NotNan<f64>]>(samples) };
        let (indices, prios) = self.0.multi_sample(samples);
        Ok((
            indices.into_pyarray(py),
            unsafe { mem::transmute::<Vec<NotNan<f64>>, Vec<f64>>(prios) }.into_pyarray(py),
        ))
    }

    #[pyo3(text_signature = "(num_samples)")]
    #[inline]
    pub fn random_multi_sample<'py>(
        &self,
        py: Python<'py>,
        num_samples: usize,
    ) -> PyResult<(&'py numpy::PyArray1<usize>, &'py numpy::PyArray1<f64>)> {
        let (indices, prios) = self.0.random_multi_sample(num_samples);
        Ok((
            indices.into_pyarray(py),
            unsafe { mem::transmute::<Vec<NotNan<f64>>, Vec<f64>>(prios) }.into_pyarray(py),
        ))
    }

    #[inline]
    pub fn __str__(&self) -> PyResult<String> {
        use std::fmt::Write;

        let mut result = format!("[{}", self.0[0]);
        if 100 < self.0.len() {
            for &x in &self.0[1..20] {
                write!(result, ", {x}").unwrap();
            }
            write!(result, ", ...").unwrap();
            for &x in &self.0[self.0.len() - 10..] {
                write!(result, ", {x}").unwrap();
            }
        } else {
            for &x in &self.0[1..] {
                write!(result, ", {x}").unwrap();
            }
        }
        result += "]";
        Ok(result)
    }

    #[inline]
    pub fn __repr__(&self) -> PyResult<String> {
        use std::fmt::Write;

        let mut result = format!("Priorities([{}", self.0[0]);
        if 100 < self.0.len() {
            for &x in &self.0[1..20] {
                write!(result, ", {x}").unwrap();
            }
            result += ", ...";
            for &x in &self.0[self.0.len() - 10..] {
                write!(result, ", {x}").unwrap();
            }
        } else {
            for &x in &self.0[1..] {
                write!(result, ", {x}").unwrap();
            }
        }
        result += "])";
        Ok(result)
    }
}
