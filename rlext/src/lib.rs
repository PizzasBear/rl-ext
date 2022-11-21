use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// pub mod buffer;
// pub mod gae;
pub mod priorities;

pub fn traverse(next: &[usize], mut start: usize, num: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(num);
    loop {
        indices.push(start);
        if num <= indices.len() {
            break;
        }
        start = next[start];
    }
    indices
}

pub fn par_traverse(next: &[usize], starts: &[usize], num: usize) -> Array2<usize> {
    let mut result = Array2::zeros((starts.len(), num));
    starts
        .par_iter()
        .zip(result.axis_iter_mut(ndarray::Axis(0)).into_par_iter())
        .for_each(|(&(mut i), mut result)| {
            result[0] = i;
            for indices in result.iter_mut().skip(1) {
                i = next[i];
                *indices = i;
            }
        });
    result
}

/// Calculates the GAE (Generalized Advantage Estimation).
///
/// # Returns
/// A tuple containing the returns and advantages in that order.
pub fn gae(
    values: ArrayView1<f32>,
    rewards: ArrayView1<f32>,
    terminated: ArrayView1<bool>,
    truncated: ArrayView1<bool>,
    lambda: f32,
    discount: f32,
    normalize_advantages: bool,
) -> (Array1<f32>, Array1<f32>) {
    const EPSILON: f32 = 1e-6;

    let len = rewards.len();
    assert_eq!(values.len(), len + 1);
    assert_eq!(terminated.len(), len);
    assert_eq!(truncated.len(), len);

    let mut advantages = Array1::zeros(len);
    let mut returns = Array1::zeros(len);
    let mut next_advantage = 0.;
    azip!((
        &reward in &rewards.slice(s![..;-1]),
        values in values.slice(s![..;-1]).axis_windows(Axis(0), 2),
        &terminated in &terminated.slice(s![..;-1]),
        &truncated in &truncated.slice(s![..;-1]),
        return_ in &mut returns.slice_mut(s![..;-1]),
        advantage in &mut advantages.slice_mut(s![..;-1]),
    ) {
        *return_ = if terminated {
            reward
        } else if truncated {
            reward + discount * values[0]
        } else {
            reward + discount * (values[0] + lambda * next_advantage)
        };
        next_advantage = *return_ - values[1];
        *advantage = next_advantage;
    });

    if normalize_advantages {
        advantages -= advantages.mean().unwrap();
        advantages /= advantages.std(0.) + EPSILON;
    }

    (returns, advantages)
}

/// Calculates the GAE (Generalized Advantage Estimation) of parallel environments.
///
/// # Returns
/// A tuple containing the returns and advantages in that order.
pub fn par_gae(
    values: ArrayView2<f32>,
    rewards: ArrayView2<f32>,
    terminated: ArrayView2<bool>,
    truncated: ArrayView2<bool>,
    lambda: f32,
    discount: f32,
    normalize_advantages: bool,
) -> (Array2<f32>, Array2<f32>) {
    const EPSILON: f32 = 1e-6;

    let (envs, len) = rewards.dim();
    assert_eq!(values.dim(), (envs, len + 1));
    assert_eq!(terminated.dim(), (envs, len));
    assert_eq!(truncated.dim(), (envs, len));

    let mut returns = Array2::zeros((envs, len));
    let mut advantages = Array2::zeros((envs, len));
    ndarray::par_azip!((
        rewards in rewards.axis_iter(Axis(0)),
        values in values.axis_iter(Axis(0)),
        terminated in terminated.axis_iter(Axis(0)),
        truncated in truncated.axis_iter(Axis(0)),
        mut returns in returns.axis_iter_mut(Axis(0)),
        mut advantages in advantages.axis_iter_mut(Axis(0)),
    ) {
        let mut next_advantage = 0.;
        azip!((
            &reward in &rewards.slice(s![..;-1]),
            values in values.slice(s![..;-1]).axis_windows(Axis(0), 2),
            &terminated in &terminated.slice(s![..;-1]),
            &truncated in &truncated.slice(s![..;-1]),
            return_ in &mut returns.slice_mut(s![..;-1]),
            advantage in &mut advantages.slice_mut(s![..;-1]),
        ) {
            *return_ = if terminated {
                reward
            } else if truncated {
                reward + discount * values[0]
            } else {
                reward + discount * (values[0] + lambda * next_advantage)
            };
            next_advantage = *return_ - values[1];
            *advantage = next_advantage;
        });
    });

    if normalize_advantages {
        advantages -= advantages.mean().unwrap();
        advantages /= advantages.std(0.) + EPSILON;
    }
    (returns, advantages)
}

#[pymodule]
pub fn rlext(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "traverse", text_signature = "(next, start, num)")]
    fn traverse_py<'py>(
        py: Python<'py>,
        next: PyReadonlyArray1<usize>,
        start: usize,
        num: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        Ok(traverse(next.as_slice()?, start, num).into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "par_traverse", text_signature = "(next, starts, num)")]
    fn par_traverse_py<'py>(
        py: Python<'py>,
        next: PyReadonlyArray1<usize>,
        starts: PyReadonlyArray1<usize>,
        num: usize,
    ) -> PyResult<&'py PyArray2<usize>> {
        Ok(par_traverse(next.as_slice()?, starts.as_slice()?, num).into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(
        name = "gae",
        text_signature = "(values, rewards, terminated, truncated, lambda_, discount, normalize_advantages)"
    )]
    /// Calculates the GAE (Generalized Advantage Estimation).
    ///
    /// # Returns
    /// A tuple containing the returns and advantages in that order.
    fn gae_py<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<f32>,
        rewards: PyReadonlyArray1<f32>,
        terminated: PyReadonlyArray1<bool>,
        truncated: PyReadonlyArray1<bool>,
        lambda_: f32,
        discount: f32,
        normalize_advantages: bool,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<f32>)> {
        let len = rewards.len();
        if values.len() == len + 1 && terminated.len() == len && truncated.len() == len {
            let (returns, advantages) = gae(
                values.as_array(),
                rewards.as_array(),
                terminated.as_array(),
                truncated.as_array(),
                lambda_,
                discount,
                normalize_advantages,
            );
            Ok((returns.into_pyarray(py), advantages.into_pyarray(py)))
        } else {
            todo!()
            // Err(pyo3::exceptions::PyUnicodeError)
        }
    }

    #[pyfn(m)]
    #[pyo3(
        name = "par_gae",
        text_signature = "(values, rewards, terminated, truncated, lambda_, discount, normalize_advantages)"
    )]
    /// Calculates the GAE (Generalized Advantage Estimation) of parallel environments.
    ///
    /// # Returns
    /// A tuple containing the returns and advantages in that order.
    fn par_gae_py<'py>(
        py: Python<'py>,
        values: PyReadonlyArray2<f32>,
        rewards: PyReadonlyArray2<f32>,
        terminated: PyReadonlyArray2<bool>,
        truncated: PyReadonlyArray2<bool>,
        lambda_: f32,
        discount: f32,
        normalize_advantages: bool,
    ) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<f32>)> {
        let values = values.as_array();
        let rewards = rewards.as_array();
        let terminated = terminated.as_array();
        let truncated = truncated.as_array();

        let (envs, len) = rewards.dim();
        if values.dim() == (envs, len + 1)
            && terminated.dim() == (envs, len)
            && truncated.dim() == (envs, len)
        {
            let (returns, advantages) = par_gae(
                values,
                rewards,
                truncated,
                terminated,
                lambda_,
                discount,
                normalize_advantages,
            );
            Ok((returns.into_pyarray(py), advantages.into_pyarray(py)))
        } else {
            todo!()
            // Err(pyo3::exceptions::PyUnicodeError)
        }
    }

    m.add_class::<priorities::PrioritiesPy>()?;
    // m.add_class::<priorities::ParPrioritiesPy>()?;
    // m.add_class::<buffer::ReplayBufferPy>()?;
    // m.add_class::<buffer::ParReplayBufferPy>()?;
    // m.add_class::<buffer::StateDTypePy>()?;
    // m.add_class::<buffer::ActionTypePy>()?;
    // m.add_class::<gae::GaePy>()?;
    // m.add_class::<gae::ParGaePy>()?;

    Ok(())
}
