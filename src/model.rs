//! SVM model types.

use crate::{
    consts::{
        DEFAULT_CACHE_SIZE, DEFAULT_COEF0, DEFAULT_COST, DEFAULT_DEGREE, DEFAULT_MODEL_EPSILON,
        DEFAULT_NU, DEFAULT_PROBABILITY_ESTIMATES, DEFAULT_SHRINKING, DEFAULT_TERMINATION_EPSILON,
    },
    data::SvmNodes,
    error::Error,
    init::{KernelInit, ModelInit, SvmInit},
    state::{Trained, Untrained},
};
use num_derive::FromPrimitive;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    ffi::{CStr, CString},
    num::NonZeroUsize,
    os::raw::c_int,
    ptr::NonNull,
    str::FromStr,
};

/// The SVM model.
#[derive(Debug)]
pub struct Svm<State> {
    pub(crate) state: State,
}

impl Svm<Untrained> {
    pub fn kind(&self) -> SvmKind {
        num::FromPrimitive::from_usize(self.state.params.svm_type as usize).unwrap()
    }

    /// Trains the model on given dataset.
    pub fn fit<X, Y>(&self, x: X, y: Y) -> Result<Svm<Trained>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
        Y: AsRef<[f64]>,
    {
        // disable message from libsvm
        crate::disable_print_in_libsvm();

        let Svm {
            state:
                Untrained {
                    gamma_opt,
                    mut params,
                    weight_labels,
                    weights,
                },
        } = self;

        // transoform x
        let x_nodes = x.try_into()?;
        let x_slice = x_nodes
            .end_indexes
            .iter()
            .cloned()
            .scan(0, |from, to| {
                let prev_from = *from;
                *from = to;
                Some((prev_from, to))
            })
            .map(|(from, to)| {
                x_nodes.nodes.get(from..to).unwrap().as_ptr() as *mut libsvm_sys::svm_node
            })
            .collect::<Vec<_>>();
        let x_ptr = x_slice.as_ptr() as *mut *mut libsvm_sys::svm_node;

        // transoform y
        let y_slice = y.as_ref();
        let y_ptr = y_slice.as_ptr() as *mut f64;

        if x_nodes.end_indexes.len() != y_slice.len() {
            return Err(Error::InvalidData {
                reason: "the size of data and label does not match".into(),
            });
        }

        // costruct problem
        let problem = libsvm_sys::svm_problem {
            l: x_nodes.end_indexes.len() as c_int,
            x: x_ptr,
            y: y_ptr,
        };

        // compute gamma if necessary
        let gamma = gamma_opt.unwrap_or_else(|| (x_nodes.n_features as f64).recip());

        // update raw params
        // we store pointers late to avoid move after saving pointers
        params.gamma = gamma;
        params.weight_label = weight_labels.as_ptr() as *mut _;
        params.weight = weights.as_ptr() as *mut _;

        unsafe {
            let ptr = libsvm_sys::svm_check_parameter(&problem, &params);

            if ptr != std::ptr::null() {
                let err_msg = CStr::from_ptr(ptr)
                    .to_str()
                    .map_err(|err| Error::InternalError {
                        reason: format!("failed to decode error mesage: {:?}", err),
                    })?;
                return Err(Error::InvalidHyperparameter {
                    reason: format!("svm_check_parameter() failed: {}", err_msg),
                });
            }
        }

        // train model
        let model_ptr = unsafe {
            NonNull::new(libsvm_sys::svm_train(&problem, &params)).ok_or_else(|| {
                Error::InternalError {
                    reason: "svm_train() returns null pointer".into(),
                }
            })?
        };

        Ok(Svm {
            state: Trained {
                model_ptr,
                nodes_opt: Some(x_nodes),
            },
        })
    }

    /// Runs cross validation on given data.
    pub fn cross_validate<X, Y>(&self, x: X, y: Y, n_folds: NonZeroUsize) -> Result<Vec<f64>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
        Y: AsRef<[f64]>,
    {
        // disable message from libsvm
        crate::disable_print_in_libsvm();

        let Svm {
            state:
                Untrained {
                    gamma_opt,
                    mut params,
                    weight_labels,
                    weights,
                },
        } = self;

        // transoform x
        let x_nodes = x.try_into()?;
        let x_slice = x_nodes
            .end_indexes
            .iter()
            .cloned()
            .scan(0, |from, to| {
                let prev_from = *from;
                *from = to;
                Some((prev_from, to))
            })
            .map(|(from, to)| {
                x_nodes.nodes.get(from..to).unwrap().as_ptr() as *mut libsvm_sys::svm_node
            })
            .collect::<Vec<_>>();
        let x_ptr = x_slice.as_ptr() as *mut *mut libsvm_sys::svm_node;

        // transoform y
        let y_slice = y.as_ref();
        let y_ptr = y_slice.as_ptr() as *mut f64;

        let n_records = {
            if x_nodes.end_indexes.len() != y_slice.len() {
                return Err(Error::InvalidData {
                    reason: "the size of data and label does not match".into(),
                });
            }
            y_slice.len()
        };

        // costruct problem
        let problem = libsvm_sys::svm_problem {
            l: x_nodes.end_indexes.len() as c_int,
            x: x_ptr,
            y: y_ptr,
        };

        // compute gamma if necessary
        let gamma = gamma_opt.unwrap_or_else(|| (x_nodes.n_features as f64).recip());

        // update raw params
        // we store pointers late to avoid move after saving pointers
        params.gamma = gamma;
        params.weight_label = weight_labels.as_ptr() as *mut _;
        params.weight = weights.as_ptr() as *mut _;

        // run cross validation
        let target = unsafe {
            let mut target = std::iter::repeat(0.0).take(n_records).collect::<Vec<_>>();
            libsvm_sys::svm_cross_validation(
                &problem,
                &params,
                n_folds.get() as c_int,
                target.as_mut_ptr(),
            );
            target
        };

        Ok(target)
    }
}

impl Svm<Trained> {
    /// Gets the type of the model.
    pub fn kind(&self) -> SvmKind {
        num::FromPrimitive::from_usize(unsafe {
            libsvm_sys::svm_get_svm_type(self.state.model_ptr.as_ptr()) as usize
        })
        .unwrap()
    }

    /// Gets the number of output classes.
    pub fn nr_classes(&self) -> usize {
        unsafe { libsvm_sys::svm_get_nr_class(self.state.model_ptr.as_ptr()) as usize }
    }

    /// Gets the label indexes.
    pub fn labels(&self) -> Vec<isize> {
        unsafe {
            let mut labels = std::iter::repeat(0)
                .take(self.nr_classes())
                .collect::<Vec<_>>();
            libsvm_sys::svm_get_labels(self.state.model_ptr.as_ptr(), labels.as_mut_ptr());
            labels
                .into_iter()
                .map(|label| label as isize)
                .collect::<Vec<_>>()
        }
    }

    /// Gets the label indexes.
    pub fn support_vectors(&self) -> Vec<Vec<(usize, f64)>> {
        let n_sv = self.nr_sv();
        let node_heads = unsafe {
            std::slice::from_raw_parts::<*mut libsvm_sys::svm_node>(
                self.state.model_ptr.as_ref().SV,
                n_sv,
            )
        };
        node_heads
            .iter()
            .cloned()
            .map(|mut node_ptr| {
                std::iter::from_fn(|| unsafe {
                    let libsvm_sys::svm_node { index, value } = *node_ptr;
                    node_ptr = node_ptr.add(1);
                    if index != -1 {
                        Some((index as usize, value))
                    } else {
                        None
                    }
                })
                .fuse()
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Gets the number of support vectors.
    pub fn nr_sv(&self) -> usize {
        unsafe { libsvm_sys::svm_get_nr_sv(self.state.model_ptr.as_ptr()) as usize }
    }

    /// Gets coefficients for SVs in decision functions
    pub fn sv_coef(&self) -> Vec<&[f64]> {
        let n_classes = self.nr_classes();
        let n_sv = self.nr_sv();
        unsafe {
            std::slice::from_raw_parts::<*mut f64>(
                self.state.model_ptr.as_ref().sv_coef,
                n_classes - 1,
            )
            .iter()
            .cloned()
            .map(|ptr| std::slice::from_raw_parts::<f64>(ptr, n_sv))
            .collect::<Vec<_>>()
        }
    }

    /// Gets constants in decision functions
    pub fn rho(&self) -> &[f64] {
        let n_classes = self.nr_classes();
        let n_rhos = n_classes * (n_classes - 1) / 2;
        unsafe { std::slice::from_raw_parts(self.state.model_ptr.as_ref().rho, n_rhos) }
    }

    /// Gets the support vector indexes in training data.
    pub fn get_sv_indexes(&self) -> Vec<usize> {
        let n_sv = self.nr_sv();
        let indexes = unsafe {
            let mut indexes = vec![0; n_sv];
            libsvm_sys::svm_get_sv_indices(self.state.model_ptr.as_ptr(), indexes.as_mut_ptr());
            indexes
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>()
        };
        indexes
    }

    /// Predicts the output for given data.
    pub fn predict<X>(&self, x: X) -> Result<Vec<f64>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
    {
        let x_nodes = x.try_into()?;
        let n_features = self.nr_classes();
        if x_nodes.n_features > n_features {
            return Err(Error::InvalidData {
                reason: format!(
                    "too many features in input data, expect {} features but get {}",
                    n_features, x_nodes.n_features
                ),
            });
        }

        let predictions = {
            x_nodes
                .end_indexes
                .iter()
                .cloned()
                .scan(0, |from, to| {
                    let prev_from = *from;
                    *from = to;
                    Some((prev_from, to))
                })
                .map(|(from, to)| {
                    x_nodes.nodes.get(from..to).unwrap().as_ptr() as *mut libsvm_sys::svm_node
                })
                .map(|node_ptr| unsafe {
                    libsvm_sys::svm_predict(self.state.model_ptr.as_ptr(), node_ptr)
                })
                .collect::<Vec<_>>()
        };

        Ok(predictions)
    }

    /// Predicts the output with decision values for given data.
    pub fn predict_with_values<X>(&self, x: X) -> Result<Vec<(f64, Vec<f64>)>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
    {
        let x_nodes = x.try_into()?;
        let n_features = self.nr_classes();
        if x_nodes.n_features > n_features {
            return Err(Error::InvalidData {
                reason: format!(
                    "too many features in input data, expect {} features but get {}",
                    n_features, x_nodes.n_features
                ),
            });
        }

        let predictions = {
            x_nodes
                .end_indexes
                .iter()
                .cloned()
                .scan(0, |from, to| {
                    let prev_from = *from;
                    *from = to;
                    Some((prev_from, to))
                })
                .map(|(from, to)| {
                    x_nodes.nodes.get(from..to).unwrap().as_ptr() as *mut libsvm_sys::svm_node
                })
                .map(|node_ptr| unsafe {
                    let n_classes = self.nr_classes();
                    let n_dec_values = n_classes * (n_classes - 1) / 2;
                    let mut dec_values = vec![0f64; n_dec_values];
                    let pred = libsvm_sys::svm_predict_values(
                        self.state.model_ptr.as_ptr(),
                        node_ptr,
                        dec_values.as_mut_ptr(),
                    );
                    (pred, dec_values)
                })
                .collect::<Vec<_>>()
        };

        Ok(predictions)
    }

    /// Predicts the output with decision values for given data.
    pub fn predict_with_probability<X>(&self, x: X) -> Result<Vec<(f64, Vec<f64>)>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
    {
        let x_nodes = x.try_into()?;
        let n_features = self.nr_classes();
        if x_nodes.n_features > n_features {
            return Err(Error::InvalidData {
                reason: format!(
                    "too many features in input data, expect {} features but get {}",
                    n_features, x_nodes.n_features
                ),
            });
        }

        let predictions = {
            x_nodes
                .end_indexes
                .iter()
                .cloned()
                .scan(0, |from, to| {
                    let prev_from = *from;
                    *from = to;
                    Some((prev_from, to))
                })
                .map(|(from, to)| {
                    x_nodes.nodes.get(from..to).unwrap().as_ptr() as *mut libsvm_sys::svm_node
                })
                .map(|node_ptr| unsafe {
                    let n_classes = self.nr_classes();
                    let mut probability_estimates = vec![0f64; n_classes];
                    let pred = libsvm_sys::svm_predict_values(
                        self.state.model_ptr.as_ptr(),
                        node_ptr,
                        probability_estimates.as_mut_ptr(),
                    );
                    (pred, probability_estimates)
                })
                .collect::<Vec<_>>()
        };

        Ok(predictions)
    }
}

impl FromStr for Svm<Trained> {
    type Err = Error;

    fn from_str(desc: &str) -> Result<Self, Self::Err> {
        let model_ptr = unsafe {
            let cstring = CString::new(desc.bytes().chain(vec![0]).collect::<Vec<_>>()).unwrap();
            let raw = cstring.into_raw();
            let model_ptr = libsvm_sys::svm_load_model(raw);
            CString::from_raw(raw);
            NonNull::new(model_ptr).ok_or_else(|| Error::InternalError {
                reason: "svm_load_model() returns null pointer".into(),
            })?
        };

        Ok(Svm {
            state: Trained {
                model_ptr,
                nodes_opt: None,
            },
        })
    }
}

/// The model type.
#[derive(Clone, Copy, Debug, FromPrimitive)]
pub enum SvmKind {
    CSvc = libsvm_sys::C_SVC as isize,
    NuSvc = libsvm_sys::NU_SVC as isize,
    NuSvr = libsvm_sys::NU_SVR as isize,
    OneClass = libsvm_sys::ONE_CLASS as isize,
    EpsilonSvr = libsvm_sys::EPSILON_SVR as isize,
}

#[derive(Debug, Clone)]
pub(crate) struct SvmParams {
    pub model: ModelParams,
    pub kernel: KernelParams,
    pub cache_size: usize,
    pub probability_estimates: bool,
    pub shrinking: bool,
    pub termination_eps: f64,
    pub label_weights: HashMap<isize, f64>,
}

impl TryFrom<&SvmInit> for SvmParams {
    type Error = Error;

    fn try_from(init: &SvmInit) -> Result<Self, Self::Error> {
        let SvmInit {
            model,
            kernel,
            cache_size,
            probability_estimates,
            shrinking,
            termination_eps,
            label_weights,
        } = init;

        let svm = Self {
            model: model.as_ref().map(From::from).unwrap_or(Default::default()),
            kernel: kernel
                .as_ref()
                .map(From::from)
                .unwrap_or(Default::default()),
            cache_size: cache_size.unwrap_or(DEFAULT_CACHE_SIZE),
            probability_estimates: probability_estimates.unwrap_or(DEFAULT_PROBABILITY_ESTIMATES),
            shrinking: shrinking.unwrap_or(DEFAULT_SHRINKING),
            termination_eps: termination_eps.unwrap_or(DEFAULT_TERMINATION_EPSILON),
            label_weights: label_weights.clone().unwrap_or(HashMap::new()),
        };

        if svm.cache_size <= 0 {
            return Err(Error::InvalidHyperparameter {
                reason: "cache_size must be positive".into(),
            });
        }

        if svm.termination_eps <= 0.0 {
            return Err(Error::InvalidHyperparameter {
                reason: "termination_eps must be positive".into(),
            });
        }

        Ok(svm)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ModelParams {
    CSvc { cost: f64 },
    NuSvc { nu: f64 },
    NuSvr { nu: f64 },
    OneClass { nu: f64 },
    EpsilonSvr { epsilon: f64 },
}

impl Default for ModelParams {
    fn default() -> Self {
        ModelParams::CSvc { cost: DEFAULT_COST }
    }
}

impl From<&ModelInit> for ModelParams {
    fn from(init: &ModelInit) -> Self {
        match init {
            ModelInit::CSvc { cost } => Self::CSvc {
                cost: cost.unwrap_or(DEFAULT_COST),
            },
            ModelInit::NuSvc { nu } => ModelParams::NuSvc {
                nu: nu.unwrap_or(DEFAULT_NU),
            },
            ModelInit::NuSvr { nu } => ModelParams::NuSvr {
                nu: nu.unwrap_or(DEFAULT_NU),
            },
            ModelInit::OneClass { nu } => ModelParams::OneClass {
                nu: nu.unwrap_or(DEFAULT_NU),
            },
            ModelInit::EpsilonSvr { epsilon } => ModelParams::EpsilonSvr {
                epsilon: epsilon.unwrap_or(DEFAULT_MODEL_EPSILON),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum KernelParams {
    Linear,
    Polynomial {
        gamma: Option<f64>,
        coef0: f64,
        degree: usize,
    },
    Rbf {
        gamma: Option<f64>,
    },
    Sigmoid {
        gamma: Option<f64>,
        coef0: f64,
    },
    // Precomputed,
}

impl Default for KernelParams {
    fn default() -> Self {
        KernelParams::Rbf { gamma: None }
    }
}

impl From<&KernelInit> for KernelParams {
    fn from(init: &KernelInit) -> Self {
        match *init {
            KernelInit::Linear => KernelParams::Linear,
            KernelInit::Polynomial {
                gamma,
                coef0,
                degree,
            } => KernelParams::Polynomial {
                gamma,
                coef0: coef0.unwrap_or(DEFAULT_COEF0),
                degree: degree.unwrap_or(DEFAULT_DEGREE),
            },
            KernelInit::Rbf { gamma } => KernelParams::Rbf { gamma },
            KernelInit::Sigmoid { gamma, coef0 } => KernelParams::Sigmoid {
                gamma,
                coef0: coef0.unwrap_or(DEFAULT_COEF0),
            },
        }
    }
}
