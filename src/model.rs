use crate::{
    consts::{
        DEFAULT_CACHE_SIZE, DEFAULT_COEF0, DEFAULT_COST, DEFAULT_DEGREE, DEFAULT_MODEL_EPSILON,
        DEFAULT_NU, DEFAULT_PROBABILITY_ESTIMATES, DEFAULT_SHRINKING, DEFAULT_TERMINATION_EPSILON,
    },
    data::SvmNodes,
    error::Error,
    init::{KernelInit, ModelInit, SvmInit},
    state::{SvmState, Trained, Untrained},
};
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    ffi::CString,
    os::raw::c_int,
    ptr::NonNull,
    str::FromStr,
};

#[derive(Debug, Clone)]
pub struct Svm<State> {
    pub(crate) state: State,
}

impl Svm<Untrained> {
    pub fn fit<X, Y>(self, x: X, y: Y) -> Result<Svm<Trained>, Error>
    where
        X: TryInto<SvmNodes, Error = Error>,
        Y: AsRef<[f64]>,
    {
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

        // n_features
        let n_features: c_int = x_nodes
            .n_features
            .try_into()
            .map_err(|_| Error::InvalidData {
                reason: format!("the number of features is too large"),
            })?;

        // costruct problem
        let problem = libsvm_sys::svm_problem {
            l: n_features,
            x: x_ptr,
            y: y_ptr,
        };

        // compute gamma if necessary
        let gamma = gamma_opt.unwrap_or_else(|| (n_features as f64).recip());

        // update raw params
        // we store pointers late to avoid move after saving pointers
        params.gamma = gamma;
        params.weight_label = weight_labels.as_ptr() as *mut _;
        params.weight = weights.as_ptr() as *mut _;

        // train model
        let model_ptr = unsafe {
            NonNull::new(libsvm_sys::svm_train(&problem, &params)).ok_or_else(|| {
                Error::InternalError {
                    reason: "svm_train() returns null pointer".into(),
                }
            })?
        };

        Ok(Svm {
            state: Trained { model_ptr },
        })
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
            state: Trained { model_ptr },
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SvmParams {
    pub model: ModelParams,
    pub kernel: KernelParams,
    pub cache_size: usize,
    pub probability_estimates: bool,
    pub shrinking: bool,
    pub termination_eps: f64,
    pub label_weights: HashMap<usize, f64>,
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
