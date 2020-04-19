//! The hyperparameter initializers.

use crate::{
    consts::{DEFAULT_COEF0, DEFAULT_COST, DEFAULT_DEGREE, DEFAULT_MODEL_EPSILON, DEFAULT_NU},
    error::Error,
    model::{KernelParams, ModelParams, Svm, SvmParams},
    state::Untrained,
};
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
};

/// The model type initializer.
#[derive(Clone, Debug)]
pub enum ModelInit {
    CSvc { cost: Option<f64> },
    NuSvc { nu: Option<f64> },
    NuSvr { nu: Option<f64> },
    OneClass { nu: Option<f64> },
    EpsilonSvr { epsilon: Option<f64> },
}

/// The kernel initializer.
#[derive(Clone, Debug)]
pub enum KernelInit {
    Linear,
    Polynomial {
        gamma: Option<f64>,
        coef0: Option<f64>,
        degree: Option<usize>,
    },
    Rbf {
        gamma: Option<f64>,
    },
    Sigmoid {
        gamma: Option<f64>,
        coef0: Option<f64>,
    },
    // Precomputed,
}

/// The SVM model initializer.
#[derive(Debug, Clone)]
pub struct SvmInit {
    pub model: Option<ModelInit>,
    pub kernel: Option<KernelInit>,
    pub cache_size: Option<usize>,
    pub probability_estimates: Option<bool>,
    pub shrinking: Option<bool>,
    pub termination_eps: Option<f64>,
    pub label_weights: Option<HashMap<usize, f64>>,
}

impl Default for SvmInit {
    fn default() -> Self {
        Self {
            model: None,
            kernel: None,
            cache_size: None,
            probability_estimates: None,
            shrinking: None,
            termination_eps: None,
            label_weights: None,
        }
    }
}

impl SvmInit {
    /// Builds SVM model from the initializer.
    pub fn build(&self) -> Result<Svm<Untrained>, Error> {
        let SvmParams {
            model,
            kernel,
            cache_size,
            probability_estimates,
            shrinking,
            termination_eps,
            label_weights,
        } = TryFrom::try_from(self)?;

        let (svm_type, cost, nu, model_epsilon) = match model {
            ModelParams::CSvc { cost } => {
                (libsvm_sys::C_SVC, cost, DEFAULT_NU, DEFAULT_MODEL_EPSILON)
            }
            ModelParams::NuSvc { nu } => {
                (libsvm_sys::NU_SVC, DEFAULT_COST, nu, DEFAULT_MODEL_EPSILON)
            }
            ModelParams::OneClass { nu } => (
                libsvm_sys::ONE_CLASS,
                DEFAULT_COST,
                nu,
                DEFAULT_MODEL_EPSILON,
            ),
            ModelParams::EpsilonSvr { epsilon } => {
                (libsvm_sys::EPSILON_SVR, DEFAULT_COST, DEFAULT_NU, epsilon)
            }
            ModelParams::NuSvr { nu } => {
                (libsvm_sys::NU_SVR, DEFAULT_COST, nu, DEFAULT_MODEL_EPSILON)
            }
        };

        let (kernel_type, gamma_opt, coef0, degree) = match kernel {
            KernelParams::Linear => (libsvm_sys::LINEAR, None, DEFAULT_COEF0, DEFAULT_DEGREE),
            KernelParams::Polynomial {
                gamma,
                coef0,
                degree,
            } => (libsvm_sys::POLY, gamma, coef0, degree),
            KernelParams::Rbf { gamma } => (libsvm_sys::RBF, gamma, DEFAULT_COEF0, DEFAULT_DEGREE),
            KernelParams::Sigmoid { gamma, coef0 } => {
                (libsvm_sys::SIGMOID, gamma, coef0, DEFAULT_DEGREE)
            }
        };

        let (nr_weight, weight_labels, weights) = {
            let nr_weight: i32 =
                label_weights
                    .len()
                    .try_into()
                    .map_err(|err| Error::InvalidHyperparameter {
                        reason: format!("invalid number of weights: {}", err),
                    })?;

            let (labels, weights) = label_weights.into_iter().fold(Ok((vec![], vec![])), |result, (index, weight)| {
                let (mut labels, mut weights) = result?;

                if weight < 0.0 || weight > 1.0 {
                    return Err(Error::InvalidHyperparameter {
                        reason: format!("the label weights in label_weights must be in range of [0, 1], but found {}", weight)
                    });
                }

                labels.push(index as i32);
                weights.push(weight);

                Ok((labels, weights))
            })?;

            (nr_weight, labels, weights)
        };

        let params = libsvm_sys::svm_parameter {
            svm_type: svm_type as i32,
            kernel_type: kernel_type as i32,
            degree: degree
                .try_into()
                .map_err(|err| Error::InvalidHyperparameter {
                    reason: format!("invalid degree parameter: {}", err),
                })?,
            gamma: gamma_opt.unwrap_or(0.0),
            coef0,
            cache_size: cache_size as f64,
            eps: termination_eps,
            C: cost,
            nu,
            p: model_epsilon,
            shrinking: shrinking as i32,
            probability: probability_estimates as i32,
            nr_weight,
            weight_label: std::ptr::null_mut(),
            weight: std::ptr::null_mut(),
        };

        let svm = Svm {
            state: Untrained {
                gamma_opt,
                params,
                weight_labels,
                weights,
            },
        };

        Ok(svm)
    }
}
