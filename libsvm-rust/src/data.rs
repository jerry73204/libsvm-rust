//! Data types and type conversion implementations.

use crate::error::Error;
use std::{
    convert::{TryFrom, TryInto},
    os::raw::c_int,
};

/// Sparse storage of two dimensional array
#[derive(Debug, Clone)]
pub struct SvmNodes {
    pub(crate) n_features: usize,
    pub(crate) nodes: Vec<libsvm_sys::svm_node>,
    pub(crate) end_indexes: Vec<usize>,
}

impl TryFrom<&[f64]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[f64]) -> Result<Self, Self::Error> {
        let n_features = from.len();
        let nodes = {
            let mut nodes = from
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, value)| libsvm_sys::svm_node {
                    index: index as c_int,
                    value,
                })
                .collect::<Vec<_>>();
            nodes.push(libsvm_sys::svm_node {
                index: -1,
                value: 0.0,
            });
            nodes
        };
        let end_indexes = vec![nodes.len()];

        Ok(SvmNodes {
            n_features,
            nodes,
            end_indexes,
        })
    }
}

impl TryFrom<&[&[f64]]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[&[f64]]) -> Result<Self, Self::Error> {
        let n_features = from
            .get(0)
            .ok_or_else(|| Error::InvalidData {
                reason: "data without features is not allowed".into(),
            })?
            .len();

        let (nodes, end_indexes) = from.iter().fold(Ok((vec![], vec![])), |result, row| {
            let (mut nodes, mut end_indexes) = result?;

            if row.len() != n_features {
                return Err(Error::InvalidData {
                    reason: "the number of features must be consistent".into(),
                });
            }

            nodes.extend(row.iter().cloned().enumerate().map(|(index, value)| {
                libsvm_sys::svm_node {
                    index: index as c_int,
                    value,
                }
            }));
            nodes.push(libsvm_sys::svm_node {
                index: -1,
                value: 0.0,
            });
            end_indexes.push(nodes.len());

            Ok((nodes, end_indexes))
        })?;

        Ok(SvmNodes {
            n_features,
            nodes,
            end_indexes,
        })
    }
}

impl TryFrom<&[Vec<f64>]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[Vec<f64>]) -> Result<Self, Self::Error> {
        from.iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
    }
}

impl TryFrom<&[&Vec<f64>]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[&Vec<f64>]) -> Result<Self, Self::Error> {
        from.iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
    }
}

impl<const N_FEATURES: usize> TryFrom<&[&[f64; N_FEATURES]]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[&[f64; N_FEATURES]]) -> Result<Self, Self::Error> {
        from.iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
    }
}

impl<const N_FEATURES: usize> TryFrom<&[[f64; N_FEATURES]]> for SvmNodes {
    type Error = Error;

    fn try_from(from: &[[f64; N_FEATURES]]) -> Result<Self, Self::Error> {
        from.iter()
            .map(|row| row.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_svm_nodes() -> Result<(), Error> {
        {
            let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
            SvmNodes::try_from(data.as_slice())?;
        }

        {
            let data = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
            SvmNodes::try_from(data.as_slice())?;
        }

        Ok(())
    }
}
