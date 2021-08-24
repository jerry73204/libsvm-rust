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
                reason: format!("data without features is not allowed"),
            })?
            .len();

        let (nodes, end_indexes) = from.iter().fold(Ok((vec![], vec![])), |result, row| {
            let (mut nodes, mut end_indexes) = result?;

            if row.len() != n_features {
                return Err(Error::InvalidData {
                    reason: format!("the number of features must be consistent"),
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

impl TryFrom<&str> for SvmNodes {
    type Error = Error;
    /// the param format is  `index1:value1 index2:value2 ...`
    /// or
    /// ```ignore
    /// index1:value1 index2:value2 ... \n
    /// index1:value1 index2:value2 ... \n
    /// ...
    /// ```
    /// and the index starts 0
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut max_features = 0;
        let mut nodes_final = vec![];
        let mut end_indexes_final = vec![];
        let lines = value.split("\n").collect::<Vec<&str>>();
        for line in lines {
            // the line format is `index1:value1 index2:value2 ...`
            let line_split: Vec<&str> = line.split(" ").collect();
            let mut nodes_tmp = vec![];
            for index_value in line_split {
                let index_value: Vec<&str> = index_value.split(":").collect();
                if index_value.len() != 2 {
                    return Err(Error::InvalidLine {
                        reason: format!(
                            "expected format `index1:value1 index2:value2 ...`,\
                but this line is {}",
                            value
                        ),
                    });
                }
                let index = index_value[0].parse::<usize>();
                if index.is_err() {
                    return Err(Error::InvalidLine {
                        reason: format!("index must be number and >= 0"),
                    });
                }
                let value = index_value[1].parse::<f64>();
                if value.is_err() {
                    return Err(Error::InvalidLine {
                        reason: format!("value must be number"),
                    });
                }
                nodes_tmp.push(libsvm_sys::svm_node {
                    index: index.unwrap() as c_int,
                    value: value.unwrap(),
                });
            }
            // index start 0, so the num of feature need +1
            let n_features = nodes_tmp.last().unwrap().index as usize + 1;
            nodes_tmp.push(libsvm_sys::svm_node {
                index: -1,
                value: 0.0,
            });

            max_features = std::cmp::max(max_features, n_features);
            //current end index = last nodes length + current nodes length
            end_indexes_final.push(nodes_final.len() + nodes_tmp.len());
            nodes_final.extend(nodes_tmp);
        }
        Ok(SvmNodes {
            n_features: max_features,
            nodes: nodes_final,
            end_indexes: end_indexes_final,
        })
    }
}

#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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

#[cfg(feature = "ndarray")]
mod try_from_ndarray {
    use super::*;
    use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};

    impl<S> TryFrom<&ArrayBase<S, Ix2>> for SvmNodes
    where
        S: Data<Elem = f64>,
    {
        type Error = Error;

        fn try_from(from: &ArrayBase<S, Ix2>) -> Result<Self, Self::Error> {
            let n_features = from.ncols();
            let (nodes, end_indexes) = from.axis_iter(Axis(0)).fold(
                (vec![], vec![]),
                |(mut nodes, mut end_indexes), row| {
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

                    (nodes, end_indexes)
                },
            );

            Ok(SvmNodes {
                n_features,
                nodes,
                end_indexes,
            })
        }
    }

    impl<S> TryFrom<ArrayBase<S, Ix2>> for SvmNodes
    where
        S: Data<Elem = f64>,
    {
        type Error = Error;

        fn try_from(from: ArrayBase<S, Ix2>) -> Result<Self, Self::Error> {
            (&from).try_into()
        }
    }

    impl<S> TryFrom<&ArrayBase<S, Ix1>> for SvmNodes
    where
        S: Data<Elem = f64>,
    {
        type Error = Error;

        fn try_from(from: &ArrayBase<S, Ix1>) -> Result<Self, Self::Error> {
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

    impl<S> TryFrom<ArrayBase<S, Ix1>> for SvmNodes
    where
        S: Data<Elem = f64>,
    {
        type Error = Error;

        fn try_from(from: ArrayBase<S, Ix1>) -> Result<Self, Self::Error> {
            (&from).try_into()
        }
    }
}

#[cfg(feature = "nalgebra")]
mod try_from_nalgebra {
    use super::*;
    use nalgebra::{storage::Storage, Dim, Matrix};

    impl<R, C, S> TryFrom<&Matrix<f64, R, C, S>> for SvmNodes
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        type Error = Error;

        fn try_from(from: &Matrix<f64, R, C, S>) -> Result<Self, Self::Error> {
            let n_features = from.nrows();

            let (nodes, end_indexes) =
                from.row_iter()
                    .fold((vec![], vec![]), |(mut nodes, mut end_indexes), row| {
                        nodes.extend(
                            row.iter()
                                .cloned()
                                .enumerate()
                                .map(|(index, value)| libsvm_sys::svm_node {
                                    index: index as c_int,
                                    value,
                                })
                                .chain(std::iter::once(libsvm_sys::svm_node {
                                    index: -1,
                                    value: 0.0,
                                })),
                        );
                        end_indexes.push(nodes.len());

                        (nodes, end_indexes)
                    });

            Ok(SvmNodes {
                n_features,
                nodes,
                end_indexes,
            })
        }
    }

    impl<R, C, S> TryFrom<Matrix<f64, R, C, S>> for SvmNodes
    where
        R: Dim,
        C: Dim,
        S: Storage<f64, R, C>,
    {
        type Error = Error;

        fn try_from(from: Matrix<f64, R, C, S>) -> Result<Self, Self::Error> {
            (&from).try_into()
        }
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

        #[cfg(feature = "nightly")]
        {
            let data = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
            SvmNodes::try_from(data.as_slice())?;
        }

        Ok(())
    }

    #[test]
    fn str_to_svm_nodes() -> Result<(), Error> {
        let data = format!("{}\n{}\n{}", "0:1 1:0", "0:0 1:1", "0:0 1:0");
        let nodes = SvmNodes::try_from(data.as_str())?;
        assert_eq!(format!("{:?}", nodes),
                   "SvmNodes { n_features: 2, nodes: [svm_node { index: 0, value: 1.0 }, svm_node { index: 1, value: 0.0 }, svm_node { index: -1, value: 0.0 }, svm_node { index: 0, value: 0.0 }, svm_node { index: 1, value: 1.0 }, svm_node { index: -1, value: 0.0 }, svm_node { index: 0, value: 0.0 }, svm_node { index: 1, value: 0.0 }, svm_node { index: -1, value: 0.0 }], end_indexes: [3, 6, 9] }"
        );
        Ok(())
    }

    #[test]
    fn str_to_svm_nodes_error() -> Result<(), Error> {
        let data = format!("{}", "-0 0:1");
        let nodes = SvmNodes::try_from(data.as_str());
        assert_eq!(format!("{:?}", nodes), "Err(InvalidLine { reason: \"expected format `index1:value1 index2:value2 ...`,but this line is -0 0:1\" })");

        let data = format!("{}", "-1:0 0:1");
        let nodes = SvmNodes::try_from(data.as_str());
        assert_eq!(
            format!("{:?}", nodes),
            "Err(InvalidLine { reason: \"index must be number and >= 0\" })"
        );

        let data = format!("{}", "0:0 a:1");
        let nodes = SvmNodes::try_from(data.as_str());
        assert_eq!(
            format!("{:?}", nodes),
            "Err(InvalidLine { reason: \"index must be number and >= 0\" })"
        );

        let data = format!("{}", "0:he 0:1");
        let nodes = SvmNodes::try_from(data.as_str());
        assert_eq!(
            format!("{:?}", nodes),
            "Err(InvalidLine { reason: \"value must be number\" })"
        );
        Ok(())
    }
}
