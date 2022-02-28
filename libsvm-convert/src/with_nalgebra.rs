use crate::r#trait::{SvmTryFrom, SvmTryInto};
use libsvm::{Error, SvmNodes};
use nalgebra::{storage::Storage, Dim, Matrix};
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};
use std::os::raw::c_int;

impl<R, C, S> SvmTryFrom<&Matrix<f64, R, C, S>> for SvmNodes
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    type Error = Error;

    fn svm_try_from(from: &Matrix<f64, R, C, S>) -> Result<Self, Self::Error> {
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

impl<R, C, S> SvmTryFrom<Matrix<f64, R, C, S>> for SvmNodes
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    type Error = Error;

    fn svm_try_from(from: Matrix<f64, R, C, S>) -> Result<Self, Self::Error> {
        Self::svm_try_from(&from)
    }
}
