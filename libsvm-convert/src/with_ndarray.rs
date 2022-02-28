use crate::r#trait::{SvmTryFrom, SvmTryInto};
use libsvm::{Error, SvmNodes};
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};
use std::os::raw::c_int;

impl<S> SvmTryFrom<&ArrayBase<S, Ix2>> for SvmNodes
where
    S: Data<Elem = f64>,
{
    type Error = Error;

    fn svm_try_from(from: &ArrayBase<S, Ix2>) -> Result<Self, Self::Error> {
        let n_features = from.ncols();
        let (nodes, end_indexes) =
            from.axis_iter(Axis(0))
                .fold((vec![], vec![]), |(mut nodes, mut end_indexes), row| {
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
                });

        Ok(SvmNodes {
            n_features,
            nodes,
            end_indexes,
        })
    }
}

impl<S> SvmTryFrom<ArrayBase<S, Ix2>> for SvmNodes
where
    S: Data<Elem = f64>,
{
    type Error = Error;

    fn svm_try_from(from: ArrayBase<S, Ix2>) -> Result<Self, Self::Error> {
        Self::svm_try_from(&from)
    }
}

impl<S> SvmTryFrom<&ArrayBase<S, Ix1>> for SvmNodes
where
    S: Data<Elem = f64>,
{
    type Error = Error;

    fn svm_try_from(from: &ArrayBase<S, Ix1>) -> Result<Self, Self::Error> {
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

impl<S> SvmTryFrom<ArrayBase<S, Ix1>> for SvmNodes
where
    S: Data<Elem = f64>,
{
    type Error = Error;

    fn svm_try_from(from: ArrayBase<S, Ix1>) -> Result<Self, Self::Error> {
        Self::svm_try_from(&from)
    }
}
