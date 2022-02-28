use anyhow::{anyhow, bail, Result};
use cfg_if::cfg_if;
use once_cell::sync::OnceCell;
use std::{
    env,
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    #[cfg(feature = "codegen")]
    codegen()?;

    #[cfg(feature = "link")]
    link()?;

    Ok(())
}

#[cfg(feature = "codegen")]
fn codegen() -> Result<()> {
    let include_dir = libsvm_include()?;
    let header_file = include_dir.join("libsvm").join("svm.h");
    let out_file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings.rs");

    let builder = bindgen::Builder::default()
        .header(format!("{}", header_file.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks));
    let bindings = builder
        .generate()
        .map_err(|_| anyhow!("Unable to generate bindings"))?;

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_file)
        .map_err(|_| anyhow!("unable to write bindings"))?;

    Ok(())
}

#[cfg(feature = "link")]
fn link() -> Result<()> {
    let library_dir = libsvm_library()?;
    println!("cargo:rustc-link-search={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=svm");
    Ok(())
}

fn libsvm_include() -> Result<&'static Path> {
    static LIBSVM_INCLUDE: OnceCell<PathBuf> = OnceCell::new();

    LIBSVM_INCLUDE.get_or_try_init(|| {
        if let Some(path) = get_env_path("LIBSVM_INCLUDE") {
            return Ok(path);
        }

        let path = {
            cfg_if! {
                if #[cfg(target_os = "linux")] {
                    libsvm_path()?.join("include")
                } else {
                    bail!("Unable to find libsvm library. Try to set LIBSVM_PATH environment variable.");
                }
            }
        };

        Ok(path)
    })
    .map(|path| path.as_ref())
}

fn libsvm_library() -> Result<&'static Path> {
    static LIBSVM_LIBRARY: OnceCell<PathBuf> = OnceCell::new();

    LIBSVM_LIBRARY.get_or_try_init(|| {
        if let Some(path) = get_env_path("LIBSVM_LIBRARY") {
            return Ok(path);
        }

        let path = {
            cfg_if! {
                if #[cfg(target_os = "linux")] {
                    libsvm_path()?.join("lib")
                } else {
                    bail!("Unable to find libsvm library. Try to set LIBSVM_PATH environment variable.");
                }
            }
        };

        Ok(path)
    })
    .map(|path| path.as_ref())
}

fn libsvm_path() -> Result<&'static Path> {
    static LIBSVM_PATH: OnceCell<PathBuf> = OnceCell::new();

    LIBSVM_PATH.get_or_try_init(|| {
        if let Some(path) = get_env_path("LIBSVM_PATH") {
            return Ok(path);
        }

        let path = {
            cfg_if! {
                if #[cfg(target_os = "linux")] {
                    PathBuf::from("/usr")
                } else {
                    bail!("Unable to find libsvm library. Try to set LIBSVM_PATH environment variable.");
                }
            }
        };

        Ok(path)
    }).map(|path| path.as_ref())
}

fn get_env_path(name: &str) -> Option<PathBuf> {
    println!("rerun-if-env-changed={}", name);
    env::var_os(name).map(PathBuf::from)
}
