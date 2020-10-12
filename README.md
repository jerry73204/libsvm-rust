# libsvm Bindings for Rust

The crate provides high level API for [cjlin1's libsvm](https://github.com/cjlin1/libsvm),
which is based on [Yu Wei Wu's libsvm-sys](https://crates.io/crates/libsvm-sys).

## Usage

Add this crate your `Cargo.toml`.

```toml
[dependencies]
libsvm = "0.3"
```

## Cargo Features

- **full**: Enable most available cargo features except **nightly**.
- **nalgebra**: Enable conversions from nalgebra types
- **ndarray**: Enable conversions from ndarray types
- **nightly**: Enable nightly features, especially for conversions from array of arbitrary size.

For example, if you would like to enable nalgebra support, add the feature in your `Cargo.toml`.

```toml
libsvm = { version = "0.3", features = ["with-nalgebra"] }
```

## License

The crate is licensed under BSD-3-clause. You can see the [license file](LICENSE)
included in the repository.

The crate links the work from Chih-Chung Chang and Chih-Jen Lin. The upstream libsvm
license can be found in `licenses` directory. Here is the citation text to respect
their contribution.

```
Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
vector machines. ACM Transactions on Intelligent Systems and
Technology, 2:27:1--27:27, 2011. Software available at
http://www.csie.ntu.edu.tw/~cjlin/libsvm
```
