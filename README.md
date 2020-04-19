# libsvm Bindings for Rust

The crate provides high level API for [cjlin1's libsvm](https://github.com/cjlin1/libsvm),
which is based on [Yu Wei Wu's libsvm-sys](https://crates.io/crates/libsvm-sys).

## Usage

Append this line to `[dependencies]` in `Cargo.toml`.

```
libsvm = "^0.1.0"
```

The following extra features add support for nalgebra and ndarray types.
The methods in this crate will accept matrix or multi-dimensional array types,
and convert them internally.

 *with-nalgebra*
 *with-ndarray*

For example, if you would like to enable nalgebra support, add the line in your
`Cargo.toml`.

```
libsvm = { version = "^0.1.0", features = ["with-nalgebra"] }
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
