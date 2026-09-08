# UpdatableFactorizations.jl

[![CI](https://github.com/el-oso/UpdatableFactorizations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/UpdatableFactorizations.jl/actions/workflows/CI.yml)
[![Coverage Status](https://coveralls.io/repos/github/el-oso/UpdatableFactorizations.jl/badge.svg?branch=master)](https://coveralls.io/github/el-oso/UpdatableFactorizations.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Maintains Cholesky, LU, and QR factorizations under rank-1 update and
downdate, and under row/column insertion and deletion, without recomputing
the factorization from scratch.

The package extends `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for its own factorization
types, rather than replacing the existing `Cholesky` methods those functions
already provide.

Every routine's provenance — the paper it derives from, and the reference
implementations consulted during development — is recorded on the
[provenance page](https://el-oso.github.io/UpdatableFactorizations.jl/dev/provenance).

## Installation

```julia
using Pkg
Pkg.add("UpdatableFactorizations")
```

## License

MIT. See [LICENSE](LICENSE).
