# pyoptsparse-ml

`pyoptsparse_ml` provides bridges between the MDO Lab aerodynamic optimization stack [pyOptSparse](https://github.com/mdolab/pyoptsparse), and pretrained ML surrogates (e.g. [AeroTransformer](https://github.com/yangyunjia/floGen)). It provide multiple strategies that evaluates aerodynamic functions and sensitivities from a pretrained model, plus a multi-fidelity wrapper (`SolverCombined`) that periodically verifies or corrects those predictions with a high-fidelity CFD solver such as ADflow.

Author: Yunjia Yang, (TUM)   contact: [yunjia.yang@tum.de](mailto:yunjia.yang@tum.de)

## Features

- **Custom drop-in ML aero solver** — Just implement subclass `AeroModel` and plug in any PyTorch (or ensemble) predictor. Our code will implements the ADflow-like `evalFunctions` / `evalFunctionsSens` interface used by pyOptSparse optimization scripts.
- **Multi-fidelity ML + CFD** — `SolverCombined` schedules CFD sampling and merges results with ML through verification, replacement, linear correction, or online Gradient-Enhanced Kriging (GEK).


## Dependencies

| Role            | Package                                                          |
| --------------- | ------------------------------------------------------------- |
| Basics          | `numpy`, `torch`                |
| MDOlab basics   | `baseclasses`, `pygeo` (for CST methods, you need the forked version at [pygeo](https://github.com/YangYunjia/pygeo)), `pyoptsparse`, `multipoint`|
| MDOlab solver   | `mpi4py`, `adflow`, `idwarp` (Parallel / CFD verification, optional) |
| Surrogate model | Default is `flowGen` (but you could hook your own APIs)        |




## Package layout

```
pyoptsparse_ml/
├── mlsolver.py      # AeroModel template + MLSolver (AeroSolver surrogate)
├── combine.py       # SolverCombined: ML + CFD multi-fidelity orchestration
└── sm/
    └── gek.py       # Online GEK correction on CFD−ML deltas
examples/
├── cst/             # CST parameterization + optional ADflow verification
├── ffd/             # FFD deformation + ML prediction
└── src/             # Shared geometry / mesh inputs
docs/
└── GEK_METHODS.md   # GEK correction modes (detailed)
```



## Installation

Requires Python 3 and the usual MDO Lab stack (`baseclasses`, `pygeo`, `pyoptsparse`, optionally `adflow` / `idwarp` / `multipoint` for CFD examples). PyTorch is required for the ML backend.

```bash
git clone <repo-url> pyoptsparse-ml
cd pyoptsparse-ml
pip install -e .
```



## Quick start



### 1. Wrap a pretrained model with `MLSolver`

```python
from baseclasses import AeroProblem
from pyoptsparse_ml.mlsolver import MLSolver
from flowvae.app.wing.api import SuperWingAPI  # example AeroModel

ap = AeroProblem(
    name="wing", alpha=1.5, mach=0.85, reynolds=2e7,
    evalFuncs=["cl", "cd", "cmz"],
    # ... refs, areas, etc.
)

mlSolver = MLSolver(
    output_keys=["cl", "cd", "cmz"],
    condition_keys={
        "alpha": [0, 5],
        "mach": [0.75, 0.90],
        "reynolds": 20000000,
    },
    options={
        "output_dir": "output",
        "sens_mode": "BP",   # or "FD"
        "fd_step": 1e-5,
    },
    device="cuda:0",
    comm=comm,
)
mlSolver.setModel(SuperWingAPI(model_version="finetune20", device="cuda:0"), ap=ap)
mlSolver.setDVGeo(DVGeo)
```

Implement your own model by subclassing `AeroModel` and providing `load_model` / `predict`.

### 2. Multi-fidelity optimization with `SolverCombined`

```python
from pyoptsparse_ml.combine import SolverCombined

combined = SolverCombined(
    opt=optProb,
    ap=ap,
    comm=comm,
    output_dir="output",
    cfd_frequency=10,      # Ni: ML-only stretch
    cfd_iter=1,            # Nj: CFD every Nj steps in the CFD window
    cfd_include_mode=11,   # see modes below
    gek_options={"max_points": 100, "max_dims": 40},
)
# Wire wrap_cruiseFuncs / wrap_cruiseFuncsSens into your multipoint objective
```



#### CFD include modes


| Mode | Behavior                                     |
| ---- | -------------------------------------------- |
| `0`  | CFD verification only (log / history)        |
| `1`  | Pass CFD failure signal to the optimizer     |
| `2`  | Single-point replacement of ML values by CFD |
| `3`  | Linear correction using value + gradient     |
| `4`  | Linear correction without gradient-to-value  |
| `10` | Online GEK on CFD−ML deltas (funcs + sens)   |
| `11` | GEK with stable x/y normalization            |
| `12` | Active-subspace + normalized GEK             |


See [docs/GEK_METHODS.md](docs/GEK_METHODS.md) for the GEK formulation and runtime flow.

## Examples

Both examples mirror a standard MDO Lab aero-opt script: multipoint setup, design variables, ML solver, optional ADflow sampling, and pyOptSparse.


| Example                                                | Description                                 |
| ------------------------------------------------------ | ------------------------------------------- |
| `[examples/cst/aero_opt.py](examples/cst/aero_opt.py)` | CST airfoil/wing DVs via `DVGeometryCustom` |
| `[examples/ffd/aero_opt.py](examples/ffd/aero_opt.py)` | FFD volume deformation via `DVGeometry`     |


Typical flags:

```bash
# Pure ML optimization
python examples/ffd/aero_opt.py -o output --mlModel finetune20

# ML + periodic ADflow with online GEK (mode 11)
python examples/ffd/aero_opt.py -o output -i 10 -j 1 -s 11 --mlModel finetune20
```

Shared inputs live under `examples/src/` (e.g. FFD box, tip geometry, `input.json`). Volume meshes such as `wing_vol.cgns` must be supplied separately for CFD runs.

## License

GNU Lesser General Public License v2.1 — see [LICENSE](LICENSE).

## Authors

Aerolab — 