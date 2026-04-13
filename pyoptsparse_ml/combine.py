'''
Combine the ML solver and the CFD solver for multi-fidelity optimization.
'''


import os
import time
import json
from typing import Any, Dict, Optional, Callable
from collections import OrderedDict

import numpy as np
from pyoptsparse import OPT

from baseclasses import AeroProblem
from .sm.gek import _OnlineGEKDeltaModel

def _jsonify_value(val):
    if isinstance(val, np.ndarray):
        if val.shape == ():
            return float(val)
        return val.tolist()
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, dict):
        return {k: _jsonify_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonify_value(v) for v in val]
    return val

def _append_cfd_hist(path, row):

    if not os.path.exists(path):
        data = {key: [] for key in row}
    else:
        with open(path, "r", encoding="ascii") as f:
            data = json.load(f)
    for key in row:
        data.setdefault(key, [])
        data[key].append(_jsonify_value(row[key]))
    with open(path, "w", encoding="ascii") as f:
        json.dump(data, f)

class SolverCombined():
    """
    This class is a wrapper to combine the ML solver with a traditional CFD solver (e.g. ADflow) for multi-fidelity optimization. 
    The user can specify which functions to evaluate with the ML solver and which to evaluate with the CFD solver, and this class 
    will route the calls accordingly. This allows leveraging the speed of the ML solver for some functions while retaining the 
    accuracy of the CFD solver for others.
    
    """
    include_modes = {
        0: "just have a verfication",
        1: "CFD fail signal will be passed to optimizor",
        2: "single point replacement",
        3: "linear correction (with gradient to value)",
        4: "linear correction (without gradient to value)",
        10: "online GEK correction (train on CFD-ML delta from funcs + funcSens)",
        11: "online GEK correction with stable normalization (x/y normalization + safe kernel)",
        12: "active-subspace + normalized GEK correction",
    }
    # Default options for GEK modes (10/11/12). Pass `gek_options={...}` to override.
    gek_default_options = {
        "max_points": 100,        # GEK sample window length
        "max_dims": 40,           # Max DV dims passed into GEK before fitting
        "nugget": 1e-8,           # Covariance diagonal regularization
        "log_interval": 1,        # Print GEK prediction logs every N calls
        "xscale_floor": 1e-3,     # Mode11/12 x-normalization minimum scale
        "yscale_floor": 1e-8,     # Mode11/12 y-normalization minimum scale
        "ell_floor": 0.2,         # Mode11/12 minimum GP length scale
        "exp_clip": 60.0,         # Mode11/12 RBF exponent clipping magnitude
        "active_rank": 8,         # Mode12 max active-subspace rank
        "active_energy": 0.95,    # Mode12 target eig-energy for rank selection
    }

    def __init__(self, opt: OPT, ap: AeroProblem, comm=None, output_dir: str = "output",
                 cfd_frequency: int = 10, cfd_include_mode: int = 1, cfd_iter: int = 1,
                 gek_options: Optional[Dict[str, Any]] = None):
        '''

        majorIter updated at end of iteration
        0, 1, .... cfd_f-1, cfd_f, cfd_f+1, ... cfd_f+cfd_i
        ML         ML       ML     CFD          CFD         

        '''

        self.opt = opt
        self.ap = ap
        self.comm = comm
        self.cfd_frequency = cfd_frequency
        self.cfd_include_mode = cfd_include_mode
        self.cfd_iter = cfd_iter
        self.output_dir = output_dir
        self.enter_CFD = False
        # First-order correction model (CFD - ML) updated at each CFD run
        self._correction: dict = None
        # online GEK correction
        self.gek_options = dict(SolverCombined.gek_default_options)
        if gek_options is not None:
            unknown = set(gek_options.keys()) - set(self.gek_options.keys())
            if unknown:
                raise ValueError(f"Unknown GEK options: {sorted(unknown)}")
            self.gek_options.update(gek_options)
        if self.cfd_include_mode == 12:
            self._gek_tag = "GEK(AS)"
        elif self.cfd_include_mode == 11:
            self._gek_tag = "GEK(norm)"
        else:
            self._gek_tag = "GEK"
        self._gek = _OnlineGEKDeltaModel(
            ap=self.ap,
            comm=self.comm,
            stable_norm=(self.cfd_include_mode in [11, 12]),
            use_active_subspace=(self.cfd_include_mode == 12),
            **self.gek_options,
        )
        self._pending_gek = None

        if self.comm.rank == 0:
            print("==============================")
            print("ML-CFD combined solver defined")
            print(f"Current combination mode = {cfd_include_mode}")
            print(f"    -> {SolverCombined.include_modes.get(cfd_include_mode, 'unknown mode')}")
            print(f"ML every {cfd_frequency:d} / {(cfd_frequency + cfd_iter):d}; CFD every {cfd_iter:d} / {(cfd_frequency + cfd_iter):d}")
            if cfd_include_mode in [10, 11, 12]:
                print(
                    f"{self._gek_tag} settings: "
                    f"max_points={self.gek_options['max_points']}, "
                    f"max_dims={self.gek_options['max_dims']}, "
                    f"nugget={self.gek_options['nugget']:.1e}, "
                    f"log_interval={self.gek_options['log_interval']}"
                )
                if cfd_include_mode in [11, 12]:
                    print(
                        f"{self._gek_tag} stable norm: xscale_floor={self.gek_options['xscale_floor']:.1e}, "
                        f"yscale_floor={self.gek_options['yscale_floor']:.1e}, "
                        f"ell_floor={self.gek_options['ell_floor']:.2f}, exp_clip={self.gek_options['exp_clip']:.1f}"
                    )
                if cfd_include_mode == 12:
                    print(
                        f"{self._gek_tag} active subspace: "
                        f"rank={self.gek_options['active_rank']}, energy={self.gek_options['active_energy']:.2f}"
                    )
            print("==============================")

    @staticmethod
    def _copy_design_vars(x: Dict[str, Any]) -> OrderedDict:
        copied = OrderedDict()
        for k, v in x.items():
            arr = np.asarray(v, dtype=float)
            copied[k] = float(arr.reshape(-1)[0]) if arr.size == 1 else arr.copy()
        return copied

    @staticmethod
    def update_funcs(ap: AeroProblem, ml_funcs, cfd_funcs):
        
        for func in ap.evalFuncs:
            ml_funcs[f"{ap.name}_{func}"] = cfd_funcs[f"{ap.name}_{func}"]
        if "fail" in cfd_funcs:
            ml_funcs["fail"] = cfd_funcs["fail"]

        return ml_funcs

    def wrap_cruiseFuncs(self, _ml_solver: Callable, _cfd_solver: Callable, _pre: Optional[Callable] = None, _post: Optional[Callable] = None):
        '''
        This function will return a function that is called by the optimizer at each iteration with the current design variables `x`.
        It combines the ML solver and the CFD solver according to the specified function keys and the optimization iteration number.
        
        '''
        def cruiseFuncs(x):
 

            opt_iter = self.opt.iterCounter
            major_opt_iter = self.opt.majorIterCounter

            self.ap.setDesignVars(x)

            if _pre is not None:
                _pre(self.ap, x)

            funcs = _ml_solver(self.ap)

            trigger_portion = (major_opt_iter + 1) % (self.cfd_frequency + self.cfd_iter)
            self.enter_CFD = ((self.cfd_iter <= 0) and (major_opt_iter+1 >= self.cfd_frequency)) or \
                             (self.cfd_iter > 0 and self.cfd_frequency > 0 and trigger_portion >= self.cfd_frequency)

            self.enter_CFD = self.comm.bcast(self.enter_CFD, root=0)

            if self.comm is not None and self.comm.rank == 0:
                print(f'Optimization iteration {opt_iter}; Major iteration {major_opt_iter}; enter CFD: {self.enter_CFD}')

            if self.enter_CFD:
                cfd_funcs = {}
                if self.comm is not None and self.comm.rank == 0:
                    print(f"")
                    print(f">>>>>>>>>>>>>>>>>>>>>>>>")
                    print(f"Running Solver (evaluation {opt_iter} / major iteration {major_opt_iter})")

                # print(f'Rank {comm.rank}: Starting CFD solve...')
                cfd_funcs = _cfd_solver(self.ap)

                if self.comm is not None and self.comm.rank == 0:
                    row = {"iter": opt_iter, "time": time.time() - self.opt.startTime, "major_iter": major_opt_iter, "ml_fail": funcs.get("fail", 0.0), "cfd_fail": cfd_funcs.get("fail", 0.0)}
                    # update ML and CFD predictions to the writting row
                    for func in self.ap.evalFuncs:
                        row[f"ml_{self.ap.name}_{func}"] = funcs[f"{self.ap.name}_{func}"]
                        row[f"cfd_{self.ap.name}_{func}"] = cfd_funcs[f"{self.ap.name}_{func}"]

                    # Store xuser variables into cfd.hst with a consistent prefix
                    for k, v in x.items():
                        row[f"xuser_{k}"] = v

                    _append_cfd_hist(os.path.join(self.output_dir, "cfd.hst"), row)
                    # simluated_iters.append(opt_iter)
                    print('current cfd funcs', row)

                if self.cfd_include_mode == 0:
                    self.ap.solveFailed = False
                    self.ap.fatalFail = False
                elif self.cfd_include_mode == 1:
                    pass # to update fail information
                elif self.cfd_include_mode == 2:
                    # use CFD solver results to replace ML's
                    # print(f"rank {self.comm.rank} enter 2")
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)
                elif self.cfd_include_mode in [10, 11, 12]:
                    # GEK modes: keep ML and CFD function values at the same CFD point.
                    # The paired sensitivities are added in wrap_cruiseFuncsSens.
                    # For GEK training, keep both ML and CFD values at the CFD point.
                    self._pending_gek = {
                        "x": self._copy_design_vars(x),
                        "ml_funcs": OrderedDict((f"{self.ap.name}_{func}", funcs[f"{self.ap.name}_{func}"]) for func in self.ap.evalFuncs),
                        "cfd_funcs": OrderedDict((f"{self.ap.name}_{func}", cfd_funcs[f"{self.ap.name}_{func}"]) for func in self.ap.evalFuncs),
                    }
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)
                elif self.cfd_include_mode in [3, 4]:
                    # build first-order correction model and return CFD results
                    self._correction = {
                        "x": x,
                        "funcs": {},
                        "funcSens": OrderedDict()
                    }
                    for func in self.ap.evalFuncs:
                        key = f"{self.ap.name}_{func}"
                        if key in cfd_funcs and key in funcs:
                            self._correction["funcs"][key] = cfd_funcs[key] - funcs[key]
                    funcs = self.update_funcs(self.ap, funcs, cfd_funcs)

            else:
                # print(f'Mode: {self.cfd_include_mode}, _correction is None?: {bool(self._correction)}')
                if self.cfd_include_mode in [3, 4] and self._correction:
                    # apply correction model to ML results between CFD updates
                    for func in self.ap.evalFuncs:
                        key = f"{self.ap.name}_{func}"
                        funcs[key] += self._correction["funcs"][key]

                        if self.cfd_include_mode == 3:
                            for dv, delta in self._correction["funcSens"][key].items():
                                # print(dv, x[dv].shape, self._correction["x"][dv].shape, delta.shape)
                                contrib = np.dot(delta, (x[dv] - self._correction["x"][dv]))
                                if isinstance(contrib, np.ndarray) and contrib.shape in [(), (1,)]:
                                    contrib = float(contrib.reshape(-1)[0])
                                funcs[key] += contrib
                elif self.cfd_include_mode in [10, 11, 12]:
                    # Non-CFD iteration: use GEK to predict correction delta and add it to ML output.
                    if self.comm is None or self.comm.rank == 0:
                        pred_funcs, _ = self._gek.predict(x, log_kind="func", tag=self._gek_tag)
                    else:
                        pred_funcs = None
                    if self.comm is not None:
                        pred_funcs = self.comm.bcast(pred_funcs, root=0)
                    if pred_funcs is not None:
                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            if key in pred_funcs and key in funcs:
                                funcs[key] += pred_funcs[key]

            # input()
            return funcs
        
        return cruiseFuncs
    
    def wrap_cruiseFuncsSens(self, _ml_solver: Callable, _cfd_solver: Callable, _pre: Optional[Callable] = None, _post: Optional[Callable] = None):
        '''
        This function will return a function that is called by the optimizer at each iteration with the current design variables `x`.
        It combines the ML solver and the CFD solver according to the specified function keys and the optimization iteration number.
        
        '''

        def cruiseFuncsSens(x, funcs):
            
            if _pre is not None:
                _pre(self.ap, x, funcs)

            funcsSens = _ml_solver(self.ap, x=x, funcs=funcs)

            if self.cfd_include_mode in [0, 1]:
                pass
            else:
                if self.enter_CFD:

                    if self.comm is not None and self.comm.rank == 0:
                        print(f"")
                        print(f">>>>>>>>>>>>>>>>>>>>>>>>")
                        print(f"Running Solver Adjoint")  

                    cfd_funcsSens = _cfd_solver(self.ap, x=x, funcs=funcs)

                    if self.comm is not None and self.comm.rank == 0:
                        row = {"ml_sen_fail": funcsSens.get("fail", 0.0), "cfd_sen_fail": cfd_funcsSens.get("fail", 0.0)}
                        for func in self.ap.evalFuncs:
                            row[f"ml_sen_{self.ap.name}_{func}"] = funcsSens[f"{self.ap.name}_{func}"]
                            row[f"cfd_sen_{self.ap.name}_{func}"] = cfd_funcsSens[f"{self.ap.name}_{func}"]
                        
                        print('current cfd recording funcsSens', row)
                        _append_cfd_hist(os.path.join(self.output_dir, "cfd.hst"), row)

                    if self.cfd_include_mode == 2:
                        # Update funcsSens with CFD sensitivities
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)
                    elif self.cfd_include_mode in [10, 11, 12]:
                        # CFD iteration: build one GEK sample from (CFD-ML) value and gradient deltas.
                        if self._pending_gek is not None:
                            if self.comm is None or self.comm.rank == 0:
                                func_delta = OrderedDict()
                                sens_delta = OrderedDict()
                                for func in self.ap.evalFuncs:
                                    key = f"{self.ap.name}_{func}"
                                    func_delta[key] = self._pending_gek["cfd_funcs"][key] - self._pending_gek["ml_funcs"][key]
                                    sens_delta[key] = OrderedDict()
                                    for dv, cfd_val in cfd_funcsSens[key].items():
                                        sens_delta[key][dv] = cfd_val - funcsSens[key][dv]
                                self._gek.add_sample_logged(self._pending_gek["x"], func_delta, sens_delta, tag=self._gek_tag)
                        self._pending_gek = None
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)

                    elif self.cfd_include_mode >= 3:

                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            # for initialization
                            if key not in self._correction["funcSens"]:
                                self._correction["funcSens"][key] = OrderedDict()
                            for dv, cfd_val in cfd_funcsSens[key].items():
                                self._correction["funcSens"][key][dv] = cfd_val - funcsSens[key][dv]
                        # Ensure funcsSens at CFD point matches high-fidelity results
                        funcsSens = self.update_funcs(self.ap, funcsSens, cfd_funcsSens)

                else:
                    if self.cfd_include_mode in [3,4] and self._correction:
                        for func in self.ap.evalFuncs:
                            key = f"{self.ap.name}_{func}"
                            for dv, delta in self._correction["funcSens"][key].items():
                                funcsSens[key][dv] += delta
                    elif self.cfd_include_mode in [10, 11, 12]:
                        # Non-CFD iteration: correct ML sensitivities with GEK-predicted gradient deltas.
                        if self.comm is None or self.comm.rank == 0:
                            _, pred_sens = self._gek.predict(x, log_kind="grad", tag=self._gek_tag)
                        else:
                            pred_sens = None
                        if self.comm is not None:
                            pred_sens = self.comm.bcast(pred_sens, root=0)
                        if pred_sens is not None:
                            for func in self.ap.evalFuncs:
                                key = f"{self.ap.name}_{func}"
                                if key not in pred_sens or key not in funcsSens:
                                    continue
                                for dv, dval in pred_sens[key].items():
                                    if dv in funcsSens[key]:
                                        funcsSens[key][dv] += dval

            # if comm.rank == 0:
            #     print('current cfd funcs sen', funcsSens)
            return funcsSens
        
        return cruiseFuncsSens
