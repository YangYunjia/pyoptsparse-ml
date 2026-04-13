'''
Online GEK model for delta = CFD - ML using function values and gradients.
'''

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable
from collections import OrderedDict

import numpy as np

from baseclasses import AeroSolver, AeroProblem

class _OnlineGEKDeltaModel:
    """
    Online GEK model for delta = CFD - ML using function values and gradients.

    Args:
        ap: AeroProblem handle (used for objective/constraint keys).
        max_points: Sliding-window size for online training samples.
        max_dims: Maximum retained DV dimensions before GEK fitting.
        nugget: Diagonal jitter added to covariance for numerical stability.
        stable_norm: Enable x/y normalization for stable small-sample fitting.
        xscale_floor: Lower bound for per-dimension x normalization scale.
        yscale_floor: Lower bound for output-delta normalization scale.
        ell_floor: Lower bound for GP length scales in normalized space.
        exp_clip: Clamp exponent in RBF kernel evaluation to avoid underflow.
        use_active_subspace: Learn and apply gradient-based active subspace.
        active_rank: Hard cap for active-subspace dimension.
        active_energy: Target cumulative eigenvalue energy for subspace size.
        log_interval: Prediction-log print interval (stateful, used by helper methods).
    """

    def __init__(
        self,
        ap: AeroProblem,
        comm=None,
        max_points: int = 20,
        max_dims: int = 40,
        nugget: float = 1e-8,
        stable_norm: bool = False,
        xscale_floor: float = 1e-3,
        yscale_floor: float = 1e-8,
        ell_floor: float = 0.2,
        exp_clip: float = 60.0,
        use_active_subspace: bool = False,
        active_rank: int = 8,
        active_energy: float = 0.95,
        log_interval: int = 1,
    ):
        self.ap = ap
        self.comm = comm
        self.max_points = max(2, int(max_points))
        self.max_dims = max(1, int(max_dims))
        self.nugget = float(nugget)
        self.stable_norm = bool(stable_norm)
        self.xscale_floor = float(xscale_floor)
        self.yscale_floor = float(yscale_floor)
        self.ell_floor = float(ell_floor)
        self.exp_clip = float(exp_clip)
        self.use_active_subspace = bool(use_active_subspace)
        self.active_rank = max(1, int(active_rank))
        self.active_energy = float(active_energy)
        self.log_interval = max(1, int(log_interval))

        self._layout_ready = False
        self._dv_names: List[str] = []
        self._dv_shapes: Dict[str, Tuple[int, ...]] = {}
        self._offsets: Dict[str, Tuple[int, int]] = {}
        self._full_dim = 0
        self._active_idx = None

        self._X: List[np.ndarray] = []
        # Per-function training buffers:
        #   Y[key]: scalar delta samples
        #   G[key]: gradient-delta samples (flattened reduced-DV space)
        self._Y: Dict[str, List[float]] = OrderedDict()
        self._G: Dict[str, List[np.ndarray]] = OrderedDict()
        self._alpha: Dict[str, np.ndarray] = OrderedDict()
        self._Xmat = None
        self._ell = None
        self._x_mu = None
        self._x_scale = None
        self._W = None
        self._as_dim = 0
        self._y_mu: Dict[str, float] = OrderedDict()
        self._y_scale: Dict[str, float] = OrderedDict()
        self._n_train = 0
        self._fitted = False
        self._just_initialized_layout = False
        self._predict_counter = 0
        self._not_ready_warned = False

    def has_model(self) -> bool:
        return self._fitted and self._n_train >= 2 and self._Xmat is not None

    def _is_rank0(self) -> bool:
        return (self.comm is None) or (self.comm.rank == 0)

    def _init_layout(self, x: Dict[str, Any], sens_delta_by_func: Dict[str, Dict[str, Any]]):
        # Build a stable flatten/unflatten map from structured DVs -> one vector.
        self._dv_names = list(x.keys())
        pos = 0
        for dv in self._dv_names:
            arr = np.asarray(x[dv], dtype=float)
            self._dv_shapes[dv] = arr.shape
            self._offsets[dv] = (pos, pos + arr.size)
            pos += arr.size
        self._full_dim = pos

        if self._full_dim <= self.max_dims:
            self._active_idx = np.arange(self._full_dim, dtype=np.int64)
        else:
            # Keep dimensions with strongest accumulated sensitivity magnitude.
            score = np.zeros((self._full_dim,), dtype=float)
            for _, grad_dict in sens_delta_by_func.items():
                score += np.abs(self._flatten_grad_full(grad_dict))
            if np.all(score <= 0.0):
                self._active_idx = np.arange(self.max_dims, dtype=np.int64)
            else:
                self._active_idx = np.argsort(score)[-self.max_dims:]
                self._active_idx.sort()
        self._layout_ready = True
        self._just_initialized_layout = True

    def _flatten_x_full(self, x: Dict[str, Any]) -> np.ndarray:
        flat = np.zeros((self._full_dim,), dtype=float)
        for dv in self._dv_names:
            s0, s1 = self._offsets[dv]
            flat[s0:s1] = np.asarray(x[dv], dtype=float).reshape(-1)
        return flat

    def _flatten_grad_full(self, grad_dict: Dict[str, Any]) -> np.ndarray:
        flat = np.zeros((self._full_dim,), dtype=float)
        for dv in self._dv_names:
            if dv not in grad_dict:
                continue
            s0, s1 = self._offsets[dv]
            flat[s0:s1] = np.asarray(grad_dict[dv], dtype=float).reshape(-1)
        return flat

    def _flatten_x(self, x: Dict[str, Any]) -> np.ndarray:
        return self._flatten_x_full(x)[self._active_idx]

    def _flatten_grad(self, grad_dict: Dict[str, Any]) -> np.ndarray:
        return self._flatten_grad_full(grad_dict)[self._active_idx]

    def _unflatten_grad(self, grad_red: np.ndarray) -> OrderedDict:
        grad_full = np.zeros((self._full_dim,), dtype=float)
        grad_full[self._active_idx] = grad_red
        out = OrderedDict()
        for dv in self._dv_names:
            s0, s1 = self._offsets[dv]
            out[dv] = grad_full[s0:s1].reshape(self._dv_shapes[dv]).copy()
        return out

    def _build_covariance(self, X: np.ndarray, ell: np.ndarray) -> np.ndarray:
        # GEK block covariance:
        # [K_ff  K_fg]
        # [K_gf  K_gg]
        n, d = X.shape
        nd = n * d
        diff = X[:, None, :] - X[None, :, :]
        inv_l2 = 1.0 / np.maximum(ell * ell, 1e-16)
        expo = -0.5 * np.sum((diff * np.sqrt(inv_l2)[None, None, :]) ** 2, axis=2)
        if self.stable_norm:
            expo = np.clip(expo, -self.exp_clip, 0.0)
        kff = np.exp(expo)

        kfg = np.zeros((n, nd), dtype=float)
        for j in range(n):
            b = j * d
            kfg[:, b:b + d] = kff[:, [j]] * (diff[:, j, :] * inv_l2[None, :])

        kgg = np.zeros((nd, nd), dtype=float)
        eye_l = np.diag(inv_l2)
        for i in range(n):
            bi = i * d
            for j in range(n):
                bj = j * d
                rij = diff[i, j, :]
                kgg[bi:bi + d, bj:bj + d] = kff[i, j] * (eye_l - np.outer(rij * inv_l2, rij * inv_l2))

        K = np.vstack([np.hstack([kff, kfg]), np.hstack([kfg.T, kgg])])
        K[np.diag_indices_from(K)] += self.nugget
        return K

    def _fit(self):
        self._fitted = False
        self._alpha = OrderedDict()
        n = len(self._X)
        self._n_train = n
        if n < 2:
            return

        X_raw = np.vstack(self._X)
        if self.stable_norm:
            # Normalize query space to avoid extreme distances / kernel underflow.
            self._x_mu = np.mean(X_raw, axis=0)
            x_std = np.std(X_raw, axis=0)
            self._x_scale = np.maximum(x_std, self.xscale_floor)
            X = (X_raw - self._x_mu[None, :]) / self._x_scale[None, :]
        else:
            self._x_mu = np.zeros((X_raw.shape[1],), dtype=float)
            self._x_scale = np.ones((X_raw.shape[1],), dtype=float)
            X = X_raw

        d = X.shape[1]
        if self.use_active_subspace and d > 1:
            # Learn active subspace from gradient covariance in normalized space.
            grads_all = []
            for func in self.ap.evalFuncs:
                key = f"{self.ap.name}_{func}"
                if key in self._G and len(self._G[key]) == n:
                    g_raw = np.vstack(self._G[key])
                    if self.stable_norm:
                        y_raw = np.asarray(self._Y[key], dtype=float)
                        y_scale = max(float(np.std(y_raw)), self.yscale_floor)
                        g_norm = (g_raw * self._x_scale[None, :]) / y_scale
                    else:
                        g_norm = g_raw
                    grads_all.append(g_norm)
            if grads_all:
                G = np.vstack(grads_all)
                C = (G.T @ G) / max(G.shape[0], 1)
                eigvals, eigvecs = np.linalg.eigh(C)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                if np.sum(eigvals) > 0.0:
                    cum = np.cumsum(np.clip(eigvals, 0.0, None)) / np.sum(np.clip(eigvals, 0.0, None))
                    k_energy = int(np.searchsorted(cum, self.active_energy) + 1)
                else:
                    k_energy = 1
                k = min(self.active_rank, k_energy, d)
                self._W = eigvecs[:, :k]
                self._as_dim = int(k)
                X_use = X @ self._W
            else:
                self._W = np.eye(d)
                self._as_dim = d
                X_use = X
        else:
            self._W = np.eye(d)
            self._as_dim = d
            X_use = X

        ell = np.std(X_use, axis=0)
        ell = np.maximum(np.where(ell > 1e-12, ell, 1.0), self.ell_floor if self.stable_norm else 1e-12)
        K = self._build_covariance(X_use, ell)

        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key not in self._Y or key not in self._G:
                continue
            if len(self._Y[key]) != n or len(self._G[key]) != n:
                continue
            y_raw = np.asarray(self._Y[key], dtype=float)
            g_raw = np.vstack(self._G[key])
            if self.stable_norm:
                # Train in normalized output space to keep scales comparable.
                y_mu = float(np.mean(y_raw))
                y_scale = max(float(np.std(y_raw)), self.yscale_floor)
                self._y_mu[key] = y_mu
                self._y_scale[key] = y_scale
                y = (y_raw - y_mu) / y_scale
                # dz/dx_norm = (dy/dx_raw) * x_scale / y_scale
                g_norm = (g_raw * self._x_scale[None, :]) / y_scale
            else:
                self._y_mu[key] = 0.0
                self._y_scale[key] = 1.0
                y = y_raw
                g_norm = g_raw
            # Active-subspace gradient: dy/dz = dy/dx * W
            g = g_norm @ self._W
            g = g.reshape(-1)
            obs = np.concatenate([y, g])
            try:
                self._alpha[key] = np.linalg.solve(K, obs)
            except np.linalg.LinAlgError:
                self._alpha[key] = np.linalg.lstsq(K, obs, rcond=1e-12)[0]

        self._Xmat = X_use
        self._ell = ell
        self._fitted = len(self._alpha) > 0

    def add_sample(self, x: Dict[str, Any], func_delta: Dict[str, float], sens_delta: Dict[str, Dict[str, Any]]):
        if not self._layout_ready:
            self._init_layout(x, sens_delta)

        self._X.append(self._flatten_x(x))
        if len(self._X) > self.max_points:
            self._X.pop(0)

        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            self._Y.setdefault(key, [])
            self._G.setdefault(key, [])
            self._Y[key].append(float(func_delta.get(key, 0.0)))
            self._G[key].append(self._flatten_grad(sens_delta.get(key, {})))
            if len(self._Y[key]) > self.max_points:
                self._Y[key].pop(0)
            if len(self._G[key]) > self.max_points:
                self._G[key].pop(0)

        self._fit()

    def add_sample_logged(self, x: Dict[str, Any], func_delta: Dict[str, float], sens_delta: Dict[str, Dict[str, Any]], tag: str = "GEK"):
        """Train on one sample and print training messages on rank0."""
        self.add_sample(x, func_delta, sens_delta)
        if self._is_rank0():
            for msg in self.consume_training_messages(tag=tag):
                print(msg)

    def model_info(self) -> Dict[str, Any]:
        red_dim = int(self._active_idx.size) if self._active_idx is not None else 0
        n = int(self._n_train)
        n_obs = n * (1 + red_dim)
        return {
            "n_train": n,
            "full_dim": int(self._full_dim),
            "red_dim": red_dim,
            "n_obs": n_obs,
            "fitted_funcs": len(self._alpha),
            "layout_initialized": bool(self._just_initialized_layout),
            "stable_norm": self.stable_norm,
            "active_subspace": self.use_active_subspace,
            "as_dim": int(self._as_dim),
            "log_interval": int(self.log_interval),
        }

    def consume_layout_init_flag(self) -> bool:
        flag = self._just_initialized_layout
        self._just_initialized_layout = False
        return flag

    def consume_not_ready_message(self, tag: str = "GEK") -> Optional[str]:
        if self.has_model() or self._not_ready_warned:
            return None
        self._not_ready_warned = True
        info = self.model_info()
        return f"{tag} predict skipped: model not ready (samples={info['n_train']}, need >=2)."

    def consume_training_messages(self, tag: str = "GEK") -> List[str]:
        info = self.model_info()
        msgs: List[str] = []
        if self.consume_layout_init_flag():
            msgs.append(f"{tag} layout initialized: full_dim={info['full_dim']}, reduced_dim={info['red_dim']}.")
        msgs.append(
            f"{tag} correction updated: "
            f"samples={info['n_train']}, obs={info['n_obs']}, "
            f"fitted_funcs={info['fitted_funcs']}/{len(self.ap.evalFuncs)}, "
            f"as_dim={info['as_dim']}"
        )
        self._not_ready_warned = False
        return msgs

    def consume_predict_func_message(self, pred_funcs: Optional[Dict[str, float]], tag: str = "GEK") -> Optional[str]:
        if pred_funcs is None:
            return None
        self._predict_counter += 1
        if self._predict_counter % self.log_interval != 0:
            return None
        msg = [f"{tag} predict #{self._predict_counter}"]
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key in pred_funcs:
                msg.append(f"{func}:d={pred_funcs[key]:+.6e}")
        return " | ".join(msg)

    def consume_predict_grad_message(self, pred_sens: Optional[Dict[str, Dict[str, Any]]], tag: str = "GEK") -> Optional[str]:
        if pred_sens is None:
            return None
        self._predict_counter += 1
        if self._predict_counter % self.log_interval != 0:
            return None
        norms = []
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key in pred_sens:
                norm_val = 0.0
                for _, g in pred_sens[key].items():
                    norm_val += float(np.linalg.norm(np.asarray(g, dtype=float)))
                norms.append(f"{func}:|dg|={norm_val:.3e}")
        return f"{tag} grad predict | " + " | ".join(norms)

    def predict(self, x: Dict[str, Any], log_kind: Optional[str] = None, tag: str = "GEK"):
        if not self.has_model():
            if self._is_rank0() and log_kind is not None:
                msg = self.consume_not_ready_message(tag=tag)
                if msg:
                    print(msg)
            return None, None

        xq_raw = self._flatten_x(x)
        if self.stable_norm:
            xq = (xq_raw - self._x_mu) / self._x_scale
        else:
            xq = xq_raw
        X = self._Xmat
        ell = self._ell
        n, d = X.shape
        # Predict in active coordinates z when active subspace is enabled.
        x_use = xq @ self._W
        diff = X - x_use[None, :]
        inv_l2 = 1.0 / np.maximum(ell * ell, 1e-16)
        expo = -0.5 * np.sum((diff * np.sqrt(inv_l2)[None, :]) ** 2, axis=1)
        if self.stable_norm:
            expo = np.clip(expo, -self.exp_clip, 0.0)
        kff = np.exp(expo)

        k_grad_train = (x_use[None, :] - X) * inv_l2[None, :] * kff[:, None]
        ks_f = np.concatenate([kff, k_grad_train.reshape(-1)])

        pred_f = OrderedDict()
        pred_g = OrderedDict()
        eye_l = np.diag(inv_l2)
        for func in self.ap.evalFuncs:
            key = f"{self.ap.name}_{func}"
            if key not in self._alpha:
                continue
            alpha = self._alpha[key]
            pred_z = float(np.dot(ks_f, alpha))
            if self.stable_norm:
                # Undo output normalization back to physical CFD-ML delta.
                pred_f[key] = pred_z * self._y_scale[key] + self._y_mu[key]
            else:
                pred_f[key] = pred_z

            grad_z = np.zeros((d,), dtype=float)
            for b in range(d):
                kf_gb = diff[:, b] * inv_l2[b] * kff
                kg_gb = np.zeros((n, d), dtype=float)
                for i in range(n):
                    ri = X[i] - x_use
                    kg_gb[i, :] = kff[i] * (eye_l[:, b] - (ri * inv_l2) * (ri[b] * inv_l2[b]))
                ks_g = np.concatenate([kf_gb, kg_gb.reshape(-1)])
                grad_z[b] = float(np.dot(ks_g, alpha))
            # Back-map: dy/dx_norm = dy/dz * W^T
            grad_x_norm = grad_z @ self._W.T
            if self.stable_norm:
                # Undo x/y normalization to recover dy/dx in original DV space.
                grad_red = (grad_x_norm * self._y_scale[key]) / self._x_scale
            else:
                grad_red = grad_x_norm
            pred_g[key] = self._unflatten_grad(grad_red)

        if self._is_rank0() and log_kind == "func":
            msg = self.consume_predict_func_message(pred_f, tag=tag)
            if msg:
                print(msg)
        elif self._is_rank0() and log_kind == "grad":
            msg = self.consume_predict_grad_message(pred_g, tag=tag)
            if msg:
                print(msg)

        return pred_f, pred_g

