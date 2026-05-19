from typing import List, Optional

import torch
import torch.nn.functional as F


class MergeMethod:
    """Base class for model merging techniques"""

    def __init__(
        self,
        num_models: int = None,
        weights: Optional[List[float]] = None,
    ):
        self.num_models = num_models
        self.weights = weights
        self._validate_inputs()

    def _validate_inputs(self):
        """Common validation for all merging techniques"""
        # Weight handling
        if self.weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            total = sum(self.weights)
            if total == 0:
                raise ValueError("Weights cannot sum to zero")
            self.weights = [w / total for w in self.weights]

        if len(self.weights) != self.num_models:
            raise ValueError(
                f"Weights count ({len(self.weights)}) must match model count ({self.num_models})"
            )

    def merge_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Main merge method to be implemented by subclasses"""
        raise NotImplementedError


class LinearMerger(MergeMethod):
    """Linear Interpolation merger"""

    def merge_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Weighted average implementation with dtype preservation"""
        # Preserve original dtype
        original_dtype = tensors[0].dtype

        # Convert to float32 for computation if needed
        work_tensors = [tensor.float() for tensor in tensors]

        blended = work_tensors[0] * self.weights[0]
        for tensor, weight in zip(work_tensors[1:], self.weights[1:]):
            blended += tensor * weight

        # Convert back to original dtype
        return blended.to(original_dtype)


class SlerpMerger(MergeMethod):
    """Spherical Linear Interpolation merger (2 models only)"""

    def __init__(self, t: float):
        super().__init__(num_models=2)
        self.t = t
        self._validate_slerp_specific()

    def _validate_slerp_specific(self):
        """SLERP-specific validation"""
        if not 0 <= self.t <= 1:
            raise ValueError("SLERP interpolation factor 't' must be between 0 and 1")

    def merge_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return self._slerp_implementation(self.t, tensors[0], tensors[1])

    @staticmethod
    def _slerp_implementation(
        t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995
    ) -> torch.Tensor:
        """Core SLERP implementation encapsulated within SlerpMerger"""
        # Input validation
        assert v0.shape == v1.shape, "Tensors must have matching dimensions"
        assert v0.device == v1.device, "Tensors must be on the same device"

        # Preserve original dtype
        original_dtype = v0.dtype
        v0 = v0.float()
        v1 = v1.float()

        # Normalization with epsilon to avoid division by zero
        epsilon = 1e-8
        norm_v0 = torch.linalg.norm(v0)
        norm_v1 = torch.linalg.norm(v1)

        v0_normalized = v0 / (norm_v0 + epsilon)
        v1_normalized = v1 / (norm_v1 + epsilon)

        # Dot product and angle calculation
        dot = (v0_normalized * v1_normalized).sum()
        if dot.abs() > dot_threshold:
            return (1 - t) * v0 + t * v1

        theta_0 = torch.arccos(dot.clamp(-1.0, 1.0))
        sin_theta_0 = torch.sin(theta_0)

        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)

        # Coefficients calculation with epsilon guard
        s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + epsilon)
        s1 = sin_theta_t / (sin_theta_0 + epsilon)

        return (s0 * v0 + s1 * v1).to(original_dtype)


class TaskArithmetic(MergeMethod):
    """Task Arithmetic merger that implements consensus-based model merging"""

    def __init__(
        self,
        num_models: int = None,
        weights: Optional[List[float]] = None,
        densities: Optional[List[float]] = None,
        sparsification_method: Optional[str] = None,
        sparsification_rescale: bool = True,
        consensus_method: str = "count",
        normalize_deltas: bool = True,
    ):
        """Initialize TaskArithmetic merger.

        Args:
            weights: Optional weights for each model. If None, equal weights are used.
            densities: Optional densities for sparsification. Required if sparsification is enabled.
            consensus_method: Method for consensus calculation ('count' or 'sum').
            sparsification_method: Optional method for sparsifying deltas.
            normalize_deltas: Whether to normalize the merged deltas by the sum of weights.
            sparsification_rescale: Whether to rescale sparse updates to maintain magnitude.
            num_models: Number of models to merge.
        """
        super().__init__(weights=weights, num_models=num_models)
        self.consensus_method = consensus_method
        self.sparsification_method = sparsification_method
        self.normalize_deltas = normalize_deltas
        self.sparsification_rescale = sparsification_rescale
        self.densities = densities
        self._validate_task_specific()

    def _validate_task_specific(self):
        """Task-specific validation"""
        if self.consensus_method not in [None, "count", "sum"]:
            raise ValueError("Consensus method must be either 'count' or 'sum'")

        implemented_methods = [None, "magnitude", "random"]
        if self.sparsification_method not in implemented_methods:
            raise ValueError(
                f"Invalid or unimplemented sparsification method: {self.sparsification_method}"
            )

        # If sparsification is enabled, validate densities
        if self.sparsification_method:
            if self.densities is None:
                raise ValueError(
                    "Densities must be provided when sparsification is enabled."
                )
            if len(self.densities) != self.num_models:
                raise ValueError(
                    f"The length of densities ({len(self.densities)}) must match the number of models ({self.num_models})."
                )

    def merge_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Merge input tensors using task arithmetic.

        This implementation aligns with GeneralizedTaskArithmeticMerge, using weighted
        combinations of model deltas with optional consensus filtering and normalization.

        Args:
            tensors: List of tensors to merge, with the first tensor being the base.

        Returns:
            Merged tensor combining the base with weighted deltas from other models.
        """
        base_tensor = tensors[0]
        # Preserve original dtype
        original_dtype = base_tensor.dtype

        # Convert to float32 for computation if needed
        base_tensor = base_tensor.float()
        work_tensors = [tensor.float() for tensor in tensors[1:]]

        deltas = []
        weights_list = []

        # Calculate deltas and collect weights
        for i, (tensor, weight) in enumerate(
            zip(work_tensors, self.weights[1:]), start=1
        ):
            delta = tensor - base_tensor
            if self.sparsification_method:
                delta = self._sparsify(delta, i)
            deltas.append(delta * weight)
            weights_list.append(weight)

        if not deltas:  # Only base tensor
            return base_tensor.to(original_dtype)

        # Convert lists to tensors for vectorized operations
        stacked_deltas = torch.stack(deltas, dim=0)
        weights = torch.tensor(
            weights_list, dtype=stacked_deltas.dtype, device=stacked_deltas.device
        )
        # Expand weights tensor to match deltas dimensions
        while len(stacked_deltas.shape) > len(weights.shape):
            weights = weights.unsqueeze(-1)

        if self.consensus_method:
            # Get consensus mask and apply it
            mask = self._get_consensus_mask(stacked_deltas)
            mixed_delta = (stacked_deltas * mask).sum(dim=0)

            if self.normalize_deltas:
                # Calculate divisor based on weights and mask
                divisor = (weights * mask).sum(dim=0)
                # Prevent division by zero
                divisor[divisor == 0] = 1
                mixed_delta = mixed_delta / divisor
        else:
            mixed_delta = stacked_deltas.sum(dim=0)

            if self.normalize_deltas:
                # Calculate divisor based on weights only
                divisor = weights.sum(dim=0)
                # Prevent division by near-zero values
                divisor[divisor.abs() < 1e-8] = 1
                mixed_delta = mixed_delta / divisor

        result = base_tensor + mixed_delta
        # Convert back to original dtype
        return result.to(original_dtype)

    def _sparsify(self, tensor: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Apply sparsification to tensor"""
        if not self.sparsification_method:
            return tensor

        density = self.densities[model_idx]

        if self.sparsification_method == "magnitude":
            k = int(density * tensor.numel())
            mask = torch.zeros_like(tensor)
            topk = torch.argsort(tensor.abs().view(-1), descending=True)[:k]
            mask.view(-1)[topk] = 1
            return tensor * mask
        elif self.sparsification_method == "random":
            # Use work_dtype based on tensor properties
            if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
                work_dtype = tensor.dtype
            else:
                # torch.bernoulli not implemented for float16 on CPU, upcast to float32
                work_dtype = torch.float32

            mask = torch.bernoulli(
                torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
            )
            res = tensor.to(work_dtype) * mask
            if self.sparsification_rescale:
                res /= density

            return res.to(tensor.dtype)

        return tensor

    def _get_consensus_mask(self, deltas: torch.Tensor) -> torch.Tensor:
        """Get consensus mask based on specified method"""
        # Get the sign (-1, 0, or 1) of each delta (change) across all models
        sign = deltas.sign()

        if self.consensus_method == "sum":
            sign_weight = deltas.sum(dim=0)
            majority_sign = (sign_weight >= 0).to(sign.dtype) * 2 - 1
            return sign == majority_sign
        else:  # count
            majority_sign = (sign.sum(dim=0) >= 0).to(sign.dtype) * 2 - 1
            return sign == majority_sign

class FusionMerger(MergeMethod):
    """
    Fusion merger (2 models only) that selects salient parameter deltas based on
    magnitude × KL scores and applies a binary mask using a dynamic threshold.

    Args:
        method: 'iqr' or 'knee' for thresholding strategy. Default: 'iqr'.
        max_samples: Max elements considered when estimating thresholds. Default: 1e6.
        rectify_embeddings: If True and both tensors are 2D with mismatched shapes,
                            trim to the common submatrix before merging. Default: True.
        on_shape_mismatch: Behavior when shapes differ and not rectified.
                           'skip' -> return base tensor, 'error' -> raise. Default: 'skip'.
    """

    # -----------------------------
    # Private auxiliary classes
    # -----------------------------
    class __DynamicThreshold:
        """Dynamic IQR thresholding over (optionally sampled) scores."""

        def __init__(self, max_samples: int = 1_000_000):
            self.max_samples = max_samples

        def __sample(self, flat: torch.Tensor) -> torch.Tensor:
            if flat.numel() <= self.max_samples:
                return flat
            # Random without replacement
            torch.manual_seed(42)
            idx = torch.randperm(flat.numel(), device=flat.device)[: self.max_samples]
            return flat[idx]

        def __sampled_quantiles(self, tensor: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(-1)
            
            flat = self.__sample(flat)
            y, _ = torch.sort(flat)
            if y.numel() == 0:
                # No data -> force select nothing
                return torch.full_like(q, float("inf"))
            idx = (q * max(y.numel() - 1, 0)).long()
            return y[idx]

        def get_threshold(self, scores: torch.Tensor) -> torch.Tensor:
            qs = self.__sampled_quantiles(scores, torch.tensor([0.25, 0.5, 0.75], device=scores.device))
            q1, median, q3 = qs
            iqr = q3 - q1
            return median + 1.5 * iqr

    class __KneeElbowThreshold:
        """Knee/Elbow detection on the sorted, normalized score curve."""

        def __init__(self, max_samples: int = 1_000_000, eps: float = 1e-12):
            self.max_samples = max_samples
            self.eps = eps

        def __sample(self, flat: torch.Tensor) -> torch.Tensor:
            if flat.numel() <= self.max_samples:
                return flat
            # Random without replacement
            torch.manual_seed(42)
            idx = torch.randperm(flat.numel(), device=flat.device)[: self.max_samples]
            return flat[idx]

        def get_threshold(self, scores: torch.Tensor) -> torch.Tensor:
            # Flatten and (maybe) subsample
            flat = scores.reshape(-1)
            flat = self.__sample(flat)

            # Sort ascending: y
            y, _ = torch.sort(flat)

            n = y.numel()
            if n <= 2:
                # Degenerate fallback: pick max as threshold to select only the top element(s)
                return y.max()

            # Normalize x in [0,1]
            x = torch.linspace(0.0, 1.0, steps=n, device=y.device)

            # Normalize y to [0,1] (avoid div by zero)
            ymin = y[0]
            ymax = y[-1]
            denom = torch.clamp(ymax - ymin, min=self.eps)
            y_norm = (y - ymin) / denom

            # Deviation from diagonal
            d = y_norm - x

            # Choose knee/elbow:
            # If any positive deviation exists, take the strongest knee (argmax).
            # Otherwise, take the most negative deviation (argmin) as elbow.
            pos_exists = (d > 0).any()
            idx = torch.argmax(d) if pos_exists else torch.argmin(d)

            # Threshold is the *original* (un-normalized) y value at knee/elbow index.
            threshold = y[idx]

            return threshold

    # -----------------------------
    # FusionMerger implementation
    # -----------------------------
    def __init__(
        self,
        method: str = "knee",
        max_samples: int = 1_000_000,
        rectify_embeddings: bool = True,
        on_shape_mismatch: str = "skip",
    ):
        super().__init__(num_models=2)  # 2 models only (base, target)
        self.method = method
        self.rectify_embeddings = rectify_embeddings
        if on_shape_mismatch not in ("skip", "error"):
            raise ValueError("on_shape_mismatch must be 'skip' or 'error'")
        self.on_shape_mismatch = on_shape_mismatch
        self.__fusion = self.__make_threshold_strategy(method, max_samples=max_samples)

    def merge_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        assert len(tensors) == 2, "FusionMerger supports exactly 2 tensors (base, target)"
        base_tensor, target_tensor = tensors

        # Shape handling (optional rectification for 2D)
        bt, tt = base_tensor, target_tensor
        if bt.shape != tt.shape:
            if self.rectify_embeddings and bt.ndim == 2 and tt.ndim == 2:
                h = min(bt.shape[0], tt.shape[0])
                w = min(bt.shape[1], tt.shape[1])
                bt = bt[:h, :w]
                tt = tt[:h, :w]
            else:
                if self.on_shape_mismatch == "error":
                    raise ValueError(f"Shape mismatch: {bt.shape} vs {tt.shape}")
                return base_tensor  # 'skip' -> no-op

        if bt.device != tt.device:
            raise AssertionError("Tensors must be on the same device")
        device = bt.device  # noqa: F841 (kept for parity with other mergers)
        original_dtype = bt.dtype

        # Compute in float32
        btf = bt.float()
        ttf = tt.float()

        # Scores and mask
        scores = self.__get_scores(ttf, btf)
        threshold = self.__fusion.get_threshold(scores)
        mask = (scores >= threshold).to(dtype=btf.dtype)

        # Apply masked delta and restore dtype
        fused = btf + (ttf - btf) * mask
        return fused.to(original_dtype)

    # -----------------------------
    # Private helpers
    # -----------------------------
    @staticmethod
    def __get_scores(params: torch.Tensor, base_params: torch.Tensor) -> torch.Tensor:
        """Importance scores combining magnitude and KL divergence (over last dim)."""
        diff = (params - base_params).abs()
        if len(params.shape) >= 2:
            log_p = F.log_softmax(params, dim=-1)
            log_q = F.log_softmax(base_params, dim=-1)
            p = log_p.exp()
            kl_div = torch.sum(p * (log_p - log_q), dim=-1)
            
            # Broadcasting logic
            if len(params.shape) > len(kl_div.shape):
                kl_div = kl_div.unsqueeze(-1)
            
            scores = diff * kl_div
        else:
            scores = diff
        return scores

    def __make_threshold_strategy(self, method: str, *, max_samples: int):
        m = (method or "iqr").lower()
        if m == "knee":
            return self.__KneeElbowThreshold(max_samples=max_samples)
        if m == "iqr":
            return self.__DynamicThreshold(max_samples=max_samples)
        raise ValueError(f"Unknown thresholding method: {method}. Choose from ['iqr', 'knee'].")

