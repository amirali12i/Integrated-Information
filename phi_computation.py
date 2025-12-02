"""
Approximate Φ (Integrated Information) Computation for Vision-Language Models

This module implements Algorithm 1 from:
"Theoretical Frameworks for Quantifying Integrated Information in Vision-Enhanced 
Large Language Models as Proxies for Emergent Consciousness"

Authors: Amirali Ghajari, Maicol Ochoa
Universidad Europea de Madrid

Key Features:
- Subnetwork sampling with cross-modal connectivity constraints
- Spectral clustering for bipartition (Fiedler vector method)
- Gaussian KDE with Silverman bandwidth for density estimation
- Variational upper bound for KL divergence computation
- Heuristic confidence interval scaling for large models

IMPORTANT LIMITATIONS:
- Validated only on networks ≤20 nodes
- 9 orders of magnitude gap to billion-parameter models
- Bootstrap 95% CI for scaling exponent: [0.21, 0.52]
- Results should be interpreted as preliminary observations
"""

import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.neighbors import KernelDensity
from typing import Tuple, List, Dict, Optional, Union
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhiResult:
    """Container for Φ computation results with uncertainty estimates."""
    phi_mean: float
    ci_lower: float
    ci_upper: float
    ci_scaled_lower: float
    ci_scaled_upper: float
    n_samples: int
    n_subnetworks: int
    subnetwork_size: int
    model_params: int
    phi_samples: np.ndarray
    
    def __repr__(self):
        return (f"PhiResult(Φ={self.phi_mean:.3f}, "
                f"95% CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
                f"Scaled CI=[{self.ci_scaled_lower:.3f}, {self.ci_scaled_upper:.3f}])")


@dataclass 
class SubnetworkConfig:
    """Configuration for subnetwork sampling."""
    size: int = 20  # Nodes per subnetwork (validation limit)
    n_subnetworks: int = 100  # Number of subnetworks to sample
    min_visual_ratio: float = 0.3  # Minimum fraction of visual nodes
    min_linguistic_ratio: float = 0.3  # Minimum fraction of linguistic nodes
    partition_depth: int = 3  # Depth of partition search


@dataclass
class CIScalingConfig:
    """Configuration for confidence interval scaling (heuristic).
    
    IMPORTANT: This scaling is heuristic, not theoretically justified.
    See Section 2.2.1 of the paper for derivation and sensitivity analysis.
    
    Sensitivity analysis shows γ ∈ {0.10, 0.15, 0.20} all preserve main findings.
    """
    gamma: float = 0.15  # CI scaling factor (heuristic)
    n_val: int = 20  # Validation limit (nodes)
    
    def compute_scaling_factor(self, n_params: int) -> float:
        """Compute CI scaling factor based on model size.
        
        Formula: 1 + γ · log₁₀(N / N_val)
        
        For 7B params: factor ≈ 2.27
        For 100B params: factor ≈ 2.70
        """
        if n_params <= self.n_val:
            return 1.0
        return 1.0 + self.gamma * np.log10(n_params / self.n_val)


class HiddenStateExtractor(ABC):
    """Abstract base class for extracting hidden states from VLLMs."""
    
    @abstractmethod
    def extract(self, visual_input: np.ndarray, 
                linguistic_input: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract hidden states from model.
        
        Returns:
            Dictionary with keys:
            - 'hidden_states': Shape (n_layers, n_tokens, hidden_dim)
            - 'attention_weights': Shape (n_layers, n_heads, n_tokens, n_tokens)
            - 'visual_token_indices': Indices of visual tokens
            - 'linguistic_token_indices': Indices of linguistic tokens
        """
        pass
    
    @abstractmethod
    def get_param_count(self) -> int:
        """Return total parameter count of the model."""
        pass


class MockHiddenStateExtractor(HiddenStateExtractor):
    """Mock extractor for testing and demonstration."""
    
    def __init__(self, n_layers: int = 32, hidden_dim: int = 768,
                 n_visual_tokens: int = 256, n_linguistic_tokens: int = 512,
                 param_count: int = 7_000_000_000):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_visual_tokens = n_visual_tokens
        self.n_linguistic_tokens = n_linguistic_tokens
        self._param_count = param_count
        
    def extract(self, visual_input: np.ndarray,
                linguistic_input: np.ndarray) -> Dict[str, np.ndarray]:
        n_tokens = self.n_visual_tokens + self.n_linguistic_tokens
        
        # Generate mock hidden states with structure
        # Visual and linguistic tokens have different distributions
        hidden_states = np.zeros((self.n_layers, n_tokens, self.hidden_dim))
        
        for layer in range(self.n_layers):
            # Visual tokens
            hidden_states[layer, :self.n_visual_tokens, :] = (
                np.random.randn(self.n_visual_tokens, self.hidden_dim) * 0.5 +
                np.sin(np.linspace(0, 2*np.pi, self.hidden_dim)) * (layer / self.n_layers)
            )
            # Linguistic tokens
            hidden_states[layer, self.n_visual_tokens:, :] = (
                np.random.randn(self.n_linguistic_tokens, self.hidden_dim) * 0.5 +
                np.cos(np.linspace(0, 2*np.pi, self.hidden_dim)) * (layer / self.n_layers)
            )
            
        # Generate attention weights with cross-modal structure
        attention_weights = np.zeros((self.n_layers, 12, n_tokens, n_tokens))
        for layer in range(self.n_layers):
            # Attention strength increases with depth
            cross_modal_strength = 0.1 + 0.8 * (layer / self.n_layers)
            
            # Self-attention within modalities
            attention_weights[layer, :, :self.n_visual_tokens, :self.n_visual_tokens] = (
                np.random.rand(12, self.n_visual_tokens, self.n_visual_tokens) * 0.5
            )
            attention_weights[layer, :, self.n_visual_tokens:, self.n_visual_tokens:] = (
                np.random.rand(12, self.n_linguistic_tokens, self.n_linguistic_tokens) * 0.5
            )
            
            # Cross-modal attention
            attention_weights[layer, :, :self.n_visual_tokens, self.n_visual_tokens:] = (
                np.random.rand(12, self.n_visual_tokens, self.n_linguistic_tokens) * cross_modal_strength
            )
            attention_weights[layer, :, self.n_visual_tokens:, :self.n_visual_tokens] = (
                np.random.rand(12, self.n_linguistic_tokens, self.n_visual_tokens) * cross_modal_strength
            )
            
            # Normalize rows
            for head in range(12):
                row_sums = attention_weights[layer, head].sum(axis=1, keepdims=True)
                attention_weights[layer, head] /= (row_sums + 1e-10)
        
        return {
            'hidden_states': hidden_states,
            'attention_weights': attention_weights,
            'visual_token_indices': np.arange(self.n_visual_tokens),
            'linguistic_token_indices': np.arange(self.n_visual_tokens, n_tokens)
        }
    
    def get_param_count(self) -> int:
        return self._param_count


class SubnetworkSampler:
    """Sample subnetworks with cross-modal connectivity constraints."""
    
    def __init__(self, config: SubnetworkConfig):
        self.config = config
        
    def sample_subnetworks(self, 
                          attention_weights: np.ndarray,
                          visual_indices: np.ndarray,
                          linguistic_indices: np.ndarray,
                          seed: Optional[int] = None) -> List[np.ndarray]:
        """Sample subnetworks using attention-weighted sampling.
        
        Args:
            attention_weights: Shape (n_layers, n_heads, n_tokens, n_tokens)
            visual_indices: Indices of visual tokens
            linguistic_indices: Indices of linguistic tokens
            seed: Random seed for reproducibility
            
        Returns:
            List of subnetwork node indices
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_tokens = attention_weights.shape[2]
        
        # Compute node importance from attention (average across layers and heads)
        avg_attention = attention_weights.mean(axis=(0, 1))
        node_importance = avg_attention.sum(axis=0) + avg_attention.sum(axis=1)
        node_importance /= node_importance.sum()
        
        subnetworks = []
        
        for _ in range(self.config.n_subnetworks):
            # Calculate required nodes from each modality
            min_visual = int(np.ceil(self.config.size * self.config.min_visual_ratio))
            min_linguistic = int(np.ceil(self.config.size * self.config.min_linguistic_ratio))
            remaining = self.config.size - min_visual - min_linguistic
            
            # Sample visual nodes
            visual_importance = node_importance[visual_indices]
            visual_importance /= visual_importance.sum()
            visual_nodes = np.random.choice(
                visual_indices, 
                size=min_visual,
                replace=False,
                p=visual_importance
            )
            
            # Sample linguistic nodes
            linguistic_importance = node_importance[linguistic_indices]
            linguistic_importance /= linguistic_importance.sum()
            linguistic_nodes = np.random.choice(
                linguistic_indices,
                size=min_linguistic,
                replace=False,
                p=linguistic_importance
            )
            
            # Sample remaining nodes from either modality
            selected = set(visual_nodes) | set(linguistic_nodes)
            available = [i for i in range(n_tokens) if i not in selected]
            available_importance = node_importance[available]
            available_importance /= available_importance.sum()
            
            remaining_nodes = np.random.choice(
                available,
                size=remaining,
                replace=False,
                p=available_importance
            )
            
            subnetwork = np.concatenate([visual_nodes, linguistic_nodes, remaining_nodes])
            subnetworks.append(subnetwork)
            
        return subnetworks


class SpectralPartitioner:
    """Partition networks using spectral clustering (Fiedler vector method)."""
    
    def __init__(self, partition_depth: int = 3):
        self.partition_depth = partition_depth
        
    def compute_fiedler_partition(self, 
                                  adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bipartition using the Fiedler vector (2nd eigenvector of Laplacian).
        
        Args:
            adjacency: Adjacency/attention matrix of the subnetwork
            
        Returns:
            Tuple of (partition_0_indices, partition_1_indices)
        """
        n = adjacency.shape[0]
        
        # Compute degree matrix
        degree = np.diag(adjacency.sum(axis=1))
        
        # Compute Laplacian: L = D - A
        laplacian = degree - adjacency
        
        # Compute eigenvectors
        try:
            if n > 10:
                # Use sparse solver for larger matrices
                eigenvalues, eigenvectors = eigsh(laplacian, k=2, which='SM')
            else:
                eigenvalues, eigenvectors = eigh(laplacian)
                
            # Fiedler vector is the eigenvector corresponding to 2nd smallest eigenvalue
            fiedler_idx = np.argsort(eigenvalues)[1]
            fiedler_vector = eigenvectors[:, fiedler_idx]
            
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}. Using random partition.")
            fiedler_vector = np.random.randn(n)
        
        # Partition by sign of Fiedler vector
        partition_0 = np.where(fiedler_vector >= 0)[0]
        partition_1 = np.where(fiedler_vector < 0)[0]
        
        # Ensure non-empty partitions
        if len(partition_0) == 0:
            partition_0 = np.array([0])
            partition_1 = np.arange(1, n)
        elif len(partition_1) == 0:
            partition_1 = np.array([n-1])
            partition_0 = np.arange(0, n-1)
            
        return partition_0, partition_1
    
    def find_optimal_partition(self,
                              hidden_states: np.ndarray,
                              adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find optimal bipartition using spectral clustering with refinement.
        
        Args:
            hidden_states: Hidden state vectors for the subnetwork
            adjacency: Adjacency matrix
            
        Returns:
            Tuple of (partition_0_indices, partition_1_indices)
        """
        # Get initial partition from Fiedler vector
        best_partition = self.compute_fiedler_partition(adjacency)
        best_kl = self._compute_partition_kl(hidden_states, best_partition)
        
        # Greedy refinement
        for _ in range(self.partition_depth):
            improved = False
            p0, p1 = best_partition
            
            # Try moving each node to the other partition
            for i in range(len(p0)):
                if len(p0) <= 1:
                    continue
                    
                # Create new partition with node moved
                new_p0 = np.delete(p0, i)
                new_p1 = np.append(p1, p0[i])
                new_partition = (new_p0, new_p1)
                
                new_kl = self._compute_partition_kl(hidden_states, new_partition)
                
                if new_kl < best_kl:
                    best_kl = new_kl
                    best_partition = new_partition
                    improved = True
                    break
                    
            for i in range(len(p1)):
                if len(p1) <= 1:
                    continue
                    
                new_p1 = np.delete(p1, i)
                new_p0 = np.append(p0, p1[i])
                new_partition = (new_p0, new_p1)
                
                new_kl = self._compute_partition_kl(hidden_states, new_partition)
                
                if new_kl < best_kl:
                    best_kl = new_kl
                    best_partition = new_partition
                    improved = True
                    break
                    
            if not improved:
                break
                
        return best_partition
    
    def _compute_partition_kl(self,
                             hidden_states: np.ndarray,
                             partition: Tuple[np.ndarray, np.ndarray]) -> float:
        """Compute KL divergence for a given partition (helper for optimization)."""
        p0, p1 = partition
        
        if len(p0) == 0 or len(p1) == 0:
            return float('inf')
            
        # Simplified KL computation for optimization
        h0 = hidden_states[p0]
        h1 = hidden_states[p1]
        
        # Use variance as proxy for information content
        var_joint = np.var(hidden_states, axis=0).sum()
        var_p0 = np.var(h0, axis=0).sum() if len(p0) > 1 else 0
        var_p1 = np.var(h1, axis=0).sum() if len(p1) > 1 else 0
        
        # Information lost is related to variance reduction
        return var_joint - (var_p0 * len(p0) + var_p1 * len(p1)) / len(hidden_states)


class KLDivergenceEstimator:
    """Estimate KL divergence using Gaussian KDE with variational bounds."""
    
    def __init__(self, bandwidth_method: str = 'silverman'):
        self.bandwidth_method = bandwidth_method
        
    def estimate_kl_divergence(self,
                              hidden_states: np.ndarray,
                              partition: Tuple[np.ndarray, np.ndarray]) -> float:
        """Estimate KL divergence between joint and factorized distributions.
        
        Uses variational upper bound from Aguilera et al. (2021):
        D_KL ≈ 0.5 * (tr(Σ_P^{-1}Σ) + (μ_P - μ)^T Σ_P^{-1}(μ_P - μ) - d + ln(|Σ_P|/|Σ|))
        
        Args:
            hidden_states: Hidden state vectors
            partition: Tuple of partition indices
            
        Returns:
            Estimated KL divergence (Φ for this partition)
        """
        p0, p1 = partition
        n = len(hidden_states)
        
        if len(p0) == 0 or len(p1) == 0:
            return 0.0
            
        # Compute statistics of joint distribution
        mu_joint = hidden_states.mean(axis=0)
        
        # Use diagonal covariance for computational efficiency
        var_joint = np.var(hidden_states, axis=0) + 1e-6  # Regularization
        
        # Compute statistics of factorized distribution
        h0 = hidden_states[p0]
        h1 = hidden_states[p1]
        
        mu_p0 = h0.mean(axis=0) if len(p0) > 0 else np.zeros_like(mu_joint)
        mu_p1 = h1.mean(axis=0) if len(p1) > 0 else np.zeros_like(mu_joint)
        
        var_p0 = np.var(h0, axis=0) + 1e-6 if len(p0) > 1 else np.ones_like(var_joint)
        var_p1 = np.var(h1, axis=0) + 1e-6 if len(p1) > 1 else np.ones_like(var_joint)
        
        # Weighted combination for factorized distribution
        w0 = len(p0) / n
        w1 = len(p1) / n
        
        mu_factorized = w0 * mu_p0 + w1 * mu_p1
        var_factorized = w0 * var_p0 + w1 * var_p1
        
        # Variational upper bound for KL divergence (diagonal case)
        d = len(mu_joint)
        
        # tr(Σ_P^{-1}Σ)
        trace_term = np.sum(var_joint / var_factorized)
        
        # (μ_P - μ)^T Σ_P^{-1}(μ_P - μ)
        mu_diff = mu_factorized - mu_joint
        quadratic_term = np.sum(mu_diff**2 / var_factorized)
        
        # ln(|Σ_P|/|Σ|) for diagonal matrices
        log_det_term = np.sum(np.log(var_factorized) - np.log(var_joint))
        
        kl_divergence = 0.5 * (trace_term + quadratic_term - d + log_det_term)
        
        return max(0.0, kl_divergence)  # KL divergence is non-negative


class PhiComputer:
    """Main class for computing Φ (integrated information) for VLLMs.
    
    Implements Algorithm 1 from the paper with all components:
    - Subnetwork sampling with cross-modal constraints
    - Spectral clustering for bipartition
    - KL divergence estimation with variational bounds
    - Confidence interval scaling for large models
    """
    
    def __init__(self,
                 subnetwork_config: Optional[SubnetworkConfig] = None,
                 ci_config: Optional[CIScalingConfig] = None,
                 seed: int = 42):
        self.subnetwork_config = subnetwork_config or SubnetworkConfig()
        self.ci_config = ci_config or CIScalingConfig()
        self.seed = seed
        
        self.sampler = SubnetworkSampler(self.subnetwork_config)
        self.partitioner = SpectralPartitioner(self.subnetwork_config.partition_depth)
        self.kl_estimator = KLDivergenceEstimator()
        
    def compute_phi(self,
                   extractor: HiddenStateExtractor,
                   visual_inputs: List[np.ndarray],
                   linguistic_inputs: List[np.ndarray],
                   n_samples: int = 1000,
                   verbose: bool = True) -> PhiResult:
        """Compute approximate Φ for a VLLM.
        
        Args:
            extractor: Hidden state extractor for the model
            visual_inputs: List of visual inputs (images)
            linguistic_inputs: List of linguistic inputs (text)
            n_samples: Number of samples to use
            verbose: Whether to print progress
            
        Returns:
            PhiResult with mean Φ and confidence intervals
        """
        np.random.seed(self.seed)
        
        n_params = extractor.get_param_count()
        phi_samples = []
        
        # Sample inputs
        n_available = min(len(visual_inputs), len(linguistic_inputs))
        sample_indices = np.random.choice(n_available, size=min(n_samples, n_available), replace=True)
        
        for idx, sample_idx in enumerate(sample_indices):
            if verbose and idx % 100 == 0:
                logger.info(f"Processing sample {idx}/{len(sample_indices)}")
                
            # Extract hidden states
            extracted = extractor.extract(
                visual_inputs[sample_idx],
                linguistic_inputs[sample_idx]
            )
            
            hidden_states = extracted['hidden_states']
            attention_weights = extracted['attention_weights']
            visual_indices = extracted['visual_token_indices']
            linguistic_indices = extracted['linguistic_token_indices']
            
            # Sample subnetworks (only on first sample for efficiency)
            if idx == 0:
                subnetworks = self.sampler.sample_subnetworks(
                    attention_weights,
                    visual_indices,
                    linguistic_indices,
                    seed=self.seed
                )
            
            # Compute Φ for each subnetwork
            subnetwork_phis = []
            
            for subnetwork in subnetworks:
                # Get hidden states and attention for subnetwork
                # Use last layer (where integration is typically highest)
                sub_hidden = hidden_states[-1, subnetwork, :]
                sub_attention = attention_weights[-1, :, subnetwork, :][:, :, subnetwork].mean(axis=0)
                
                # Find optimal partition
                partition = self.partitioner.find_optimal_partition(sub_hidden, sub_attention)
                
                # Compute KL divergence (Φ for this subnetwork)
                phi = self.kl_estimator.estimate_kl_divergence(sub_hidden, partition)
                subnetwork_phis.append(phi)
                
            # Average across subnetworks
            phi_sample = np.mean(subnetwork_phis)
            phi_samples.append(phi_sample)
            
        phi_samples = np.array(phi_samples)
        
        # Compute statistics
        phi_mean = np.mean(phi_samples)
        phi_std = np.std(phi_samples)
        
        # Standard 95% CI
        ci_margin = 1.96 * phi_std / np.sqrt(len(phi_samples))
        ci_lower = phi_mean - ci_margin
        ci_upper = phi_mean + ci_margin
        
        # Scaled CI (heuristic for large models)
        scaling_factor = self.ci_config.compute_scaling_factor(n_params)
        scaled_margin = ci_margin * scaling_factor
        ci_scaled_lower = phi_mean - scaled_margin
        ci_scaled_upper = phi_mean + scaled_margin
        
        if verbose:
            logger.info(f"Φ = {phi_mean:.3f} ± {ci_margin:.3f}")
            logger.info(f"Scaled CI (factor={scaling_factor:.2f}): [{ci_scaled_lower:.3f}, {ci_scaled_upper:.3f}]")
            
        return PhiResult(
            phi_mean=phi_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_scaled_lower=ci_scaled_lower,
            ci_scaled_upper=ci_scaled_upper,
            n_samples=len(phi_samples),
            n_subnetworks=len(subnetworks),
            subnetwork_size=self.subnetwork_config.size,
            model_params=n_params,
            phi_samples=phi_samples
        )
    
    def compute_layer_wise_phi(self,
                              extractor: HiddenStateExtractor,
                              visual_input: np.ndarray,
                              linguistic_input: np.ndarray) -> Dict[int, float]:
        """Compute Φ for each layer separately.
        
        Args:
            extractor: Hidden state extractor
            visual_input: Single visual input
            linguistic_input: Single linguistic input
            
        Returns:
            Dictionary mapping layer index to Φ value
        """
        extracted = extractor.extract(visual_input, linguistic_input)
        hidden_states = extracted['hidden_states']
        attention_weights = extracted['attention_weights']
        visual_indices = extracted['visual_token_indices']
        linguistic_indices = extracted['linguistic_token_indices']
        
        # Sample subnetworks
        subnetworks = self.sampler.sample_subnetworks(
            attention_weights,
            visual_indices,
            linguistic_indices,
            seed=self.seed
        )
        
        n_layers = hidden_states.shape[0]
        layer_phis = {}
        
        for layer in range(n_layers):
            layer_phi_values = []
            
            for subnetwork in subnetworks[:10]:  # Use subset for efficiency
                sub_hidden = hidden_states[layer, subnetwork, :]
                sub_attention = attention_weights[layer, :, subnetwork, :][:, :, subnetwork].mean(axis=0)
                
                partition = self.partitioner.find_optimal_partition(sub_hidden, sub_attention)
                phi = self.kl_estimator.estimate_kl_divergence(sub_hidden, partition)
                layer_phi_values.append(phi)
                
            layer_phis[layer] = np.mean(layer_phi_values)
            
        return layer_phis


def compute_geometric_phi(hidden_states: np.ndarray, 
                         n_samples: int = 10000) -> float:
    """Compute geometric integrated information |Φ_G|.
    
    Φ_G = ∫_M |κ(x)|ρ(x) dσ(x)
    
    where κ is Gaussian curvature, ρ is probability density.
    
    Args:
        hidden_states: Shape (n_tokens, hidden_dim)
        n_samples: Number of points to sample for curvature estimation
        
    Returns:
        Absolute geometric integrated information
    """
    n_tokens, hidden_dim = hidden_states.shape
    
    # Sample points for curvature estimation
    sample_indices = np.random.choice(n_tokens, size=min(n_samples, n_tokens), replace=True)
    sampled_states = hidden_states[sample_indices]
    
    # Estimate probability density using KDE
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    # Use PCA for high-dimensional data
    if hidden_dim > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        sampled_states_reduced = pca.fit_transform(sampled_states)
    else:
        sampled_states_reduced = sampled_states
        
    kde.fit(sampled_states_reduced)
    log_density = kde.score_samples(sampled_states_reduced)
    density = np.exp(log_density)
    
    # Estimate curvature using local Hessian approximation
    # This is a simplified approximation
    curvatures = []
    
    for i in range(len(sampled_states_reduced)):
        # Find nearest neighbors
        point = sampled_states_reduced[i]
        distances = np.linalg.norm(sampled_states_reduced - point, axis=1)
        nearest_indices = np.argsort(distances)[1:min(20, len(distances))]
        neighbors = sampled_states_reduced[nearest_indices]
        
        if len(neighbors) < 3:
            curvatures.append(0.0)
            continue
            
        # Estimate local curvature from neighbor distribution
        centered = neighbors - point
        cov = np.cov(centered.T) + 1e-6 * np.eye(centered.shape[1])
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Gaussian curvature approximation (product of principal curvatures)
        # For high-dim, use ratio of eigenvalues as proxy
        if len(eigenvalues) >= 2:
            curvature = eigenvalues[-1] / (eigenvalues[0] + 1e-10) - 1.0
        else:
            curvature = 0.0
            
        curvatures.append(curvature)
        
    curvatures = np.array(curvatures)
    
    # Integrate |κ|ρ
    phi_geom = np.mean(np.abs(curvatures) * density)
    
    return phi_geom


# Convenience function for quick computation
def quick_phi_estimate(extractor: HiddenStateExtractor,
                      visual_inputs: List[np.ndarray],
                      linguistic_inputs: List[np.ndarray],
                      n_samples: int = 100,
                      seed: int = 42) -> PhiResult:
    """Quick Φ estimate with default settings.
    
    Args:
        extractor: Model hidden state extractor
        visual_inputs: Visual inputs
        linguistic_inputs: Linguistic inputs
        n_samples: Number of samples (default 100 for speed)
        seed: Random seed
        
    Returns:
        PhiResult with Φ estimate
    """
    computer = PhiComputer(seed=seed)
    return computer.compute_phi(
        extractor, 
        visual_inputs, 
        linguistic_inputs,
        n_samples=n_samples,
        verbose=False
    )


if __name__ == "__main__":
    # Demonstration with mock data
    print("=" * 60)
    print("Φ Computation Demo")
    print("=" * 60)
    
    # Create mock extractor (simulating LLaVA-7B)
    extractor = MockHiddenStateExtractor(
        n_layers=32,
        hidden_dim=768,
        n_visual_tokens=256,
        n_linguistic_tokens=512,
        param_count=7_000_000_000
    )
    
    # Create mock inputs
    n_samples = 100
    visual_inputs = [np.random.randn(224, 224, 3) for _ in range(n_samples)]
    linguistic_inputs = [np.random.randn(512, 768) for _ in range(n_samples)]
    
    # Compute Φ
    computer = PhiComputer(seed=42)
    result = computer.compute_phi(
        extractor,
        visual_inputs,
        linguistic_inputs,
        n_samples=50,  # Reduced for demo
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Φ (mean): {result.phi_mean:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"Scaled CI: [{result.ci_scaled_lower:.4f}, {result.ci_scaled_upper:.4f}]")
    print(f"Samples: {result.n_samples}")
    print(f"Subnetworks: {result.n_subnetworks}")
    print(f"Model params: {result.model_params:,}")
