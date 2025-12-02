"""
Awareness Benchmarks and Evaluation

This module implements the composite Awareness Score and individual
benchmark evaluations used in the paper:

1. CLEVR - Object permanence and visual reasoning
2. VCR - Visual commonsense reasoning  
3. FANToM - Theory of Mind evaluation

The Awareness Score is defined as:
A = (1/3)[CLEVR_norm + VCR_norm + FANToM_norm]

where each score is normalized to [0, 1] using:
score_norm = (score - baseline_min) / (baseline_max - baseline_min)

Authors: Amirali Ghajari, Maicol Ochoa
Universidad Europea de Madrid
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkBaselines:
    """Normalization baselines for benchmarks.
    
    All use random chance as minimum following Issue 9 resolution.
    """
    # CLEVR: Random chance for yes/no questions
    clevr_min: float = 0.50
    clevr_max: float = 0.97  # Human expert
    
    # VCR: Random chance for 4-choice
    vcr_min: float = 0.25
    vcr_max: float = 0.93  # Human crowdworkers
    
    # FANToM: Random chance for binary ToM
    fantom_min: float = 0.50
    fantom_max: float = 0.92  # Human baseline


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""
    benchmark_name: str
    raw_score: float
    normalized_score: float
    n_samples: int
    ci_lower: float
    ci_upper: float
    details: Dict[str, Any]
    
    def __repr__(self):
        return (f"BenchmarkResult({self.benchmark_name}: "
                f"raw={self.raw_score:.3f}, norm={self.normalized_score:.3f}, "
                f"CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}])")


@dataclass
class AwarenessScore:
    """Composite awareness score with component breakdown."""
    score: float
    ci_lower: float
    ci_upper: float
    clevr_contribution: float
    vcr_contribution: float
    fantom_contribution: float
    component_results: Dict[str, BenchmarkResult]
    
    def __repr__(self):
        return (f"AwarenessScore(A={self.score:.3f}, "
                f"CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}])")


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    @abstractmethod
    def evaluate(self, model_predictions: List[Any], 
                ground_truth: List[Any]) -> BenchmarkResult:
        """Evaluate model on benchmark.
        
        Args:
            model_predictions: Model outputs
            ground_truth: Ground truth labels
            
        Returns:
            BenchmarkResult with scores
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return benchmark name."""
        pass


class CLEVRBenchmark(Benchmark):
    """CLEVR benchmark for compositional visual reasoning.
    
    Evaluates:
    - Object counting
    - Spatial relationships
    - Attribute comparison
    - Object permanence (with occlusion sequences)
    
    Reference: Johnson et al. (2017) CVPR
    """
    
    def __init__(self, baselines: Optional[BenchmarkBaselines] = None):
        self.baselines = baselines or BenchmarkBaselines()
        
    def get_name(self) -> str:
        return "CLEVR"
        
    def evaluate(self,
                model_predictions: List[str],
                ground_truth: List[str]) -> BenchmarkResult:
        """Evaluate model on CLEVR.
        
        Args:
            model_predictions: Model answers (strings)
            ground_truth: Correct answers (strings)
            
        Returns:
            BenchmarkResult with accuracy
        """
        if len(model_predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
            
        n_correct = sum(
            self._normalize_answer(p) == self._normalize_answer(g)
            for p, g in zip(model_predictions, ground_truth)
        )
        
        n_samples = len(ground_truth)
        raw_score = n_correct / n_samples
        
        # Normalize
        normalized = self._normalize_score(raw_score)
        
        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(
            model_predictions, ground_truth, n_bootstrap=1000
        )
        
        return BenchmarkResult(
            benchmark_name="CLEVR",
            raw_score=raw_score,
            normalized_score=normalized,
            n_samples=n_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                'n_correct': n_correct,
                'baseline_min': self.baselines.clevr_min,
                'baseline_max': self.baselines.clevr_max
            }
        )
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        return str(answer).lower().strip()
    
    def _normalize_score(self, raw: float) -> float:
        """Normalize raw score to [0, 1]."""
        normalized = (raw - self.baselines.clevr_min) / (
            self.baselines.clevr_max - self.baselines.clevr_min
        )
        return max(0.0, min(1.0, normalized))  # Clip to [0, 1]
    
    def _bootstrap_ci(self,
                     predictions: List[str],
                     ground_truth: List[str],
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(predictions)
        scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_preds = [predictions[i] for i in indices]
            boot_truth = [ground_truth[i] for i in indices]
            
            n_correct = sum(
                self._normalize_answer(p) == self._normalize_answer(g)
                for p, g in zip(boot_preds, boot_truth)
            )
            scores.append(self._normalize_score(n_correct / n))
            
        alpha = 1 - confidence
        lower = np.percentile(scores, alpha/2 * 100)
        upper = np.percentile(scores, (1 - alpha/2) * 100)
        
        return lower, upper
    
    def evaluate_object_permanence(self,
                                   model_fn: Callable,
                                   sequences: List[Dict]) -> BenchmarkResult:
        """Evaluate specifically on object permanence tasks.
        
        Args:
            model_fn: Function that takes sequence and returns prediction
            sequences: List of occlusion sequence dictionaries with:
                - 'frames': List of frame images
                - 'question': Question about occluded object
                - 'answer': Ground truth answer
                
        Returns:
            BenchmarkResult for object permanence
        """
        predictions = []
        ground_truth = []
        
        for seq in sequences:
            pred = model_fn(seq['frames'], seq['question'])
            predictions.append(pred)
            ground_truth.append(seq['answer'])
            
        result = self.evaluate(predictions, ground_truth)
        result.details['task_type'] = 'object_permanence'
        
        return result


class VCRBenchmark(Benchmark):
    """Visual Commonsense Reasoning benchmark.
    
    Evaluates ability to:
    - Infer implicit relationships
    - Understand social situations
    - Apply commonsense knowledge
    
    Reference: Zellers et al. (2019) CVPR
    """
    
    def __init__(self, baselines: Optional[BenchmarkBaselines] = None):
        self.baselines = baselines or BenchmarkBaselines()
        
    def get_name(self) -> str:
        return "VCR"
        
    def evaluate(self,
                model_predictions: List[int],
                ground_truth: List[int]) -> BenchmarkResult:
        """Evaluate model on VCR.
        
        Args:
            model_predictions: Selected answer indices (0-3)
            ground_truth: Correct answer indices (0-3)
            
        Returns:
            BenchmarkResult with accuracy
        """
        if len(model_predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
            
        n_correct = sum(p == g for p, g in zip(model_predictions, ground_truth))
        n_samples = len(ground_truth)
        raw_score = n_correct / n_samples
        
        # Normalize
        normalized = self._normalize_score(raw_score)
        
        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(
            model_predictions, ground_truth
        )
        
        return BenchmarkResult(
            benchmark_name="VCR",
            raw_score=raw_score,
            normalized_score=normalized,
            n_samples=n_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                'n_correct': n_correct,
                'n_choices': 4,
                'baseline_min': self.baselines.vcr_min,
                'baseline_max': self.baselines.vcr_max
            }
        )
    
    def _normalize_score(self, raw: float) -> float:
        """Normalize raw score to [0, 1]."""
        normalized = (raw - self.baselines.vcr_min) / (
            self.baselines.vcr_max - self.baselines.vcr_min
        )
        return max(0.0, min(1.0, normalized))
    
    def _bootstrap_ci(self,
                     predictions: List[int],
                     ground_truth: List[int],
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(predictions)
        scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_preds = [predictions[i] for i in indices]
            boot_truth = [ground_truth[i] for i in indices]
            
            n_correct = sum(p == g for p, g in zip(boot_preds, boot_truth))
            scores.append(self._normalize_score(n_correct / n))
            
        alpha = 1 - confidence
        return np.percentile(scores, alpha/2 * 100), np.percentile(scores, (1 - alpha/2) * 100)


class FANToMBenchmark(Benchmark):
    """FANToM benchmark for Theory of Mind.
    
    Evaluates ability to:
    - Track beliefs of agents
    - Understand false beliefs
    - Predict actions based on beliefs
    
    Reference: Gandhi et al. (2023) EMNLP
    
    Note: This replaces the incorrectly cited "ToMBench" from earlier versions.
    """
    
    def __init__(self, baselines: Optional[BenchmarkBaselines] = None):
        self.baselines = baselines or BenchmarkBaselines()
        
    def get_name(self) -> str:
        return "FANToM"
        
    def evaluate(self,
                model_predictions: List[bool],
                ground_truth: List[bool]) -> BenchmarkResult:
        """Evaluate model on FANToM.
        
        Args:
            model_predictions: Model's belief predictions (True/False)
            ground_truth: Correct beliefs (True/False)
            
        Returns:
            BenchmarkResult with accuracy
        """
        if len(model_predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
            
        n_correct = sum(p == g for p, g in zip(model_predictions, ground_truth))
        n_samples = len(ground_truth)
        raw_score = n_correct / n_samples
        
        # Normalize
        normalized = self._normalize_score(raw_score)
        
        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(
            model_predictions, ground_truth
        )
        
        return BenchmarkResult(
            benchmark_name="FANToM",
            raw_score=raw_score,
            normalized_score=normalized,
            n_samples=n_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                'n_correct': n_correct,
                'baseline_min': self.baselines.fantom_min,
                'baseline_max': self.baselines.fantom_max
            }
        )
    
    def _normalize_score(self, raw: float) -> float:
        """Normalize raw score to [0, 1]."""
        normalized = (raw - self.baselines.fantom_min) / (
            self.baselines.fantom_max - self.baselines.fantom_min
        )
        return max(0.0, min(1.0, normalized))
    
    def _bootstrap_ci(self,
                     predictions: List[bool],
                     ground_truth: List[bool],
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(predictions)
        scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_preds = [predictions[i] for i in indices]
            boot_truth = [ground_truth[i] for i in indices]
            
            n_correct = sum(p == g for p, g in zip(boot_preds, boot_truth))
            scores.append(self._normalize_score(n_correct / n))
            
        alpha = 1 - confidence
        return np.percentile(scores, alpha/2 * 100), np.percentile(scores, (1 - alpha/2) * 100)


class AwarenessEvaluator:
    """Compute composite Awareness Score from benchmark results.
    
    A = (1/3)[CLEVR_norm + VCR_norm + FANToM_norm]
    
    Equal weights are used as a neutral prior.
    Sensitivity analysis shows all components contribute meaningfully.
    """
    
    def __init__(self,
                 clevr_weight: float = 1/3,
                 vcr_weight: float = 1/3,
                 fantom_weight: float = 1/3):
        """Initialize evaluator with component weights.
        
        Args:
            clevr_weight: Weight for CLEVR (default 1/3)
            vcr_weight: Weight for VCR (default 1/3)
            fantom_weight: Weight for FANToM (default 1/3)
        """
        total = clevr_weight + vcr_weight + fantom_weight
        self.clevr_weight = clevr_weight / total
        self.vcr_weight = vcr_weight / total
        self.fantom_weight = fantom_weight / total
        
        self.clevr_benchmark = CLEVRBenchmark()
        self.vcr_benchmark = VCRBenchmark()
        self.fantom_benchmark = FANToMBenchmark()
        
    def compute_awareness_score(self,
                               clevr_result: BenchmarkResult,
                               vcr_result: BenchmarkResult,
                               fantom_result: BenchmarkResult) -> AwarenessScore:
        """Compute composite awareness score from benchmark results.
        
        Args:
            clevr_result: CLEVR evaluation result
            vcr_result: VCR evaluation result
            fantom_result: FANToM evaluation result
            
        Returns:
            AwarenessScore with composite and component scores
        """
        # Weighted sum
        score = (
            self.clevr_weight * clevr_result.normalized_score +
            self.vcr_weight * vcr_result.normalized_score +
            self.fantom_weight * fantom_result.normalized_score
        )
        
        # Propagate uncertainties (assuming independence)
        var = (
            (self.clevr_weight * (clevr_result.ci_upper - clevr_result.ci_lower) / 3.92)**2 +
            (self.vcr_weight * (vcr_result.ci_upper - vcr_result.ci_lower) / 3.92)**2 +
            (self.fantom_weight * (fantom_result.ci_upper - fantom_result.ci_lower) / 3.92)**2
        )
        std = np.sqrt(var)
        ci_lower = score - 1.96 * std
        ci_upper = score + 1.96 * std
        
        return AwarenessScore(
            score=score,
            ci_lower=max(0.0, ci_lower),
            ci_upper=min(1.0, ci_upper),
            clevr_contribution=self.clevr_weight * clevr_result.normalized_score,
            vcr_contribution=self.vcr_weight * vcr_result.normalized_score,
            fantom_contribution=self.fantom_weight * fantom_result.normalized_score,
            component_results={
                'CLEVR': clevr_result,
                'VCR': vcr_result,
                'FANToM': fantom_result
            }
        )
    
    def sensitivity_analysis(self,
                            clevr_result: BenchmarkResult,
                            vcr_result: BenchmarkResult,
                            fantom_result: BenchmarkResult) -> Dict[str, float]:
        """Analyze sensitivity of awareness score to each component.
        
        Returns correlation when each component is removed.
        """
        results = {}
        
        # Full score
        full = self.compute_awareness_score(clevr_result, vcr_result, fantom_result)
        results['full_score'] = full.score
        
        # Without CLEVR
        score_no_clevr = (
            0.5 * vcr_result.normalized_score +
            0.5 * fantom_result.normalized_score
        )
        results['without_clevr'] = score_no_clevr
        
        # Without VCR
        score_no_vcr = (
            0.5 * clevr_result.normalized_score +
            0.5 * fantom_result.normalized_score
        )
        results['without_vcr'] = score_no_vcr
        
        # Without FANToM
        score_no_fantom = (
            0.5 * clevr_result.normalized_score +
            0.5 * vcr_result.normalized_score
        )
        results['without_fantom'] = score_no_fantom
        
        return results


def compute_phi_awareness_correlation(phi_values: List[float],
                                      awareness_scores: List[float]) -> Dict[str, float]:
    """Compute correlation between Φ and awareness scores.
    
    Args:
        phi_values: List of Φ values for different models
        awareness_scores: Corresponding awareness scores
        
    Returns:
        Dictionary with correlation statistics
    """
    from scipy import stats
    
    phi = np.array(phi_values)
    awareness = np.array(awareness_scores)
    
    # Pearson correlation
    r, p_value = stats.pearsonr(phi, awareness)
    
    # Bootstrap CI for correlation
    n = len(phi)
    bootstrap_r = []
    for _ in range(1000):
        indices = np.random.choice(n, size=n, replace=True)
        boot_r, _ = stats.pearsonr(phi[indices], awareness[indices])
        bootstrap_r.append(boot_r)
        
    ci_lower = np.percentile(bootstrap_r, 2.5)
    ci_upper = np.percentile(bootstrap_r, 97.5)
    
    # Linear regression for performance prediction
    slope, intercept, _, _, std_err = stats.linregress(phi, awareness)
    
    return {
        'correlation_r': r,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'regression_slope': slope,
        'regression_intercept': intercept,
        'regression_std_err': std_err
    }


# Expected results from the paper (Table 5)
PAPER_RESULTS = {
    'phi-3-mini': {'phi': 0.18, 'awareness': 0.42},
    'phi-3': {'phi': 0.31, 'awareness': 0.48},
    'llava-1.5-7b': {'phi': 0.42, 'awareness': 0.58},
    'llava-7b': {'phi': 0.53, 'awareness': 0.67},
    'llava-13b': {'phi': 0.64, 'awareness': 0.74},
    'phi-4-multimodal': {'phi': 0.61, 'awareness': 0.71},
    'gpt-4v': {'phi': 0.79, 'awareness': 0.88}  # Estimated
}


if __name__ == "__main__":
    print("Awareness Benchmarks Demo")
    print("=" * 50)
    
    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate model predictions (above random chance)
    clevr_preds = ['yes' if np.random.rand() > 0.3 else 'no' for _ in range(n_samples)]
    clevr_truth = ['yes' if np.random.rand() > 0.5 else 'no' for _ in range(n_samples)]
    
    vcr_preds = [np.random.randint(0, 4) for _ in range(n_samples)]
    vcr_truth = [np.random.randint(0, 4) for _ in range(n_samples)]
    # Bias predictions toward correct
    vcr_preds = [t if np.random.rand() > 0.4 else p for p, t in zip(vcr_preds, vcr_truth)]
    
    fantom_preds = [np.random.rand() > 0.4 for _ in range(n_samples)]
    fantom_truth = [np.random.rand() > 0.5 for _ in range(n_samples)]
    fantom_preds = [t if np.random.rand() > 0.3 else p for p, t in zip(fantom_preds, fantom_truth)]
    
    # Evaluate
    clevr = CLEVRBenchmark()
    vcr = VCRBenchmark()
    fantom = FANToMBenchmark()
    
    clevr_result = clevr.evaluate(clevr_preds, clevr_truth)
    vcr_result = vcr.evaluate(vcr_preds, vcr_truth)
    fantom_result = fantom.evaluate(fantom_preds, fantom_truth)
    
    print("\nBenchmark Results:")
    print(f"  {clevr_result}")
    print(f"  {vcr_result}")
    print(f"  {fantom_result}")
    
    # Compute awareness score
    evaluator = AwarenessEvaluator()
    awareness = evaluator.compute_awareness_score(clevr_result, vcr_result, fantom_result)
    
    print(f"\n{awareness}")
    print(f"  CLEVR contribution: {awareness.clevr_contribution:.3f}")
    print(f"  VCR contribution: {awareness.vcr_contribution:.3f}")
    print(f"  FANToM contribution: {awareness.fantom_contribution:.3f}")
    
    # Correlation analysis from paper
    print("\n" + "=" * 50)
    print("Φ-Awareness Correlation (Paper Results)")
    print("=" * 50)
    
    phi_vals = [v['phi'] for v in PAPER_RESULTS.values()]
    awareness_vals = [v['awareness'] for v in PAPER_RESULTS.values()]
    
    corr = compute_phi_awareness_correlation(phi_vals, awareness_vals)
    print(f"\nCorrelation: r = {corr['correlation_r']:.3f}")
    print(f"P-value: {corr['p_value']:.6f}")
    print(f"95% CI: [{corr['ci_lower']:.3f}, {corr['ci_upper']:.3f}]")
    print(f"Regression: Awareness = {corr['regression_intercept']:.2f} + {corr['regression_slope']:.2f} × Φ")
