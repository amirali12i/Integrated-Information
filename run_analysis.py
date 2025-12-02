"""
Main Analysis Pipeline for VLLM Φ Research

This script orchestrates the complete analysis pipeline:
1. Model loading and hidden state extraction
2. Φ computation with uncertainty quantification
3. Benchmark evaluation (CLEVR, VCR, FANToM)
4. Scaling law analysis
5. Results visualization and export

Usage:
    python run_analysis.py --model llava-7b --n_samples 1000
    python run_analysis.py --all_models --output results/
    python run_analysis.py --demo  # Run with mock data

Authors: Amirali Ghajari, Maicol Ochoa
Universidad Europea de Madrid
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Local imports
from phi_computation import (
    PhiComputer, PhiResult, SubnetworkConfig, CIScalingConfig,
    MockHiddenStateExtractor, compute_geometric_phi
)
from model_extractors import (
    create_extractor, MODEL_REGISTRY, get_model_info,
    LLaVAExtractor, Phi3Extractor, GPT4VEstimator
)
from benchmarks import (
    CLEVRBenchmark, VCRBenchmark, FANToMBenchmark,
    AwarenessEvaluator, AwarenessScore, BenchmarkResult,
    compute_phi_awareness_correlation, PAPER_RESULTS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Complete analysis pipeline for VLLM Φ research."""
    
    def __init__(self, 
                 output_dir: str = "results",
                 seed: int = 42,
                 device: str = "cuda"):
        """Initialize pipeline.
        
        Args:
            output_dir: Directory for results
            seed: Random seed for reproducibility
            device: Compute device ('cuda' or 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = device
        
        np.random.seed(seed)
        
        # Initialize components
        self.phi_computer = PhiComputer(
            subnetwork_config=SubnetworkConfig(
                size=20,
                n_subnetworks=100,
                min_visual_ratio=0.3,
                min_linguistic_ratio=0.3
            ),
            ci_config=CIScalingConfig(gamma=0.15, n_val=20),
            seed=seed
        )
        
        self.awareness_evaluator = AwarenessEvaluator()
        
        self.results = {}
        
    def analyze_model(self,
                     model_key: str,
                     visual_inputs: List[np.ndarray],
                     linguistic_inputs: List[np.ndarray],
                     clevr_data: Optional[Dict] = None,
                     vcr_data: Optional[Dict] = None,
                     fantom_data: Optional[Dict] = None,
                     n_phi_samples: int = 1000) -> Dict[str, Any]:
        """Run complete analysis for a single model.
        
        Args:
            model_key: Key in MODEL_REGISTRY
            visual_inputs: Visual inputs for Φ computation
            linguistic_inputs: Linguistic inputs
            clevr_data: CLEVR evaluation data
            vcr_data: VCR evaluation data
            fantom_data: FANToM evaluation data
            n_phi_samples: Number of samples for Φ estimation
            
        Returns:
            Dictionary with all results
        """
        logger.info(f"Analyzing model: {model_key}")
        
        model_info = get_model_info(model_key)
        
        # Create extractor
        try:
            extractor = create_extractor(
                model_info['name'],
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Could not load model {model_key}: {e}")
            logger.info("Using mock extractor")
            extractor = MockHiddenStateExtractor(
                param_count=model_info['params']
            )
            
        # Compute Φ
        logger.info("Computing Φ...")
        phi_result = self.phi_computer.compute_phi(
            extractor,
            visual_inputs,
            linguistic_inputs,
            n_samples=n_phi_samples
        )
        
        # Compute layer-wise Φ
        logger.info("Computing layer-wise Φ...")
        layer_phi = self.phi_computer.compute_layer_wise_phi(
            extractor,
            visual_inputs[0],
            linguistic_inputs[0]
        )
        
        # Compute geometric Φ
        logger.info("Computing geometric Φ...")
        extracted = extractor.extract(visual_inputs[0], linguistic_inputs[0])
        phi_geom = compute_geometric_phi(extracted['hidden_states'][-1])
        
        # Benchmark evaluations
        benchmark_results = {}
        
        if clevr_data:
            logger.info("Evaluating on CLEVR...")
            clevr_bench = CLEVRBenchmark()
            benchmark_results['CLEVR'] = clevr_bench.evaluate(
                clevr_data['predictions'],
                clevr_data['ground_truth']
            )
            
        if vcr_data:
            logger.info("Evaluating on VCR...")
            vcr_bench = VCRBenchmark()
            benchmark_results['VCR'] = vcr_bench.evaluate(
                vcr_data['predictions'],
                vcr_data['ground_truth']
            )
            
        if fantom_data:
            logger.info("Evaluating on FANToM...")
            fantom_bench = FANToMBenchmark()
            benchmark_results['FANToM'] = fantom_bench.evaluate(
                fantom_data['predictions'],
                fantom_data['ground_truth']
            )
            
        # Compute awareness score if all benchmarks available
        awareness_score = None
        if len(benchmark_results) == 3:
            awareness_score = self.awareness_evaluator.compute_awareness_score(
                benchmark_results['CLEVR'],
                benchmark_results['VCR'],
                benchmark_results['FANToM']
            )
            
        results = {
            'model_key': model_key,
            'model_name': model_info['name'],
            'param_count': model_info['params'],
            'phi': {
                'mean': phi_result.phi_mean,
                'ci_lower': phi_result.ci_lower,
                'ci_upper': phi_result.ci_upper,
                'ci_scaled_lower': phi_result.ci_scaled_lower,
                'ci_scaled_upper': phi_result.ci_scaled_upper,
                'n_samples': phi_result.n_samples
            },
            'phi_geometric': phi_geom,
            'layer_phi': layer_phi,
            'peak_layer': max(layer_phi, key=layer_phi.get),
            'benchmarks': {
                name: {
                    'raw_score': r.raw_score,
                    'normalized_score': r.normalized_score,
                    'ci_lower': r.ci_lower,
                    'ci_upper': r.ci_upper
                }
                for name, r in benchmark_results.items()
            },
            'awareness_score': {
                'score': awareness_score.score if awareness_score else None,
                'ci_lower': awareness_score.ci_lower if awareness_score else None,
                'ci_upper': awareness_score.ci_upper if awareness_score else None
            } if awareness_score else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results[model_key] = results
        return results
        
    def analyze_scaling(self) -> Dict[str, Any]:
        """Analyze scaling law from collected results.
        
        Returns:
            Dictionary with scaling analysis
        """
        if len(self.results) < 3:
            logger.warning("Need at least 3 models for scaling analysis")
            return {}
            
        params = []
        phis = []
        
        for key, result in self.results.items():
            params.append(result['param_count'])
            phis.append(result['phi']['mean'])
            
        params = np.array(params)
        phis = np.array(phis)
        
        # Fit power law: Φ = α * N^β
        log_params = np.log10(params)
        log_phis = np.log10(phis + 1e-10)  # Avoid log(0)
        
        # Linear regression in log space
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, log_phis)
        
        beta = slope
        alpha = 10 ** intercept
        
        # Bootstrap CI for β
        n = len(params)
        beta_samples = []
        for _ in range(10000):
            indices = np.random.choice(n, size=n, replace=True)
            boot_slope, _, _, _, _ = stats.linregress(
                log_params[indices], log_phis[indices]
            )
            beta_samples.append(boot_slope)
            
        beta_ci_lower = np.percentile(beta_samples, 2.5)
        beta_ci_upper = np.percentile(beta_samples, 97.5)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'beta_ci_lower': beta_ci_lower,
            'beta_ci_upper': beta_ci_upper,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'n_models': n,
            'formula': f'Φ = {alpha:.2e} × N^{beta:.2f}'
        }
        
    def run_demo(self) -> Dict[str, Any]:
        """Run demo analysis with mock data.
        
        Returns:
            Demo results
        """
        logger.info("Running demo analysis with mock data...")
        
        # Create mock inputs
        n_samples = 50
        visual_inputs = [np.random.randn(224, 224, 3) for _ in range(n_samples)]
        linguistic_inputs = [np.random.randn(512, 768) for _ in range(n_samples)]
        
        # Create mock benchmark data
        np.random.seed(self.seed)
        
        clevr_data = {
            'predictions': ['yes' if np.random.rand() > 0.25 else 'no' for _ in range(100)],
            'ground_truth': ['yes' if np.random.rand() > 0.5 else 'no' for _ in range(100)]
        }
        
        vcr_data = {
            'predictions': [np.random.randint(0, 4) for _ in range(100)],
            'ground_truth': [np.random.randint(0, 4) for _ in range(100)]
        }
        # Bias toward correct
        vcr_data['predictions'] = [
            t if np.random.rand() > 0.35 else p 
            for p, t in zip(vcr_data['predictions'], vcr_data['ground_truth'])
        ]
        
        fantom_data = {
            'predictions': [np.random.rand() > 0.35 for _ in range(100)],
            'ground_truth': [np.random.rand() > 0.5 for _ in range(100)]
        }
        fantom_data['predictions'] = [
            t if np.random.rand() > 0.3 else p
            for p, t in zip(fantom_data['predictions'], fantom_data['ground_truth'])
        ]
        
        # Analyze mock model
        results = self.analyze_model(
            'llava-7b',
            visual_inputs,
            linguistic_inputs,
            clevr_data=clevr_data,
            vcr_data=vcr_data,
            fantom_data=fantom_data,
            n_phi_samples=50
        )
        
        return results
        
    def export_results(self, filename: str = "results.json"):
        """Export all results to JSON.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
            
        export_data = {
            'results': convert(self.results),
            'scaling_analysis': convert(self.analyze_scaling()) if len(self.results) >= 3 else None,
            'metadata': {
                'seed': self.seed,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Results exported to {output_path}")
        
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        for model_key, results in self.results.items():
            print(f"\n{model_key.upper()}")
            print("-" * 40)
            print(f"Parameters: {results['param_count']:,}")
            print(f"Φ: {results['phi']['mean']:.3f} "
                  f"[{results['phi']['ci_lower']:.3f}, {results['phi']['ci_upper']:.3f}]")
            print(f"Φ_G (geometric): {results['phi_geometric']:.3f}")
            print(f"Peak layer: {results['peak_layer']}")
            
            if results['benchmarks']:
                print("\nBenchmark Scores:")
                for name, scores in results['benchmarks'].items():
                    print(f"  {name}: {scores['normalized_score']:.3f} "
                          f"(raw: {scores['raw_score']:.3f})")
                          
            if results['awareness_score']:
                print(f"\nAwareness Score: {results['awareness_score']['score']:.3f}")
                
        if len(self.results) >= 3:
            scaling = self.analyze_scaling()
            print("\n" + "=" * 70)
            print("SCALING ANALYSIS")
            print("=" * 70)
            print(f"Formula: {scaling['formula']}")
            print(f"β = {scaling['beta']:.3f} [{scaling['beta_ci_lower']:.3f}, {scaling['beta_ci_upper']:.3f}]")
            print(f"R² = {scaling['r_squared']:.3f}")


def reproduce_paper_table5():
    """Reproduce Table 5 from the paper (expected values)."""
    print("\n" + "=" * 70)
    print("TABLE 5: Summary of Experimental Results (Expected Values)")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'Φ':>8} {'|Φ_G|':>8} {'Awareness':>10}")
    print("-" * 70)
    
    expected_results = [
        ('Phi-3-Mini', '3.8B', 0.18, 0.15, 0.42),
        ('Phi-3', '5B', 0.31, 0.27, 0.48),
        ('LLaVA-1.5-7B', '7B', 0.42, 0.38, 0.58),
        ('LLaVA-7B', '7B', 0.53, 0.49, 0.67),
        ('LLaVA-13B', '13B', 0.64, 0.61, 0.74),
        ('Phi-4-MM', '14B', 0.61, 0.58, 0.71),
        ('GPT-4V*', '~100B', 0.79, 0.75, 0.88),
    ]
    
    for model, params, phi, phi_g, awareness in expected_results:
        print(f"{model:<20} {params:>12} {phi:>8.2f} {phi_g:>8.2f} {awareness:>10.2f}")
        
    print("-" * 70)
    print("* GPT-4V values are estimates (±0.10)")
    print("\nNote: Φ computed with CI scaling factor γ=0.15")
    print("Bootstrap 95% CI for β: [0.21, 0.52]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLLM Φ Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py --demo
    python run_analysis.py --model llava-7b --n_samples 1000
    python run_analysis.py --reproduce_table5
        """
    )
    
    parser.add_argument('--model', type=str, 
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model to analyze')
    parser.add_argument('--all_models', action='store_true',
                       help='Analyze all registered models')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with mock data')
    parser.add_argument('--reproduce_table5', action='store_true',
                       help='Print expected Table 5 values')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for Φ computation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Compute device')
    
    args = parser.parse_args()
    
    if args.reproduce_table5:
        reproduce_paper_table5()
        return
        
    pipeline = AnalysisPipeline(
        output_dir=args.output,
        seed=args.seed,
        device=args.device
    )
    
    if args.demo:
        results = pipeline.run_demo()
        pipeline.print_summary()
        pipeline.export_results('demo_results.json')
        
    elif args.model:
        # Would need actual data here
        logger.error("Full analysis requires visual and benchmark data. Use --demo for demonstration.")
        
    elif args.all_models:
        logger.error("Full analysis requires visual and benchmark data. Use --demo for demonstration.")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
