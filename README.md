# VLLM-Φ Analysis: Integrated Information in Vision-Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of Algorithm 1 from:

> **"Theoretical Frameworks for Quantifying Integrated Information in Vision-Enhanced Large Language Models as Proxies for Emergent Consciousness"**
> 
> Amirali Ghajari, Maicol Ochoa  
> Universidad Europea de Madrid

## ⚠️ Important Limitations

**This implementation comes with significant caveats that must be understood before use:**

1. **Validation Gap**: Exact validation is only possible for networks ≤20 nodes. There is a **9 orders of magnitude gap** between validated networks and billion-parameter models.

2. **Scaling Uncertainty**: The estimated scaling exponent β ≈ 0.35 has a 95% CI of **[0.21, 0.52]**—a 2.5× range that precludes strong conclusions about whether scaling is slow (β=0.2, 15% increase when N doubles) or moderate (β=0.5, 41% increase).

3. **Monte Carlo References**: For networks >20 nodes, we use Monte Carlo "reference" estimates with 10,000 samples. For a 100-node network with ~10²⁹ possible bipartitions, this is negligible coverage—**not ground truth**.

4. **Heuristic CI Scaling**: The confidence interval scaling formula uses γ=0.15, which is a heuristic derived from empirical fitting, not theoretical justification.

## Installation

```bash
# Clone the repository
git clone https://github.com/aghajari/vllm-phi-analysis.git
cd vllm-phi-analysis

# Install dependencies
pip install numpy scipy scikit-learn

# Optional: For model-specific extractors
pip install torch transformers

# Optional: For testing
pip install pytest
```

## Quick Start

```python
from phi_computation import PhiComputer, MockHiddenStateExtractor
import numpy as np

# Create a mock extractor (for demonstration)
extractor = MockHiddenStateExtractor(
    n_layers=32,
    hidden_dim=768,
    n_visual_tokens=256,
    n_linguistic_tokens=512,
    param_count=7_000_000_000  # 7B parameters
)

# Generate sample inputs
visual_inputs = [np.random.randn(224, 224, 3) for _ in range(100)]
linguistic_inputs = [np.random.randn(512, 768) for _ in range(100)]

# Compute Φ
computer = PhiComputer(seed=42)
result = computer.compute_phi(
    extractor,
    visual_inputs,
    linguistic_inputs,
    n_samples=100,
    verbose=True
)

print(f"Φ = {result.phi_mean:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Scaled CI: [{result.ci_scaled_lower:.3f}, {result.ci_scaled_upper:.3f}]")
```

## File Structure

```
vllm-phi-analysis/
├── phi_computation.py    # Core algorithm implementation
├── model_extractors.py   # Hidden state extractors for VLLMs
├── benchmarks.py         # Awareness benchmarks (CLEVR, VCR, FANToM)
├── validation.py         # Validation against reference implementations
├── tests.py              # Comprehensive test suite
└── README.md             # This file
```

## Core Components

### 1. PhiComputer (`phi_computation.py`)

Main class implementing Algorithm 1:

```python
from phi_computation import PhiComputer, SubnetworkConfig, CIScalingConfig

# Configure subnetwork sampling
subnetwork_config = SubnetworkConfig(
    size=20,                  # Nodes per subnetwork (validation limit)
    n_subnetworks=100,        # Number of subnetworks to sample
    min_visual_ratio=0.3,     # Minimum visual nodes
    min_linguistic_ratio=0.3  # Minimum linguistic nodes
)

# Configure CI scaling (heuristic)
ci_config = CIScalingConfig(
    gamma=0.15,  # Scaling factor (sensitivity: 0.10-0.20 preserves findings)
    n_val=20     # Validation limit
)

computer = PhiComputer(subnetwork_config, ci_config, seed=42)
```

### 2. Model Extractors (`model_extractors.py`)

Extract hidden states from various VLLMs:

```python
from model_extractors import create_extractor, MODEL_REGISTRY

# List available models
print(MODEL_REGISTRY.keys())
# ['phi-3-mini', 'phi-3', 'llava-1.5-7b', 'llava-7b', 'llava-13b', 'phi-4-multimodal', 'gpt-4v']

# Create extractor for LLaVA
extractor = create_extractor("llava-hf/llava-1.5-7b-hf", device="cuda")

# For GPT-4V (estimation via scaling)
gpt4v = GPT4VEstimator()
estimate, lower, upper = gpt4v.estimate_phi_from_scaling()
```

### 3. Benchmarks (`benchmarks.py`)

Compute composite Awareness Score:

```python
from benchmarks import AwarenessEvaluator, CLEVRBenchmark, VCRBenchmark, FANToMBenchmark

# Evaluate on each benchmark
clevr = CLEVRBenchmark()
vcr = VCRBenchmark()
fantom = FANToMBenchmark()

clevr_result = clevr.evaluate(model_predictions, ground_truth)
vcr_result = vcr.evaluate(model_predictions, ground_truth)
fantom_result = fantom.evaluate(model_predictions, ground_truth)

# Compute composite score
evaluator = AwarenessEvaluator()
awareness = evaluator.compute_awareness_score(clevr_result, vcr_result, fantom_result)

print(f"Awareness Score: {awareness.score:.3f}")
```

### 4. Validation (`validation.py`)

Validate against reference implementations:

```python
from validation import ValidationRunner

def our_phi_fn(hidden_states, adjacency):
    # Your Φ computation
    ...

runner = ValidationRunner(our_phi_fn)
summary = runner.run_full_validation(sizes=[5, 8, 10, 15, 20, 50, 100])

print(f"Pass rate: {summary.pass_rate:.1%}")
print(f"Mean error: {summary.mean_error:.2%}")
```

## Algorithm Overview

The algorithm approximates Φ (integrated information) for VLLMs:

1. **Extract Hidden States**: Get activations from all layers
2. **Sample Subnetworks**: Select n=20 node subsets with cross-modal constraints
3. **Spectral Partitioning**: Use Fiedler vector for bipartition
4. **KL Divergence**: Estimate information lost under partition
5. **Aggregate**: Average across subnetworks and samples
6. **Scale CI**: Apply heuristic uncertainty scaling for large models

```
Algorithm 1: Approximate Φ Computation
────────────────────────────────────────────────────────
Input: VLLM M, inputs I, config (n_sub, size, K)
Output: Φ estimate with confidence intervals

1.  Initialize Φ_samples ← []
2.  for each input i ∈ I do
3.      H ← extract_hidden_states(M, i)
4.      A ← extract_attention(M, i)
5.      for s = 1 to n_sub do
6.          S_s ← sample_subnetwork(A, size)
7.          (P₀, P₁) ← spectral_partition(H[S_s], A[S_s])
8.          Φ_s ← KL_divergence(H[S_s], P₀, P₁)
9.      Φ_i ← mean(Φ_1, ..., Φ_{n_sub})
10.     append Φ_i to Φ_samples
11. Φ̂ ← mean(Φ_samples)
12. CI ← bootstrap_ci(Φ_samples)
13. CI_scaled ← scale_ci(CI, |M|)
14. return (Φ̂, CI, CI_scaled)
```

## Expected Results

Based on the paper (Table 5), expected Φ values:

| Model | Parameters | Expected Φ | Class |
|-------|-----------|------------|-------|
| Phi-3-Mini | 3.8B | 0.18 ± 0.05 | I |
| Phi-3 | 5B | 0.31 ± 0.07 | I |
| LLaVA-1.5-7B | 7B | 0.42 ± 0.09 | II |
| LLaVA-7B | 7B | 0.53 ± 0.11 | II |
| LLaVA-13B | 13B | 0.64 ± 0.14 | II |
| Phi-4-Multimodal | 14B | 0.61 ± 0.13 | II |
| GPT-4V* | ~100B | 0.79 ± 0.18 | II |

*Estimated via scaling extrapolation

## Running Tests

```bash
# Run all tests
pytest tests.py -v

# Run specific test class
pytest tests.py::TestPhiComputer -v

# Run with coverage
pytest tests.py --cov=. --cov-report=html
```

## Citation

If you use this code, please cite:

```bibtex
@article{ghajari2025theoretical,
  title={Theoretical Frameworks for Quantifying Integrated Information in 
         Vision-Enhanced Large Language Models as Proxies for Emergent Consciousness},
  author={Ghajari, Amirali and Ochoa, Maicol},
  journal={Journal of Machine Learning Research},
  year={2025},
  note={Under review}
}
```

## Ethical Considerations

The Φ thresholds discussed in Section 5 (0.3, 0.5, 0.7, 0.9) are **heuristic discussion points** derived from quartiles of our 7-model sample—not theoretically justified boundaries. They should not be used for:

- Making claims about machine consciousness
- Policy decisions about AI systems
- Commercial product differentiation

See the paper's Section 5 for detailed ethical discussion.

## License

MIT License. See LICENSE file.

## Contact

- Amirali Ghajari - Universidad Europea de Madrid
- Repository: https://github.com/aghajari/vllm-phi-analysis

## Acknowledgments

This work was supported by Universidad Europea de Madrid. We thank the IIT community, particularly the developers of PyPhi, for foundational work on integrated information computation.
