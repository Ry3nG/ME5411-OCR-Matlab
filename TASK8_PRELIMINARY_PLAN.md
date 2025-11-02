# Task 8 Preliminary Plan: Sensitivity Analysis

**Date**: 2025-11-02
**Status**: Planning Phase
**Goal**: Experiment with preprocessing and hyperparameter tuning to analyze sensitivity

---

## Task 8 Requirements

From the project brief:
> In carrying out Step 7, also experiment with **pre-processing of the data** (e.g., padding/resizing the input images) as well as with **hyperparameter tuning**. In your report, discuss your findings and how **sensitive your approach is to these changes**.

**Key Objectives**:
1. Test different preprocessing approaches
2. Tune hyperparameters systematically
3. Analyze sensitivity of the model to each change
4. Compare effectiveness and efficiency
5. Document findings in report

---

## Experiment Categories

### Category A: Preprocessing Experiments

#### A1. Input Image Size
**Research Question**: How does input resolution affect accuracy and training time?

**Experiments**:
- [ ] 64×64 (quarter resolution)
- [ ] 96×96 (3/4 resolution)
- [ ] 124×124 (current baseline)
- [ ] 156×156 (1.25× resolution)
- [ ] 192×192 (1.5× resolution)

**Metrics to Track**:
- Test accuracy
- Training time per epoch
- Model parameters count
- Memory usage

**Expected Insights**:
- Trade-off between resolution and computation
- Minimum resolution needed for good performance
- Diminishing returns of higher resolution

---

#### A2. Padding Strategies
**Research Question**: Does padding method affect character recognition?

**Experiments**:
- [ ] Zero padding (current)
- [ ] Edge replication padding
- [ ] Reflection padding
- [ ] Mean value padding

**Metrics to Track**:
- Per-class accuracy (especially boundary-sensitive characters)
- Overall test accuracy

**Expected Insights**:
- Impact of boundary artifacts on learning
- Which characters are most affected

---

#### A3. Normalization Methods
**Research Question**: How does normalization affect training stability and accuracy?

**Experiments**:
- [ ] No normalization (if currently used)
- [ ] Min-max normalization [0, 1]
- [ ] Z-score normalization (mean=0, std=1)
- [ ] Per-image normalization
- [ ] Global dataset normalization

**Metrics to Track**:
- Training convergence speed
- Final accuracy
- Training stability (loss variance)

**Expected Insights**:
- Best normalization for grayscale character images
- Impact on gradient flow

---

#### A4. Data Augmentation (if implemented)
**Research Question**: Does augmentation improve generalization?

**Experiments**:
- [ ] No augmentation (baseline)
- [ ] Rotation only (±15°, ±25°)
- [ ] Translation only (±5%, ±10%)
- [ ] Combined rotation + translation
- [ ] Elastic deformation
- [ ] Noise injection

**Metrics to Track**:
- Training vs test accuracy gap
- Per-class robustness
- Training time increase

**Expected Insights**:
- Does augmentation help with limited dataset?
- Which augmentation works best for characters?

---

### Category B: Network Architecture Experiments

#### B1. Convolutional Layer Depth
**Research Question**: How many conv layers are optimal?

**Experiments**:
- [ ] 2 conv layers
- [ ] 3 conv layers (current baseline)
- [ ] 4 conv layers
- [ ] 5 conv layers

**Fixed Variables**: Keep filters progression, pooling, FC layers similar

**Metrics to Track**:
- Accuracy
- Training time
- Overfitting tendency

---

#### B2. Filter Count Progression
**Research Question**: How does filter count affect capacity?

**Experiments**:
- [ ] Small: 8→16→32
- [ ] Baseline: 16→32→64 (improved design)
- [ ] Large: 32→64→128
- [ ] Very Large: 64→128→256

**Metrics to Track**:
- Accuracy improvement vs parameters
- Training time scaling
- Memory requirements

**Expected Insights**:
- Diminishing returns of more filters
- Computational vs accuracy trade-off

---

#### B3. Filter Size
**Research Question**: 3×3 vs 5×5 vs 7×7?

**Experiments**:
- [ ] All 3×3 (modern approach)
- [ ] All 5×5 (current)
- [ ] All 7×7 (traditional)
- [ ] Mixed: 7→5→3 (coarse to fine)
- [ ] Mixed: 3→3→3 (deeper with same receptive field)

**Metrics to Track**:
- Accuracy
- Parameter efficiency
- Inference speed

---

#### B4. Pooling Strategy
**Research Question**: Impact of pooling size and placement?

**Experiments**:
- [ ] No pooling (stride in conv)
- [ ] All 2×2 (current baseline)
- [ ] Progressive: 2→2→4
- [ ] Aggressive: 4→4→2
- [ ] Average pooling vs Max pooling

**Metrics to Track**:
- Spatial information retention
- Accuracy on characters with fine details

---

#### B5. Fully Connected Layer Design
**Research Question**: How much FC capacity is needed?

**Experiments**:
- [ ] Direct: 9216→7 (no hidden layer)
- [ ] Small: 9216→64→7
- [ ] Baseline: 9216→128→7
- [ ] Large: 9216→256→128→7
- [ ] Very Large: 9216→512→256→7

**Metrics to Track**:
- Accuracy
- Overfitting tendency
- Parameter count

---

### Category C: Regularization Experiments

#### C1. Dropout Rate
**Research Question**: Optimal dropout strength?

**Experiments**:
- [ ] No dropout (0.0)
- [ ] Light: 0.1
- [ ] Moderate: 0.2
- [ ] Strong: 0.3 (current)
- [ ] Very strong: 0.5
- [ ] Dropout on multiple layers

**Metrics to Track**:
- Train-test accuracy gap
- Convergence speed
- Final test accuracy

---

#### C2. L2 Regularization (Weight Decay)
**Research Question**: Does L2 help or hurt?

**Experiments**:
- [ ] No L2 (current)
- [ ] Weak: λ = 1e-5
- [ ] Moderate: λ = 1e-4, 1e-3
- [ ] Strong: λ = 1e-2
- [ ] Combined with dropout

**Metrics to Track**:
- Generalization gap
- Weight magnitude distribution

---

### Category D: Training Hyperparameters

#### D1. Learning Rate
**Research Question**: Optimal initial learning rate?

**Experiments**:
- [ ] Very low: 0.01
- [ ] Low: 0.05
- [ ] Baseline: 0.1
- [ ] High: 0.2
- [ ] Very high: 0.5

**Metrics to Track**:
- Convergence speed
- Training stability
- Final accuracy

---

#### D2. Learning Rate Schedule
**Research Question**: Which decay strategy works best?

**Experiments**:
- [ ] Constant (no decay)
- [ ] Linear decay (current)
- [ ] Exponential decay
- [ ] Step decay (drop every N epochs)
- [ ] Cosine annealing
- [ ] Warm restart

**Metrics to Track**:
- Training curve smoothness
- Final convergence
- Training time to reach target accuracy

---

#### D3. Batch Size
**Research Question**: Impact on convergence and generalization?

**Experiments**:
- [ ] Small: 32
- [ ] Medium: 64
- [ ] Baseline: 128
- [ ] Large: 256
- [ ] Very large: 512

**Metrics to Track**:
- Training time per epoch
- Generalization (small batch theory)
- Memory usage
- Convergence stability

---

#### D4. Optimizer
**Research Question**: SGD with momentum vs alternatives?

**Experiments**:
- [ ] SGD (no momentum)
- [ ] SGD + Momentum 0.9 (current)
- [ ] Different momentum: 0.8, 0.95, 0.99
- [ ] (If available) Adam, RMSprop, AdaGrad

**Note**: May need to implement other optimizers if not available

---

#### D5. Training Duration
**Research Question**: Are 30 epochs enough?

**Experiments**:
- [ ] Short: 15 epochs
- [ ] Baseline: 30 epochs
- [ ] Long: 50 epochs
- [ ] Very long: 100 epochs
- [ ] Early stopping based on validation

**Metrics to Track**:
- When does accuracy plateau?
- Signs of overfitting at later epochs
- Training time vs accuracy trade-off

---

## Recommended Experiment Priorities

### Priority 1: High Impact, Must Do
1. **Input image size** (A1) - Directly addresses task requirement
2. **Dropout rate** (C1) - Key regularization parameter
3. **Learning rate** (D1) - Most critical hyperparameter
4. **Batch size** (D3) - Affects both speed and accuracy

### Priority 2: Important, Should Do
5. **Filter count** (B2) - Understand capacity needs
6. **Learning rate schedule** (D2) - Optimization efficiency
7. **Training duration** (D5) - Efficiency analysis

### Priority 3: Nice to Have, Time Permitting
8. **Normalization** (A3)
9. **FC layer design** (B5)
10. **Filter size** (B3)

### Priority 4: Research Extensions
11. Data augmentation (A4) - if implemented
12. L2 regularization (C2)
13. Pooling strategy (B4)

---

## Experimental Framework Design

### Approach 1: One-at-a-Time (Recommended)
- Change one variable at a time from baseline
- Easy to interpret results
- Clear cause-effect relationships
- More experiments needed but clearer insights

### Approach 2: Grid Search
- Test combinations of 2-3 key hyperparameters
- More comprehensive but exponentially more experiments
- Example: LR × Batch Size × Dropout
  - 4 × 4 × 4 = 64 experiments (infeasible)

### Approach 3: Random Search
- Randomly sample from hyperparameter space
- More efficient than grid for high dimensions
- May miss optimal combinations

### Approach 4: Progressive Refinement
1. Start with Priority 1 experiments
2. Pick best configuration from each
3. Use as new baseline
4. Continue with Priority 2 experiments

**Recommendation**: Use Approach 1 (one-at-a-time) for Priority 1-2 experiments.

---

## Implementation Strategy

### Code Structure
```
src/
├── task7_1.m              (baseline CNN)
├── task7_2.m              (baseline non-CNN)
├── task8_experiment.m     (main experiment runner)
├── task8_preprocessing.m  (preprocessing variations)
├── task8_hyperparams.m    (hyperparameter sweep)
└── utils/
    ├── run_experiment.m   (generic experiment wrapper)
    ├── compare_results.m  (result comparison tool)
    └── plot_sensitivity.m (sensitivity visualization)
```

### Experiment Tracking
Each experiment should save:
- Configuration (JSON format)
- Results (accuracy, time, etc.)
- Model checkpoint
- Training curves
- Unique experiment ID

### Results Comparison
Create comparison tables/plots:
- Bar charts: accuracy vs hyperparameter
- Line plots: training curves overlay
- Heatmaps: 2D hyperparameter sensitivity
- Scatter: accuracy vs training time

---

## Report Structure for Task 8

### Section 1: Introduction
- Brief recap of baseline CNN (Task 7.1)
- Motivation for sensitivity analysis
- Overview of experiments

### Section 2: Methodology
- Description of experimental framework
- Variables tested
- Metrics used
- Fixed baseline configuration

### Section 3: Results

#### 3.1 Preprocessing Experiments
- Input size results
- Best preprocessing configuration

#### 3.2 Hyperparameter Tuning
- Learning rate sensitivity
- Batch size impact
- Regularization effects

#### 3.3 Comparative Analysis
- Which parameters are most sensitive?
- Which have minimal impact?
- Interactions between parameters

### Section 4: Discussion
- Interpretation of sensitivity patterns
- Practical recommendations
- Trade-offs (accuracy vs speed vs memory)
- Robustness analysis

### Section 5: Conclusion
- Optimal configuration found
- Key insights learned
- Applicability to similar tasks

---

## Time Estimates

Assuming each experiment takes ~20-30 minutes:

| Priority | # Experiments | Total Time |
|----------|---------------|------------|
| Priority 1 (4 groups × ~4 variants) | ~16 | 8 hours |
| Priority 2 (3 groups × ~4 variants) | ~12 | 6 hours |
| Priority 3 (3 groups × ~4 variants) | ~12 | 6 hours |
| **Feasible subset (P1 + P2)** | **~28** | **~14 hours** |

**Practical Plan**: Focus on Priority 1 experiments (~16 runs, ~8 hours compute time)

---

## Success Criteria

- [ ] At least 10 different experiments completed
- [ ] At least 3 preprocessing variations tested
- [ ] At least 3 hyperparameter variations tested
- [ ] Clear sensitivity analysis with visualizations
- [ ] Identification of most and least sensitive parameters
- [ ] Recommendations for optimal configuration
- [ ] Discussion of trade-offs in report

---

## Notes for Execution
- Start AFTER Task 7.1 improvements are complete and validated
- Use improved CNN as baseline (not original design)
- Can run experiments in parallel if multiple machines available
- Keep detailed logs for reproducibility
- Budget time for report writing (~4-6 hours)
- Some experiments may need to run overnight

---

## Questions to Answer in Report

1. How sensitive is the CNN to input image resolution?
2. What is the minimum image size that maintains >90% accuracy?
3. Does the model benefit more from architectural changes or hyperparameter tuning?
4. What is the optimal learning rate and schedule?
5. Is dropout at 0.3 optimal, or should it be adjusted?
6. How much does batch size affect convergence and final accuracy?
7. What trade-offs exist between training time and accuracy?
8. Which parameters can be set "good enough" without fine-tuning?
9. Are there any unexpected interactions between parameters?
10. What configuration would you recommend for a production deployment?

---

**Status**: Ready for execution after Task 7.1 completion
**Next Step**: Complete Task 7.1 improvements first
**Estimated Start Date**: After Task 7.1 validation
