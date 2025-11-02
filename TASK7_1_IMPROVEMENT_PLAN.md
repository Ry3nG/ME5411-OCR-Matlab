# Task 7.1 CNN Improvement Plan

**Date**: 2025-11-02
**Status**: Ready to Execute
**Goal**: Redesign CNN architecture to achieve better classification performance

---

## 1. Current Status Analysis

### Current Architecture
```
Input: 124×124×1
├─ Conv1: 5×5×4,  ReLU, MaxPool 4×4 → 30×30×4
├─ Conv2: 5×5×8,  ReLU, MaxPool 2×2 → 13×13×8
├─ Conv3: 5×5×16, ReLU, MaxPool 3×3 → 3×3×16
├─ FC1:   144→100, ReLU, Dropout 0.2
├─ FC2:   100→50,  ReLU
└─ Output: 50→7,   Softmax
```

### Current Performance (Baseline)
- **Overall Accuracy**: 94.79%
- **Training Time**: 17.75 minutes (30 epochs)
- **Per-class Accuracy**:
  - Digits (0,4,7,8): 94.12% - 98.82% ✓ Good
  - Letters (A): 93.73% ✓ Acceptable
  - Letters (D,H): 90.20% - 90.98% ⚠️ Weaker performance

### Identified Issues

#### Issue 1: Too Few Filters ⚠️ Critical
- Current: 4 → 8 → 16 (total 28 filters)
- Problem: Insufficient feature extraction capacity for character recognition
- Evidence: Low accuracy on similar characters (D, H)
- Impact: Model cannot learn enough discriminative features

#### Issue 2: Aggressive First Pooling ⚠️ Critical
- Current: 4×4 pooling (120×120 → 30×30)
- Problem: Loses 75% spatial information immediately
- Impact: Fine details (strokes, curves) may be destroyed
- Standard practice: Use 2×2 pooling for gradual downsampling

#### Issue 3: Inefficient FC Layer Design ⚠️ Moderate
- Current: 144 → 100 → 50
- Problem: Expanding from 144 to 100 serves no purpose
- Standard practice: Monotonically decreasing dimensions
- Adds unnecessary parameters without benefit

#### Issue 4: Suboptimal Regularization ⚠️ Minor
- Current: Dropout 0.2 only on FC1
- Could be stronger to prevent overfitting

---

## 2. Improved Architecture Design

### Proposed Architecture
```
Input: 124×124×1
├─ Conv1: 5×5×16, ReLU, MaxPool 2×2 → 60×60×16
├─ Conv2: 5×5×32, ReLU, MaxPool 2×2 → 28×28×32
├─ Conv3: 5×5×64, ReLU, MaxPool 2×2 → 12×12×64
├─ Flatten: 12×12×64 = 9216
├─ FC1:   9216→128, ReLU, Dropout 0.3
└─ Output: 128→7,   Softmax
```

### Key Improvements

| Aspect | Before | After | Rationale |
|--------|--------|-------|-----------|
| Filters | 4→8→16 | 16→32→64 | 4× capacity, stronger feature learning |
| Pool1 | 4×4 | 2×2 | Preserve spatial details |
| Pool2 | 2×2 | 2×2 | Keep consistent |
| Pool3 | 3×3 | 2×2 | Standard uniform pooling |
| FC layers | 144→100→50 | 9216→128 | Cleaner design, direct projection |
| Dropout | 0.2 | 0.3 | Stronger regularization |

### Expected Benefits
- ✓ Better feature extraction (especially for D, H characters)
- ✓ Preserve stroke details with gentle pooling
- ✓ Simpler, cleaner architecture
- ✓ More robust regularization
- ⚠️ Slightly longer training time (estimated 25-30 min)

---

## 3. Code Modification Checklist

### File: `src/task7_1.m`

#### Modification 1: Update CNN Architecture (Lines 69-77)
```matlab
% OLD
cnn.layers = {
    struct('type', 'input')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 4, 'poolDim', 4, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 8, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 3, 'actiFunc', 'relu')
    struct('type', 'Linear', 'hiddenUnits', 100, 'actiFunc', 'relu', 'dropout', 0.2)
    struct('type', 'Linear', 'hiddenUnits', 50, 'actiFunc', 'relu')
    struct('type', 'output', 'softmax', 1)
};

% NEW
cnn.layers = {
    struct('type', 'input')  % Input layer: 124x124x1
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 16, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 32, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Conv2D', 'filterDim', 5, 'numFilters', 64, 'poolDim', 2, 'actiFunc', 'relu')
    struct('type', 'Linear', 'hiddenUnits', 128, 'actiFunc', 'relu', 'dropout', 0.3)
    struct('type', 'output', 'softmax', 1)
};
```

#### Modification 2: Update Architecture Print (Lines 79-86)
```matlab
% NEW
fprintf('CNN Architecture:\n');
fprintf('  Input: 124x124x1\n');
fprintf('  Conv1: 5x5x16, ReLU, MaxPool 2x2 -> 60x60x16\n');
fprintf('  Conv2: 5x5x32, ReLU, MaxPool 2x2 -> 28x28x32\n');
fprintf('  Conv3: 5x5x64, ReLU, MaxPool 2x2 -> 12x12x64\n');
fprintf('  FC1: 9216 -> 128, ReLU, Dropout(0.3)\n');
fprintf('  Output: 128 -> 7, Softmax\n\n');
```

#### Modification 3: Update Training Configuration (Optional)
Consider adjusting hyperparameters due to increased model capacity:
- Keep epochs: 30 (start conservatively)
- Keep batch size: 128
- Keep learning rate: 0.1 → 1e-5 (linear decay)
- Monitor for overfitting with new dropout 0.3

**Note**: If initial results show underfitting, may increase epochs to 40-50.

---

## 4. Execution Plan

### Phase 1: Code Modification (5 min)
- [ ] Backup current `task7_1.m` (copy to `task7_1_v1_old.m`)
- [ ] Apply Modification 1: Update CNN layers
- [ ] Apply Modification 2: Update print statements
- [ ] Verify code syntax (no errors)

### Phase 2: Training (25-35 min estimated)
- [ ] Run training: `matlab -batch "run('src/task7_1.m')"`
- [ ] Monitor output for:
  - Training accuracy progression
  - Validation accuracy
  - Loss convergence
  - Any warnings/errors
- [ ] Save timestamp of training run for reference

### Phase 3: Results Analysis (10 min)
- [ ] Read `output/task7_1/[timestamp]/results.txt`
- [ ] Compare with baseline:
  - Overall accuracy (target: >95%, baseline: 94.79%)
  - Per-class accuracy (focus on D, H improvement)
  - Training time
- [ ] Check training curves if saved
- [ ] Analyze confusion patterns

### Phase 4: Generate Figures (15 min)
Need to generate/update figures for report:
- [ ] Model architecture diagram
- [ ] Training/validation accuracy curves
- [ ] Training/validation loss curves
- [ ] Confusion matrix
- [ ] Per-class accuracy bar chart
- [ ] Sample predictions (correct and incorrect)

**Figure Save Locations**:
- Output: `/output/task7_1/[timestamp]/`
- Report: `/ME5411-Project-Report/figs/task7_1/`

### Phase 5: Report Updates (30 min)

#### Update 1: Method Section
File: `/ME5411-Project-Report/task7.tex` (or similar)

**Update CNN Architecture Description**:
```latex
% OLD architecture description needs to be replaced
% NEW description:
The CNN consists of three convolutional blocks followed by
a fully-connected classifier. Each convolutional block contains:
\begin{itemize}
    \item 5×5 convolutional layer with ReLU activation
    \item 2×2 max pooling with stride 2
\end{itemize}

The number of filters increases progressively: 16 → 32 → 64,
allowing the network to learn increasingly complex features.
After the third pooling layer, the 12×12×64 feature maps are
flattened into a 9216-dimensional vector, which is fed into
a 128-unit fully-connected layer with 30\% dropout for
regularization. The final softmax layer produces class
probabilities for the 7 character classes.
```

**Update Architecture Table/Figure**:
- Replace old architecture specifications
- Update layer dimensions
- Update parameter counts

#### Update 2: Results Section
**Update Performance Metrics**:
```latex
The improved CNN achieved an overall test accuracy of
XX.XX\% on the validation set, representing a X.XX percentage
point improvement over the baseline design.

Per-class accuracy results are shown in Table~\ref{tab:task7_1_results}.
Notable improvements were observed for characters D and H, which
increased from 90.20\% and 90.98\% to XX.XX\% and XX.XX\%
respectively, demonstrating the benefit of increased model
capacity and preserved spatial resolution.
```

**Update Figures**:
- Figure: CNN architecture diagram → Update layer specifications
- Figure: Accuracy curves → Replace with new training results
- Table: Per-class accuracy → Update all numbers

#### Update 3: Discussion Section
**Add Architecture Design Rationale**:
```latex
The CNN architecture was designed with several key considerations:

\textbf{Filter Progression (16→32→64):} Progressive increase in
filter count allows the network to learn hierarchical features,
from simple edges in early layers to complex character-specific
patterns in deeper layers.

\textbf{Uniform 2×2 Pooling:} Consistent pooling strategy provides
gradual spatial downsampling, preserving important stroke details
while reducing computational cost. Earlier designs with 4×4 pooling
were found to lose critical fine-grained information.

\textbf{Single FC Layer:} A single 128-unit fully-connected layer
with dropout provides sufficient classification capacity while
avoiding unnecessary complexity. This design choice balances
model expressiveness with training efficiency.
```

**Update Findings**:
- Add comparison with baseline architecture
- Discuss which classes improved and why
- Mention training time trade-offs
- Note any failure cases

#### Update 4: Algorithm Pseudocode (if present)
- Update layer dimensions in forward pass
- Update backpropagation dimensions if shown

---

## 5. Success Criteria

### Minimum Acceptable Results
- [ ] Overall accuracy ≥ 95% (improvement of 0.21%)
- [ ] D class accuracy ≥ 93% (improvement of 2.8%)
- [ ] H class accuracy ≥ 93% (improvement of 2%)
- [ ] Training time ≤ 40 minutes (less than 2× baseline)
- [ ] No degradation in other classes (maintain ≥94%)

### Target Results
- [ ] Overall accuracy ≥ 96%
- [ ] All classes ≥ 94%
- [ ] Training time ≤ 30 minutes

### If Results Are Poor
**Fallback Plan**:
1. Check for implementation bugs in CNN layers
2. Try training for more epochs (40-50)
3. Adjust learning rate (try 0.05 or 0.15)
4. Verify data preprocessing is correct
5. Consider intermediate architecture (8→16→32 filters)

---

## 6. Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Code modification | 5 min | 5 min |
| Training | 30 min | 35 min |
| Results analysis | 10 min | 45 min |
| Generate figures | 15 min | 60 min |
| Update report | 30 min | 90 min |
| **Total** | **90 min** | **1.5 hours** |

---

## 7. Deliverables Checklist

- [ ] Modified `src/task7_1.m` with new architecture
- [ ] Training output in `output/task7_1/[timestamp]/`
- [ ] Updated figures in `ME5411-Project-Report/figs/task7_1/`
- [ ] Updated report sections (method, results, discussion)
- [ ] Performance comparison table (old vs new)
- [ ] This document updated with actual results

---

## 8. Post-Execution: Record Actual Results

**Actual Training Time**: ___ minutes
**Overall Accuracy**: ____%
**Improvement over Baseline**: +____%

**Per-class Results**:
| Class | Baseline | New | Improvement |
|-------|----------|-----|-------------|
| 0 | 98.82% | ___% | ___% |
| 4 | 97.25% | ___% | ___% |
| 7 | 98.43% | ___% | ___% |
| 8 | 94.12% | ___% | ___% |
| A | 93.73% | ___% | ___% |
| D | 90.20% | ___% | ___% |
| H | 90.98% | ___% | ___% |

**Key Observations**:
-
-
-

**Unexpected Issues**:
-
-

---

## Notes
- Keep baseline model saved for comparison in Task 8
- Document any hyperparameter adjustments made during training
- If accuracy is exceptionally high (>98%), consider if there's data leakage
- Monitor for overfitting: if training acc >> test acc, may need more regularization
