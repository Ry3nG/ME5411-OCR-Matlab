# Task 8 Implementation Plan: Sensitivity Analysis

**Last Updated**: 2025-11-03
**Status**: Ready for Execution
**Goal**: Systematic ablation study on preprocessing and hyperparameters

---

## Task 8 Requirements

From ME5411_CA_2025.pdf:
> In carrying out Step 7, also experiment with **pre-processing of the data** (e.g., padding/resizing the input images) as well as with **hyperparameter tuning**. In your report, discuss your findings and how **sensitive your approach is to these changes**.

**Key Deliverables**:
1. Ablation experiments on preprocessing (data augmentation)
2. Ablation experiments on critical hyperparameters
3. Sensitivity analysis with clear visualizations
4. Concise report section (4-5 pages) following project guidelines

---

## Baseline Configuration (Task 7.1)

```
Architecture:
  Input: 124×124×1
  Conv1: 5×5×16, ReLU, MaxPool 2×2 → 60×60×16
  Conv2: 5×5×32, ReLU, MaxPool 2×2 → 28×28×32
  Conv3: 5×5×64, ReLU, MaxPool 2×2 → 12×12×64
  FC1: 9216 → 128, ReLU, Dropout(0.3)
  Output: 128 → 7, Softmax

Hyperparameters:
  - Image size: 124×124
  - Epochs: 30
  - Batch size: 128
  - Learning rate: 0.1 → 1e-5 (linear decay)
  - Momentum: 0.9
  - Dropout: 0.3 (FC layer only)
  - L2 regularization: disabled
  - Data augmentation: disabled

Performance (expected):
  - Test accuracy: ~XX% (to be measured)
  - Training time: ~25-30 min/run
```

---

## Ablation Study Design

### Approach: One-Factor-at-a-Time (OFAT)
- Change **one variable** at a time from baseline
- Clear cause-effect relationships
- Easy to interpret and visualize
- **Baseline (exp00) included** - saves to output/task7_1 directory
- Total: **13 experiments, ~5.5 hours compute time**

---

## Part A: Preprocessing Ablations (5 experiments)

**Research Question**: How does data augmentation affect generalization?

### Experiment Group A: Data Augmentation

| Exp ID | Name | Translation | Rotation | Scale | Description |
|--------|------|-------------|----------|-------|-------------|
| **exp00** | baseline | None | None | None | No augmentation (baseline) |
| **exp01** | translation | ±2% | None | None | Translation only |
| **exp02** | rotation | None | ±15° | None | Rotation only |
| **exp03** | scale | None | None | 0.9-1.1× | Scale only |
| **exp04** | all_aug | ±2% | ±15° | 0.9-1.1× | All combined |

**Configuration Details**:
```matlab
% All experiments use probability = 0.5 for applying transforms

% exp01: Translation only
random_trans.prob = 0.5;
random_trans.trans_ratio = 0.02;     % ±2% shift
random_trans.rot_range = [0 0];
random_trans.scale_ratio = [1 1];

% exp02: Rotation only
random_trans.prob = 0.5;
random_trans.trans_ratio = 0;
random_trans.rot_range = [-15 15];   % ±15 degrees
random_trans.scale_ratio = [1 1];

% exp03: Scale only
random_trans.prob = 0.5;
random_trans.trans_ratio = 0;
random_trans.rot_range = [0 0];
random_trans.scale_ratio = [0.9 1.1]; % 90%-110%

% exp04: All combined
random_trans.prob = 0.5;
random_trans.trans_ratio = 0.02;
random_trans.rot_range = [-15 15];
random_trans.scale_ratio = [0.9 1.1];
```

**Fixed Variables**: img_dim=124, lr=0.1, lr_method='linear', batch_size=128, dropout=0.3

**Metrics to Track**:
- Test accuracy (primary)
- Train-test accuracy gap (overfitting indicator)
- Training time
- Per-class accuracy (identify which classes benefit most)

**Expected Insights**:
1. Does augmentation improve generalization? (compare exp00 vs exp04)
2. Which augmentation is most effective? (compare exp01, 02, 03)
3. Is there diminishing return or over-augmentation? (exp04 vs best individual)

---

## Part B: Hyperparameter Ablations (8 experiments)

**Research Question**: Which hyperparameters are most sensitive?

### Experiment Group B1: Learning Rate Magnitude (3 experiments)

| Exp ID | Name | LR | LR Schedule | Batch Size | Other |
|--------|------|----|-------------|------------|-------|
| **exp00** | baseline | 0.1 | linear | 128 | (reference) |
| **exp05** | lr0.05 | 0.05 | linear | 128 | Half LR |
| **exp06** | lr0.2 | 0.2 | linear | 128 | Double LR |
| **exp07** | lr0.3 | 0.3 | linear | 128 | Triple LR |

**Fixed Variables**: no augmentation, img_dim=124, dropout=0.3

**Expected Insights**:
- Optimal learning rate for convergence speed and final accuracy
- Too low: slow convergence
- Too high: unstable training or divergence

---

### Experiment Group B2: Learning Rate Schedule (3 experiments)

| Exp ID | Name | LR | LR Schedule | Details |
|--------|------|----|-------------|---------|
| **exp00** | baseline | 0.1 | linear | 0.1 → 1e-5 over 30 epochs |
| **exp08** | cosine | 0.1 | cosine | Cosine annealing |
| **exp09** | exp_decay | 0.1 | exponential | γ=0.95 per epoch |
| **exp10** | step_decay | 0.1 | step | ×0.1 every 10 epochs |

**Configuration Details**:
```matlab
% exp08: Cosine annealing
options.lr_method = 'cosine';
options.lr_max = 0.1;
options.lr_min = 1e-5;
% LR = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / total_epochs))

% exp09: Exponential decay
options.lr_method = 'exp';
options.lr_decay_rate = 0.95;  % LR = LR * 0.95 per epoch

% exp10: Step decay
options.lr_method = 'step';
options.lr_step_size = 10;     % Every 10 epochs
options.lr_gamma = 0.1;        % Multiply by 0.1
```

**Fixed Variables**: no augmentation, img_dim=124, batch_size=128, dropout=0.3

**Expected Insights**:
- Which schedule achieves best final accuracy?
- Training curve smoothness and stability
- Time to reach target accuracy (e.g., 90%)

---

### Experiment Group B3: Batch Size (2 experiments)

| Exp ID | Name | Batch Size | LR | Note |
|--------|------|------------|-----|------|
| **exp00** | baseline | 128 | 0.1 | (reference) |
| **exp11** | bs64 | 64 | 0.1 | Smaller batch |
| **exp12** | bs256 | 256 | 0.1 | Larger batch |

**Fixed Variables**: no augmentation, img_dim=124, lr=0.1, lr_method='linear', dropout=0.3

**Expected Insights**:
- Small batch: better generalization? slower training?
- Large batch: faster epoch? worse generalization?
- Memory and time trade-offs

---

## Experiment Summary Table

| Group | Exp ID | Name | Aug | LR | LR Schedule | BS | Time (est) |
|-------|--------|------|-----|----|-----------|----|------------|
| **A** | exp00 | baseline | none | 0.1 | linear | 128 | 25 min |
| **A** | exp01 | translation | trans | 0.1 | linear | 128 | 25 min |
| **A** | exp02 | rotation | rot | 0.1 | linear | 128 | 25 min |
| **A** | exp03 | scale | scale | 0.1 | linear | 128 | 25 min |
| **A** | exp04 | all_aug | all | 0.1 | linear | 128 | 25 min |
| **B1** | exp05 | lr0.05 | none | 0.05 | linear | 128 | 25 min |
| **B1** | exp06 | lr0.2 | none | 0.2 | linear | 128 | 25 min |
| **B1** | exp07 | lr0.3 | none | 0.3 | linear | 128 | 25 min |
| **B2** | exp08 | cosine | none | 0.1 | cosine | 128 | 25 min |
| **B2** | exp09 | exp_decay | none | 0.1 | exp | 128 | 25 min |
| **B2** | exp10 | step_decay | none | 0.1 | step | 128 | 25 min |
| **B3** | exp11 | bs64 | none | 0.1 | linear | 64 | 30 min |
| **B3** | exp12 | bs256 | none | 0.1 | linear | 256 | 20 min |
| | | | | | **Total** | | **~5.5 hrs** |

**Note**: exp00 (baseline) saves to `output/task7_1/` instead of `output/task8/exp00/` for consistency.

---

## Execution Strategy

### Phase 1: Complete Run (all 13 experiments)
Run exp00-12 in sequence
- **Time**: ~5.5 hours total
- **exp00** saves to output/task7_1 directory
- **exp01-12** save to output/task8/exp##/ directories

### Execution Priority:
1. **Baseline first** (exp00) - establishes reference performance
2. **Preprocessing experiments** (exp01-04) - directly addresses project requirement
3. **Hyperparameter tuning** (exp05-12) - identifies sensitive parameters

### Execution Options:
1. **Sequential**: Run experiments one by one (easier to monitor)
2. **Overnight batch**: Queue all 13 experiments to run overnight (recommended)
3. **Parallel** (if multiple GPUs/machines): Run groups in parallel

---

## Implementation

### Code Structure
```
src/
├── task7_1.m                    # Baseline CNN (existing)
├── task8_run_experiments.m      # Main experiment runner (NEW)
├── task8_single_experiment.m    # Single experiment wrapper (NEW)
└── utils/
    ├── plot_task8_results.m     # Visualization script (NEW)
    └── compare_experiments.m    # Results comparison (NEW)
```

### Experiment Runner Template
```matlab
% task8_run_experiments.m
% Note: exp00 (baseline) saves to output/task7_1 instead of output/task8/exp00

experiments = {
    % [id, name, aug_trans, aug_rot, aug_scale, lr, lr_method, batch_size]
    {'exp00', 'baseline',     0,    [0 0],      [1 1],     0.1,  'linear', 128}
    {'exp01', 'translation',  0.02, [0 0],      [1 1],     0.1,  'linear', 128}
    {'exp02', 'rotation',     0,    [-15 15],   [1 1],     0.1,  'linear', 128}
    {'exp03', 'scale',        0,    [0 0],      [0.9 1.1], 0.1,  'linear', 128}
    {'exp04', 'all_aug',      0.02, [-15 15],   [0.9 1.1], 0.1,  'linear', 128}
    {'exp05', 'lr0.05',       0,    [0 0],      [1 1],     0.05, 'linear', 128}
    {'exp06', 'lr0.2',        0,    [0 0],      [1 1],     0.2,  'linear', 128}
    {'exp07', 'lr0.3',        0,    [0 0],      [1 1],     0.3,  'linear', 128}
    {'exp08', 'cosine',       0,    [0 0],      [1 1],     0.1,  'cosine', 128}
    {'exp09', 'exp_decay',    0,    [0 0],      [1 1],     0.1,  'exp',    128}
    {'exp10', 'step_decay',   0,    [0 0],      [1 1],     0.1,  'step',   128}
    {'exp11', 'bs64',         0,    [0 0],      [1 1],     0.1,  'linear', 64}
    {'exp12', 'bs256',        0,    [0 0],      [1 1],     0.1,  'linear', 256}
};

for i = 1:length(experiments)
    fprintf('Running %s (%d/%d)...\n', experiments{i}{2}, i, length(experiments));
    task8_single_experiment(experiments{i});
end
```

### Data Storage Format
- **exp00** saves to: `output/task7_1/[timestamp]/`
- **exp01-12** save to: `output/task8/[exp_id]/`

Each experiment directory contains:
output/task8/exp00/
├── cnn.mat                 # Trained model
├── predictions.mat         # Test predictions
├── hyper_params.json       # Configuration
├── results.txt             # Text summary
├── training_curve.png      # Training/test accuracy plot
└── confusion_matrix.png    # Per-class results
```

### Results Aggregation
Create `output/task8/summary.mat` containing:
```matlab
summary = struct();
summary.exp_ids = {'exp00', 'exp01', ...};
summary.test_acc = [0.XX, 0.XX, ...];
summary.train_acc = [0.XX, 0.XX, ...];
summary.train_time = [1500, 1520, ...];  % seconds
summary.configs = {...};  % Full config structs
```

---

## Visualization Plan

### Figure 1: Data Augmentation Comparison
- **Type**: Bar chart
- **X-axis**: Augmentation method (none, trans, rot, scale, all)
- **Y-axis**: Test accuracy
- **Error bars**: Train-test gap (optional)
- **Insight**: Which augmentation helps most?

### Figure 2: Training Curves - Augmentation
- **Type**: Line plot overlay
- **Lines**: exp00 (none) vs exp04 (all)
- **X-axis**: Epoch
- **Y-axis**: Accuracy (train and test)
- **Insight**: Does augmentation reduce overfitting?

### Figure 3: Learning Rate Sensitivity
- **Type**: Bar chart + line plot
- **X-axis**: LR value (0.05, 0.1, 0.2, 0.3)
- **Y-axis**: Test accuracy
- **Secondary**: Training time
- **Insight**: Optimal LR and speed trade-off

### Figure 4: Learning Rate Schedule Comparison
- **Type**: Training curves overlay
- **Lines**: linear (baseline), cosine, exp, step
- **X-axis**: Epoch
- **Y-axis**: Test accuracy
- **Insight**: Which schedule converges best?

### Figure 5: Batch Size Effect
- **Type**: Scatter plot
- **X-axis**: Batch size (64, 128, 256)
- **Y-axis**: Test accuracy
- **Color/size**: Training time per epoch
- **Insight**: Speed vs generalization trade-off

### Figure 6: Overall Sensitivity Summary
- **Type**: Horizontal bar chart
- **Y-axis**: Parameter varied (aug, LR, schedule, BS)
- **X-axis**: Accuracy range (max - min)
- **Insight**: Most vs least sensitive parameters

---

## Report Structure (Task 8 Section)

**Target**: 4-5 pages, following CLAUDE.md concise style

```latex
\section{Task 8: Sensitivity Analysis}

\subsection{Methodology}
% 0.5 page
- Baseline configuration from Task 7.1
- One-factor-at-a-time ablation approach
- Two groups: preprocessing (augmentation) and hyperparameters
- Metrics: test accuracy, training time, generalization gap

\subsection{Results}

\subsubsection{Preprocessing Sensitivity}
% 1-1.5 pages
- Figure: Augmentation comparison bar chart
- Figure: Training curves (none vs all)
- Observation: 1-2 sentences on findings

\subsubsection{Hyperparameter Sensitivity}
% 1.5-2 pages
- Figure: LR magnitude comparison
- Figure: LR schedule comparison
- Figure: Batch size effect
- Observation: 1-2 sentences per experiment group

\subsubsection{Sensitivity Summary}
% 0.5 page
- Figure: Overall sensitivity comparison
- Table: Best configuration from each group

\subsection{Discussion and Conclusion}
% 1 page, NO subsections
- Brief explanation: Why certain parameters are more sensitive
- Key findings (3-5 bullet points):
  • Most sensitive parameter: [X] (±Y% accuracy change)
  • Least sensitive: [Z]
  • Recommended config: [...]
  • Trade-off: augmentation adds +X% accuracy but +Y% time
  • Generalization: [observation on train-test gap]
- Computational note: 13 experiments, ~5.5 hours total
- Conclusion (2-3 sentences): Summary and practical recommendation
```

---

## Success Criteria

- [ ] 13 experiments completed (exp00-12)
- [ ] All results saved in structured format
- [ ] 6 visualization figures generated
- [ ] Clear identification of most/least sensitive parameters
- [ ] Recommended optimal configuration
- [ ] Concise 4-5 page report section
- [ ] All figures copied to report directory

---

## Key Questions to Answer

1. **Does data augmentation improve generalization?** (exp00 vs exp04)
2. **Which augmentation technique is most effective?** (exp01-03)
3. **What is the optimal learning rate?** (exp05-07)
4. **Which LR schedule converges best?** (exp08-10)
5. **How does batch size affect performance?** (exp11-12)
6. **Which parameter is most sensitive?** (compare all groups)
7. **What trade-offs exist?** (accuracy vs time, generalization vs speed)
8. **Recommended configuration for production?** (final recommendation)

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Preparation** | ✅ Complete | Experiment scripts implemented |
| **Execution** | 5.5 hours | Run all 13 experiments (exp00-12, overnight) |
| **Analysis** | 2 hours | Generate visualizations and summary |
| **Report Writing** | 2-3 hours | Write concise 4-5 page section |
| **Total** | **~10 hours** | From execution to final report |

---

## Notes

- **exp00 (baseline)** saves to `output/task7_1/` directory (timestamped subdirectory)
- **exp01-12** save to `output/task8/exp##/` directories
- **Use exp00 as reference**: All comparisons relative to baseline
- **Resume support**: Safe to re-run if interrupted - skips completed experiments
- **Monitor experiments**: Check master_log.txt for real-time status
- **Save everything**: Models, configs, plots, logs for reproducibility
- **Focus on clarity**: Visualizations should be self-explanatory
- **Be concise**: Report should be dense with information, not verbose

---

**Status**: ✅ Implementation Complete, Ready to Run
**Next Action**: Run `task8.m` in tmux overnight
**Dependencies**: Dataset files (train.mat, test.mat) exist
