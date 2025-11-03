# Task 8: Sensitivity Analysis - Current Status

**Last Updated**: 2025-11-03 22:45
**Status**: 🚀 **RUNNING IN PARALLEL** (2 groups, exp08-12)

---

## 📊 Progress Summary

### ✅ Completed (7/13)
- ✅ exp01 (translation) - 95.69%
- ✅ exp02 (rotation) - 95.74%
- ✅ exp03 (scale) - **96.13%** ⭐ Best so far
- ✅ exp04 (all_aug) - 95.69%
- ✅ exp05 (lr0.05) - 95.52%
- ✅ exp06 (lr0.2) - 95.85%
- ✅ exp07 (lr0.3) - 96.30%

### 🔄 Running (2/13) - Parallel Execution
- 🔄 **Group 1** (tmux: `task8_group1`): exp08 (cosine), exp09 (exp_decay)
- 🔄 **Group 2** (tmux: `task8_group2`): exp10 (step_decay), exp11 (bs64), exp12 (bs256)

### ⏳ Pending (4/13)
- exp00 (baseline) - deferred
- exp09, exp11, exp12 - queued in respective groups

**ETA**: ~7.5 hours (morning ~6-7am on 2025-11-04)

---

## 🎯 Experiment Design

**Goal**: Systematic ablation study on preprocessing and hyperparameters

### Group A: Data Augmentation (exp01-04)
| ID | Name | Translation | Rotation | Scale | Test Acc |
|----|------|-------------|----------|-------|----------|
| exp01 | translation | ±2% | - | - | ✅ 95.69% |
| exp02 | rotation | - | ±15° | - | ✅ 95.74% |
| exp03 | scale | - | - | 0.9-1.1× | ✅ 96.13% ⭐ |
| exp04 | all_aug | ±2% | ±15° | 0.9-1.1× | ✅ 95.69% |

### Group B1: Learning Rate Magnitude (exp05-07)
| ID | Name | LR | Test Acc |
|----|------|----|----------|
| exp05 | lr0.05 | 0.05 | ✅ 95.52% |
| exp06 | lr0.2 | 0.2 | ✅ 95.85% |
| exp07 | lr0.3 | 0.3 | ✅ 96.30% ⭐ |

### Group B2: LR Schedule (exp08-10)
| ID | Name | Schedule | Status |
|----|------|----------|--------|
| exp08 | cosine | Cosine annealing | 🔄 Running (Group 1) |
| exp09 | exp_decay | Exponential (γ=0.95) | ⏳ Queued (Group 1) |
| exp10 | step_decay | Step (÷10 every 10 ep) | 🔄 Running (Group 2) |

### Group B3: Batch Size (exp11-12)
| ID | Name | Batch Size | Status |
|----|------|------------|--------|
| exp11 | bs64 | 64 | ⏳ Queued (Group 2) |
| exp12 | bs256 | 256 | ⏳ Queued (Group 2) |

**Baseline Reference** (exp00):
- No augmentation, LR=0.1, linear decay, BS=128
- Saves to `output/task7_1/` directory
- Deferred (will use existing Task 7.1 results)

---

## 🚀 Parallel Execution Strategy

**Optimization**: Running 2 experiments simultaneously to utilize CPU cores efficiently

### Before Optimization
- Sequential execution: 1 experiment at a time
- CPU usage: ~30%
- Time for 5 remaining experiments: ~15 hours
- ETA: 2025-11-04 evening

### After Optimization (Current)
- **Parallel execution**: 2 groups running simultaneously
- **CPU usage**: ~60% (14 cores, no Parallel Computing Toolbox)
- **Time for 5 remaining experiments**: ~7.5 hours
- **ETA**: 2025-11-04 morning 6-7am

### Implementation
```bash
# Group 1: exp08, exp09
tmux attach -t task8_group1

# Group 2: exp10, exp11, exp12
tmux attach -t task8_group2
```

**Key Design**:
- Groups have **no overlap** → zero collision risk
- Each experiment takes ~2.5 hours
- Group 1: 2 experiments = 5 hours
- Group 2: 3 experiments = 7.5 hours
- Both finish by morning

---

## 📁 Output Structure

```
output/
├── task7_1/
│   └── [timestamp]/         ← exp00 (baseline, if needed)
└── task8/
    ├── master_log.txt       ← Master execution log
    ├── summary.mat          ← Final results (after all complete)
    ├── summary.txt
    ├── task8_group1.log     ← Group 1 parallel log
    ├── task8_group2.log     ← Group 2 parallel log
    ├── exp01/ ... exp07/    ← Completed experiments
    │   ├── cnn_final.mat    ← Trained model
    │   ├── results.mat      ← Results struct
    │   ├── training_log.txt ← Training log
    │   └── ...
    └── exp08/ ... exp12/    ← Running/pending
```

---

## 🔍 Monitoring Commands

### Check Progress
```bash
# View Group 1 (exp08, exp09)
tail -f ~/code/ME5411/ME5411-OCR-Matlab/output/task8_group1.log

# View Group 2 (exp10, exp11, exp12)
tail -f ~/code/ME5411/ME5411-OCR-Matlab/output/task8_group2.log

# Attach to tmux sessions
tmux attach -t task8_group1  # Ctrl+B, D to detach
tmux attach -t task8_group2

# List all task8 sessions
tmux list-sessions | grep task8
```

### Check Completion Status
```bash
cd ~/code/ME5411/ME5411-OCR-Matlab

# Count completed experiments
ls output/task8/exp*/results.mat 2>/dev/null | wc -l

# Check which are done
for exp in exp{01..12}; do
    [ -f "output/task8/$exp/results.mat" ] && echo "✅ $exp" || echo "⏳ $exp"
done
```

### CPU/Memory Usage
```bash
# Check CPU usage
top -b -n 1 | grep -E "(Cpu|matlab)"

# Check memory
free -h
```

---

## 🎓 Key Findings (So Far)

### Preprocessing Sensitivity (Group A)
- **Best**: exp03 (scale only) - 96.13%
- **Worst**: exp05 (lr0.05) - 95.52%
- **Range**: 0.61% difference
- **Observation**: Scale augmentation most effective; combined augmentation surprisingly not better

### Learning Rate Sensitivity (Group B1)
- **Best**: exp07 (lr0.3) - 96.30% ⭐ Overall best
- **Baseline**: exp06 (lr0.2) - 95.85%
- **Worst**: exp05 (lr0.05) - 95.52%
- **Range**: 0.78% difference
- **Observation**: Higher LR (0.3) performs best; LR appears moderately sensitive

### Preliminary Insights
1. **Most sensitive parameter so far**: Learning rate magnitude (0.78% range)
2. **Scale augmentation**: Surprisingly effective alone
3. **High LR tolerance**: Model benefits from aggressive LR=0.3
4. **Training time**: Consistent ~140-155 min per experiment

---

## 🛠️ Scripts Reference

### Main Scripts
- `src/task8_run_group1.m` - Runs exp08, exp09 (Group 1)
- `src/task8_run_group2.m` - Runs exp10, exp11, exp12 (Group 2)
- `src/task8_single_experiment.m` - Single experiment wrapper
- `src/task8_run_experiments.m` - Original sequential runner (not used)

### If Something Goes Wrong

**Restart a group**:
```bash
# Kill the tmux session
tmux kill-session -t task8_group1

# Restart (will skip completed experiments)
cd ~/code/ME5411/ME5411-OCR-Matlab
tmux new -s task8_group1 "matlab -batch \"run('src/task8_run_group1.m')\" 2>&1 | tee output/task8_group1.log"
```

**Re-run a specific experiment**:
```bash
# Delete its results
rm output/task8/exp08/results.mat

# Restart the group (will only re-run exp08)
# ... same as above
```

---

## 📝 Next Steps (After Completion)

### 1. Generate Summary
- Load all results into `output/task8/summary.mat`
- Create comparison tables

### 2. Visualizations (6 figures needed)
- Data augmentation bar chart (exp01-04)
- Training curves (augmentation effect)
- LR magnitude comparison (exp05-07)
- LR schedule curves (exp08-10)
- Batch size scatter plot (exp11-12)
- Overall sensitivity summary (all groups)

### 3. Report Writing
- Write Task 8 section (4-5 pages)
- Follow CLAUDE.md concise style
- Include all 6 figures
- Discussion: sensitivity insights, recommendations

### 4. Final Deliverable
- Copy figures to `/home/gong-zerui/code/ME5411/ME5411-Project-Report/figs/task8/`
- Update LaTeX report
- Complete analysis and conclusions

---

## ⚡ Why Parallel Execution?

### Problem Identified
- **CPU cores**: 14 physical cores (20 with hyperthreading)
- **Actual usage**: Only ~30% (due to serial for-loops in forward/backward passes)
- **Bottleneck**: Not data loading (already in RAM), but serial image processing
- **No Parallel Computing Toolbox**: Can't use `parfor` to parallelize within experiment

### Solution
- **Run 2 experiments simultaneously** (experiment-level parallelism)
- **Each uses ~30% CPU** → total ~60% CPU utilization
- **Halves total time**: 15 hours → 7.5 hours
- **Safe**: Groups have no overlap, zero collision risk

---

## 📌 Important Notes

- **exp00 (baseline)** not yet run - can use existing Task 7.1 results if needed
- **Experiment duration**: ~2.5 hours each (30 epochs, batch size 128)
- **Memory per experiment**: ~1.5 GB
- **Total available memory**: 26 GB (plenty of headroom)
- **Random seed**: Not fixed - slight variation between runs is normal
- **Resume support**: All scripts check for existing `results.mat` and skip completed work

---

**Status**: ✅ Parallel execution successfully launched at 22:45
**Monitor**: `tmux attach -t task8_group1` or `task8_group2`
**ETA**: Morning 6-7am, then generate visualizations and write report 🚀
