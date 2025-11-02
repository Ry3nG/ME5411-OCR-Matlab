# Task 8 Experiment Runner - Quick Start Guide

## Overview
Automated experiment runner for Task 8 sensitivity analysis with 12 ablation experiments testing preprocessing (data augmentation) and hyperparameter variations.

## Quick Start

### Option 1: Run in tmux (Recommended for overnight runs)
```bash
cd /home/gong-zerui/code/ME5411/ME5411-OCR-Matlab
tmux new -s task8
matlab -batch "run('task8.m')" 2>&1 | tee output/task8_run.log
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t task8
```

### Option 2: Run in MATLAB
```matlab
cd('/home/gong-zerui/code/ME5411/ME5411-OCR-Matlab')
run('task8.m')
```

### Option 3: Direct command
```bash
cd /home/gong-zerui/code/ME5411/ME5411-OCR-Matlab
matlab -batch "run('task8.m')"
```

## Features

### ✅ Automatic Resume
- Checks for completed experiments before starting
- Skips experiments that have `results.mat` in their output directory
- Safe to re-run if interrupted

### ✅ Real-time Logging
- Each experiment logs to: `output/task8/exp##/training_log.txt`
- Master log: `output/task8/master_log.txt`
- Console output with progress indicators

### ✅ Robust Error Handling
- Continues to next experiment if one fails
- Saves error logs to: `output/task8/exp##/error_log.txt`
- Reports failed experiments in final summary

### ✅ Complete Results Saving
Each experiment saves:
- `cnn_final.mat` - Trained model
- `cnn_best_acc.mat` - Best accuracy checkpoint
- `results.mat` - Results summary struct
- `results.txt` - Human-readable summary
- `config.json` - Experiment configuration
- `predictions.mat` - Test set predictions
- `loss_ar.mat`, `acc_train.mat`, `acc_test.mat`, `lr_ar.mat` - Training curves

## Experiments

### Group A: Preprocessing (Data Augmentation)
- `exp01`: Translation only (±2%)
- `exp02`: Rotation only (±15°)
- `exp03`: Scale only (0.9-1.1×)
- `exp04`: All augmentation combined

### Group B1: Learning Rate Magnitude
- `exp05`: LR = 0.05 (half baseline)
- `exp06`: LR = 0.2 (double baseline)
- `exp07`: LR = 0.3 (triple baseline)

### Group B2: Learning Rate Schedule
- `exp08`: Cosine annealing
- `exp09`: Exponential decay (γ=0.95)
- `exp10`: Step decay (÷10 every 10 epochs)

### Group B3: Batch Size
- `exp11`: Batch size = 64
- `exp12`: Batch size = 256

**Note**: `exp00` (baseline: no aug, LR=0.1, linear decay, BS=128) is running separately in another tmux session.

## Time Estimates
- Each experiment: ~20-30 minutes
- Total for 12 experiments: ~5 hours
- Recommended: Run overnight in tmux

## Output Structure
```
output/task8/
├── summary.mat              # Complete results for all experiments
├── summary.txt              # Human-readable summary table
├── master_log.txt           # Master execution log
├── exp01/
│   ├── cnn_final.mat
│   ├── cnn_best_acc.mat
│   ├── results.mat
│   ├── results.txt
│   ├── config.json
│   ├── training_log.txt
│   ├── predictions.mat
│   ├── loss_ar.mat
│   ├── acc_train.mat
│   ├── acc_test.mat
│   └── lr_ar.mat
├── exp02/
│   └── ...
...
└── exp12/
    └── ...
```

## Monitoring Progress

### Check which experiments are done
```bash
ls -d output/task8/exp*/results.mat
```

### View master log
```bash
tail -f output/task8/master_log.txt
```

### View specific experiment log
```bash
tail -f output/task8/exp01/training_log.txt
```

### Check summary (after completion)
```bash
cat output/task8/summary.txt
```

## Troubleshooting

### If experiment fails
- Check `output/task8/exp##/error_log.txt` for error details
- Check `output/task8/exp##/training_log.txt` for last output
- Re-run `task8.m` - it will skip completed experiments and retry failed ones

### If you need to re-run a specific experiment
```bash
# Delete its results to force re-run
rm output/task8/exp05/results.mat
matlab -batch "run('task8.m')"  # Will only run exp05
```

### If you need to run a single experiment manually
```matlab
cd('/home/gong-zerui/code/ME5411/ME5411-OCR-Matlab')
addpath(genpath('src'))
config = {'exp01', 'translation', 0.02, [0 0], [1 1], 0.1, 'linear', 128};
result = task8_single_experiment(config);
```

## Expected Results

After completion, you will have:
1. **12 trained CNN models** (one per experiment)
2. **Complete training curves** for each experiment
3. **Summary statistics** comparing all experiments
4. **Per-class accuracy** for each experiment
5. **Timing information** for each experiment

This data will be used to:
- Generate sensitivity analysis figures
- Identify most/least sensitive parameters
- Write the Task 8 report section

## Next Steps

After all experiments complete:
1. Load `output/task8/summary.mat` in MATLAB
2. Generate comparison visualizations
3. Analyze sensitivity patterns
4. Write report section

## Notes
- Dataset must be pre-loaded in `data/train.mat` and `data/test.mat`
- Uses same CNN architecture as Task 7.1 baseline
- All experiments use 30 epochs
- Random seed not fixed - slight variation between runs is normal
