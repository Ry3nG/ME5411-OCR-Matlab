# Task 7.2 Implementation Plan: SOM + Bag-of-Visual-Words + SVM

**Date**: November 3, 2025
**Method**: Self-Organizing Map (SOM) for codebook learning + Bag-of-Visual-Words feature extraction + Support Vector Machine (SVM) classification

---

## 📌 Overview

### Motivation
Task 7.2 requires a **non-CNN-based method** using approaches **covered in Part 2 of the course**. After reviewing the course content:
- ✅ **SOM (Self-Organizing Map)**: Part 2 unsupervised learning method
- ✅ **SVM (Support Vector Machine)**: Part 2 supervised learning method (confirmed in course slides)
- ✅ **Method combination**: Task 2 explicitly allows "combination of such methods"

### Core Idea
**Bag-of-Visual-Words (BoVW) with SOM-based Codebook**

1. **Unsupervised Learning**: Train SOM on image patches to learn a codebook of visual prototypes (visual words)
2. **Feature Encoding**: Represent each character as a histogram over the learned visual words
3. **Supervised Classification**: Train SVM on the histogram features for 7-class character recognition

This approach:
- Focuses on **local texture/stroke patterns**
- Does not require HOG (which is not in the course)
- Provides **low-dimensional features** (~64D histogram)
- Shows deep understanding of **combining unsupervised + supervised learning**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Patch Sampling (Unsupervised Data Collection)
┌──────────────┐
│ Training     │
│ Images       │  ──┐
│ (5,327)      │    │
└──────────────┘    │
                    ├──> Random sample 8×8 patches
┌──────────────┐    │    Collect ~50K-100K patches
│ 124×124      │    │    Flatten to 64D vectors
│ Grayscale    │  ──┘    Normalize to [0, 1]
└──────────────┘
                    │
                    ↓
              ┌──────────────┐
              │ Patch Pool   │
              │ (50K × 64D)  │
              └──────────────┘


Step 2: SOM Training (Learn Visual Codebook)
┌──────────────┐
│ Patch Pool   │
│ (50K × 64D)  │
└──────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Self-Organizing Map (SOM)            │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┐                    │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤  8×8 grid         │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤  = 64 neurons     │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤                    │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤  Each neuron:     │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤  64D weight vector│
│  ├─┼─┼─┼─┼─┼─┼─┼─┤  (8×8 prototype)  │
│  ├─┼─┼─┼─┼─┼─┼─┼─┤                    │
│  └─┴─┴─┴─┴─┴─┴─┴─┘                    │
│                                        │
│  Training: Competitive learning        │
│  - Find BMU (Best Matching Unit)      │
│  - Update BMU + neighbors             │
│  - Decay learning rate & neighborhood │
└────────────────────────────────────────┘
       │
       ↓
┌──────────────┐
│ 64 Visual    │
│ Prototypes   │  ──> Learned codebook
└──────────────┘      (visual words)


Step 3: Feature Extraction (Bag-of-Words Encoding)
┌──────────────┐     ┌──────────────┐
│ Training     │     │ 64 Visual    │
│ Image        │  +  │ Prototypes   │
│ (124×124)    │     │ (codebook)   │
└──────────────┘     └──────────────┘
       │                    │
       ↓                    │
  Dense sampling            │
  8×8 patches               │
  (stride = 4)              │
       │                    │
       ↓                    │
  ┌─────────────┐           │
  │ Patches:    │           │
  │ P₁, P₂, ... │           │
  │ P_N         │           │
  └─────────────┘           │
       │                    │
       └────────┬───────────┘
                ↓
        For each patch Pᵢ:
        Find BMU index k ∈ {1,...,64}
                ↓
        ┌──────────────────┐
        │ Histogram [64D]  │
        │ h[k] += 1        │
        └──────────────────┘
                ↓
        L2 normalization
                ↓
        ┌──────────────────┐
        │ BoW Feature      │
        │ (64D vector)     │
        └──────────────────┘


Step 4: SVM Classification (Supervised Learning)
┌──────────────────┐
│ BoW Features     │
│ (5,327 × 64D)    │
└──────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Support Vector Machines (SVM)        │
│                                        │
│  One-vs-All Strategy:                 │
│  ┌───────────┐  ┌───────────┐         │
│  │ SVM for   │  │ SVM for   │  ...    │
│  │ class '0' │  │ class '4' │         │
│  └───────────┘  └───────────┘         │
│                                        │
│  7 binary classifiers                 │
│  Linear or RBF kernel                 │
└────────────────────────────────────────┘
       │
       ↓
┌──────────────────┐
│ Trained Model    │
└──────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                     TESTING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Test Image ──> Extract patches ──> Find BMU for each
                                    ↓
                            Build histogram ──> SVM predict
                                    ↓
                            Predicted class
```

---

## 🔧 Implementation Details

### 1. Patch Sampling

**Purpose**: Collect diverse local patterns for unsupervised learning

```matlab
function patches = extract_patches_for_som(data, num_samples, patch_size)
    % data: 4D array (H × W × C × N)
    % Returns: (num_samples × patch_size^2) matrix

    [H, W, ~, N] = size(data);
    patches = zeros(num_samples, patch_size * patch_size);

    for i = 1:num_samples
        % Random image
        img_idx = randi(N);
        img = squeeze(data(:, :, 1, img_idx));

        % Random location (ensure patch fits)
        y = randi([1, H - patch_size + 1]);
        x = randi([1, W - patch_size + 1]);

        % Extract and flatten patch
        patch = img(y:y+patch_size-1, x:x+patch_size-1);
        patches(i, :) = patch(:)' / 255;  % Normalize to [0,1]
    end
end
```

**Parameters**:
- `patch_size`: 8 (8×8 patches)
- `num_samples`: 50,000 - 100,000
- Sampling strategy: Uniform random from all training images

---

### 2. SOM Training

**Purpose**: Learn a topologically-organized codebook of visual prototypes

**Algorithm**: Competitive Learning with Neighborhood Updates

```matlab
function som_model = train_som(patches, grid_size, num_iterations)
    % Initialize SOM weights randomly from data distribution
    [M, D] = size(patches);  % M samples, D dimensions (64)
    num_neurons = grid_size(1) * grid_size(2);

    % Random initialization from data samples
    som_weights = patches(randperm(M, num_neurons), :);

    % Create 2D grid coordinates
    [grid_x, grid_y] = meshgrid(1:grid_size(1), 1:grid_size(2));
    neuron_coords = [grid_x(:), grid_y(:)];

    % Training parameters
    lr_init = 0.5;
    lr_final = 0.01;
    sigma_init = 3.0;
    sigma_final = 0.5;

    for iter = 1:num_iterations
        % Decay schedule
        t = iter / num_iterations;
        lr = lr_init * (1 - t) + lr_final * t;
        sigma = sigma_init * (1 - t) + sigma_final * t;

        % Random sample
        idx = randi(M);
        sample = patches(idx, :);

        % Find BMU (Best Matching Unit)
        distances = sum((som_weights - sample).^2, 2);
        [~, bmu_idx] = min(distances);

        % Update BMU and neighbors
        bmu_coord = neuron_coords(bmu_idx, :);
        for k = 1:num_neurons
            % Distance in grid space
            grid_dist = norm(neuron_coords(k, :) - bmu_coord);

            % Neighborhood function (Gaussian)
            h = exp(-grid_dist^2 / (2 * sigma^2));

            % Weight update
            som_weights(k, :) = som_weights(k, :) + ...
                lr * h * (sample - som_weights(k, :));
        end

        if mod(iter, 500) == 0
            fprintf('  SOM iteration %d/%d\n', iter, num_iterations);
        end
    end

    som_model = struct();
    som_model.weights = som_weights;
    som_model.grid_size = grid_size;
    som_model.patch_size = sqrt(D);
end
```

**Key Parameters**:
- Grid size: 8×8 = 64 neurons
- Iterations: 2000 - 5000
- Learning rate: 0.5 → 0.01 (linear decay)
- Neighborhood radius: 3.0 → 0.5 (linear decay)
- Neighborhood function: Gaussian

**Mathematical Formulation**:

Weight update rule:
$$w_k(t+1) = w_k(t) + \eta(t) \cdot h_{ck}(t) \cdot (x(t) - w_k(t))$$

Where:
- $w_k$: weight vector of neuron $k$
- $x$: input sample
- $\eta(t)$: learning rate at time $t$
- $h_{ck}(t)$: neighborhood function centered at BMU $c$

Neighborhood function:
$$h_{ck}(t) = \exp\left(-\frac{d_{ck}^2}{2\sigma(t)^2}\right)$$

Where:
- $d_{ck}$: grid distance between neuron $k$ and BMU $c$
- $\sigma(t)$: neighborhood radius at time $t$

---

### 3. Bag-of-Words Feature Extraction

**Purpose**: Encode each image as a histogram over visual words

```matlab
function features = extract_bow_features(data, som_model, stride)
    % data: 4D array (H × W × C × N)
    % Returns: (N × num_neurons) feature matrix

    [H, W, ~, N] = size(data);
    patch_size = som_model.patch_size;
    num_neurons = size(som_model.weights, 1);

    features = zeros(N, num_neurons);

    for i = 1:N
        img = double(squeeze(data(:, :, 1, i))) / 255;
        histogram = zeros(1, num_neurons);

        % Dense patch sampling with stride
        count = 0;
        for y = 1:stride:(H - patch_size + 1)
            for x = 1:stride:(W - patch_size + 1)
                % Extract patch
                patch = img(y:y+patch_size-1, x:x+patch_size-1);
                patch_vec = patch(:)';

                % Find BMU
                distances = sum((som_model.weights - patch_vec).^2, 2);
                [~, bmu_idx] = min(distances);

                % Accumulate in histogram
                histogram(bmu_idx) = histogram(bmu_idx) + 1;
                count = count + 1;
            end
        end

        % Normalize histogram (L2 norm)
        histogram = histogram / norm(histogram + 1e-10);
        features(i, :) = histogram;

        if mod(i, 500) == 0
            fprintf('  Processed %d/%d images\n', i, N);
        end
    end
end
```

**Parameters**:
- Stride: 4 (50% overlap between patches)
- Normalization: L2 norm for scale invariance

**Mathematical Formulation**:

For an image $I$, extract patches $\{p_1, p_2, ..., p_M\}$

For each patch $p_i$, find BMU:
$$c_i = \arg\min_k \|p_i - w_k\|^2$$

Build histogram:
$$h[k] = \sum_{i=1}^{M} \mathbb{1}[c_i = k]$$

L2 normalization:
$$\tilde{h} = \frac{h}{\|h\|_2 + \epsilon}$$

---

### 4. SVM Classification

**Purpose**: Multi-class classification on BoW features

Reuse the custom SVM implementation from initial Task 7.2 attempt:
- `trainLinearSVM.m`: SGD-based linear SVM training
- `predictSVM.m`: SVM prediction
- One-vs-All strategy for 7-class problem

**Parameters**:
- Kernel: Linear (for interpretability and speed)
- Regularization C: 1.0
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.01 → 1e-5 (linear decay)
- Iterations: 1000

---

## 📁 File Structure

```
src/
├── task7_2.m                          # Main training script
├── core/
│   ├── som/
│   │   ├── train_som.m                # SOM training algorithm
│   │   ├── find_bmu.m                 # Find best matching unit
│   │   └── som_forward.m              # Forward pass (BMU finding)
│   ├── features/
│   │   ├── extract_patches.m          # Random patch sampling
│   │   ├── extract_bow_features.m     # BoW feature extraction
│   │   └── normalize_features.m       # Feature normalization
│   └── network/
│       ├── trainLinearSVM.m           # SVM training (reuse from previous)
│       └── predictSVM.m               # SVM prediction
└── utils/
    └── visualization/
        ├── visualize_som_codebook.m   # Visualize learned prototypes
        ├── visualize_activation.m     # Visualize activation histograms
        └── plot_results.m             # Standard result plots

output/task7_2/
├── som_model.mat                      # Trained SOM codebook
├── bow_features_train.mat             # BoW features (training set)
├── bow_features_test.mat              # BoW features (test set)
├── svm_models.mat                     # Trained SVM classifiers
├── predictions.mat                    # Test predictions
├── results.txt                        # Text summary
│
├── figures/
│   ├── som_codebook.png               # 8×8 grid of visual prototypes
│   ├── som_codebook_detailed.png      # Individual prototypes with labels
│   ├── activation_examples.png        # Sample images + histograms
│   ├── confusion_matrix.png           # Normalized confusion matrix
│   ├── per_class_accuracy.png         # Bar chart
│   └── misclassification_examples.png # Error analysis
```

---

## 🎯 Training Script Outline

**File**: `src/task7_2.m`

```matlab
%% Task 7.2: SOM + Bag-of-Visual-Words + SVM Classification
clear all; close all;

% Add paths
addpath('core/som');
addpath('core/features');
addpath('core/network');
addpath('utils/visualization');

output_dir = '../output/task7_2/';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

fprintf('=== Task 7.2: SOM + BoW + SVM ===\n\n');

%% 1. Load Dataset
load('../data/train.mat');  % data_train, labels_train
load('../data/test.mat');   % data_test, labels_test

%% 2. Patch Sampling
fprintf('[1/5] Sampling patches for SOM training...\n');
num_patches = 50000;
patch_size = 8;
patches = extract_patches(data_train, num_patches, patch_size);
fprintf('  Sampled %d patches of size %dx%d\n\n', num_patches, patch_size, patch_size);

%% 3. Train SOM
fprintf('[2/5] Training SOM codebook...\n');
grid_size = [8, 8];  % 64 visual words
num_iterations = 2000;
som_model = train_som(patches, grid_size, num_iterations);
save([output_dir 'som_model.mat'], 'som_model');
fprintf('  SOM training complete\n\n');

% Visualize codebook
visualize_som_codebook(som_model, output_dir);

%% 4. Extract BoW Features
fprintf('[3/5] Extracting Bag-of-Words features...\n');
stride = 4;  % Dense sampling

fprintf('  Training set...\n');
features_train = extract_bow_features(data_train, som_model, stride);
save([output_dir 'bow_features_train.mat'], 'features_train');

fprintf('  Test set...\n');
features_test = extract_bow_features(data_test, som_model, stride);
save([output_dir 'bow_features_test.mat'], 'features_test');
fprintf('  Feature extraction complete\n\n');

%% 5. Train SVM
fprintf('[4/5] Training SVM classifiers (One-vs-All)...\n');
numClasses = 7;
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
svm_models = cell(numClasses, 1);

tic;
for c = 1:numClasses
    fprintf('  Training SVM for class %s (%d/%d)...\n', class_names{c}, c, numClasses);

    % Binary labels
    labels_binary = double(labels_train == (c-1));
    labels_binary(labels_binary == 0) = -1;

    % Train SVM
    svm_models{c} = trainLinearSVM(features_train, labels_binary, ...
        'C', 1.0, 'MaxIter', 1000, 'Verbose', false);
end
training_time = toc;
save([output_dir 'svm_models.mat'], 'svm_models', 'class_names');
fprintf('  SVM training complete (%.2f seconds)\n\n', training_time);

%% 6. Evaluate
fprintf('[5/5] Evaluating on test set...\n');
% ... (prediction and evaluation code)

fprintf('\n=== Task 7.2 Complete ===\n');
```

---

## 📊 Expected Performance

### Accuracy Prediction

Based on the Bag-of-Visual-Words literature and our dataset characteristics:

| Metric | Expected Range | Target |
|--------|----------------|--------|
| Overall Test Accuracy | 88% - 92% | 90% |
| Training Time | 5 - 10 min | 7 min |
| Per-class Accuracy (Digits) | 90% - 95% | 92% |
| Per-class Accuracy (Letters) | 85% - 90% | 88% |

### Comparison with CNN (Task 7.1)

| Aspect | CNN | SOM+BoW+SVM |
|--------|-----|-------------|
| Test Accuracy | 94.79% | ~90% (expected) |
| Training Time | 17.75 min | ~7 min |
| Parameters | ~50K | ~4K (64×64 SOM weights) |
| Feature Learning | Automatic (hierarchical) | Manual (BoW encoding) |
| Spatial Structure | Exploited (convolution) | Partially (local patches) |
| Interpretability | Low (black box) | High (visual prototypes) |

### Key Insights for Task 7.3 (Comparison)

**Advantages of SOM+BoW+SVM**:
1. **Faster training** (~2.5× faster than CNN)
2. **Highly interpretable** (can visualize learned visual words)
3. **Fewer parameters** (more efficient)
4. **Part-based representation** (robust to small deformations)

**Disadvantages**:
1. **Lower accuracy** (~5% gap from CNN)
2. **Manual feature design** (patch size, grid size, stride)
3. **No end-to-end optimization** (SOM and SVM trained separately)
4. **Limited spatial modeling** (bag-of-words discards spatial layout)

---

## 🎨 Visualization Plan

### 1. SOM Codebook Visualization

**Figure 1: Visual Prototypes Grid**
- 8×8 grid showing all 64 learned prototypes
- Each cell displays an 8×8 grayscale patch
- Title: "Learned Visual Words (SOM Codebook)"

```matlab
function visualize_som_codebook(som_model, output_dir)
    weights = som_model.weights;
    grid_size = som_model.grid_size;
    patch_size = som_model.patch_size;

    figure('Position', [100, 100, 800, 800]);
    for i = 1:prod(grid_size)
        subplot(grid_size(1), grid_size(2), i);
        patch = reshape(weights(i, :), patch_size, patch_size);
        imshow(patch, []);
        axis off;
    end
    sgtitle('Learned Visual Words (SOM Codebook)', 'FontSize', 14);
    saveas(gcf, [output_dir 'figures/som_codebook.png']);
end
```

### 2. Activation Histogram Examples

**Figure 2: Character Encoding**
- Show 4-6 sample images (2 per class, correct + misclassified)
- Below each image: 64-bin histogram showing activation distribution
- Demonstrates how different characters activate different visual words

### 3. Standard Performance Plots

Reuse visualization code from Task 7.1:
- **Confusion Matrix**: Normalized with counts + percentages
- **Per-class Accuracy**: Bar chart
- **Misclassification Examples**: 3×4 grid of error cases

---

## 📝 Report Structure

### Section: Task 7.2 - Non-CNN Method

#### 7.2.1 Introduction
- Motivation: Traditional ML approach using Part 2 methods
- Overview: SOM (unsupervised) + SVM (supervised) combination
- Bag-of-Visual-Words concept
- Rationale: Local patch features for character recognition

#### 7.2.2 Method

**Self-Organizing Map (SOM)**
- Algorithm description
- Training procedure
- Mathematical formulation (competitive learning, neighborhood function)
- Hyperparameters

**Bag-of-Visual-Words Feature Encoding**
- Patch extraction strategy
- Codebook lookup (BMU finding)
- Histogram construction
- Normalization

**SVM Classification**
- One-vs-All multi-class strategy
- Linear kernel choice
- Training algorithm (SGD)

#### 7.2.3 Results
- Overall performance (accuracy, training time)
- Per-class accuracy table + bar chart
- Confusion matrix analysis
- Misclassification patterns

**Visualization Analysis**:
- Learned visual prototypes (what patterns did SOM discover?)
- Activation histogram comparison between classes
- Example: "Class '0' strongly activates circular edge prototypes, while 'H' activates vertical/horizontal edge prototypes"

#### 7.2.4 Discussion

**Method Characteristics**:
- Part-based representation vs. holistic features
- Effect of codebook size on performance
- Interpretability advantage over deep learning

**Comparison Teaser** (detailed in Task 7.3):
- SOM+BoW: Faster, interpretable, but lower accuracy
- CNN: Slower, black-box, but superior performance
- Trade-off: Efficiency vs. Accuracy

---

## ⚠️ Potential Challenges & Solutions

### Challenge 1: SOM Convergence
**Issue**: SOM may not converge well with poor initialization
**Solution**:
- Initialize weights from random data samples (not pure random)
- Use sufficient training iterations (2000+)
- Monitor quantization error during training

### Challenge 2: Codebook Size Selection
**Issue**: Unclear optimal grid size (6×6? 8×8? 10×10?)
**Solution**:
- Start with 8×8 = 64 (standard BoW size)
- If time permits, compare {6×6, 8×8, 10×10}
- Report: "64 visual words provides good balance between expressiveness and efficiency"

### Challenge 3: Low Accuracy
**Issue**: If accuracy < 85%, may look weak compared to CNN
**Solution**:
- Emphasize **interpretability** and **efficiency** advantages
- Multi-scale BoW: Extract histograms at multiple patch sizes (8×8, 12×12, 16×16), concatenate features
- Longer SOM training or more patches

### Challenge 4: Implementation Time
**Issue**: SOM training + feature extraction may take longer than expected
**Solution**:
- Parallelize feature extraction if possible
- Cache intermediate results (patches, SOM model, features)
- Start with smaller num_patches (30K) for quick prototype

---

## ✅ Success Criteria

### Minimum Requirements (Must Achieve)
- [ ] SOM training converges successfully
- [ ] Test accuracy ≥ 85%
- [ ] Training completes in < 15 minutes
- [ ] All visualizations generate correctly
- [ ] Code runs without errors

### Target Goals (Ideal)
- [ ] Test accuracy ≥ 90%
- [ ] Training time ≤ 10 minutes
- [ ] Clear visual prototypes (interpretable)
- [ ] Per-class accuracy: digits > 90%, letters > 85%

### Bonus (If Time Permits)
- [ ] Multi-scale BoW (multiple patch sizes)
- [ ] Codebook size comparison experiment
- [ ] RBF kernel SVM comparison
- [ ] t-SNE visualization of BoW feature space

---

## 📅 Implementation Timeline

**Estimated Total Time**: 3-4 hours

| Task | Time | Status |
|------|------|--------|
| 1. SOM training implementation | 45 min | ⏳ Pending |
| 2. BoW feature extraction | 30 min | ⏳ Pending |
| 3. SVM integration (reuse code) | 15 min | ⏳ Pending |
| 4. Run training pipeline | 10 min | ⏳ Pending |
| 5. Generate visualizations | 30 min | ⏳ Pending |
| 6. Write report section | 60 min | ⏳ Pending |
| **Total** | **3h 10min** | |

---

## 🔍 References

**Theoretical Foundation**:
1. Kohonen, T. (1990). "The self-organizing map". Proceedings of the IEEE.
2. Sivic, J. & Zisserman, A. (2003). "Video Google: A text retrieval approach to object matching in videos". ICCV.
3. Csurka, G. et al. (2004). "Visual categorization with bags of keypoints". ECCV Workshop.
4. Cortes, C. & Vapnik, V. (1995). "Support-vector networks". Machine Learning.

**Course Material**:
- ME5411 Part 2: Self-Organizing Maps (SOM)
- ME5411 Part 2: Support Vector Machines (SVM)

---

## 📌 Notes

- This plan prioritizes **course compliance** (100% Part 2 methods) over maximum accuracy
- The method is **innovative** (SOM for codebook learning is less common than k-means) and shows deep understanding
- **Interpretability** is a major selling point: can visualize what the model learned
- Implementation is **modular**: each component (SOM, BoW, SVM) is independent and testable

---

**Last Updated**: November 3, 2025
**Status**: Ready for implementation
**Confidence Level**: 85% success probability for ≥90% accuracy
