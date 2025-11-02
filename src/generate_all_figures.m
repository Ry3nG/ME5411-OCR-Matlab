% Generate All Final Figures for Report
% Consolidated script to generate only the figures we need

close all;
clear;

% Add paths
addpath('core/network');
addpath('utils');

fprintf('=== Generating All Final Report Figures ===\n\n');

%% Configuration
log_path = '../output/task7_1/11-02_21-09-20/';
output_dir = '../output/final_figures/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load model and data
fprintf('Loading model and data...\n');
load([log_path 'cnn_best_acc.mat'], 'cnn');
load('../data/train.mat', 'data_train', 'labels_train');
load('../data/test.mat', 'data_test', 'labels_test');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% 1. Training curves from log
fprintf('\n1. Generating training curves...\n');
log_file = '../output/logs/task7_1_training.log';
fid = fopen(log_file, 'r');
train_acc_history = [];
test_acc_history = [];

if fid ~= -1
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Epoch')
            tokens = regexp(line, 'Epoch (\d+): acc_test:([\d.]+) acc_train:([\d.]+)', 'tokens');
            if ~isempty(tokens)
                test_acc_history(end+1) = str2double(tokens{1}{2});
                train_acc_history(end+1) = str2double(tokens{1}{3});
            end
        end
    end
    fclose(fid);
end

% Accuracy curve
figure('Position', [100, 100, 800, 500]);
plot(1:length(train_acc_history), train_acc_history * 100, 'b-', 'LineWidth', 2);
hold on;
plot(1:length(test_acc_history), test_acc_history * 100, 'r-', 'LineWidth', 2);
grid on;
xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
legend('Training', 'Validation', 'Location', 'southeast');
ylim([0, 100]);
saveas(gcf, [output_dir 'accuracy_curve.png']);
fprintf('   Saved: accuracy_curve.png\n');
close(gcf);

%% 2. Make predictions and compute confusion matrix
fprintf('\n2. Computing predictions and confusion matrix...\n');
cnn.train_mode = false;
[preds, ~] = predict(cnn, data_test);
preds = preds - 1;  % Convert to 0-indexed

confusion_mat = zeros(num_classes, num_classes);
for i = 1:length(preds)
    true_label = labels_test(i) + 1;
    pred_label = preds(i) + 1;
    confusion_mat(true_label, pred_label) = confusion_mat(true_label, pred_label) + 1;
end

confusion_mat_norm = confusion_mat ./ sum(confusion_mat, 2);

% Confusion matrix
figure('Position', [100, 100, 700, 600]);
imagesc(confusion_mat_norm);
colormap(flipud(gray));
colorbar;
caxis([0, 1]);

for i = 1:num_classes
    for j = 1:num_classes
        count = confusion_mat(i, j);
        percentage = confusion_mat_norm(i, j) * 100;
        if percentage > 50
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%d\n(%.1f%%)', count, percentage), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', text_color, 'FontSize', 10);
    end
end

xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
xticks(1:num_classes);
yticks(1:num_classes);
xticklabels(class_names);
yticklabels(class_names);
axis square;
saveas(gcf, [output_dir 'confusion_matrix.png']);
fprintf('   Saved: confusion_matrix.png\n');
close(gcf);

%% 3. Misclassification examples (diverse, no repetition)
fprintf('\n3. Generating misclassification examples...\n');
misclassified_idx = find(preds ~= labels_test);

% Collect unique pairs
pairs = struct('key', {}, 'true', {}, 'pred', {}, 'idx', {});
for i = 1:length(misclassified_idx)
    idx = misclassified_idx(i);
    t = labels_test(idx);
    p = preds(idx);
    key = sprintf('%d_%d', t, p);

    found = false;
    for j = 1:length(pairs)
        if strcmp(pairs(j).key, key)
            found = true;
            break;
        end
    end

    if ~found
        pairs(end+1).key = key;
        pairs(end).true = t;
        pairs(end).pred = p;
        pairs(end).idx = idx;
    end
end

% Plot up to 12 unique pairs
num_show = min(12, length(pairs));
figure('Position', [100, 100, 1200, 300 * ceil(num_show/4)]);

for i = 1:num_show
    subplot(ceil(num_show/4), 4, i);
    imshow(data_test(:,:,:,pairs(i).idx), []);
    title(sprintf('%s \\rightarrow %s', ...
        class_names{pairs(i).true+1}, class_names{pairs(i).pred+1}), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');
    axis off;
end

saveas(gcf, [output_dir 'misclassification_examples.png']);
fprintf('   Saved: misclassification_examples.png\n');
close(gcf);

%% 4. Feature maps (individual files)
fprintf('\n4. Generating feature map images...\n');
fm_dir = [output_dir 'feature_maps/'];
if ~exist(fm_dir, 'dir')
    mkdir(fm_dir);
end

% Select 2 misclass + 2 correct
d_to_4 = misclassified_idx(labels_test(misclassified_idx) == 5 & preds(misclassified_idx) == 1);
eight_to_a = misclassified_idx(labels_test(misclassified_idx) == 3 & preds(misclassified_idx) == 4);
d_correct = find(preds == labels_test & labels_test == 5);
zero_correct = find(preds == labels_test & labels_test == 0);

samples = [];
if ~isempty(d_to_4), samples(end+1).idx = d_to_4(1); samples(end).name = 'misclass_D_to_4'; end
if ~isempty(eight_to_a), samples(end+1).idx = eight_to_a(1); samples(end).name = 'misclass_8_to_A'; end
if ~isempty(d_correct), samples(end+1).idx = d_correct(1); samples(end).name = 'correct_D'; end
if ~isempty(zero_correct), samples(end+1).idx = zero_correct(1); samples(end).name = 'correct_0'; end

for s = 1:length(samples)
    img = data_test(:,:,:,samples(s).idx);

    % Forward pass
    cnn.train_mode = false;
    cnn = forward(cnn, img, struct('train_mode', false));

    c1 = cnn.layers{2}.activations;  % 30x30x4
    c3 = cnn.layers{3}.activations;  % 13x13x8
    c5 = cnn.layers{4}.activations;  % 3x3x16

    % Input
    figure('Position', [100, 100, 200, 200]);
    imshow(img, []);
    axis off;
    saveas(gcf, [fm_dir samples(s).name '_input.png']);
    close(gcf);

    % C1: 2x2
    figure('Position', [100, 100, 400, 400]);
    for f = 1:4
        subplot(2, 2, f);
        imagesc(c1(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(gcf, [fm_dir samples(s).name '_c1.png']);
    close(gcf);

    % C3: 2x4
    figure('Position', [100, 100, 800, 400]);
    for f = 1:8
        subplot(2, 4, f);
        imagesc(c3(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(gcf, [fm_dir samples(s).name '_c3.png']);
    close(gcf);

    % C5: 4x4
    figure('Position', [100, 100, 600, 600]);
    for f = 1:16
        subplot(4, 4, f);
        imagesc(c5(:,:,f));
        colormap(gray);
        axis off;
    end
    saveas(gcf, [fm_dir samples(s).name '_c5.png']);
    close(gcf);

    fprintf('   Saved: %s (4 files)\n', samples(s).name);
end

fprintf('\n=== Generation Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('\nFigures for report:\n');
fprintf('  - accuracy_curve.png\n');
fprintf('  - confusion_matrix.png\n');
fprintf('  - misclassification_examples.png\n');
fprintf('  - feature_maps/ (16 individual images)\n');
