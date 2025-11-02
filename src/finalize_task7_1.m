% Finalize Task 7.1: Generate visualizations from best saved model
% This script loads the best model and generates all required figures

close all;
clear;

% Add paths
addpath('core/network');
addpath('utils');

% Configuration
log_path = '../output/task7_1/11-02_19-28-01/';
fig_output = log_path + "figures/";
if ~exist(fig_output, 'dir')
    mkdir(fig_output);
end

fprintf('=== Task 7.1 Results Finalization ===\n');
fprintf('Loading best model from: %s\n\n', log_path);

%% Load best model and data
load([log_path 'cnn_best_acc.mat'], 'cnn');
load('../data/test.mat', 'data_test', 'labels_test');

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% Parse training history from log file
log_file = '/tmp/matlab_task7_final.log';
fid = fopen(log_file, 'r');
train_acc_history = [];
test_acc_history = [];
loss_history = [];

if fid ~= -1
    fprintf('Parsing training log...\n');
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Epoch')
            % Parse: "Epoch X: acc_test:Y acc_train:Z"
            tokens = regexp(line, 'Epoch (\d+): acc_test:([\d.]+) acc_train:([\d.]+)', 'tokens');
            if ~isempty(tokens)
                epoch = str2double(tokens{1}{1});
                test_acc = str2double(tokens{1}{2});
                train_acc = str2double(tokens{1}{3});
                test_acc_history(end+1) = test_acc;
                train_acc_history(end+1) = train_acc;
            end
        elseif contains(line, 'loss:')
            % Parse loss from iteration lines (for approximate loss curve)
            tokens = regexp(line, 'loss:([\d.]+)', 'tokens');
            if ~isempty(tokens) && mod(length(loss_history), 10) == 0
                % Sample every 10th iteration for loss curve
                loss_history(end+1) = str2double(tokens{1}{1});
            end
        end
    end
    fclose(fid);
    fprintf('Parsed %d epochs\n', length(test_acc_history));
else
    fprintf('Warning: Could not open log file\n');
end

%% Make predictions on test set
fprintf('Running predictions on test set...\n');
cnn.train_mode = false;
[preds, probs] = predict(cnn, data_test);
test_acc = sum(preds == labels_test) / length(preds);

fprintf('Test accuracy: %.4f (%.2f%%)\n\n', test_acc, test_acc * 100);

%% 1. Training and Validation Accuracy Curve
if ~isempty(train_acc_history)
    figure('Position', [100, 100, 800, 500]);
    plot(1:length(train_acc_history), train_acc_history * 100, 'b-', 'LineWidth', 2);
    hold on;
    plot(1:length(test_acc_history), test_acc_history * 100, 'r-', 'LineWidth', 2);
    grid on;
    xlabel('Epoch', 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    title('CNN Training and Validation Accuracy', 'FontSize', 14);
    legend('Training', 'Validation', 'Location', 'southeast');
    ylim([0, 100]);
    saveas(gcf, fig_output + "accuracy_curve.png");
    fprintf('Saved: accuracy_curve.png\n');
end

%% 2. Training Loss Curve (sampled)
if ~isempty(loss_history) && length(loss_history) > 10
    figure('Position', [100, 100, 800, 500]);
    plot(1:length(loss_history), loss_history, 'b-', 'LineWidth', 2);
    grid on;
    xlabel('Training Iteration (sampled)', 'FontSize', 12);
    ylabel('Cross-Entropy Loss', 'FontSize', 12);
    title('CNN Training Loss', 'FontSize', 14);
    saveas(gcf, fig_output + "loss_curve.png");
    fprintf('Saved: loss_curve.png\n');
end

%% 3. Confusion Matrix
confusion_mat = zeros(num_classes, num_classes);
for i = 1:length(preds)
    true_label = labels_test(i) + 1;
    pred_label = preds(i) + 1;
    % Safety check for valid indices
    if true_label >= 1 && true_label <= num_classes && pred_label >= 1 && pred_label <= num_classes
        confusion_mat(true_label, pred_label) = confusion_mat(true_label, pred_label) + 1;
    end
end

% Normalize by row
confusion_mat_norm = confusion_mat ./ sum(confusion_mat, 2);

figure('Position', [100, 100, 700, 600]);
imagesc(confusion_mat_norm);
colormap(flipud(gray));
colorbar;
caxis([0, 1]);

% Add text annotations
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
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', text_color, ...
            'FontSize', 10);
    end
end

xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
title('Confusion Matrix (Normalized)', 'FontSize', 14);
xticks(1:num_classes);
yticks(1:num_classes);
xticklabels(class_names);
yticklabels(class_names);
axis square;
saveas(gcf, fig_output + "confusion_matrix.png");
fprintf('Saved: confusion_matrix.png\n');

%% 4. Per-Class Accuracy
class_accuracies = zeros(num_classes, 1);
for i = 0:num_classes-1
    idx = (labels_test == i);
    if sum(idx) > 0
        class_accuracies(i+1) = sum(preds(idx) == labels_test(idx)) / sum(idx) * 100;
    end
end

figure('Position', [100, 100, 700, 500]);
bar(class_accuracies, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
for i = 1:num_classes
    text(i, class_accuracies(i) + 2, sprintf('%.1f%%', class_accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end
grid on;
xlabel('Character Class', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('Per-Class Accuracy', 'FontSize', 14);
xticks(1:num_classes);
xticklabels(class_names);
ylim([0, 105]);
saveas(gcf, fig_output + "per_class_accuracy.png");
fprintf('Saved: per_class_accuracy.png\n');

%% 5. Summary Statistics
fprintf('\n=== Results Summary ===\n');
if ~isempty(train_acc_history)
    fprintf('Final Training Accuracy: %.2f%%\n', train_acc_history(end) * 100);
    fprintf('Final Validation Accuracy: %.2f%%\n', test_acc_history(end) * 100);
    fprintf('Best Validation Accuracy: %.2f%% (Epoch %d)\n', ...
        max(test_acc_history) * 100, find(test_acc_history == max(test_acc_history), 1));
end
fprintf('Test Set Accuracy: %.2f%%\n', test_acc * 100);
fprintf('\nPer-Class Accuracy:\n');
for i = 1:num_classes
    fprintf('  Class %s: %.2f%%\n', class_names{i}, class_accuracies(i));
end

%% Save summary to file
summary_file = fopen(fig_output + "summary.txt", 'w');
fprintf(summary_file, 'Task 7.1 CNN Character Classification - Results Summary\n');
fprintf(summary_file, '=======================================================\n\n');
if ~isempty(train_acc_history)
    fprintf(summary_file, 'Training Epochs: %d\n', length(train_acc_history));
    fprintf(summary_file, 'Final Training Accuracy: %.2f%%\n', train_acc_history(end) * 100);
    fprintf(summary_file, 'Final Validation Accuracy: %.2f%%\n', test_acc_history(end) * 100);
    fprintf(summary_file, 'Best Validation Accuracy: %.2f%% (Epoch %d)\n', ...
        max(test_acc_history) * 100, find(test_acc_history == max(test_acc_history), 1));
end
fprintf(summary_file, 'Test Set Accuracy: %.2f%%\n\n', test_acc * 100);
fprintf(summary_file, 'Per-Class Accuracy:\n');
for i = 1:num_classes
    fprintf(summary_file, '  Class %s: %.2f%%\n', class_names{i}, class_accuracies(i));
end
fprintf(summary_file, '\nConfusion Matrix (counts):\n');
fprintf(summary_file, '     ');
for i = 1:num_classes
    fprintf(summary_file, '%6s', class_names{i});
end
fprintf(summary_file, '\n');
for i = 1:num_classes
    fprintf(summary_file, '%4s ', class_names{i});
    for j = 1:num_classes
        fprintf(summary_file, '%6d', confusion_mat(i, j));
    end
    fprintf(summary_file, '\n');
end
fclose(summary_file);

fprintf('\n=== Finalization Complete ===\n');
fprintf('All visualizations saved to: %s\n', fig_output);
