% Visualization script for Task 7.1 CNN results
% Generates training curves, confusion matrix, and per-class accuracy plots

close all;
clear;

% Add paths
addpath('utils');

% Prompt for log directory
fprintf('=== Task 7.1 Visualization ===\n');
fprintf('Enter the log directory path (e.g., ../output/task7_1/11-02_19-28-01/):\n');
log_path = input('Log path: ', 's');

if ~endsWith(log_path, '/')
    log_path = [log_path '/'];
end

fprintf('\nLoading results from: %s\n', log_path);

% Load training history
if ~exist([log_path 'training_history.mat'], 'file')
    error('training_history.mat not found. Training may not have completed.');
end

load([log_path 'training_history.mat']);  % train_acc_history, test_acc_history, loss_history

% Load predictions
load([log_path 'predictions.mat']);  % preds, labels_test
load([log_path 'cnn.mat']);  % cnn model

class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};
num_classes = length(class_names);

%% Create output directory for figures
fig_output = log_path + "figures/";
if ~exist(fig_output, 'dir')
    mkdir(fig_output);
end

%% 1. Training and Validation Accuracy Curve
figure('Position', [100, 100, 800, 500]);
plot(1:length(train_acc_history), train_acc_history * 100, 'b-', 'LineWidth', 2);
hold on;
plot(1:length(test_acc_history), test_acc_history * 100, 'r-', 'LineWidth', 2);
grid on;
xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('CNN Training and Validation Accuracy', 'FontSize', 14);
legend('Training Accuracy', 'Validation Accuracy', 'Location', 'southeast');
ylim([0, 100]);
saveas(gcf, fig_output + "accuracy_curve.png");
fprintf('Saved: accuracy_curve.png\n');

%% 2. Training Loss Curve
figure('Position', [100, 100, 800, 500]);
plot(1:length(loss_history), loss_history, 'b-', 'LineWidth', 2);
grid on;
xlabel('Epoch', 'FontSize', 12);
ylabel('Cross-Entropy Loss', 'FontSize', 12);
title('CNN Training Loss', 'FontSize', 14);
saveas(gcf, fig_output + "loss_curve.png");
fprintf('Saved: loss_curve.png\n');

%% 3. Confusion Matrix
confusion_mat = zeros(num_classes, num_classes);
for i = 1:length(preds)
    true_label = labels_test(i) + 1;  % Convert 0-indexed to 1-indexed
    pred_label = preds(i) + 1;
    confusion_mat(true_label, pred_label) = confusion_mat(true_label, pred_label) + 1;
end

% Normalize by row (true labels)
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

%% 4. Per-Class Accuracy Bar Chart
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
% Add value labels on bars
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

%% 5. Combined Training Overview
figure('Position', [100, 100, 1200, 400]);

% Subplot 1: Accuracy
subplot(1, 3, 1);
plot(1:length(train_acc_history), train_acc_history * 100, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:length(test_acc_history), test_acc_history * 100, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Epoch');
ylabel('Accuracy (%)');
title('Training Progress');
legend('Train', 'Val', 'Location', 'southeast');
ylim([0, 100]);

% Subplot 2: Loss
subplot(1, 3, 2);
plot(1:length(loss_history), loss_history, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');

% Subplot 3: Per-class accuracy
subplot(1, 3, 3);
bar(class_accuracies, 'FaceColor', [0.2, 0.6, 0.8]);
grid on;
xlabel('Class');
ylabel('Accuracy (%)');
title('Per-Class Accuracy');
xticks(1:num_classes);
xticklabels(class_names);
ylim([0, 100]);

saveas(gcf, fig_output + "training_overview.png");
fprintf('Saved: training_overview.png\n');

%% 6. Print Summary Statistics
fprintf('\n=== Task 7.1 Results Summary ===\n');
fprintf('Final Training Accuracy: %.2f%%\n', train_acc_history(end) * 100);
fprintf('Final Validation Accuracy: %.2f%%\n', test_acc_history(end) * 100);
fprintf('Final Loss: %.4f\n', loss_history(end));
fprintf('\nPer-Class Accuracy:\n');
for i = 1:num_classes
    fprintf('  Class %s: %.2f%%\n', class_names{i}, class_accuracies(i));
end

% Save summary to text file
summary_file = fopen(fig_output + "summary.txt", 'w');
fprintf(summary_file, 'Task 7.1 CNN Character Classification - Results Summary\n');
fprintf(summary_file, '=======================================================\n\n');
fprintf(summary_file, 'Final Training Accuracy: %.2f%%\n', train_acc_history(end) * 100);
fprintf(summary_file, 'Final Validation Accuracy: %.2f%%\n', test_acc_history(end) * 100);
fprintf(summary_file, 'Final Loss: %.4f\n', loss_history(end));
fprintf(summary_file, '\nPer-Class Accuracy:\n');
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

fprintf('\nAll visualizations saved to: %s\n', fig_output);
fprintf('=== Visualization Complete ===\n');
