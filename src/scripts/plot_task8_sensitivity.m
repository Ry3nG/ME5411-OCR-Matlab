% PLOT_TASK8_SENSITIVITY Generate sensitivity analysis visualizations for Task 8
%
% This script reads the Task 8 experiment results (results.csv) and generates
% various sensitivity plots:
%   - K (dictionary size) vs accuracy
%   - Stride vs accuracy and time
%   - Voting type comparison (hard vs soft)
%   - SPM comparison (with vs without)
%   - SVM C parameter sensitivity
%
% ME5411 Robot Vision and AI - Task 8 Visualization

function plot_task8_sensitivity()
    % Read results
    results_file = 'output/task8_non_cnn/results.csv';
    if ~exist(results_file, 'file')
        error('Results file not found: %s\nPlease run task8_non_cnn.m first', results_file);
    end

    T = readtable(results_file);
    outdir = 'output/task8_non_cnn';

    fprintf('Generating Task 8 sensitivity plots...\n');

    %% 1. Dictionary Size (K) Sensitivity
    fprintf('  [1/5] K sensitivity...\n');
    % Fix: vote=soft, spm=1, C=1
    idx = strcmp(T.vote, 'soft') & T.spm == 1 & abs(T.C - 1) < 1e-9;
    if sum(idx) > 0
        Tsub = T(idx, :);
        G = groupsummary(Tsub, 'K', 'max', 'acc');

        figure('Color', 'w', 'Position', [100, 100, 800, 500]);
        plot(G.K, G.max_acc * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
             'MarkerFaceColor', [0.2, 0.4, 0.8]);
        grid on;
        xlabel('Dictionary Size (K)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Sensitivity to Dictionary Size K (soft voting, SPM, C=1)', ...
              'FontSize', 13, 'FontWeight', 'bold');
        ylim([min(G.max_acc * 100) - 2, max(G.max_acc * 100) + 2]);
        set(gca, 'FontSize', 11);
        saveas(gcf, fullfile(outdir, 'sensitivity_K.png'));
        close(gcf);
    end

    %% 2. Voting Type Comparison (Hard vs Soft)
    fprintf('  [2/5] Voting type comparison...\n');
    % Fix: spm=1, C=1
    idx = T.spm == 1 & abs(T.C - 1) < 1e-9;
    if sum(idx) > 0
        Tsub = T(idx, :);
        G = groupsummary(Tsub, 'vote', 'mean', 'acc');

        figure('Color', 'w', 'Position', [100, 100, 800, 500]);
        bar(categorical(G.vote), G.mean_acc * 100, 'FaceColor', [0.3, 0.6, 0.4]);
        grid on;
        ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Voting Strategy Comparison (with SPM, C=1)', ...
              'FontSize', 13, 'FontWeight', 'bold');
        set(gca, 'FontSize', 11);
        ylim([min(G.mean_acc * 100) - 2, max(G.mean_acc * 100) + 2]);
        saveas(gcf, fullfile(outdir, 'voting_comparison.png'));
        close(gcf);
    end

    %% 3. Spatial Pyramid Effect (SPM vs No-SPM)
    fprintf('  [3/5] SPM comparison...\n');
    % Fix: vote=soft, C=1
    idx = strcmp(T.vote, 'soft') & abs(T.C - 1) < 1e-9;
    if sum(idx) > 0
        Tsub = T(idx, :);
        G = groupsummary(Tsub, 'spm', 'mean', 'acc');

        figure('Color', 'w', 'Position', [100, 100, 800, 500]);
        bar(categorical({'No SPM', 'With SPM (1x1+2x2)'}), G.mean_acc * 100, ...
            'FaceColor', [0.8, 0.3, 0.3]);
        grid on;
        ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Spatial Pyramid Effect (soft voting, C=1)', ...
              'FontSize', 13, 'FontWeight', 'bold');
        set(gca, 'FontSize', 11);
        ylim([min(G.mean_acc * 100) - 2, max(G.mean_acc * 100) + 2]);
        saveas(gcf, fullfile(outdir, 'spm_comparison.png'));
        close(gcf);
    end

    %% 4. SVM C Parameter Sensitivity
    fprintf('  [4/5] SVM C sensitivity...\n');
    % Fix: vote=soft, spm=1
    idx = strcmp(T.vote, 'soft') & T.spm == 1;
    if sum(idx) > 0
        Tsub = T(idx, :);
        G = groupsummary(Tsub, 'C', 'max', 'acc');

        figure('Color', 'w', 'Position', [100, 100, 800, 500]);
        semilogx(G.C, G.max_acc * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
                 'MarkerFaceColor', [0.6, 0.2, 0.6]);
        grid on;
        xlabel('SVM C Parameter', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
        title('SVM Regularization Sensitivity (soft voting, SPM)', ...
              'FontSize', 13, 'FontWeight', 'bold');
        set(gca, 'FontSize', 11);
        ylim([min(G.max_acc * 100) - 2, max(G.max_acc * 100) + 2]);
        saveas(gcf, fullfile(outdir, 'sensitivity_C.png'));
        close(gcf);
    end

    %% 5. Accuracy vs Training Time (Pareto Frontier)
    fprintf('  [5/5] Accuracy-time trade-off...\n');
    figure('Color', 'w', 'Position', [100, 100, 800, 600]);
    scatter(T.train_sec, T.acc * 100, 50, T.K, 'filled', 'MarkerEdgeColor', 'k');
    colormap('jet');
    cb = colorbar;
    cb.Label.String = 'Dictionary Size K';
    cb.Label.FontSize = 11;
    grid on;
    xlabel('Training Time (seconds)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Accuracy vs Training Time Trade-off', 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'FontSize', 11);
    saveas(gcf, fullfile(outdir, 'pareto_time_accuracy.png'));
    close(gcf);

    fprintf('Sensitivity plots saved to: %s\n', outdir);
    fprintf('  - sensitivity_K.png\n');
    fprintf('  - voting_comparison.png\n');
    fprintf('  - spm_comparison.png\n');
    fprintf('  - sensitivity_C.png\n');
    fprintf('  - pareto_time_accuracy.png\n');
end
