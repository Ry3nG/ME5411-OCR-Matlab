% task8_run_experiments.m
% Main experiment runner for Task 8 sensitivity analysis
% Runs all ablation experiments with checkpoint/resume support
%
% Usage:
%   Run from project root: matlab -batch "run('src/task8_run_experiments.m')"
%   Or in MATLAB: cd to project root, then run('src/task8_run_experiments.m')
%
% Features:
%   - Automatic resume: skips already completed experiments
%   - Real-time logging to file and console
%   - Summary statistics saved after each experiment
%   - Robust error handling

clear all; %#ok<CLALL>
close all;

%% Setup
fprintf('\n');
fprintf('=========================================\n');
fprintf('   Task 8: Sensitivity Analysis Runner  \n');
fprintf('=========================================\n');
fprintf('Started at: %s\n', datetime('now'));
fprintf('=========================================\n\n');

% Get project root
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

% Add paths
addpath(genpath('src'));

% Create output directory
output_root = fullfile('output', 'task8');
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

% Setup master log
master_log_file = fullfile(output_root, 'master_log.txt');
master_diary = fopen(master_log_file, 'a');  % Append mode for resume
fprintf(master_diary, '\n\n========== NEW RUN: %s ==========\n', datetime('now'));
fclose(master_diary);

%% Define Experiments
% Format: {exp_id, name, aug_trans, aug_rot, aug_scale, lr, lr_method, batch_size}
% Note: exp00 (baseline) saves to output/task7_1 instead of output/task8/exp00

experiments = {
    % Baseline experiment (Group 0) - saves to output/task7_1
    {'exp00', 'baseline',     0,    [0 0],      [1 1],     0.1,  'linear', 128}

    % Preprocessing experiments (Group A)
    {'exp01', 'translation',  0.02, [0 0],      [1 1],     0.1,  'linear', 128}
    {'exp02', 'rotation',     0,    [-15 15],   [1 1],     0.1,  'linear', 128}
    {'exp03', 'scale',        0,    [0 0],      [0.9 1.1], 0.1,  'linear', 128}
    {'exp04', 'all_aug',      0.02, [-15 15],   [0.9 1.1], 0.1,  'linear', 128}

    % LR magnitude experiments (Group B1)
    {'exp05', 'lr0.05',       0,    [0 0],      [1 1],     0.05, 'linear', 128}
    {'exp06', 'lr0.2',        0,    [0 0],      [1 1],     0.2,  'linear', 128}
    {'exp07', 'lr0.3',        0,    [0 0],      [1 1],     0.3,  'linear', 128}

    % LR schedule experiments (Group B2)
    {'exp08', 'cosine',       0,    [0 0],      [1 1],     0.1,  'cosine', 128}
    {'exp09', 'exp_decay',    0,    [0 0],      [1 1],     0.1,  'exp',    128}
    {'exp10', 'step_decay',   0,    [0 0],      [1 1],     0.1,  'step',   128}

    % Batch size experiments (Group B3)
    {'exp11', 'bs64',         0,    [0 0],      [1 1],     0.1,  'linear', 64}
    {'exp12', 'bs256',        0,    [0 0],      [1 1],     0.1,  'linear', 256}
};

num_experiments = length(experiments);
fprintf('Total experiments to run: %d\n', num_experiments);
fprintf('Output directory: %s\n\n', output_root);

%% Check for completed experiments
completed_experiments = {};
pending_experiments = {};

fprintf('Checking for completed experiments...\n');
for i = 1:num_experiments
    exp_id = experiments{i}{1};

    % Special handling for exp00 (baseline) - check in output/task7_1
    if strcmp(exp_id, 'exp00')
        % Check if any results.mat exists in output/task7_1 subdirectories
        task7_output = fullfile('output', 'task7_1');
        if exist(task7_output, 'dir')
            subdirs = dir(fullfile(task7_output, '*-*_*-*-*'));  % Timestamp pattern
            found_results = false;
            for j = 1:length(subdirs)
                if subdirs(j).isdir
                    results_file = fullfile(task7_output, subdirs(j).name, 'results.mat');
                    if exist(results_file, 'file')
                        found_results = true;
                        break;
                    end
                end
            end
            if found_results
                fprintf('  [✓] %s - Already completed (found in output/task7_1)\n', exp_id);
                completed_experiments{end+1} = exp_id; %#ok<AGROW>
            else
                fprintf('  [ ] %s - Pending\n', exp_id);
                pending_experiments{end+1} = i; %#ok<AGROW>
            end
        else
            fprintf('  [ ] %s - Pending\n', exp_id);
            pending_experiments{end+1} = i; %#ok<AGROW>
        end
    else
        % Normal handling for other experiments
        exp_output_dir = fullfile(output_root, exp_id);
        results_file = fullfile(exp_output_dir, 'results.mat');

        if exist(results_file, 'file')
            fprintf('  [✓] %s - Already completed (found results.mat)\n', exp_id);
            completed_experiments{end+1} = exp_id; %#ok<AGROW>
        else
            fprintf('  [ ] %s - Pending\n', exp_id);
            pending_experiments{end+1} = i; %#ok<AGROW>
        end
    end
end

fprintf('\nSummary:\n');
fprintf('  Completed: %d/%d\n', length(completed_experiments), num_experiments);
fprintf('  Pending: %d/%d\n', length(pending_experiments), num_experiments);

if isempty(pending_experiments)
    fprintf('\n✓ All experiments already completed!\n');
    fprintf('Results can be found in: %s\n', output_root);
    return;
end

fprintf('\n=========================================\n');
fprintf('Starting %d pending experiments...\n', length(pending_experiments));
fprintf('=========================================\n\n');

%% Run experiments
all_results = cell(num_experiments, 1);
failed_experiments = {};

for idx = 1:length(pending_experiments)
    exp_idx = pending_experiments{idx};
    exp_config = experiments{exp_idx};
    exp_id = exp_config{1};
    exp_name = exp_config{2};

    fprintf('\n');
    fprintf('═════════════════════════════════════════\n');
    fprintf('Experiment %d/%d: %s (%s)\n', idx, length(pending_experiments), exp_id, exp_name);
    fprintf('═════════════════════════════════════════\n');

    % Log to master log
    master_diary = fopen(master_log_file, 'a');
    fprintf(master_diary, '\n[%s] Starting %s (%s)\n', datetime('now'), exp_id, exp_name);
    fclose(master_diary);

    % Run experiment with error handling
    try
        exp_start_time = tic;

        % Run the experiment
        result = task8_single_experiment(exp_config);

        exp_duration = toc(exp_start_time);

        % Store result
        all_results{exp_idx} = result;

        % Log success
        master_diary = fopen(master_log_file, 'a');
        fprintf(master_diary, '[%s] Completed %s - Test Acc: %.4f, Time: %.1f min\n', ...
            datetime('now'), exp_id, result.test_acc, exp_duration/60);
        fclose(master_diary);

        fprintf('\n');
        fprintf('✓ Experiment %s completed successfully!\n', exp_id);
        fprintf('  Test Accuracy: %.2f%%\n', result.test_acc * 100);
        fprintf('  Training Time: %.1f minutes\n', exp_duration / 60);
        fprintf('═════════════════════════════════════════\n');

    catch ME
        % Log error
        fprintf('\n');
        fprintf('✗ Experiment %s FAILED with error:\n', exp_id);
        fprintf('  %s\n', ME.message);
        fprintf('  Continuing with next experiment...\n');
        fprintf('═════════════════════════════════════════\n');

        master_diary = fopen(master_log_file, 'a');
        fprintf(master_diary, '[%s] FAILED %s - Error: %s\n', ...
            datetime('now'), exp_id, ME.message);
        fclose(master_diary);

        failed_experiments{end+1} = struct('exp_id', exp_id, 'error', ME.message); %#ok<AGROW>

        % Save error log
        error_log_file = fullfile(output_root, exp_id, 'error_log.txt');
        if ~exist(fullfile(output_root, exp_id), 'dir')
            mkdir(fullfile(output_root, exp_id));
        end
        error_fid = fopen(error_log_file, 'w');
        fprintf(error_fid, 'Experiment failed at: %s\n\n', datetime('now'));
        fprintf(error_fid, 'Error message:\n%s\n\n', ME.message);
        fprintf(error_fid, 'Stack trace:\n');
        for k = 1:length(ME.stack)
            fprintf(error_fid, '  %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
        fclose(error_fid);
    end

    % Save intermediate summary after each experiment
    summary_file = fullfile(output_root, 'summary_intermediate.mat');
    save(summary_file, 'all_results', 'completed_experiments', 'failed_experiments');
end

%% Generate Final Summary
fprintf('\n');
fprintf('=========================================\n');
fprintf('   ALL EXPERIMENTS COMPLETE              \n');
fprintf('=========================================\n');
fprintf('Finished at: %s\n', datetime('now'));

% Collect all results (including previously completed ones)
fprintf('\nCollecting all results...\n');
for i = 1:num_experiments
    if isempty(all_results{i})
        % Load from file if not in memory
        exp_id = experiments{i}{1};

        % Special handling for exp00 - load from output/task7_1
        if strcmp(exp_id, 'exp00')
            task7_output = fullfile('output', 'task7_1');
            if exist(task7_output, 'dir')
                subdirs = dir(fullfile(task7_output, '*-*_*-*-*'));  % Timestamp pattern
                for j = 1:length(subdirs)
                    if subdirs(j).isdir
                        results_file = fullfile(task7_output, subdirs(j).name, 'results.mat');
                        if exist(results_file, 'file')
                            loaded = load(results_file);
                            all_results{i} = loaded.results;
                            fprintf('  Loaded %s from output/task7_1/%s\n', exp_id, subdirs(j).name);
                            break;
                        end
                    end
                end
            end
        else
            % Normal handling for other experiments
            results_file = fullfile(output_root, exp_id, 'results.mat');
            if exist(results_file, 'file')
                loaded = load(results_file);
                all_results{i} = loaded.results;
                fprintf('  Loaded %s from disk\n', exp_id);
            end
        end
    end
end

% Create summary table
summary = struct();
summary.exp_ids = {};
summary.exp_names = {};
summary.test_acc = [];
summary.train_acc = [];
summary.train_test_gap = [];
summary.training_time = [];
summary.configs = {};

for i = 1:num_experiments
    if ~isempty(all_results{i})
        r = all_results{i};
        summary.exp_ids{end+1} = r.exp_id;
        summary.exp_names{end+1} = r.exp_name;
        summary.test_acc(end+1) = r.test_acc;
        summary.train_acc(end+1) = r.train_acc;
        summary.train_test_gap(end+1) = r.train_test_gap;
        summary.training_time(end+1) = r.training_time;
        summary.configs{end+1} = r.config;
    end
end

% Save summary
summary_file = fullfile(output_root, 'summary.mat');
save(summary_file, 'summary', 'all_results');
fprintf('\nSummary saved to: %s\n', summary_file);

% Save summary as text
summary_txt_file = fullfile(output_root, 'summary.txt');
fid = fopen(summary_txt_file, 'w');
fprintf(fid, 'Task 8 Sensitivity Analysis - Results Summary\n');
fprintf(fid, '==============================================\n');
fprintf(fid, 'Generated: %s\n\n', datetime('now'));

fprintf(fid, '%-8s %-15s %10s %10s %10s %10s\n', ...
    'Exp ID', 'Name', 'Test Acc', 'Train Acc', 'Gap', 'Time(min)');
fprintf(fid, '%s\n', repmat('-', 1, 75));

for i = 1:length(summary.exp_ids)
    fprintf(fid, '%-8s %-15s %9.2f%% %9.2f%% %9.2f%% %10.1f\n', ...
        summary.exp_ids{i}, summary.exp_names{i}, ...
        summary.test_acc(i)*100, summary.train_acc(i)*100, ...
        summary.train_test_gap(i)*100, summary.training_time(i)/60);
end

fprintf(fid, '\n\nStatistics:\n');
fprintf(fid, '  Best test accuracy: %.2f%% (%s)\n', ...
    max(summary.test_acc)*100, summary.exp_ids{find(summary.test_acc == max(summary.test_acc), 1)});
fprintf(fid, '  Worst test accuracy: %.2f%% (%s)\n', ...
    min(summary.test_acc)*100, summary.exp_ids{find(summary.test_acc == min(summary.test_acc), 1)});
fprintf(fid, '  Average test accuracy: %.2f%%\n', mean(summary.test_acc)*100);
fprintf(fid, '  Std dev test accuracy: %.2f%%\n', std(summary.test_acc)*100);
fprintf(fid, '  Total training time: %.1f hours\n', sum(summary.training_time)/3600);

if ~isempty(failed_experiments)
    fprintf(fid, '\n\nFailed Experiments: %d\n', length(failed_experiments));
    for i = 1:length(failed_experiments)
        fprintf(fid, '  %s: %s\n', failed_experiments{i}.exp_id, failed_experiments{i}.error);
    end
end

fclose(fid);

% Display summary to console
fprintf('\n');
fprintf('==============================================\n');
fprintf('%-8s %-15s %10s %10s\n', 'Exp ID', 'Name', 'Test Acc', 'Time(min)');
fprintf('%s\n', repmat('-', 1, 50));
for i = 1:length(summary.exp_ids)
    fprintf('%-8s %-15s %9.2f%% %10.1f\n', ...
        summary.exp_ids{i}, summary.exp_names{i}, ...
        summary.test_acc(i)*100, summary.training_time(i)/60);
end
fprintf('==============================================\n');

fprintf('\nStatistics:\n');
fprintf('  Best: %.2f%% (%s)\n', max(summary.test_acc)*100, ...
    summary.exp_ids{find(summary.test_acc == max(summary.test_acc), 1)});
fprintf('  Worst: %.2f%% (%s)\n', min(summary.test_acc)*100, ...
    summary.exp_ids{find(summary.test_acc == min(summary.test_acc), 1)});
fprintf('  Average: %.2f%% ± %.2f%%\n', mean(summary.test_acc)*100, std(summary.test_acc)*100);
fprintf('  Range: %.2f%%\n', (max(summary.test_acc) - min(summary.test_acc))*100);

if ~isempty(failed_experiments)
    fprintf('\n⚠ Warning: %d experiments failed\n', length(failed_experiments));
    for i = 1:length(failed_experiments)
        fprintf('  - %s\n', failed_experiments{i}.exp_id);
    end
end

fprintf('\n✓ All results saved to: %s\n', output_root);
fprintf('  - summary.mat: MATLAB summary data\n');
fprintf('  - summary.txt: Text summary\n');
fprintf('  - master_log.txt: Complete execution log\n');
fprintf('  - exp##/: Individual experiment results\n');

fprintf('\n=========================================\n');
fprintf('Task 8 experiment suite completed!\n');
fprintf('=========================================\n\n');

% Log completion to master log
master_diary = fopen(master_log_file, 'a');
fprintf(master_diary, '\n[%s] ========== RUN COMPLETE ==========\n', datetime('now'));
fprintf(master_diary, 'Completed: %d/%d experiments\n', length(summary.exp_ids), num_experiments);
if ~isempty(failed_experiments)
    fprintf(master_diary, 'Failed: %d experiments\n', length(failed_experiments));
end
fprintf(master_diary, 'Total time: %.1f hours\n', sum(summary.training_time)/3600);
fclose(master_diary);
