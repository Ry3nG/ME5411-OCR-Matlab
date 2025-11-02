% task8.m
% Convenient launcher for Task 8 sensitivity analysis experiments
%
% Usage (from project root):
%   1. In MATLAB: run('task8.m')
%   2. In terminal: matlab -batch "run('task8.m')"
%   3. In tmux: matlab -batch "run('task8.m')" 2>&1 | tee task8_run.log
%
% This script will:
%   - Run all 12 ablation experiments (exp01-exp12)
%   - Skip already completed experiments (resume support)
%   - Save all results to output/task8/
%   - Generate summary statistics

fprintf('\n');
fprintf('╔═══════════════════════════════════════════╗\n');
fprintf('║   Task 8: Sensitivity Analysis           ║\n');
fprintf('║   Ablation Study Runner                  ║\n');
fprintf('╚═══════════════════════════════════════════╝\n');
fprintf('\n');

% Ensure we're in the project root
[script_path, ~, ~] = fileparts(mfilename('fullpath'));
if ~strcmp(pwd, script_path)
    fprintf('Changing directory to project root: %s\n', script_path);
    cd(script_path);
end

% Run the main experiment suite
run('src/task8_run_experiments.m');
