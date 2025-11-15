

































































































% Check training time from task8_2_hyper results
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

load('output/task8_2_hyper/codebook_size/results.mat');
fprintf('=== Codebook Size ===\n');
fprintf('Values: %s\n', mat2str(group_results.param_values));
fprintf('Train Time (sec): %s\n', mat2str(round(group_results.train_time, 1)));
fprintf('Train Time (min): %s\n\n', mat2str(round(group_results.train_time / 60, 2)));

load('output/task8_2_hyper/spatial_pyramid/results.mat');
fprintf('=== Spatial Pyramid ===\n');
fprintf('Values: %s\n', mat2str(group_results.param_values));
fprintf('Train Time (sec): %s\n', mat2str(round(group_results.train_time, 1)));
fprintf('Train Time (min): %s\n\n', mat2str(round(group_results.train_time / 60, 2)));

load('output/task8_2_hyper/soft_voting_sigma/results.mat');
fprintf('=== Soft Voting Sigma ===\n');
fprintf('Values: %s\n', mat2str(group_results.param_values));
fprintf('Train Time (sec): %s\n', mat2str(round(group_results.train_time, 1)));
fprintf('Train Time (min): %s\n\n', mat2str(round(group_results.train_time / 60, 2)));
