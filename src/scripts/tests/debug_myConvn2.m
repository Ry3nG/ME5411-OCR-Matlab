% Detailed test to find the specific issue
clear; close all;

% Get project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts/tests
scripts_dir = fileparts(script_dir);  % src/scripts
src_dir = fileparts(scripts_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));

fprintf('=== Detailed Test ===\n\n');

% Test 1: Check if the issue is related to size
fprintf('Test 1: Same size as original test\n');
rng(42); % Fixed seed for reproducibility
A = rand(10, 10);
B = rand(3, 3);

result_convn = convn(A, B, 'valid');
result_myConvn = myConvn(A, B, 'valid');

fprintf('convn output size: %s\n', mat2str(size(result_convn)));
fprintf('myConvn output size: %s\n', mat2str(size(result_myConvn)));
fprintf('Max absolute difference: %.15f\n', max(abs(result_convn(:) - result_myConvn(:))));
fprintf('Sample values (convn): %.10f, %.10f, %.10f\n', result_convn(1,1), result_convn(1,2), result_convn(2,1));
fprintf('Sample values (myConvn): %.10f, %.10f, %.10f\n', result_myConvn(1,1), result_myConvn(1,2), result_myConvn(2,1));

% Manual verification for position (1,1)
fprintf('\n=== Manual verification for position (1,1) ===\n');
region = A(1:3, 1:3);
manual_result = sum(sum(region .* B));
fprintf('Manual calculation: %.15f\n', manual_result);
fprintf('convn result: %.15f\n', result_convn(1,1));
fprintf('myConvn result: %.15f\n', result_myConvn(1,1));
