% Debug script for myConvn
clear; close all;

% Get project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts/tests
scripts_dir = fileparts(script_dir);  % src/scripts
src_dir = fileparts(scripts_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));

fprintf('=== Debug myConvn ===\n\n');

% Simple test case
fprintf('Test: 2D valid convolution (simple case)\n');
A = [1 2 3; 4 5 6; 7 8 9];
B = [1 0; 0 1];

fprintf('Input A:\n');
disp(A);
fprintf('Kernel B:\n');
disp(B);

result_convn = convn(A, B, 'valid');
fprintf('\nconvn result:\n');
disp(result_convn);
fprintf('Size: %s\n', mat2str(size(result_convn)));

result_myConvn = myConvn(A, B, 'valid');
fprintf('\nmyConvn result:\n');
disp(result_myConvn);
fprintf('Size: %s\n', mat2str(size(result_myConvn)));

fprintf('\nDifference:\n');
disp(result_convn - result_myConvn);

% Manual calculation
fprintf('\n=== Manual calculation ===\n');
fprintf('Expected output should be:\n');
fprintf('  (1,1): A(1:2,1:2) .* B = [1 2; 4 5] .* [1 0; 0 1] = 1*1 + 2*0 + 4*0 + 5*1 = %d\n', 1*1 + 2*0 + 4*0 + 5*1);
fprintf('  (1,2): A(1:2,2:3) .* B = [2 3; 5 6] .* [1 0; 0 1] = 2*1 + 3*0 + 5*0 + 6*1 = %d\n', 2*1 + 3*0 + 5*0 + 6*1);
fprintf('  (2,1): A(2:3,1:2) .* B = [4 5; 7 8] .* [1 0; 0 1] = 4*1 + 5*0 + 7*0 + 8*1 = %d\n', 4*1 + 5*0 + 7*0 + 8*1);
fprintf('  (2,2): A(2:3,2:3) .* B = [5 6; 8 9] .* [1 0; 0 1] = 5*1 + 6*0 + 8*0 + 9*1 = %d\n', 5*1 + 6*0 + 8*0 + 9*1);
