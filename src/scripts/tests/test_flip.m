% Test if convn flips the kernel
clear; close all;

% Get project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts/tests
scripts_dir = fileparts(script_dir);  % src/scripts
src_dir = fileparts(scripts_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));

fprintf('=== Testing if convn flips kernel ===\n\n');

rng(42);
A = rand(10, 10);
B = rand(3, 3);

% Standard convn
result_convn = convn(A, B, 'valid');

% Manual correlation (what I implemented)
region = A(1:3, 1:3);
correlation_result = sum(sum(region .* B));

% Manual convolution (flip kernel)
flipped_B = rot90(rot90(B)); % 180 degree rotation
convolution_result = sum(sum(region .* flipped_B));

fprintf('At position (1,1):\n');
fprintf('convn result: %.15f\n', result_convn(1,1));
fprintf('Correlation (no flip): %.15f\n', correlation_result);
fprintf('Convolution (with flip): %.15f\n', convolution_result);

fprintf('\nDifference:\n');
fprintf('|convn - correlation|: %.15f\n', abs(result_convn(1,1) - correlation_result));
fprintf('|convn - convolution|: %.15f\n', abs(result_convn(1,1) - convolution_result));

if abs(result_convn(1,1) - convolution_result) < 1e-10
    fprintf('\n✓ convn performs CONVOLUTION (flips kernel)\n');
else
    fprintf('\n✓ convn performs CORRELATION (no flip)\n');
end
