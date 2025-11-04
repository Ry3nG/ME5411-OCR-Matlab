% Test 3D kernel flipping behavior
clear; close all;

% Get project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts/tests
scripts_dir = fileparts(script_dir);  % src/scripts
src_dir = fileparts(scripts_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));

fprintf('=== Testing 3D convolution ===\n\n');

rng(42);
A = rand(10, 10, 3);
B = rand(5, 5, 3);

result_convn = convn(A, B, 'valid');

% Test different flipping strategies
fprintf('Testing manual calculation with different flip strategies:\n\n');

% No flip
region = A(1:5, 1:5, 1:3);
no_flip = sum(region(:) .* B(:));
fprintf('No flip: %.15f\n', no_flip);

% Flip first two dims only
B_flip_xy = B(end:-1:1, end:-1:1, :);
flip_xy = sum(region(:) .* B_flip_xy(:));
fprintf('Flip XY only: %.15f\n', flip_xy);

% Flip all three dims
B_flip_xyz = B(end:-1:1, end:-1:1, end:-1:1);
flip_xyz = sum(region(:) .* B_flip_xyz(:));
fprintf('Flip XYZ: %.15f\n', flip_xyz);

fprintf('\nconvn result at (1,1,1): %.15f\n', result_convn(1,1,1));

fprintf('\nDifferences:\n');
fprintf('|convn - no flip|: %.15f\n', abs(result_convn(1,1,1) - no_flip));
fprintf('|convn - flip XY|: %.15f\n', abs(result_convn(1,1,1) - flip_xy));
fprintf('|convn - flip XYZ|: %.15f\n', abs(result_convn(1,1,1) - flip_xyz));
