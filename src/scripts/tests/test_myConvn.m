% Quick test to verify myConvn matches convn behavior
clear; close all;

% Get project root directory
script_dir = fileparts(mfilename('fullpath'));  % src/scripts/tests
scripts_dir = fileparts(script_dir);  % src/scripts
src_dir = fileparts(scripts_dir);  % src
project_root = fileparts(src_dir);  % project root
cd(project_root);

% Add paths
addpath(genpath('src/core'));

fprintf('Testing myConvn vs convn...\n\n');

% Test 1: 2D convolution with 'valid' mode
fprintf('Test 1: 2D valid convolution\n');
A = rand(10, 10);
B = rand(3, 3);
result_convn = convn(A, B, 'valid');
result_myConvn = myConvn(A, B, 'valid');
diff1 = max(abs(result_convn(:) - result_myConvn(:)));
fprintf('  Max difference: %.10f\n', diff1);
if diff1 < 1e-10
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

% Test 2: 3D convolution with 'valid' mode
fprintf('Test 2: 3D valid convolution\n');
A = rand(10, 10, 3);
B = rand(5, 5, 3);
result_convn = convn(A, B, 'valid');
result_myConvn = myConvn(A, B, 'valid');
diff2 = max(abs(result_convn(:) - result_myConvn(:)));
fprintf('  Max difference: %.10f\n', diff2);
if diff2 < 1e-10
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

% Test 3: 2D convolution with 'full' mode
fprintf('Test 3: 2D full convolution\n');
A = rand(8, 8);
B = rand(3, 3);
result_convn = convn(A, B, 'full');
result_myConvn = myConvn(A, B, 'full');
diff3 = max(abs(result_convn(:) - result_myConvn(:)));
fprintf('  Max difference: %.10f\n', diff3);
if diff3 < 1e-10
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

% Test 4: 2D kernel on 3D input (like CNN forward pass)
fprintf('Test 4: 2D kernel on 3D input\n');
A = rand(10, 10, 16);  % Multi-channel input
B = rand(5, 5, 16);    % Multi-channel kernel
result_convn = convn(A, B, 'valid');
result_myConvn = myConvn(A, B, 'valid');
diff4 = max(abs(result_convn(:) - result_myConvn(:)));
fprintf('  Max difference: %.10f\n', diff4);
if diff4 < 1e-10
    fprintf('  ✓ PASSED\n\n');
else
    fprintf('  ✗ FAILED\n\n');
end

% Summary
fprintf('=== Test Summary ===\n');
all_passed = (diff1 < 1e-10) && (diff2 < 1e-10) && (diff3 < 1e-10) && (diff4 < 1e-10);
if all_passed
    fprintf('All tests PASSED! ✓\n');
    fprintf('myConvn is working correctly.\n');
else
    fprintf('Some tests FAILED! ✗\n');
    fprintf('Please check myConvn implementation.\n');
end
