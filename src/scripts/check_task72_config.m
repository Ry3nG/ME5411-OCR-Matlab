













load('output/task7_2/som_model.mat');
fprintf('Task 7.2 SOM:\n');
fprintf('  Grid: %dx%d\n', size(som.W, 1), size(som.W, 2));
fprintf('  Codebook size: %d\n', size(som.W, 1) * size(som.W, 2));
fprintf('  Weight shape: [%s]\n', mat2str(size(som.W)));

load('output/task7_2/results.mat');
fprintf('\nTask 7.2 Config:\n');
fprintf('  Patch size: %d\n', results.config.patch_size);
fprintf('  Stride: %d\n', results.config.stride);
fprintf('  Spatial pyramid: %d\n', results.config.spatial_pyramid);
fprintf('  Soft voting sigma: %.2f\n', results.config.sigma_bow);
