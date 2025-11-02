function features = extract_bow_features(data, som_model, stride, varargin)
% EXTRACT_BOW_FEATURES Extract Bag-of-Visual-Words features using SOM codebook
%
% Inputs:
%   data       - [H x W x C x N] image data
%   som_model  - SOM model from train_som
%   stride     - Stride for sliding window patch extraction
%   varargin   - Optional parameters:
%                'normalize' (default: true) - Apply patch normalization
%                'norm_type' (default: 'l2') - 'l1' or 'l2' histogram normalization
%                'soft_voting' (default: true) - Use soft voting with SOM neighborhood
%                'sigma_bow' (default: 1.0) - Neighborhood sigma for soft voting
%                'spatial_pyramid' (default: false) - Use 1x1 + 2x2 spatial pyramid
%                'verbose' (default: true)
%
% Outputs:
%   features - [N x D] BoW histogram features (D depends on spatial_pyramid)

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'normalize', true);
    addParameter(p, 'norm_type', 'l2');
    addParameter(p, 'soft_voting', true);
    addParameter(p, 'sigma_bow', 1.0);
    addParameter(p, 'spatial_pyramid', false);
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});

    normalize_patches = p.Results.normalize;
    norm_type = p.Results.norm_type;
    soft_voting = p.Results.soft_voting;
    sigma_bow = p.Results.sigma_bow;
    spatial_pyramid = p.Results.spatial_pyramid;
    verbose = p.Results.verbose;

    [H, W, ~, N] = size(data);
    num_neurons = som_model.num_neurons;
    patch_size = sqrt(som_model.patch_dim);

    % Precompute pairwise distances between neurons for soft voting
    if soft_voting
        neuron_coords = som_model.grid_coords;
        % Distance matrix: [num_neurons x num_neurons]
        dist_matrix = zeros(num_neurons, num_neurons);
        for i = 1:num_neurons
            for j = 1:num_neurons
                dist_matrix(i, j) = norm(neuron_coords(i, :) - neuron_coords(j, :));
            end
        end
        % Precompute Gaussian weights
        weight_matrix = exp(-dist_matrix.^2 / (2 * sigma_bow^2));
    end

    if verbose
        fprintf('Extracting BoW features from %d images...\n', N);
        fprintf('  SOM codebook: %d visual words\n', num_neurons);
        fprintf('  Patch size: %dx%d, Stride: %d\n', patch_size, patch_size, stride);
        if soft_voting
            fprintf('  Using soft voting with sigma=%.2f\n', sigma_bow);
        else
            fprintf('  Using hard voting (single BMU)\n');
        end
        if spatial_pyramid
            fprintf('  Using spatial pyramid (1x1 + 2x2)\n');
        end
    end

    % Determine feature dimension
    if spatial_pyramid
        feature_dim = num_neurons * 5;  % 1x1 + 2x2 = 5 blocks
    else
        feature_dim = num_neurons;
    end

    % Pre-allocate feature matrix
    features = zeros(N, feature_dim);

    % Process each image
    for i = 1:N
        img = double(squeeze(data(:, :, 1, i))) / 255.0;

        if spatial_pyramid
            % Extract features from 5 spatial regions
            histograms = cell(5, 1);

            % 1. Global (1x1)
            histograms{1} = extract_region_histogram(img, 1, H, 1, W, ...
                som_model, patch_size, stride, normalize_patches, ...
                soft_voting, weight_matrix);

            % 2-5. Four quadrants (2x2)
            mid_h = floor(H / 2);
            mid_w = floor(W / 2);
            regions = {[1, mid_h, 1, mid_w], ...           % Top-left
                      [1, mid_h, mid_w+1, W], ...         % Top-right
                      [mid_h+1, H, 1, mid_w], ...         % Bottom-left
                      [mid_h+1, H, mid_w+1, W]};          % Bottom-right

            for r = 1:4
                reg = regions{r};
                histograms{r+1} = extract_region_histogram(img, reg(1), reg(2), reg(3), reg(4), ...
                    som_model, patch_size, stride, normalize_patches, ...
                    soft_voting, weight_matrix);
            end

            % Concatenate all histograms
            features(i, :) = [histograms{:}];
        else
            % Single global histogram
            histogram = extract_region_histogram(img, 1, H, 1, W, ...
                som_model, patch_size, stride, normalize_patches, ...
                soft_voting, weight_matrix);
            features(i, :) = histogram;
        end

        % Progress display
        if verbose && mod(i, max(1, floor(N/10))) == 0
            fprintf('  Processed %d/%d images (%.1f%%)\n', i, N, i/N*100);
        end
    end

    if verbose
        fprintf('BoW feature extraction completed.\n');
        fprintf('  Feature dimension: %d\n', feature_dim);
        fprintf('  Feature statistics: mean=%.4f, std=%.4f\n', ...
            mean(features(:)), std(features(:)));
    end
end

function histogram = extract_region_histogram(img, row_start, row_end, col_start, col_end, ...
    som_model, patch_size, stride, normalize_patches, soft_voting, weight_matrix)
% Extract histogram from a specific region of the image

    num_neurons = som_model.num_neurons;
    histogram = zeros(1, num_neurons);

    % Sliding window extraction in the region
    for y = row_start:stride:(row_end - patch_size + 1)
        for x = col_start:stride:(col_end - patch_size + 1)
            % Extract patch
            patch = img(y:y+patch_size-1, x:x+patch_size-1);
            patch_vec = patch(:)';

            % Normalize patch (same as during SOM training)
            if normalize_patches
                patch_mean = mean(patch_vec);
                patch_std = std(patch_vec);
                if patch_std > 1e-6
                    patch_vec = (patch_vec - patch_mean) / patch_std;
                else
                    patch_vec = patch_vec - patch_mean;
                end
            end

            % Find Best Matching Unit (BMU)
            distances = sum((som_model.weights - patch_vec).^2, 2);
            [~, bmu_idx] = min(distances);

            if soft_voting
                % Soft voting: distribute votes to BMU and neighbors
                histogram = histogram + weight_matrix(bmu_idx, :);
            else
                % Hard voting: only increment BMU bin
                histogram(bmu_idx) = histogram(bmu_idx) + 1;
            end
        end
    end

    % Normalize histogram
    histogram = histogram / (norm(histogram) + 1e-10);  % L2 normalization
end
