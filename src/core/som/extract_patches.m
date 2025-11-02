function patches = extract_patches(data, labels, patch_size, num_samples, varargin)
% EXTRACT_PATCHES Extract image patches for SOM training
%
% Inputs:
%   data         - [H x W x C x N] image data
%   labels       - [N x 1] labels (can be empty for test data)
%   patch_size   - Scalar, size of square patches (e.g., 8)
%   num_samples  - Number of patches to sample
%   varargin     - Optional parameters:
%                  'normalize' (default: true) - Apply patch normalization
%                  'verbose' (default: true)
%
% Outputs:
%   patches - [num_samples x (patch_size^2)] matrix of vectorized patches

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'normalize', true);
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});

    normalize_patches = p.Results.normalize;
    verbose = p.Results.verbose;

    [H, W, ~, N] = size(data);
    patch_dim = patch_size * patch_size;

    if verbose
        fprintf('Extracting %d patches of size %dx%d from %d images...\n', ...
            num_samples, patch_size, patch_size, N);
    end

    % Pre-allocate patch storage
    patches = zeros(num_samples, patch_dim);

    % Calculate maximum valid starting positions
    max_y = H - patch_size + 1;
    max_x = W - patch_size + 1;

    if max_y <= 0 || max_x <= 0
        error('Patch size %d is too large for images of size %dx%d', ...
            patch_size, H, W);
    end

    % Random sampling
    for i = 1:num_samples
        % Randomly select an image
        img_idx = randi(N);
        img = double(squeeze(data(:, :, 1, img_idx))) / 255.0;

        % Randomly select patch location
        y = randi(max_y);
        x = randi(max_x);

        % Extract patch
        patch = img(y:y+patch_size-1, x:x+patch_size-1);
        patch_vec = patch(:)';

        % Normalize patch (zero mean, unit variance)
        if normalize_patches
            patch_mean = mean(patch_vec);
            patch_std = std(patch_vec);
            if patch_std > 1e-6
                patch_vec = (patch_vec - patch_mean) / patch_std;
            else
                patch_vec = patch_vec - patch_mean;
            end
        end

        patches(i, :) = patch_vec;

        % Progress display
        if verbose && mod(i, num_samples/10) == 0
            fprintf('  Extracted %d/%d patches (%.1f%%)\n', ...
                i, num_samples, i/num_samples*100);
        end
    end

    if verbose
        fprintf('Patch extraction completed.\n');
        fprintf('  Patch statistics: mean=%.4f, std=%.4f\n', ...
            mean(patches(:)), std(patches(:)));
    end
end
