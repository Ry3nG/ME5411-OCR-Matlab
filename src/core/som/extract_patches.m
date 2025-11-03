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
%                  'min_patch_std' (default: 0.005) - Reject patches with std below this threshold
%
% Outputs:
%   patches - [num_samples x (patch_size^2)] matrix of vectorized patches

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'normalize', true);
    addParameter(p, 'verbose', true);
    addParameter(p, 'min_patch_std', 0.005);
    parse(p, varargin{:});

    normalize_patches = p.Results.normalize;
    verbose = p.Results.verbose;
    min_patch_std = p.Results.min_patch_std;

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

    % Random sampling with blank-patch rejection
    count = 0;
    attempts = 0;
    max_attempts = num_samples * 20;
    while count < num_samples && attempts < max_attempts
        attempts = attempts + 1;

        % Randomly select an image
        img_idx = randi(N);
        img = double(squeeze(data(:, :, 1, img_idx))) / 255.0;

        % Randomly select patch location
        y = randi(max_y);
        x = randi(max_x);

        % Extract patch
        patch = img(y:y+patch_size-1, x:x+patch_size-1);
        patch_vec = patch(:)';

        % Reject low-variance (near-blank) patches
        patch_std = std(patch_vec);
        if patch_std < min_patch_std
            continue;
        end

        % Normalize patch (zero mean, unit variance)
        if normalize_patches
            patch_mean = mean(patch_vec);
            if patch_std > 1e-6
                patch_vec = (patch_vec - patch_mean) / patch_std;
            else
                patch_vec = patch_vec - patch_mean;
            end
        end

        count = count + 1;
        patches(count, :) = patch_vec;

        % Progress display
        if verbose && mod(count, max(1, floor(num_samples/10))) == 0
            fprintf('  Extracted %d/%d patches (%.1f%%)\n', ...
                count, num_samples, count/num_samples*100);
        end
    end

    if count < num_samples
        patches = patches(1:count, :);
        if verbose
            fprintf('Warning: only collected %d/%d patches after %d attempts (increase dataset or lower min_patch_std).\n', ...
                count, num_samples, attempts);
        end
    end

    if verbose
        fprintf('Patch extraction completed.\n');
        fprintf('  Patch statistics: mean=%.4f, std=%.4f\n', ...
            mean(patches(:)), std(patches(:)));
    end
end
