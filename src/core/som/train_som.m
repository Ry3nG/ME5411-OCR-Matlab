function som_model = train_som(patches, grid_size, num_iterations, varargin)
% TRAIN_SOM Train a Self-Organizing Map for visual codebook learning
%
% Inputs:
%   patches        - [M x D] matrix, M patches each of dimension D
%   grid_size      - [rows, cols] size of the SOM grid
%   num_iterations - Number of training iterations
%   varargin       - Optional parameters:
%                    'lr_init' (default: 0.5)
%                    'lr_final' (default: 0.01)
%                    'sigma_init' (default: max(grid_size)/2)
%                    'sigma_final' (default: 0.5)
%                    'verbose' (default: true)
%
% Outputs:
%   som_model - Struct containing:
%               .weights: [num_neurons x D] weight matrix
%               .grid_size: [rows, cols]
%               .grid_coords: [num_neurons x 2] neuron positions

    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'lr_init', 0.5);
    addParameter(p, 'lr_final', 0.01);
    addParameter(p, 'sigma_init', max(grid_size)/2);
    addParameter(p, 'sigma_final', 0.5);
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});

    lr_init = p.Results.lr_init;
    lr_final = p.Results.lr_final;
    sigma_init = p.Results.sigma_init;
    sigma_final = p.Results.sigma_final;
    verbose = p.Results.verbose;

    [M, D] = size(patches);
    num_neurons = grid_size(1) * grid_size(2);

    if verbose
        fprintf('Training SOM with %d neurons on %d patches...\n', num_neurons, M);
        fprintf('Grid size: %dx%d, Patch dimension: %d\n', grid_size(1), grid_size(2), D);
    end

    % Initialize weights randomly from input data
    rand_indices = randperm(M, num_neurons);
    som_weights = patches(rand_indices, :);

    % Create grid coordinates for neurons (for neighborhood computation)
    [grid_x, grid_y] = meshgrid(1:grid_size(2), 1:grid_size(1));
    neuron_coords = [grid_y(:), grid_x(:)];

    % Training loop
    if verbose
        fprintf('Starting SOM training...\n');
        tic;
    end

    for iter = 1:num_iterations
        % Compute progress
        t = iter / num_iterations;

        % Decay learning rate and neighborhood radius
        lr = lr_init * (1 - t) + lr_final * t;
        sigma = sigma_init * (1 - t) + sigma_final * t;

        % Random sample selection
        idx = randi(M);
        sample = patches(idx, :);

        % Find Best Matching Unit (BMU)
        distances = sum((som_weights - sample).^2, 2);
        [~, bmu_idx] = min(distances);
        bmu_coord = neuron_coords(bmu_idx, :);

        % Update all neurons based on distance to BMU
        for k = 1:num_neurons
            % Compute grid distance to BMU
            grid_dist = norm(neuron_coords(k, :) - bmu_coord);

            % Gaussian neighborhood function
            h = exp(-grid_dist^2 / (2 * sigma^2));

            % Update weight
            som_weights(k, :) = som_weights(k, :) + lr * h * (sample - som_weights(k, :));
        end

        % Progress display
        if verbose && mod(iter, num_iterations/10) == 0
            fprintf('  Iteration %d/%d (%.1f%%), lr=%.4f, sigma=%.2f\n', ...
                iter, num_iterations, t*100, lr, sigma);
        end
    end

    if verbose
        elapsed = toc;
        fprintf('SOM training completed in %.2f seconds\n', elapsed);
    end

    % Package model
    som_model.weights = som_weights;
    som_model.grid_size = grid_size;
    som_model.grid_coords = neuron_coords;
    som_model.num_neurons = num_neurons;
    som_model.patch_dim = D;
end
