function som = train_som_batch(X, grid_size, num_iters, varargin)
% TRAIN_SOM_BATCH Vectorized mini-batch SOM training
%
% Inputs:
%   X          - [M x D] patch matrix
%   grid_size  - [rows, cols] SOM grid dimensions
%   num_iters  - Number of training iterations
%   varargin   - Optional parameters:
%                'lr_init', 'lr_final' - Learning rate decay
%                'sigma_init', 'sigma_final' - Neighborhood radius decay
%                'batch' - Mini-batch size (default: 32)
%                'verbose' - Progress display (default: true)
%
% Outputs:
%   som - Structure with fields: weights, grid_size, grid_coords, num_neurons, patch_dim
%
% Theory (from Part-II lectures):
%   - Competition: Find Best Matching Unit (BMU) for each input
%   - Cooperation: Update BMU and neighbors using Gaussian neighborhood h(t)
%   - Adaptation: Weights move toward inputs with learning rate η(t)
%   - Both η(t) and σ(t) decay over iterations for convergence

    p = inputParser;
    addParameter(p, 'lr_init', 0.5);
    addParameter(p, 'lr_final', 0.01);
    addParameter(p, 'sigma_init', max(grid_size) / 2);
    addParameter(p, 'sigma_final', 0.5);
    addParameter(p, 'batch', 32);
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});

    lr0 = p.Results.lr_init;
    lr1 = p.Results.lr_final;
    sg0 = p.Results.sigma_init;
    sg1 = p.Results.sigma_final;
    B = p.Results.batch;
    verbose = p.Results.verbose;

    [M, D] = size(X);
    K = prod(grid_size);

    % Initialize prototypes from random data samples
    idx0 = randperm(M, K);
    W = X(idx0, :);  % [K x D]

    % Create 2D grid coordinates for neurons
    [gy, gx] = ndgrid(1:grid_size(1), 1:grid_size(2));
    coords = [gy(:), gx(:)];  % [K x 2]

    if verbose
        fprintf('Mini-batch SOM training: K=%d neurons, %d iterations, batch=%d\n', ...
            K, num_iters, B);
        tic;
    end

    for t = 1:num_iters
        % Decay learning rate and neighborhood radius (exponential schedule)
        progress = t / num_iters;
        alpha = lr0 * (1 - progress) + lr1 * progress;
        sigma = sg0 * (1 - progress) + sg1 * progress;

        % Sample mini-batch
        bi = randi(M, [B, 1]);
        Xb = X(bi, :);  % [B x D]

        % Competition: Find BMU for each sample (vectorized distance computation)
        D2 = bsxfun(@plus, sum(Xb.^2, 2), sum(W.^2, 2)') - 2 * (Xb * W');  % [B x K]
        [~, bmu] = min(D2, [], 2);  % [B x 1]

        % Cooperation + Adaptation: Update all prototypes based on batch
        dW = zeros(K, D);
        for i = 1:B
            c = coords(bmu(i), :);  % BMU coordinates
            diff = bsxfun(@minus, coords, c);  % [K x 2]
            % Gaussian neighborhood function
            h = exp(-sum(diff.^2, 2) / (2 * sigma^2));  % [K x 1]
            % Accumulate weighted updates: h(j,i) * (x_i - w_j)
            dW = dW + (h * ones(1, D)) .* bsxfun(@minus, Xb(i, :), W);
        end

        % Apply batch-averaged update with learning rate
        W = W + alpha * dW / B;

        % Progress logging
        if verbose && mod(t, max(1, floor(num_iters / 10))) == 0
            fprintf('  Iteration %d/%d (%.1f%%), lr=%.4f, sigma=%.2f\n', ...
                t, num_iters, progress * 100, alpha, sigma);
        end
    end

    if verbose
        fprintf('SOM training completed in %.2f seconds\n', toc);
    end

    % Package model
    som.weights = W;
    som.grid_size = grid_size;
    som.grid_coords = coords;
    som.num_neurons = K;
    som.patch_dim = D;
end
