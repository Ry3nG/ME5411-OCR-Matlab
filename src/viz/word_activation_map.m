function heat = word_activation_map(img, som, patchSize, stride)
% WORD_ACTIVATION_MAP Generate spatial activation heatmap for visual words
%
% Inputs:
%   img       - Grayscale image (uint8 or double)
%   som       - SOM model structure
%   patchSize - Patch size
%   stride    - Stride for patch extraction
%
% Outputs:
%   heat - [H x W] activation heatmap
%
% Theory: Shows which image regions activate specific visual words.
%         Helps interpret why certain characters are confused.

    % Ensure img is 2D
    if ndims(img) == 3
        img = squeeze(img(:, :, 1));
    end
    img = double(img) / 255;
    [H, W] = size(img);
    ps = patchSize;

    % Extract all patches with stride
    cols = my_im2col(img, [ps ps], 'sliding');
    Hwin = H - ps + 1;
    Wwin = W - ps + 1;
    [yy, xx] = ndgrid(1:Hwin, 1:Wwin);
    mask = (mod(yy - 1, stride) == 0) & (mod(xx - 1, stride) == 0);
    cols = cols(:, mask(:));
    XY = [yy(mask), xx(mask)];  % Patch starting positions

    % Normalize patches
    X = normalize_rows(cols');
    W = som.weights;

    % Find BMU for each patch
    D2 = bsxfun(@plus, sum(X.^2, 2), sum(W.^2, 2)') - 2 * (X * W');
    [~, bmu] = min(D2, [], 2);

    % Create activation map (accumulate at patch centers)
    heat = zeros(H, W);
    for i = 1:size(XY, 1)
        r = XY(i, 1) + floor(ps / 2);
        c = XY(i, 2) + floor(ps / 2);
        if r <= H && c <= W
            heat(r, c) = heat(r, c) + 1;
        end
    end

    % Smooth with Gaussian filter
    sigma = 1;
    kernel_size = ceil(3 * sigma);
    x = -kernel_size:kernel_size;
    kernel = exp(-x.^2 / (2 * sigma^2));
    kernel = kernel' * kernel;
    kernel = kernel / sum(kernel(:));
    heat = conv2(heat, kernel, 'same');
end
