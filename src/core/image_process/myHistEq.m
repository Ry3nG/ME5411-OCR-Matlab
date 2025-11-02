function img_eq = myHistEq(img)
% myHistEq - Custom implementation of histogram equalization
% This function implements histogram equalization to enhance image contrast
%
% Input:
%   img - input grayscale image (uint8)
%
% Output:
%   img_eq - histogram equalized image (uint8)

    % Convert to double for processing
    img_double = double(img);
    [rows, cols] = size(img);

    % Compute histogram
    histogram = zeros(256, 1);
    for i = 1:rows
        for j = 1:cols
            pixel_val = img(i, j) + 1; % MATLAB uses 1-based indexing
            histogram(pixel_val) = histogram(pixel_val) + 1;
        end
    end

    % Compute cumulative distribution function (CDF)
    cdf = zeros(256, 1);
    cdf(1) = histogram(1);
    for i = 2:256
        cdf(i) = cdf(i-1) + histogram(i);
    end

    % Normalize CDF to [0, 255]
    total_pixels = rows * cols;
    cdf_normalized = round((cdf / total_pixels) * 255);

    % Apply transformation
    img_eq = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            pixel_val = img(i, j) + 1;
            img_eq(i, j) = cdf_normalized(pixel_val);
        end
    end

    % Convert back to uint8
    img_eq = uint8(img_eq);
end
