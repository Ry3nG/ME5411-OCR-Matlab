function img_adapted = myAdaptHistEq(img, window_size)
% myAdaptHistEq - Custom implementation of adaptive histogram equalization (CLAHE)
% This function implements Contrast Limited Adaptive Histogram Equalization
%
% Input:
%   img - input grayscale image (uint8)
%   window_size - size of the local window (default: 8)
%
% Output:
%   img_adapted - adaptively equalized image (uint8)

    if nargin < 2
        window_size = 8;
    end

    [rows, cols] = size(img);
    img_adapted = zeros(rows, cols);

    % Process image in tiles
    half_window = floor(window_size / 2);

    for i = 1:rows
        for j = 1:cols
            % Define local window boundaries
            row_min = max(1, i - half_window);
            row_max = min(rows, i + half_window);
            col_min = max(1, j - half_window);
            col_max = min(cols, j + half_window);

            % Extract local window
            local_window = img(row_min:row_max, col_min:col_max);

            % Compute local histogram
            histogram = zeros(256, 1);
            for r = 1:size(local_window, 1)
                for c = 1:size(local_window, 2)
                    pixel_val = local_window(r, c) + 1;
                    histogram(pixel_val) = histogram(pixel_val) + 1;
                end
            end

            % Compute local CDF
            cdf = zeros(256, 1);
            cdf(1) = histogram(1);
            for k = 2:256
                cdf(k) = cdf(k-1) + histogram(k);
            end

            % Normalize CDF
            total_pixels = size(local_window, 1) * size(local_window, 2);
            if total_pixels > 0
                cdf_normalized = (cdf / total_pixels) * 255;
            else
                cdf_normalized = cdf;
            end

            % Apply transformation to center pixel
            pixel_val = img(i, j) + 1;
            img_adapted(i, j) = cdf_normalized(pixel_val);
        end
    end

    % Convert back to uint8
    img_adapted = uint8(img_adapted);
end
