function [edges, gradMag, gradDir] = mySobel(img, threshold)
% mySobel - Custom implementation of Sobel edge detection
% Detects edges using Sobel operator (gradient-based method)
%
% Input:
%   img - input grayscale image (uint8 or double)
%   threshold - threshold for edge detection (optional, default: auto)
%
% Output:
%   edges - binary edge map
%   gradMag - gradient magnitude
%   gradDir - gradient direction

    % Convert to double if needed
    if isa(img, 'uint8')
        img = double(img) / 255;
    else
        img = double(img);
    end

    % Sobel kernels for x and y directions
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [-1 -2 -1; 0 0 0; 1 2 1];

    % Compute gradients using convolution
    gradX = myConvolution(img, Gx);
    gradY = myConvolution(img, Gy);

    % Compute gradient magnitude and direction
    gradMag = sqrt(gradX.^2 + gradY.^2);
    gradDir = atan2(gradY, gradX);

    % Normalize gradient magnitude to [0, 1]
    gradMag = gradMag / max(gradMag(:));

    % Auto threshold if not provided
    if nargin < 2 || isempty(threshold)
        threshold = graythresh(gradMag);  % Otsu's method
    end

    % Apply threshold to get binary edges
    edges = gradMag > threshold;
end
