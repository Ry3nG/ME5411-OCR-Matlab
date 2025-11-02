function edges = myCanny(img, lowThreshold, highThreshold, sigma)
% myCanny - Custom implementation of Canny edge detection
% Multi-stage edge detection algorithm with non-maximum suppression
%
% Input:
%   img - input grayscale image (uint8 or double)
%   lowThreshold - low threshold for hysteresis (default: auto)
%   highThreshold - high threshold for hysteresis (default: auto)
%   sigma - Gaussian filter sigma (default: 1.4)
%
% Output:
%   edges - binary edge map

    % Default parameters
    if nargin < 4
        sigma = 1.4;
    end

    % Convert to double if needed
    if isa(img, 'uint8')
        img = double(img) / 255;
    else
        img = double(img);
    end

    % Step 1: Gaussian filtering for noise reduction
    filterSize = 2 * ceil(3 * sigma) + 1;  % Filter size based on sigma
    imgSmooth = myGaussianFilter(img, filterSize, sigma);

    % Step 2: Compute gradients using Sobel operator
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [-1 -2 -1; 0 0 0; 1 2 1];

    gradX = myConvolution(imgSmooth, Gx);
    gradY = myConvolution(imgSmooth, Gy);

    gradMag = sqrt(gradX.^2 + gradY.^2);
    gradDir = atan2(gradY, gradX);  % Range: -pi to pi

    % Step 3: Non-maximum suppression
    [rows, cols] = size(gradMag);
    nms = zeros(rows, cols);

    % Convert gradient direction to 4 discrete directions (0, 45, 90, 135 degrees)
    angle = gradDir * 180 / pi;
    angle(angle < 0) = angle(angle < 0) + 180;

    for i = 2:rows-1
        for j = 2:cols-1
            q = 255;
            r = 255;

            % Angle 0 degree (horizontal)
            if (0 <= angle(i,j) && angle(i,j) < 22.5) || (157.5 <= angle(i,j) && angle(i,j) <= 180)
                q = gradMag(i, j+1);
                r = gradMag(i, j-1);
            % Angle 45 degree
            elseif (22.5 <= angle(i,j) && angle(i,j) < 67.5)
                q = gradMag(i+1, j-1);
                r = gradMag(i-1, j+1);
            % Angle 90 degree (vertical)
            elseif (67.5 <= angle(i,j) && angle(i,j) < 112.5)
                q = gradMag(i+1, j);
                r = gradMag(i-1, j);
            % Angle 135 degree
            elseif (112.5 <= angle(i,j) && angle(i,j) < 157.5)
                q = gradMag(i-1, j-1);
                r = gradMag(i+1, j+1);
            end

            % Keep only local maxima
            if (gradMag(i,j) >= q) && (gradMag(i,j) >= r)
                nms(i,j) = gradMag(i,j);
            else
                nms(i,j) = 0;
            end
        end
    end

    % Step 4: Double threshold and edge tracking by hysteresis
    % Auto-compute thresholds if not provided
    if nargin < 2 || isempty(lowThreshold) || isempty(highThreshold)
        highThreshold = max(nms(:)) * 0.2;
        lowThreshold = highThreshold * 0.4;
    end

    % Apply double threshold
    strongEdges = nms > highThreshold;
    weakEdges = (nms >= lowThreshold) & (nms <= highThreshold);

    % Edge tracking by hysteresis
    edges = strongEdges;
    [rows, cols] = size(edges);

    for i = 2:rows-1
        for j = 2:cols-1
            if weakEdges(i,j)
                % Check if connected to strong edge
                if any(any(strongEdges(i-1:i+1, j-1:j+1)))
                    edges(i,j) = 1;
                end
            end
        end
    end
end
