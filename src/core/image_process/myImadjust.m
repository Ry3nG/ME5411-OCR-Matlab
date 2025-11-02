function img_adjusted = myImadjust(img, in_low, in_high, out_low, out_high, gamma)
% myImadjust - Custom implementation of intensity adjustment
% This function performs linear contrast stretching with optional gamma correction
%
% Input:
%   img - input grayscale image (uint8)
%   in_low - input intensity low bound (default: min intensity)
%   in_high - input intensity high bound (default: max intensity)
%   out_low - output intensity low bound (default: 0)
%   out_high - output intensity high bound (default: 255)
%   gamma - gamma value for power-law transformation (default: 1.0)
%
% Output:
%   img_adjusted - adjusted image (uint8)

    % Default parameters
    if nargin < 2 || isempty(in_low)
        in_low = double(min(img(:))) / 255;
    end
    if nargin < 3 || isempty(in_high)
        in_high = double(max(img(:))) / 255;
    end
    if nargin < 4 || isempty(out_low)
        out_low = 0;
    end
    if nargin < 5 || isempty(out_high)
        out_high = 255;
    end
    if nargin < 6 || isempty(gamma)
        gamma = 1.0;
    end

    % Convert to double and normalize to [0, 1]
    img_double = double(img) / 255;

    % Clip input values
    img_clipped = img_double;
    img_clipped(img_clipped < in_low) = in_low;
    img_clipped(img_clipped > in_high) = in_high;

    % Normalize to [0, 1] based on input range
    img_normalized = (img_clipped - in_low) / (in_high - in_low);

    % Apply gamma correction
    img_gamma = img_normalized .^ gamma;

    % Scale to output range
    img_scaled = img_gamma * (out_high - out_low) + out_low;

    % Convert back to uint8
    img_adjusted = uint8(img_scaled);
end
