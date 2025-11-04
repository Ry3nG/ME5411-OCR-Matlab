function output = myImresize(img, new_size, method)
% Custom image resize function (replacement for imresize)
% Supports nearest neighbor and bilinear interpolation
%
% Input:
%   img: Input image (grayscale or RGB)
%   new_size: [new_height, new_width] or scalar (for uniform scaling)
%   method: 'nearest' or 'bilinear' (default: 'bilinear')
% Output:
%   output: Resized image

if nargin < 3
    method = 'bilinear';
end

[old_h, old_w, channels] = size(img);

% Handle scalar input (uniform scaling)
if isscalar(new_size)
    new_h = round(old_h * new_size);
    new_w = round(old_w * new_size);
else
    new_h = new_size(1);
    new_w = new_size(2);
end

% Initialize output
output = zeros(new_h, new_w, channels);

% Calculate scale factors
scale_h = old_h / new_h;
scale_w = old_w / new_w;

switch lower(method)
    case 'nearest'
        % Nearest neighbor interpolation
        for i = 1:new_h
            for j = 1:new_w
                % Map to original coordinates
                src_i = round((i - 0.5) * scale_h + 0.5);
                src_j = round((j - 0.5) * scale_w + 0.5);

                % Clamp to valid range
                src_i = max(1, min(old_h, src_i));
                src_j = max(1, min(old_w, src_j));

                output(i, j, :) = img(src_i, src_j, :);
            end
        end

    case 'bilinear'
        % Bilinear interpolation
        for i = 1:new_h
            for j = 1:new_w
                % Map to original coordinates (center of pixels)
                src_i = (i - 0.5) * scale_h + 0.5;
                src_j = (j - 0.5) * scale_w + 0.5;

                % Get surrounding pixel coordinates
                i1 = floor(src_i);
                i2 = ceil(src_i);
                j1 = floor(src_j);
                j2 = ceil(src_j);

                % Clamp to valid range
                i1 = max(1, min(old_h, i1));
                i2 = max(1, min(old_h, i2));
                j1 = max(1, min(old_w, j1));
                j2 = max(1, min(old_w, j2));

                % Calculate interpolation weights
                wi = src_i - i1;
                wj = src_j - j1;

                % Bilinear interpolation
                for c = 1:channels
                    val11 = img(i1, j1, c);
                    val12 = img(i1, j2, c);
                    val21 = img(i2, j1, c);
                    val22 = img(i2, j2, c);

                    % Interpolate along rows
                    val1 = val11 * (1 - wj) + val12 * wj;
                    val2 = val21 * (1 - wj) + val22 * wj;

                    % Interpolate along columns
                    output(i, j, c) = val1 * (1 - wi) + val2 * wi;
                end
            end
        end

    otherwise
        error('Unsupported interpolation method. Use ''nearest'' or ''bilinear''.');
end

% Convert to same class as input
if isa(img, 'uint8')
    output = uint8(output);
elseif isa(img, 'uint16')
    output = uint16(output);
end

end
