function output = myImrotate(img, angle, method, crop_mode)
% Custom image rotation function (replacement for imrotate)
% Simple rotation using nearest neighbor or bilinear interpolation
%
% Input:
%   img: Input image (grayscale)
%   angle: Rotation angle in degrees (counterclockwise)
%   method: 'nearest' or 'bilinear' (default: 'nearest')
%   crop_mode: 'crop' to keep original size, 'loose' for full rotated image
% Output:
%   output: Rotated image

if nargin < 3
    method = 'nearest';
end
if nargin < 4
    crop_mode = 'loose';
end

[old_h, old_w] = size(img);

% Convert angle to radians
theta = -angle * pi / 180;  % Negative for counterclockwise

% Rotation matrix
cos_theta = cos(theta);
sin_theta = sin(theta);

if strcmp(crop_mode, 'crop')
    % Keep original size
    new_h = old_h;
    new_w = old_w;
    center_x = old_w / 2;
    center_y = old_h / 2;
else
    % Calculate new size to fit entire rotated image
    corners_x = [1, old_w, 1, old_w] - old_w/2;
    corners_y = [1, 1, old_h, old_h] - old_h/2;

    rotated_x = corners_x * cos_theta - corners_y * sin_theta;
    rotated_y = corners_x * sin_theta + corners_y * cos_theta;

    new_w = ceil(max(rotated_x) - min(rotated_x));
    new_h = ceil(max(rotated_y) - min(rotated_y));
    center_x = new_w / 2;
    center_y = new_h / 2;
end

% Initialize output with white background
output = ones(new_h, new_w) * 255;

% Inverse rotation for each output pixel
for i = 1:new_h
    for j = 1:new_w
        % Center coordinates
        x = j - center_x;
        y = i - center_y;

        % Apply inverse rotation
        src_x = x * cos_theta + y * sin_theta + old_w/2;
        src_y = -x * sin_theta + y * cos_theta + old_h/2;

        % Check if source is within bounds
        if src_x >= 1 && src_x <= old_w && src_y >= 1 && src_y <= old_h
            if strcmp(method, 'nearest')
                % Nearest neighbor
                src_i = round(src_y);
                src_j = round(src_x);
                output(i, j) = img(src_i, src_j);
            else
                % Bilinear interpolation
                i1 = floor(src_y);
                i2 = ceil(src_y);
                j1 = floor(src_x);
                j2 = ceil(src_x);

                i1 = max(1, min(old_h, i1));
                i2 = max(1, min(old_h, i2));
                j1 = max(1, min(old_w, j1));
                j2 = max(1, min(old_w, j2));

                wi = src_y - i1;
                wj = src_x - j1;

                val11 = img(i1, j1);
                val12 = img(i1, j2);
                val21 = img(i2, j1);
                val22 = img(i2, j2);

                val1 = val11 * (1 - wj) + val12 * wj;
                val2 = val21 * (1 - wj) + val22 * wj;

                output(i, j) = val1 * (1 - wi) + val2 * wi;
            end
        end
    end
end

% Convert to same class as input
if isa(img, 'uint8')
    output = uint8(output);
end

end
