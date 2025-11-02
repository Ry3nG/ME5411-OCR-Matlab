function bw = myImbinarize(img, threshold)
% myImbinarize - Custom implementation of image binarization
% Convert grayscale image to binary image using threshold
%
% Input:
%   img - input grayscale image (uint8 or double)
%   threshold - threshold value (0-1 for normalized, 0-255 for uint8)
%
% Output:
%   bw - binary image (logical)

    % Convert to double if needed
    if isa(img, 'uint8')
        img_double = double(img) / 255;
    else
        img_double = double(img);
    end

    % Ensure threshold is normalized
    if threshold > 1
        threshold = threshold / 255;
    end

    % Apply threshold
    bw = img_double > threshold;
end
