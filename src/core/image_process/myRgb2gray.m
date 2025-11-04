function gray = myRgb2gray(img)
% Custom RGB to grayscale conversion function (replacement for rgb2gray)
% Uses standard ITU-R BT.601 weighting: 0.299*R + 0.587*G + 0.114*B
%
% Input:
%   img: Input image (RGB or grayscale)
% Output:
%   gray: Grayscale image

if ndims(img) == 3 && size(img, 3) == 3
    % RGB image: convert using ITU-R BT.601 weights
    gray = 0.299 * double(img(:, :, 1)) + ...
           0.587 * double(img(:, :, 2)) + ...
           0.114 * double(img(:, :, 3));

    % Convert to same class as input
    if isa(img, 'uint8')
        gray = uint8(gray);
    elseif isa(img, 'uint16')
        gray = uint16(gray);
    else
        gray = double(gray);
    end
else
    % Already grayscale or single channel
    gray = img;
end

end

