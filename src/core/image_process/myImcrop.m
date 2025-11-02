function cropped = myImcrop(img, rect)
% myImcrop - Custom implementation of image cropping
% Crop image using rectangle specification
%
% Input:
%   img - input image
%   rect - rectangle [x, y, width, height]
%          where (x,y) is top-left corner (column, row)
%
% Output:
%   cropped - cropped image

    % Extract rectangle parameters
    x = round(rect(1));
    y = round(rect(2));
    width = round(rect(3));
    height = round(rect(4));

    % Get image dimensions
    [rows, cols, ~] = size(img);

    % Ensure bounds are within image
    x = max(1, x);
    y = max(1, y);
    x_end = min(cols, x + width - 1);
    y_end = min(rows, y + height - 1);

    % Crop the image
    cropped = img(y:y_end, x:x_end, :);
end
