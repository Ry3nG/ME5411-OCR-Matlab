function result = myAverageFilter(inputImage, filterSize)
    % myAverageFilter - Apply averaging filter to an image
    %
    % Inputs:
    %   inputImage - Input grayscale image
    %   filterSize - Size of the averaging filter (e.g., 3, 5, 7, 9, 11)
    %
    % Output:
    %   result - Filtered image with same size as input

    % Create averaging filter kernel
    % All elements are 1/(filterSize^2) for equal averaging
    filter = ones(filterSize, filterSize) / (filterSize * filterSize);

    % Apply convolution using custom convolution function
    result = myConvolution(double(inputImage), filter);

    % Convert back to uint8 for image display
    result = uint8(result);
end
