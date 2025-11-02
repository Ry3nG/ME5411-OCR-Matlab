function se = myStrel(shape, size_param)
% myStrel - Custom implementation of structuring element creation
% Create morphological structuring element
%
% Input:
%   shape - shape type: 'disk', 'square', 'line', 'rectangle'
%   size_param - size parameter (radius for disk, side length for square)
%   For 'line': size_param is length, and third argument is angle
%
% Output:
%   se - structuring element structure with 'Neighborhood' field

    se = struct();

    switch lower(shape)
        case 'disk'
            % Create circular disk structuring element
            radius = size_param;
            diameter = 2 * radius + 1;
            neighborhood = zeros(diameter, diameter);
            center = radius + 1;

            for i = 1:diameter
                for j = 1:diameter
                    distance = sqrt((i - center)^2 + (j - center)^2);
                    if distance <= radius
                        neighborhood(i, j) = 1;
                    end
                end
            end

        case 'square'
            % Create square structuring element
            side = size_param;
            neighborhood = ones(side, side);

        case 'rectangle'
            % Create rectangular structuring element
            % size_param should be [rows, cols]
            neighborhood = ones(size_param(1), size_param(2));

        otherwise
            error('Unsupported structuring element shape: %s', shape);
    end

    se.Neighborhood = neighborhood;
end
