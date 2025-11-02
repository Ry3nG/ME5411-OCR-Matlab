function se = myStrelLine(length, angle)
% myStrelLine - Create line structuring element
% Create a linear structuring element at specified angle
%
% Input:
%   length - length of the line
%   angle - angle in degrees (0=horizontal, 90=vertical)
%
% Output:
%   se - structuring element structure with 'Neighborhood' field

    se = struct();

    % Convert angle to radians
    angle_rad = deg2rad(angle);

    % Calculate the range for the structuring element
    max_offset = ceil(length / 2);
    size_se = 2 * max_offset + 1;
    neighborhood = zeros(size_se, size_se);

    % Generate line coordinates
    center = max_offset + 1;
    for t = -length/2:0.5:length/2
        x = round(center + t * cos(angle_rad));
        y = round(center + t * sin(angle_rad));

        % Check bounds
        if x >= 1 && x <= size_se && y >= 1 && y <= size_se
            neighborhood(y, x) = 1;
        end
    end

    se.Neighborhood = neighborhood;
end
