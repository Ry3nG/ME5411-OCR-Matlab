function CC = myBwconncomp(bw)
% myBwconncomp - Custom implementation of connected component labeling
% Find connected components in binary image using 8-connectivity
%
% Input:
%   bw - binary image (logical or uint8)
%
% Output:
%   CC - structure containing:
%        .NumObjects - number of connected components
%        .PixelIdxList - cell array of linear indices for each component

    [rows, cols] = size(bw);
    bw = logical(bw);

    % Initialize label matrix
    labels = zeros(rows, cols);
    current_label = 0;

    % 8-connectivity offsets
    offsets = [-1, -1; -1, 0; -1, 1; 0, -1; 0, 1; 1, -1; 1, 0; 1, 1];

    % Label each connected component using flood fill
    for i = 1:rows
        for j = 1:cols
            if bw(i, j) && labels(i, j) == 0
                % Start new component
                current_label = current_label + 1;

                % Flood fill using stack
                stack = [i, j];
                while ~isempty(stack)
                    % Pop pixel from stack
                    current = stack(1, :);
                    stack(1, :) = [];

                    r = current(1);
                    c = current(2);

                    % Skip if already labeled or out of bounds
                    if r < 1 || r > rows || c < 1 || c > cols
                        continue;
                    end
                    if labels(r, c) ~= 0 || ~bw(r, c)
                        continue;
                    end

                    % Label this pixel
                    labels(r, c) = current_label;

                    % Add 8-connected neighbors to stack
                    for k = 1:size(offsets, 1)
                        nr = r + offsets(k, 1);
                        nc = c + offsets(k, 2);
                        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                            if bw(nr, nc) && labels(nr, nc) == 0
                                stack = [stack; nr, nc];
                            end
                        end
                    end
                end
            end
        end
    end

    % Create output structure
    CC.NumObjects = current_label;
    CC.PixelIdxList = cell(1, current_label);
    CC.ImageSize = [rows, cols];

    % Collect pixel indices for each component
    for label = 1:current_label
        [r, c] = find(labels == label);
        CC.PixelIdxList{label} = sub2ind([rows, cols], r, c);
    end
end
