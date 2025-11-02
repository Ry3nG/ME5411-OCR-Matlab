function props = myRegionprops(CC, varargin)
% myRegionprops - Custom implementation of region properties calculation
% Calculate properties of connected components
%
% Input:
%   CC - connected components structure from myBwconncomp
%   varargin - property names: 'BoundingBox', 'Area', 'Centroid', etc.
%
% Output:
%   props - structure array with requested properties

    num_objects = CC.NumObjects;
    props = struct();

    % Get image size from CC structure
    if isfield(CC, 'ImageSize')
        img_size = CC.ImageSize;
    else
        error('CC structure must contain ImageSize field');
    end

    % Parse requested properties
    compute_bbox = any(strcmp(varargin, 'BoundingBox'));
    compute_area = any(strcmp(varargin, 'Area'));
    compute_centroid = any(strcmp(varargin, 'Centroid'));

    % Process each connected component
    for i = 1:num_objects
        pixel_list = CC.PixelIdxList{i};

        if compute_area
            props(i).Area = length(pixel_list);
        end

        if compute_bbox || compute_centroid
            if ~isempty(pixel_list)
                % Convert linear indices to subscripts
                [rows, cols] = ind2sub(img_size, pixel_list);

                if compute_centroid
                    props(i).Centroid = [mean(cols), mean(rows)];
                end

                if compute_bbox
                    % BoundingBox format: [min_col, min_row, width, height]
                    min_col = min(cols);
                    max_col = max(cols);
                    min_row = min(rows);
                    max_row = max(rows);

                    width = max_col - min_col + 1;
                    height = max_row - min_row + 1;

                    props(i).BoundingBox = [min_col, min_row, width, height];
                end
            else
                if compute_centroid
                    props(i).Centroid = [0, 0];
                end
                if compute_bbox
                    props(i).BoundingBox = [0, 0, 0, 0];
                end
            end
        end
    end
end
