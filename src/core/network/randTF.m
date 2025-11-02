function augmentedImage = randTF(image, options)
    % Random transformation for data augmentation
    % Uses built-in MATLAB functions (imrotate, imtranslate are in base MATLAB)

    [H, W] = size(image);
    augmentedImage = image;

    % Random rotation
    if isfield(options, 'rot_range')
        angle = options.rot_range(1) + ...
            (options.rot_range(2) - options.rot_range(1)) * rand();
        % imrotate is available in base MATLAB
        augmentedImage = imrotate(augmentedImage, angle, 'bilinear', 'crop');
    end

    % Random translation
    if isfield(options, 'trans_ratio')
        max_shift = round(H * options.trans_ratio);
        dx = randi([-max_shift, max_shift]);
        dy = randi([-max_shift, max_shift]);
        augmentedImage = myImtranslate(augmentedImage, [dx, dy]);
    end

    % Ensure output size matches input
    if size(augmentedImage, 1) ~= H || size(augmentedImage, 2) ~= W
        augmentedImage = imresize(augmentedImage, [H, W]);
    end
end

function out = myImtranslate(img, offset)
    % Simple translation without toolbox
    [H, W] = size(img);
    dx = offset(1);
    dy = offset(2);

    out = ones(H, W) * 255;  % White background

    % Calculate valid regions
    src_x_start = max(1, 1 - dx);
    src_x_end = min(W, W - dx);
    src_y_start = max(1, 1 - dy);
    src_y_end = min(H, H - dy);

    dst_x_start = max(1, 1 + dx);
    dst_x_end = min(W, W + dx);
    dst_y_start = max(1, 1 + dy);
    dst_y_end = min(H, H + dy);

    % Copy translated region
    out(dst_y_start:dst_y_end, dst_x_start:dst_x_end) = ...
        img(src_y_start:src_y_end, src_x_start:src_x_end);
end
