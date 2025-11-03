function cols = my_im2col(img, block_size, mode)
% MY_IM2COL Custom implementation of im2col without toolbox dependency
%
% Inputs:
%   img        - Input image [H x W]
%   block_size - [bh, bw] block dimensions
%   mode       - 'sliding' for sliding window extraction
%
% Outputs:
%   cols - [bh*bw x num_blocks] matrix of vectorized blocks
%
% Implementation: Manual sliding window extraction

    [H, W] = size(img);
    bh = block_size(1);
    bw = block_size(2);

    if strcmp(mode, 'sliding')
        % Compute number of blocks
        num_h = H - bh + 1;
        num_w = W - bw + 1;

        if num_h <= 0 || num_w <= 0
            cols = [];
            return;
        end

        num_blocks = num_h * num_w;
        cols = zeros(bh * bw, num_blocks);

        % Extract all sliding windows
        idx = 1;
        for y = 1:num_h
            for x = 1:num_w
                block = img(y:y+bh-1, x:x+bw-1);
                cols(:, idx) = block(:);
                idx = idx + 1;
            end
        end
    else
        error('Only "sliding" mode is supported');
    end
end
