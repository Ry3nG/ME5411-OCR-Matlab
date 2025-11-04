%% Task 7.3: Character Identification on Microchip Label (Image 1)
% Reuse Task 3/4/6 preprocessing pipeline and evaluate Task 7.1 CNN + Task 7.2 SOM+SVM

clear; clc; close all;

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
cd(project_root);

addpath(genpath('src/core'));
addpath(genpath('src/utils'));

%% Configuration
input_img_path = 'data/charact2.bmp';
output_dir = 'output/task7_3';
char_dir = fullfile(output_dir, 'characters');
prep_dir = fullfile(output_dir, 'prepared');
report_fig_dir = fullfile(output_dir, 'figures');

if ~exist(output_dir, 'dir'); mkdir(output_dir); end
if ~exist(char_dir, 'dir'); mkdir(char_dir); end
if ~exist(prep_dir, 'dir'); mkdir(prep_dir); end
if ~exist(report_fig_dir, 'dir'); mkdir(report_fig_dir); end

% Cropping rectangles (reuse Task 3 coordinates + mirrored top-line)
rect_bottom = [50, 200, 900, 150];  % [x, y, width, height]
rect_top = [50, 35, 900, 150];      % manually mirrored for the top line

% Segmentation parameters
min_area = 600;             % filter out tiny noise components
merge_factor = 1.9;         % width threshold for splitting fused characters
target_size = 124;          % CNN / BoW expected input resolution
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

%% Load and preprocess image
fprintf('=== Task 7.3: Character Identification ===\n');
fprintf('Loading image: %s\n', input_img_path);
img_color = imread(input_img_path);
if size(img_color, 3) == 3
    img_gray = myRgb2gray(img_color);
else
    img_gray = img_color;
end

% Save grayscale reference
imwrite(img_gray, fullfile(output_dir, 'image1_gray.png'));

% Crop top and bottom lines using task3 logic
top_gray = myImcrop(img_gray, rect_top);
bottom_gray = myImcrop(img_gray, rect_bottom);
imwrite(top_gray, fullfile(output_dir, 'top_line_gray.png'));
imwrite(bottom_gray, fullfile(output_dir, 'bottom_line_gray.png'));

% Binarize each line via Task4 operators
top_thres = myOtsuThres(top_gray);
bottom_thres = myOtsuThres(bottom_gray);
fprintf('Top line Otsu threshold: %.4f\n', top_thres);
fprintf('Bottom line Otsu threshold: %.4f\n', bottom_thres);

top_binary = myImbinarize(top_gray, top_thres);
bottom_binary = myImbinarize(bottom_gray, bottom_thres);
se_noise = myStrel('disk', 1);
top_binary_clean = myOpen(uint8(top_binary), se_noise) > 0;
bottom_binary_clean = myOpen(uint8(bottom_binary), se_noise) > 0;

imwrite(uint8(top_binary) * 255, fullfile(output_dir, 'top_line_binary_raw.png'));
imwrite(uint8(bottom_binary) * 255, fullfile(output_dir, 'bottom_line_binary_raw.png'));
imwrite(uint8(top_binary_clean) * 255, fullfile(output_dir, 'top_line_binary.png'));
imwrite(uint8(bottom_binary_clean) * 255, fullfile(output_dir, 'bottom_line_binary.png'));

%% Segment characters from each line (Task6 style)
fprintf('Segmenting characters on top line...\n');
[segments_top, next_idx] = segment_line(top_gray, top_binary_clean, rect_top, ...
    'top', char_dir, 1, min_area, merge_factor);
fprintf('  -> %d characters extracted from top line\n', numel(segments_top));

fprintf('Segmenting characters on bottom line...\n');
[segments_bottom, ~] = segment_line(bottom_gray, bottom_binary_clean, rect_bottom, ...
    'bottom', char_dir, next_idx, min_area, merge_factor);
fprintf('  -> %d characters extracted from bottom line\n', numel(segments_bottom));

segments_all = [segments_top, segments_bottom];
num_chars = numel(segments_all);
fprintf('Total segmented characters: %d\n', num_chars);

%% Prepare canvases for CNN & SOM pipelines
cnn_batch = zeros(target_size, target_size, 1, num_chars);
bow_batch = zeros(target_size, target_size, 1, num_chars, 'uint8');

for i = 1:num_chars
    char_gray = segments_all(i).gray_trim;
    [canvas_double, canvas_uint8] = prepare_char_canvas(char_gray, target_size);

    cnn_batch(:, :, 1, i) = canvas_double;
    bow_batch(:, :, 1, i) = canvas_uint8;

    cnn_name = sprintf('cnn_char_%02d.png', i);
    bow_name = sprintf('bow_char_%02d.png', i);
    imwrite(canvas_double, fullfile(prep_dir, cnn_name));
    imwrite(canvas_uint8, fullfile(prep_dir, bow_name));

    segments_all(i).cnn_input = canvas_double;
    segments_all(i).bow_input = canvas_uint8;
end

%% Load CNN model (Task 7.1)
cnn_model_path = 'output/task7_1/11-03_03-17-04/cnn_best_acc.mat';
fprintf('Loading CNN model: %s\n', cnn_model_path);
cnn_data = load(cnn_model_path, 'cnn');
cnn = cnn_data.cnn;

fprintf('Running CNN inference on %d characters...\n', num_chars);
[cnn_preds, cnn] = predict(cnn, cnn_batch);
cnn_scores = cnn.layers{end}.activations;  % softmax output [num_classes x N]

cnn_top1 = zeros(num_chars, 1);
cnn_top2 = zeros(num_chars, 1);
cnn_conf = zeros(num_chars, 1);
for i = 1:num_chars
    [sorted_vals, sorted_idx] = sort(cnn_scores(:, i), 'descend');
    cnn_top1(i) = sorted_idx(1);
    cnn_top2(i) = sorted_idx(2);
    cnn_conf(i) = sorted_vals(1);
end

%% Load SOM + SVM models (Task 7.2)
fprintf('Loading SOM + BoW + SVM models...\n');
som = load('output/task7_2/som_model.mat', 'som');
som = som.som;
svm_models = load('output/task7_2/svm_models.mat', 'models');
svm_models = svm_models.models;
pca_model = load('output/task7_2/pca_model.mat', 'pca_model');
pca_model = pca_model.pca_model;
cfg_struct = load('output/task7_2/results.mat', 'results');
cfg = cfg_struct.results.config;

fprintf('Extracting BoW features for %d characters...\n', num_chars);
F_char = extract_bow_features(bow_batch, som, cfg.stride, ...
    'normalize', true, 'norm_type', 'l2', ...
    'soft_voting', cfg.soft_voting, 'sigma_bow', cfg.sigma_bow, ...
    'spatial_pyramid', cfg.spatial_pyramid, 'verbose', false, ...
    'min_patch_std', cfg.min_patch_std);

% Apply saved PCA projection
F_char_pca = (F_char - pca_model.mu) * pca_model.W;

% Compute class scores (one-vs-rest linear SVMs)
num_classes = numel(svm_models);
svm_scores = zeros(num_chars, num_classes);
for c = 1:num_classes
    model = svm_models{c};
    svm_scores(:, c) = F_char_pca * model.w + model.b;
end

[~, svm_top1] = max(svm_scores, [], 2);
svm_top2 = zeros(num_chars, 1);
svm_margin = zeros(num_chars, 1);
for i = 1:num_chars
    [sorted_vals, sorted_idx] = sort(svm_scores(i, :), 'descend');
    svm_top2(i) = sorted_idx(2);
    svm_margin(i) = sorted_vals(1) - sorted_vals(2);
end

%% Aggregate results
table_lines = strings(num_chars, 1);
table_idx = zeros(num_chars, 1);
cnn_pred_labels = strings(num_chars, 1);
cnn_second_labels = strings(num_chars, 1);
svm_pred_labels = strings(num_chars, 1);
svm_second_labels = strings(num_chars, 1);

for i = 1:num_chars
    table_lines(i) = segments_all(i).line;
    table_idx(i) = segments_all(i).line_index;
    cnn_pred_labels(i) = string(class_names{cnn_top1(i)});
    cnn_second_labels(i) = string(class_names{cnn_top2(i)});
    svm_pred_labels(i) = string(class_names{svm_top1(i)});
    svm_second_labels(i) = string(class_names{svm_top2(i)});

    segments_all(i).cnn_pred = cnn_pred_labels(i);
    segments_all(i).cnn_conf = cnn_conf(i);
    segments_all(i).cnn_second = cnn_second_labels(i);
    segments_all(i).svm_pred = svm_pred_labels(i);
    segments_all(i).svm_margin = svm_margin(i);
    segments_all(i).svm_second = svm_second_labels(i);
end

results_table = table((1:num_chars)', table_lines, table_idx, ...
    cnn_pred_labels, cnn_conf, cnn_second_labels, ...
    svm_pred_labels, svm_margin, svm_second_labels, ...
    'VariableNames', {'CharID', 'Line', 'LineIndex', ...
    'CNN_Pred', 'CNN_Confidence', 'CNN_Second', ...
    'SOM_SVM_Pred', 'SOM_SVM_Margin', 'SOM_SVM_Second'});

writetable(results_table, fullfile(output_dir, 'task7_3_predictions.csv'));
save(fullfile(output_dir, 'task7_3_results.mat'), 'segments_all', ...
    'cnn_top1', 'cnn_top2', 'cnn_conf', 'svm_top1', 'svm_top2', 'svm_margin', ...
    'rect_top', 'rect_bottom');

%% Visualization: bounding boxes on original image
overlay_fig = figure('Name', 'Task 7.3 - Character Localization', ...
    'Color', 'white', 'Position', [100, 100, 1100, 420]);
imshow(img_gray); hold on;

for i = 1:num_chars
    bbox = segments_all(i).bbox_global;
    rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 1.8);
    text(bbox(1), bbox(2) - 8, sprintf('%02d', i), ...
        'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold');
end

text(20, 25, sprintf('Total characters: %d', num_chars), ...
    'Color', 'y', 'FontSize', 14, 'FontWeight', 'bold');

hold off;
set(gca, 'Position', [0 0 1 1]);
print(overlay_fig, fullfile(report_fig_dir, 'char_localization.png'), '-dpng', '-r150');
close(overlay_fig);

%% Visualization: per-character predictions
cols = 7;
rows = ceil(num_chars / cols);
pred_fig = figure('Name', 'Task 7.3 - Character Predictions', ...
    'Color', 'white', 'Position', [100, 100, 1600, 400]);
tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:num_chars
    nexttile;
    img_show = segments_all(i).cnn_input;
    imshow(img_show, []);
    line_tag = upper(char(segments_all(i).line));
    line_tag = line_tag(1);
    title(sprintf('#%02d %s-%d\nCNN:%s %.2f\nSOM:%s %.2f', ...
        i, line_tag, segments_all(i).line_index, ...
        segments_all(i).cnn_pred, segments_all(i).cnn_conf, ...
        segments_all(i).svm_pred, segments_all(i).svm_margin), ...
        'FontSize', 9, 'Color', 'black');
end

print(pred_fig, fullfile(report_fig_dir, 'char_predictions.png'), '-dpng', '-r150');
close(pred_fig);

fprintf('All outputs saved under %s\n', output_dir);
fprintf('Figures saved to %s\n', report_fig_dir);

%% Helper functions
function [segments, next_idx] = segment_line(gray_img, binary_img, rect_offset, ...
    line_name, char_dir, start_idx, min_area, merge_factor)
    % Segment a single line of text using connected components with width-based splitting

    CC = myBwconncomp(binary_img);
    props = myRegionprops(CC, 'BoundingBox', 'Area');

    valid_idx = [];
    for i = 1:numel(props)
        if props(i).Area >= min_area
            valid_idx(end+1) = i; %#ok<AGROW>
        end
    end

    if isempty(valid_idx)
        segments = struct('line', {}, 'line_index', {}, 'bbox_global', {}, ...
            'bbox_local', {}, 'gray_trim', {}, 'binary_trim', {}, 'save_path', {});
        next_idx = start_idx;
        return;
    end

    bboxes = zeros(numel(valid_idx), 4);
    for i = 1:numel(valid_idx)
        bboxes(i, :) = props(valid_idx(i)).BoundingBox;
    end

    % Sort left-to-right
    [~, order] = sort(bboxes(:, 1));
    bboxes = bboxes(order, :);

    % Estimate merge threshold from lower-quartile glyph width
    widths = bboxes(:, 3);
    widths_sorted = sort(widths);
    q_idx = max(1, round(0.25 * numel(widths_sorted)));
    ref_width = widths_sorted(q_idx);
    if ref_width == 0
        merge_threshold = 0;
    else
        merge_threshold = max(ref_width * merge_factor, ref_width + 20);
    end

    segments = struct('line', {}, 'line_index', {}, 'bbox_global', {}, ...
        'bbox_local', {}, 'gray_trim', {}, 'binary_trim', {}, 'save_path', {});
    next_idx = start_idx;
    line_counter = 0;

    for i = 1:size(bboxes, 1)
        bbox = bboxes(i, :);
        [char_gray, char_bw, x_base, y_base] = extract_component(gray_img, binary_img, bbox);

        if isempty(char_gray)
            continue;
        end

        [segments, next_idx, line_counter] = process_component(segments, char_gray, char_bw, ...
            rect_offset, line_name, next_idx, line_counter, char_dir, x_base, y_base, merge_threshold);
    end
end

function [segments, next_idx, line_counter] = process_component(segments, gray_img, bw_img, ...
    rect_offset, line_name, next_idx, line_counter, char_dir, x_base, y_base, merge_threshold)

    queue = struct('gray', {gray_img}, 'bw', {bw_img}, 'shift', {0});

    while ~isempty(queue)
        current = queue(1);
        queue(1) = [];

        width = size(current.bw, 2);
        split_cols = [];

        if width > merge_threshold && merge_threshold > 0
            split_cols = detect_vertical_splits(current.bw);
            if isempty(split_cols) && width > merge_threshold * 1.25
                split_cols = round(width / 2);
            end
        end

        if isempty(split_cols)
            [segments, next_idx, line_counter] = append_char_segment(segments, current.gray, current.bw, ...
                rect_offset, line_name, next_idx, line_counter, char_dir, x_base, y_base, current.shift);
        else
            bounds = [1, split_cols, width + 1];
            for idx = 1:numel(bounds) - 1
                col_start = bounds(idx);
                col_end = bounds(idx + 1) - 1;
                if col_end <= col_start
                    continue;
                end

                sub_bw = current.bw(:, col_start:col_end);
                if ~any(sub_bw(:))
                    continue;
                end
                sub_gray = current.gray(:, col_start:col_end);
                new_shift = current.shift + col_start - 1;

                queue(end+1) = struct('gray', sub_gray, 'bw', sub_bw, 'shift', new_shift); %#ok<AGROW>
            end
        end
    end
end

function split_cols = detect_vertical_splits(bw_img)
    [height, width] = size(bw_img);
    if width < 6
        split_cols = [];
        return;
    end

    col_sum = sum(bw_img, 1);
    smooth_kernel = ones(1, 5) / 5;
    col_smooth = conv(double(col_sum), smooth_kernel, 'same');

    avg_val = mean(col_smooth);
    upper_thresh = min(avg_val * 0.45, 0.25 * height);
    lower_thresh = max(upper_thresh, 0.08 * height);
    gap_threshold = max(2, lower_thresh);

    gap_mask = col_smooth <= gap_threshold;
    gap_mask = suppress_short_runs(gap_mask, 3);

    d = diff([0, gap_mask, 0]);
    start_idx = find(d == 1);
    stop_idx = find(d == -1) - 1;

    edge_buffer = 2;
    split_cols = [];
    for k = 1:numel(start_idx)
        len = stop_idx(k) - start_idx(k) + 1;
        if len < 3
            continue;
        end
        if start_idx(k) <= edge_buffer || stop_idx(k) >= width - edge_buffer
            continue;
        end
        seg_min = min(col_smooth(start_idx(k):stop_idx(k)));
        if seg_min > 0.15 * height
            continue;
        end
        split_cols(end+1) = floor((start_idx(k) + stop_idx(k)) / 2); %#ok<AGROW>
    end
    split_cols = unique(split_cols);
    max_splits = 4;
    if numel(split_cols) > max_splits
        split_cols = split_cols(1:max_splits);
    end
end

function mask = suppress_short_runs(mask, min_len)
    mask = logical(mask(:)');
    d = diff([0, mask, 0]);
    starts = find(d == 1);
    stops = find(d == -1) - 1;
    keep = false(size(mask));
    for i = 1:numel(starts)
        if stops(i) - starts(i) + 1 >= min_len
            keep(starts(i):stops(i)) = true;
        end
    end
    mask = keep;
end

function [segments, next_idx, line_counter] = append_char_segment(segments, gray_img, bw_img, ...
    rect_offset, line_name, next_idx, line_counter, char_dir, x_base, y_base, col_shift)
    % Trim blank borders and append character metadata / image exports

    [bw_trim, gray_trim, row_offset, col_offset] = trim_character(bw_img, gray_img);
    if isempty(bw_trim) || ~any(bw_trim(:))
        return;
    end
    if size(bw_trim, 1) < 5 || size(bw_trim, 2) < 3
        return;
    end

    global_x = rect_offset(1) + x_base + col_shift + col_offset - 2;
    global_y = rect_offset(2) + y_base + row_offset - 2;

    entry = struct();
    entry.line = string(line_name);
    line_counter = line_counter + 1;
    entry.line_index = line_counter;
    entry.bbox_global = [global_x, global_y, size(bw_trim, 2), size(bw_trim, 1)];
    entry.bbox_local = [x_base + col_shift + col_offset - 1, ...
        y_base + row_offset - 1, size(bw_trim, 2), size(bw_trim, 1)];
    entry.gray_trim = gray_trim;
    entry.binary_trim = bw_trim;

    filename = sprintf('%s_char_%02d.png', char(line_name), next_idx);
    imwrite(uint8(bw_trim) * 255, fullfile(char_dir, filename));
    entry.save_path = fullfile(char_dir, filename);

    segments = [segments, entry]; %#ok<AGROW>
    next_idx = next_idx + 1;
end

function [char_gray, char_bw, x, y] = extract_component(gray_img, binary_img, bbox)
    % Extract rectangular patch for a connected component
    x = max(1, floor(bbox(1)));
    y = max(1, floor(bbox(2)));
    w = min(size(binary_img, 2) - x + 1, ceil(bbox(3)));
    h = min(size(binary_img, 1) - y + 1, ceil(bbox(4)));

    char_bw = binary_img(y:y+h-1, x:x+w-1);
    char_gray = gray_img(y:y+h-1, x:x+w-1);
end

function [bw_trim, gray_trim, row_offset, col_offset] = trim_character(bw_img, gray_img)
    % Remove empty rows/columns around the binary glyph
    rows = any(bw_img, 2);
    cols = any(bw_img, 1);

    if ~any(rows) || ~any(cols)
        bw_trim = bw_img;
        gray_trim = gray_img;
        row_offset = 1;
        col_offset = 1;
        return;
    end

    row_idx = find(rows);
    col_idx = find(cols);

    row_start = row_idx(1);
    row_end = row_idx(end);
    col_start = col_idx(1);
    col_end = col_idx(end);

    bw_trim = bw_img(row_start:row_end, col_start:col_end);
    gray_trim = gray_img(row_start:row_end, col_start:col_end);
    row_offset = row_start;
    col_offset = col_start;
end

function [canvas_double, canvas_uint8] = prepare_char_canvas(char_gray, target_size)
    % Resize character patch while preserving aspect ratio and center on canvas

    if isa(char_gray, 'uint8')
        char_double = double(char_gray) / 255;
    else
        char_double = mat2gray(char_gray);
    end

    [h, w] = size(char_double);
    if h == 0 || w == 0
        canvas_double = zeros(target_size);
        canvas_uint8 = uint8(canvas_double);
        return;
    end

    margin = 4;
    scale = min((target_size - margin) / h, (target_size - margin) / w);
    resized = myImresize(char_double, scale, 'bilinear');
    [rh, rw] = size(resized);

    canvas_double = zeros(target_size);
    row_start = floor((target_size - rh) / 2) + 1;
    col_start = floor((target_size - rw) / 2) + 1;
    canvas_double(row_start:row_start+rh-1, col_start:col_start+rw-1) = resized;

    canvas_uint8 = uint8(canvas_double * 255);
end
