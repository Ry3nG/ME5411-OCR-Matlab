function [data_train, labels_train, data_test, labels_test] = loadDataset(data_path, options)
    % Note: Paths should already be added by the calling script
    % addpath(genpath('core/image_process')); is already done in caller
    labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};
    data_train = [];
    labels_train = [];
    data_test = [];
    labels_test = [];
    img_dim = options.img_dim;

    if options.load_raw
        data_path = data_path + "/dataset_2025/";
        files = dir(data_path);

        % First pass: count total images for pre-allocation
        total_train = 0;
        total_test = 0;
        for i = 1:length(files)
            curr_file = files(i).name;
            if sum(strcmp(curr_file, labels_name)) > 0
                fullPath = fullfile(data_path, curr_file);
                filesInFolder = dir(fullPath);
                num_files = sum(endsWith({filesInFolder.name}, '.png'));
                num_train = floor(num_files * options.train_ratio);
                total_train = total_train + num_train;
                total_test = total_test + (num_files - num_train);
            end
        end

        % Pre-allocate arrays
        data_train = zeros(img_dim, img_dim, total_train);
        labels_train = zeros(1, total_train);
        data_test = zeros(img_dim, img_dim, total_test);
        labels_test = zeros(1, total_test);

        train_idx = 1;
        test_idx = 1;
        processed = 0;
        total = total_train + total_test;

        for i = 1:length(files)
            curr_file = files(i).name;

            if sum(strcmp(curr_file, labels_name)) > 0
                fullPath = fullfile(data_path, curr_file);
                filesInFolder = dir(fullPath);
                num_files = sum(endsWith({filesInFolder.name}, '.png'));
                num_train = floor(num_files * options.train_ratio);

                png_count = 0;  % Counter for PNG files only
                for j = 1:length(filesInFolder)
                    filename = filesInFolder(j).name;

                    if endsWith(filename, '.png')
                        png_count = png_count + 1;  % Increment PNG counter
                        processed = processed + 1;
                        if mod(processed, 100) == 0 || processed == total
                            fprintf('Processing: %d/%d (%.1f%%)\n', processed, total, 100*processed/total);
                        end

                        img = imread(fullfile(fullPath, filename));
                        % Convert RGB to grayscale if needed
                        if size(img, 3) == 3
                            img = myRgb2gray(img);
                        end
                        label = labelChar2Num(curr_file);

                        if png_count <= num_train

                            if options.apply_rand_tf && options.rand_tf.prob > rand()
                                img = randTF(img, options.rand_tf);
                            end

                            img = myImresize(img, [img_dim, img_dim], 'bilinear');
                            img = double(img) / 255; % normalize
                            data_train(:, :, train_idx) = img;
                            labels_train(train_idx) = label;
                            train_idx = train_idx + 1;
                        else
                            img = myImresize(img, [img_dim, img_dim], 'bilinear');
                            img = double(img) / 255; % normalize
                            data_test(:, :, test_idx) = img;
                            labels_test(test_idx) = label;
                            test_idx = test_idx + 1;
                        end

                    end

                end

            end

        end

    else

        for p = 1:2

            if p == 1
                path = "../data/train/";
            else
                path = "../data/test/";
            end

            files = dir(path);

            for i = 1:length(files)
                curr_file = files(i).name;

                if sum(strcmp(curr_file, labels_name)) > 0
                    fullPath = fullfile(path, curr_file);
                    filesInFolder = dir(fullPath);

                    for j = 1:length(filesInFolder)
                        filename = filesInFolder(j).name;

                        if endsWith(filename, '.png')
                            img = imread(fullfile(fullPath, filename));
                            % Convert RGB to grayscale if needed
                            if size(img, 3) == 3
                                img = myRgb2gray(img);
                            end

                            if options.apply_rand_tf && p == 1 && options.rand_tf.prob > rand()
                                img = randTF(img, options.rand_tf);
                            end

                            img = myImresize(img, [img_dim, img_dim], 'bilinear');
                            img = double(img) / 255; % normalize
                            label = labelChar2Num(curr_file);

                            if p == 1
                                data_train = cat(3, data_train, img);
                                labels_train = cat(2, labels_train, label);
                            else
                                data_test = cat(3, data_test, img);
                                labels_test = cat(2, labels_test, label);
                            end

                        end

                    end

                end

            end

        end

        if options.shuffle
            [data_train, labels_train] = shuffleData(data_train, labels_train);
            [data_test, labels_test] = shuffleData(data_test, labels_test);
        end

    end

    data_train = reshape(data_train, img_dim, img_dim, 1, []);
    labels_train = permute(labels_train, [2, 1]);
    data_test = reshape(data_test, img_dim, img_dim, 1, []);
    labels_test = permute(labels_test, [2, 1]);

    if options.save
        % Save to data directory (relative to project root where calling script is)
        data_dir = 'data';
        if ~exist(data_dir, 'dir')
            mkdir(data_dir);
        end
        save(fullfile(data_dir, 'train.mat'), 'data_train', 'labels_train');
        save(fullfile(data_dir, 'test.mat'), 'data_test', 'labels_test');
    end

end
