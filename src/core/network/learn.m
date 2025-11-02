% @Description: Learn the CNN model
% @Input:
%   cnn: CNN model
%   data: training data
%   labels: training labels
%   data_t: testing data
%   labels_t: testing labels
%   options: options for learning
% @Output:
%   cnn_final: learned CNN model
function [cnn_final] = learn(cnn, data_train, labels_train, data_test, labels_test, options)
    % This file is modified based on UFLDL Deep Learning Tutorial
    % http://ufldl.stanford.edu/tutorial/
    %
    % Runs stochastic gradient descent with momentum to optimize the
    % parameters for the given objective.
    %
    % Parameters:
    %  funObj     -  function handle which accepts as input theta,
    %                data, labels and returns cost and gradient w.r.t
    %                to theta.
    %  theta      -  unrolled parameter vector
    %  data       -  stores data in m x n x numExamples tensor
    %  labels     -  corresponding labels in numExamples x 1 vector
    %  options    -  struct to store specific options for optimization
    %
    % Returns:
    %  opttheta   -  optimized parameter vector
    %
    % Options (* required)
    %  epochs*     - number of epochs through data
    %  lr*      - initial learning rate
    %  minibatch*  - size of minibatch
    %  momentum    - momentum constant, defualts to 0.9

    %% Setup
    theta = unrollWeights(cnn);
    assert(all(isfield(options, {'epochs', 'lr', 'minibatch'})), 'Some options not defined');

    if ~isfield(options, 'momentum')
        options.momentum = 0.9;
    end

    epochs = options.epochs;
    lr = options.lr_max;
    minibatch = options.minibatch;
    m = length(labels_train); % training set size
    iter_per_epoch = max(1, floor((m - minibatch) / minibatch) + 1);
    total_iter = iter_per_epoch * epochs;
    % Setup for momentum
    mom = 0.5;
    momIncrease = 20;
    velocity = zeros(size(theta));

    options.total_iter = total_iter;

    %%======================================================================
    it = 0;
    loss_ar = [];
    acc_train = [];
    acc_test = [];
    lr_ar = [];
    best_acc = 0;

    % For ETA calculation
    epoch_start_time = tic;
    total_time_elapsed = 0;

    for e = 1:epochs
        % randomly permute indices of data for quick minibatch sampling
        rp = randperm(m);

        epoch_iter_start = it + 1;

        for s = 1:minibatch:(m - minibatch + 1)
            it = it + 1;
            % increase momentum after momIncrease iterations
            % if it == momIncrease
            %     mom = options.momentum;
            % end
            mom = options.momentum;

            % get next randomly selected minibatch
            mb_data = data_train(:, :, :, rp(s:s + minibatch - 1));
            mb_labels = labels_train(rp(s:s + minibatch - 1));

            cnn = forward(cnn, mb_data, options);
            [cnn, curr_loss] = calcuLoss(cnn, mb_data, mb_labels, options);

            if options.train_mode
                grad = backward(cnn, mb_data, options);
                % update weights
                velocity = mom * velocity + (1 - mom) * grad;
                theta = theta - lr * velocity;
            end

            % update model
            if exist('theta', 'var')
                cnn = updateWeights(cnn, theta);
            end

            loss_ar = [loss_ar; curr_loss];

            % Display progress (every 10% of epoch or first iter)
            if mod(it - epoch_iter_start + 1, max(1, floor(iter_per_epoch/10))) == 0 || it == epoch_iter_start
                iter_in_epoch = it - epoch_iter_start + 1;
                epoch_progress = 100 * iter_in_epoch / iter_per_epoch;
                overall_progress = 100 * it / total_iter;

                % Calculate ETA (wait for at least 5 iterations for accurate estimate)
                if it >= 5
                    elapsed_total = total_time_elapsed + toc(epoch_start_time);
                    avg_time_per_iter = elapsed_total / it;
                    remaining_iters = total_iter - it;
                    eta_seconds = avg_time_per_iter * remaining_iters;
                    eta_minutes = round(eta_seconds / 60);
                    eta_hours = floor(eta_minutes / 60);
                    eta_minutes_rem = mod(eta_minutes, 60);

                    if eta_hours > 0
                        eta_str = sprintf('%dh%dm', eta_hours, eta_minutes_rem);
                    elseif eta_minutes > 0
                        eta_str = sprintf('%dm', eta_minutes);
                    else
                        eta_str = '<1m';
                    end
                else
                    eta_str = '...';
                end

                fprintf('Epoch %2d/%d | %5.1f%% | iter %3d/%3d | loss: %.4f | LR: %.5f | ETA: %s\n', ...
                    e, epochs, epoch_progress, iter_in_epoch, iter_per_epoch, curr_loss, lr, eta_str);
            end
        end

        % Epoch completed - evaluate on test and train sets
        epoch_time = toc(epoch_start_time);
        total_time_elapsed = total_time_elapsed + epoch_time;

        [preds,~] = predict(cnn, data_test);
        curr_acc_test = sum(preds == labels_test) / length(preds);
        acc_test = [acc_test; curr_acc_test];
        [preds,~] = predict(cnn, data_train);
        curr_acc_train = sum(preds == labels_train) / length(preds);
        acc_train = [acc_train; curr_acc_train];

        % Display epoch summary
        if curr_acc_test > best_acc
            best_acc = curr_acc_test;
            best_marker = ' ★ NEW BEST';

            if options.save_best_acc_model
                save(options.log_path + 'cnn_best_acc.mat', 'cnn');
                fileID = fopen(options.log_path + "results.txt", 'w');
                fprintf(fileID, 'Best accuracy: %f\n', curr_acc_test);
                fclose(fileID);
            end
        else
            best_marker = '';
        end

        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        fprintf('Epoch %2d Summary | Test: %6.2f%% | Train: %6.2f%% | Best: %6.2f%% | Time: %.1fs%s\n', ...
            e, curr_acc_test*100, curr_acc_train*100, best_acc*100, epoch_time, best_marker);
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

        lr = lrSchedule(e, epochs, options);
        lr_ar = [lr_ar; lr];

        epoch_start_time = tic;  % Reset for next epoch
    end

    opttheta = theta;
    cnn_final = updateWeights(cnn, opttheta);

    save(options.log_path + 'loss_ar.mat', 'loss_ar');
    save(options.log_path + 'acc_train.mat', 'acc_train');
    save(options.log_path + 'acc_test.mat', 'acc_test');
    save(options.log_path + 'lr_ar.mat', 'lr_ar');
    fprintf('Best accuracy: %f\n', best_acc);
end
