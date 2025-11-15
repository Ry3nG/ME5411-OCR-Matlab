




% Extract baseline samples for Task 8 report
load('data/train.mat');

classes = [0, 1, 2, 3, 4, 5, 6];  % 0-indexed: 0, 4, 7, 8, A, D, H
class_names = {'0', '4', '7', '8', 'A', 'D', 'H'};

mkdir('output/task8_augmentation/baseline');

for i = 1:length(classes)
    cls = classes(i);
    idx = find(labels_train == cls, 1);
    img = data_train(:, :, 1, idx);

    % Save
    filename = sprintf('output/task8_augmentation/baseline/class_%s.png', class_names{i});
    imwrite(img, filename);
end

fprintf('Baseline samples extracted to output/task8_augmentation/baseline/\n');
