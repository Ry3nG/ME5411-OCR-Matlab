function num = labelChar2Num(label)
    labels_name = {'0', '4', '7', '8', 'A', 'D', 'H'};
    num = find(strcmp(labels_name, label)) - 1;  % Convert to 0-indexed
end
