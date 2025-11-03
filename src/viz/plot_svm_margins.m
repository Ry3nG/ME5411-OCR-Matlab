function plot_svm_margins(margins, outpath)
% PLOT_SVM_MARGINS Visualize distribution of SVM margins
%
% Inputs:
%   margins - [N x 1] margin values (true_score - second_best_score)
%   outpath - Optional output file path
%
% Theory: Margin represents classification confidence (from SVM theory).
%         Small margins = hard examples near decision boundary
%         Large margins = easy examples far from boundary
%         Negative margins = misclassifications

    figure('Color', 'w', 'Position', [100, 100, 800, 500]);
    set(gcf, 'ToolBar', 'none', 'MenuBar', 'none');

    histogram(margins, 50, 'FaceColor', [0.2, 0.4, 0.6]);
    grid on;
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8, 0.8, 0.8]);
    xlabel('Margin (True Class Score - Second Best Score)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Number of Samples', 'FontSize', 11, 'FontWeight', 'bold');
    title('SVM Decision Margin Distribution', 'FontSize', 12, 'FontWeight', 'bold');

    % Add vertical line at margin=0
    hold on;
    yl = ylim;
    plot([0, 0], yl, 'r--', 'LineWidth', 2);
    text(0, yl(2) * 0.9, 'Decision Boundary', ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
    hold off;

    if nargin > 1
        saveas(gcf, outpath);
        close(gcf);
    end
end
