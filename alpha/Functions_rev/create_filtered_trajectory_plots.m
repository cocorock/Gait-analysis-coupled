%% create_filtered_trajectory_plots: Creates and saves plots of filtered gait trajectories.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function generates a figure with subplots showing the filtered gait
%   trajectories for hip and knee flexion of both legs. It also plots the mean
%   and standard deviation for each joint.
%
% Input:
%   processed_data - struct: Contains the filtered gait data. See
%                     'apply_filtering_and_derivatives.m' for details.
%   file_info      - struct: Contains information about the processed files,
%                     including file names and colors for plotting.
%
% Output:
%   None. A .png file of the plot is saved to the 'Plots and Figs' directory.

function create_filtered_trajectory_plots(processed_data, file_info, save_flag)
    fprintf('\n=== CREATING FILTERED TRAJECTORY PLOTS ===\n');
    
    all_right_hip_cycles = processed_data.filtered.right_hip_cycles;
    all_left_hip_cycles = processed_data.filtered.left_hip_cycles;
    all_right_knee_cycles = processed_data.filtered.right_knee_cycles;
    all_left_knee_cycles = processed_data.filtered.left_knee_cycles;
    
    right_hip_file_indices = processed_data.file_indices.right_hip;
    left_hip_file_indices = processed_data.file_indices.left_hip;
    right_knee_file_indices = processed_data.file_indices.right_knee;
    left_knee_file_indices = processed_data.file_indices.left_knee;
    
    file_colors = file_info.colors;
    amc_files = file_info.names;
    total_cycles_found = file_info.total_cycles;
    
    figure('Name', 'All Gait Cycles from All AMC Files (Filtered)', 'Position', [0, 50, 1550, 800]);
    time_standard = processed_data.time_standard;
    
    % Plot 1: Right Hip Flexion
    subplot(2,3,1);
    hold on;
    for i = 1:size(all_right_hip_cycles,1)
        file_idx = right_hip_file_indices(i);
        plot(time_standard, all_right_hip_cycles(i,:), 'Color', [file_colors(file_idx,:), 0.6], 'LineWidth', 1);
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Right Hip Flexion (degrees)');
    title(sprintf('Right Hip Flexion - All %d Cycles (Filtered)', size(all_right_hip_cycles,1)));
    grid on;
    % legend commented out
    % legend_entries = {};
    % for file_idx = 1:length(amc_files)
    %     [~, name_only, ~] = fileparts(amc_files{file_idx});
    %     legend_entries{end+1} = name_only;
    %     plot(NaN, NaN, 'Color', file_colors(file_idx,:), 'LineWidth', 2);
    % end
    % legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    
    % Plot 2: Left Hip Flexion
    subplot(2,3,2);
    hold on;
    for i = 1:size(all_left_hip_cycles,1)
        file_idx = left_hip_file_indices(i);
        plot(time_standard, all_left_hip_cycles(i,:), 'Color', [file_colors(file_idx,:), 0.6], 'LineWidth', 1);
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Left Hip Flexion (degrees)');
    title(sprintf('Left Hip Flexion - All %d Cycles (Filtered)', size(all_left_hip_cycles,1)));
    grid on;
    % legend commented out
    % for file_idx = 1:length(amc_files)
    %     plot(NaN, NaN, 'Color', file_colors(file_idx,:), 'LineWidth', 2);
    % end
    % legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    
    % Plot 3: Hip Flexion Mean + STD
    subplot(2,3,3);
    hold on;
    if ~isempty(all_right_hip_cycles)
        right_hip_mean = mean(all_right_hip_cycles,1);
        right_hip_std = std(all_right_hip_cycles,0,1);
        fill([time_standard, fliplr(time_standard)], [right_hip_mean+right_hip_std, fliplr(right_hip_mean-right_hip_std)], ...
            'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Right � STD');
        plot(time_standard, right_hip_mean, 'r-', 'LineWidth', 3, 'DisplayName', 'Right Mean');
    end
    if ~isempty(all_left_hip_cycles)
        left_hip_mean = mean(all_left_hip_cycles,1);
        left_hip_std = std(all_left_hip_cycles,0,1);
        fill([time_standard, fliplr(time_standard)], [left_hip_mean+left_hip_std, fliplr(left_hip_mean-left_hip_std)], ...
            'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Left � STD');
        plot(time_standard, left_hip_mean, 'b-', 'LineWidth', 3, 'DisplayName', 'Left Mean');
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Hip Flexion (degrees)');
    title('Hip Flexion - Combined Mean � STD (Filtered)');
    legend('Location', 'best');
    grid on;
    
    % Plot 4: Right Knee Flexion
    subplot(2,3,4);
    hold on;
    for i = 1:size(all_right_knee_cycles,1)
        file_idx = right_knee_file_indices(i);
        plot(time_standard, all_right_knee_cycles(i,:), 'Color', [file_colors(file_idx,:), 0.6], 'LineWidth', 1);
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Right Knee Flexion (degrees)');
    title(sprintf('Right Knee Flexion - All %d Cycles (Filtered)', size(all_right_knee_cycles,1)));
    grid on;
    % legend commented out
    % for file_idx = 1:length(amc_files)
    %     plot(NaN, NaN, 'Color', file_colors(file_idx,:), 'LineWidth', 2);
    % end
    % legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    
    % Plot 5: Left Knee Flexion
    subplot(2,3,5);
    hold on;
    for i = 1:size(all_left_knee_cycles,1)
        file_idx = left_knee_file_indices(i);
        plot(time_standard, all_left_knee_cycles(i,:), 'Color', [file_colors(file_idx,:), 0.6], 'LineWidth', 1);
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Left Knee Flexion (degrees)');
    title(sprintf('Left Knee Flexion - All %d Cycles (Filtered)', size(all_left_knee_cycles,1)));
    grid on;
    % legend commented out
    % for file_idx = 1:length(amc_files)
    %     plot(NaN, NaN, 'Color', file_colors(file_idx,:), 'LineWidth', 2);
    % end
    % legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    
    % Plot 6: Knee Flexion Mean + STD
    subplot(2,3,6);
    hold on;
    if ~isempty(all_right_knee_cycles)
        right_knee_mean = mean(all_right_knee_cycles,1);
        right_knee_std = std(all_right_knee_cycles,0,1);
        fill([time_standard, fliplr(time_standard)], [right_knee_mean+right_knee_std, fliplr(right_knee_mean-right_knee_std)], ...
            'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Right � STD');
        plot(time_standard, right_knee_mean, 'r-', 'LineWidth', 3, 'DisplayName', 'Right Mean');
    end
    if ~isempty(all_left_knee_cycles)
        left_knee_mean = mean(all_left_knee_cycles,1);
        left_knee_std = std(all_left_knee_cycles,0,1);
        fill([time_standard, fliplr(time_standard)], [left_knee_mean+left_knee_std, fliplr(left_knee_mean-left_knee_std)], ...
            'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Left � STD');
        plot(time_standard, left_knee_mean, 'b-', 'LineWidth', 3, 'DisplayName', 'Left Mean');
    end
    xlabel('Normalized Time (0-1)');
    ylabel('Knee Flexion (degrees)');
    title('Knee Flexion - Combined Mean � STD (Filtered)');
    legend('Location', 'best');
    grid on;
    
    sgtitle(sprintf('Combined Gait Analysis: %d Files, %d Total Cycles', length(amc_files), total_cycles_found));
    
    % Save figure
    if save_flag
        combined_fig_filename = sprintf('Plots and Figs/combined_gait_analysis_filtered_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, combined_fig_filename);
        fprintf('Filtered trajectory plot saved as: %s\n', combined_fig_filename);
    end
end