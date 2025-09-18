%% create_phase_plots: Creates and saves phase plots for gait analysis.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function generates and saves phase plots (position vs. velocity) for
%   hip and knee joints based on the processed gait data.
%
% Input:
%   processed_data - struct: Contains filtered data and derivatives.
%                     See 'apply_filtering_and_derivatives.m' for details.
%   file_info      - struct: Contains file information for plotting.
%
% Output:
%   None. A .png file of the plot is saved to the 'Plots and Figs' directory.

function create_phase_plots(processed_data, file_info, save_flag)
    fprintf('\n=== CREATING PHASE PLOTS ===\n');
    
    figure('Name', 'Phase Plots - All AMC Files', 'Position', [100, 100, 1200, 600]);
    
    subplot(1,2,1);
    hold on;
    if isfield(processed_data.filtered, 'right_hip_cycles') && ~isempty(processed_data.filtered.right_hip_cycles)
        plot_phase_cycles(processed_data.filtered.right_hip_cycles, processed_data.derivatives.right_hip_velocity, processed_data.file_indices.right_hip, file_info.colors, 'Right Hip');
    end
    if isfield(processed_data.filtered, 'left_hip_cycles') && ~isempty(processed_data.filtered.left_hip_cycles)
        plot_phase_cycles(processed_data.filtered.left_hip_cycles, processed_data.derivatives.left_hip_velocity, processed_data.file_indices.left_hip, file_info.colors, 'Left Hip');
    end
    xlabel('Hip Position (degrees)');
    ylabel('Hip Velocity (deg/s)');
    title('Hip Phase Plot - All Cycles');
    grid on;
    
    subplot(1,2,2);
    hold on;
    if isfield(processed_data.filtered, 'right_knee_cycles') && ~isempty(processed_data.filtered.right_knee_cycles)
        plot_phase_cycles(processed_data.filtered.right_knee_cycles, processed_data.derivatives.right_knee_velocity, processed_data.file_indices.right_knee, file_info.colors, 'Right Knee');
    end
    if isfield(processed_data.filtered, 'left_knee_cycles') && ~isempty(processed_data.filtered.left_knee_cycles)
        plot_phase_cycles(processed_data.filtered.left_knee_cycles, processed_data.derivatives.left_knee_velocity, processed_data.file_indices.left_knee, file_info.colors, 'Left Knee');
    end
    xlabel('Knee Position (degrees)');
    ylabel('Knee Velocity (deg/s)');
    title('Knee Phase Plot - All Cycles');
    grid on;
    
    legend_entries = {};
    for i = 1:length(file_info.names)
        [~, name_only, ~] = fileparts(file_info.names{i});
        legend_entries{end+1} = name_only;
        plot(NaN, NaN, 'Color', file_info.colors(i,:), 'LineWidth', 2);
    end
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    
    if save_flag
        phase_plot_filename = sprintf('Plots and Figs/phase_plots_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, phase_plot_filename);
        fprintf('Phase plots saved as: %s\n', phase_plot_filename);
    end
end