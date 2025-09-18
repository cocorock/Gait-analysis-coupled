%% plot_gait_kinematics_v3: Plots raw and filtered joint positions for gait cycles.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function visualizes the processed gait data from apply_filtering_V3.m.
%   It generates a single figure for each leg, displaying raw and filtered positions
%   for each of the four main joints (right/left hip/knee).
%   Each cycle is plotted twice (concatenated) to visually inspect continuity.
%
% Input:
%   processed_data - struct: The processed gait data structure from apply_filtering_V3.
%   leg_to_plot    - (optional) string: 'right', 'left', or 'both'. Defaults to 'both'.
%
% Output:
%   One or two figures displaying the position data.

function plot_gait_kinematics_v3(processed_data, leg_to_plot)
    if nargin < 2
        leg_to_plot = 'both';
    end
    
    fprintf('\n=== PLOTTING GAIT KINEMATICS (V3) ===\n');
    
    time_standard = processed_data.time_standard;
    cycle_duration = time_standard(end);
    % Create a time vector for plotting two full cycles to check continuity
    time_double = [time_standard, time_standard(2:end) + cycle_duration];
    
    joint_names = {'Right Hip', 'Left Hip', 'Right Knee', 'Left Knee'};
    joint_fields_raw = {'right_hip_flex', 'left_hip_flex', 'right_knee_flex', 'left_knee_flex'};
    joint_fields_filtered = {'right_hip_flex_filtered', 'left_hip_flex_filtered', 'right_knee_flex_filtered', 'left_knee_flex_filtered'};
    
    % --- Plot Right Leg Cycles ---
    if (strcmp(leg_to_plot, 'right') || strcmp(leg_to_plot, 'both')) && isfield(processed_data, 'right_leg_cycles') && ~isempty(processed_data.right_leg_cycles)
        fprintf('  Plotting Right Leg Positions...\n');
        num_cycles = length(processed_data.right_leg_cycles);
        colors = hsv(num_cycles); % Use HSV colormap for distinct cycle colors

        figure('Name', 'Right Leg Angular Positions', 'Position', [50, 50, 1200, 800]);
        sgtitle('Right Leg Cycles - Joint Positions (Raw vs Filtered)');

        for j = 1:length(joint_names)
            subplot(2, 2, j);
            hold on;
            h_for_legend = gobjects(2, 1);
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                raw_data = processed_data.right_leg_cycles(i).(joint_fields_raw{j});
                raw_data_double = [raw_data, raw_data(2:end)];
                filtered_data = processed_data.right_leg_cycles(i).(joint_fields_filtered{j});
                filtered_data_double = [filtered_data, filtered_data(2:end)];

                p_filtered = plot(time_double, filtered_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
                p_raw = plot(time_double, raw_data_double, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
                if i == 1, h_for_legend(1) = p_raw; h_for_legend(2) = p_filtered; end
            end
            xline(cycle_duration, '--k', 'HandleVisibility', 'off'); % Mark cycle junction
            hold off;
            title(joint_names{j});
            xlabel('Time (s)');
            ylabel('Angle (degrees)');
            grid on;
            xlim([0, 2 * cycle_duration]);
            if j == 1 && num_cycles > 0, legend(h_for_legend, {'Raw', 'Filtered'}, 'Location', 'best'); end
        end
    else
        fprintf('  No right leg cycles to plot or plotting skipped.\n');
    end
    
    % --- Plot Left Leg Cycles ---
    if (strcmp(leg_to_plot, 'left') || strcmp(leg_to_plot, 'both')) && isfield(processed_data, 'left_leg_cycles') && ~isempty(processed_data.left_leg_cycles)
        fprintf('  Plotting Left Leg Positions...\n');
        num_cycles = length(processed_data.left_leg_cycles);
        colors = hsv(num_cycles); % Use a colormap for distinct cycle colors

        figure('Name', 'Left Leg Angular Positions', 'Position', [400, 250, 1200, 800]);
        sgtitle('Left Leg Cycles - Joint Positions (Raw vs Filtered)');

        for j = 1:length(joint_names)
            subplot(2, 2, j);
            hold on;
            h_for_legend = gobjects(2, 1);
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                raw_data = processed_data.left_leg_cycles(i).(joint_fields_raw{j});
                raw_data_double = [raw_data, raw_data(2:end)];
                filtered_data = processed_data.left_leg_cycles(i).(joint_fields_filtered{j});
                filtered_data_double = [filtered_data, filtered_data(2:end)];

                p_filtered = plot(time_double, filtered_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
                p_raw = plot(time_double, raw_data_double, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
                if i == 1, h_for_legend(1) = p_raw; h_for_legend(2) = p_filtered; end
            end
            xline(1, '--k', 'HandleVisibility', 'off'); % Mark cycle junction
            hold off;
            title(joint_names{j});
            xlabel('Normalized Cycle Time (2 Cycles)');
            ylabel('Angle (degrees)');
            grid on;
            xlim([0, 2]);
            if j == 1 && num_cycles > 0, legend(h_for_legend, {'Raw', 'Filtered'}, 'Location', 'best'); end
        end
    else
        fprintf('  No left leg cycles to plot or plotting skipped.\n');
    end
    
    fprintf('Plotting complete!\n');
end