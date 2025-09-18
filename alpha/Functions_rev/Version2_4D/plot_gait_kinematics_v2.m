%% plot_gait_kinematics_v2: Plots raw/filtered positions, velocities, and accelerations for gait cycles in tabs.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function visualizes the processed gait data from apply_filtering_and_derivatives_v2.m.
%   It generates a single figure for each leg, with 'Positions', 'Velocities', 
%   and 'Accelerations' organized into separate tabs.
%   Each cycle is plotted twice (concatenated) to visually inspect the continuity from end to start.
%   Each cycle is given a unique color. Raw data is plotted as a dashed line.
%
% Input:
%   processed_data - struct: The processed gait data structure from apply_filtering_and_derivatives_v2.
%   leg_to_plot    - (optional) string: 'right', 'left', or 'both'. Defaults to 'both'.
%
% Output:
%   One or two figures with tabbed plots displaying the kinematic data.

function plot_gait_kinematics_v2(processed_data, leg_to_plot)
    if nargin < 2
        leg_to_plot = 'both';
    end
    
    fprintf('\n=== PLOTTING GAIT KINEMATICS (V2) IN TABS ===\n');
    
    time_standard = processed_data.time_standard;
    % Create a time vector for plotting two full cycles to check continuity
    time_double = [time_standard, time_standard(2:end) + 1];
    
    joint_names = {'Right Hip', 'Left Hip', 'Right Knee', 'Left Knee'};
    joint_fields_raw = {'right_hip_flex', 'left_hip_flex', 'right_knee_flex', 'left_knee_flex'};
    joint_fields_filtered = {'right_hip_flex_filtered', 'left_hip_flex_filtered', 'right_knee_flex_filtered', 'left_knee_flex_filtered'};
    joint_fields_velocity = {'right_hip_flex_velocity', 'left_hip_flex_velocity', 'right_knee_flex_velocity', 'left_knee_flex_velocity'};
    joint_fields_acceleration = {'right_hip_flex_acceleration', 'left_hip_flex_acceleration', 'right_knee_flex_acceleration', 'left_knee_flex_acceleration'};
    
    % --- Plot Right Leg Cycles ---
    if (strcmp(leg_to_plot, 'right') || strcmp(leg_to_plot, 'both')) && isfield(processed_data, 'right_leg_cycles') && ~isempty(processed_data.right_leg_cycles)
        fprintf('  Plotting Right Leg Cycles...\n');
        num_cycles = length(processed_data.right_leg_cycles);
        colors = hsv(num_cycles); % Use HSV colormap for distinct cycle colors

        % Create a single figure and a tab group
        fig_right = figure('Name', 'Right Leg Angular Kinematics', 'Position', [50, 50, 1200, 800]);
        tab_group = uitabgroup(fig_right);
        
        % Create tabs for each kinematic type
        tab_pos = uitab(tab_group, 'Title', 'Positions');
        tab_vel = uitab(tab_group, 'Title', 'Velocities');
        tab_accel = uitab(tab_group, 'Title', 'Accelerations');

        % Loop through each joint and create subplots on each tab
        for j = 1:length(joint_names)
            % --- POSITIONS SUBPLOT ---
            ax_pos = subplot(2, 2, j, 'Parent', tab_pos);
            hold(ax_pos, 'on');
            h_for_legend = gobjects(2, 1);
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                raw_data = processed_data.right_leg_cycles(i).(joint_fields_raw{j});
                raw_data_double = [raw_data, raw_data(2:end)];
                filtered_data = processed_data.right_leg_cycles(i).(joint_fields_filtered{j});
                filtered_data_double = [filtered_data, filtered_data(2:end)];

                p_filtered = plot(ax_pos, time_double, filtered_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
                p_raw = plot(ax_pos, time_double, raw_data_double, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
                if i == 1, h_for_legend(1) = p_raw; h_for_legend(2) = p_filtered; end
            end
            hold(ax_pos, 'off');
            title(ax_pos, joint_names{j});
            xlabel(ax_pos, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_pos, 'Angle (degrees)');
            grid(ax_pos, 'on');
            xlim(ax_pos, [0, 2]);
            if j == 1 && num_cycles > 0, legend(ax_pos, h_for_legend, {'Raw', 'Filtered'}, 'Location', 'best'); end

            % --- VELOCITIES SUBPLOT ---
            ax_vel = subplot(2, 2, j, 'Parent', tab_vel);
            hold(ax_vel, 'on');
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                velocity_data = processed_data.right_leg_cycles(i).(joint_fields_velocity{j});
                velocity_data_double = [velocity_data, velocity_data(2:end)];
                plot(ax_vel, time_double, velocity_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
            end
            hold(ax_vel, 'off');
            title(ax_vel, joint_names{j});
            xlabel(ax_vel, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_vel, 'Angular Velocity (deg/s)');
            grid(ax_vel, 'on');
            xlim(ax_vel, [0, 2]);

            % --- ACCELERATIONS SUBPLOT ---
            ax_accel = subplot(2, 2, j, 'Parent', tab_accel);
            hold(ax_accel, 'on');
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                accel_data = processed_data.right_leg_cycles(i).(joint_fields_acceleration{j});
                accel_data_double = [accel_data, accel_data(2:end)];
                plot(ax_accel, time_double, accel_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
            end
            hold(ax_accel, 'off');
            title(ax_accel, joint_names{j});
            xlabel(ax_accel, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_accel, 'Angular Acceleration (deg/s^2)');
            grid(ax_accel, 'on');
            xlim(ax_accel, [0, 2]);
        end
    else
        fprintf('  No right leg cycles to plot or plotting skipped.\n');
    end
    
    % --- Plot Left Leg Cycles ---
    if (strcmp(leg_to_plot, 'left') || strcmp(leg_to_plot, 'both')) && isfield(processed_data, 'left_leg_cycles') && ~isempty(processed_data.left_leg_cycles)
        fprintf('  Plotting Left Leg Cycles...\n');
        num_cycles = length(processed_data.left_leg_cycles);
        colors = hsv(num_cycles); % Use a colormap for distinct cycle colors

        % Create a single figure and a tab group
        fig_left = figure('Name', 'Left Leg Angular Kinematics', 'Position', [400, 250, 1200, 800]);
        tab_group = uitabgroup(fig_left);
        
        % Create tabs for each kinematic type
        tab_pos = uitab(tab_group, 'Title', 'Positions');
        tab_vel = uitab(tab_group, 'Title', 'Velocities');
        tab_accel = uitab(tab_group, 'Title', 'Accelerations');

        % Loop through each joint and create subplots on each tab
        for j = 1:length(joint_names)
            % --- POSITIONS SUBPLOT ---
            ax_pos = subplot(2, 2, j, 'Parent', tab_pos);
            hold(ax_pos, 'on');
            h_for_legend = gobjects(2, 1);
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                raw_data = processed_data.left_leg_cycles(i).(joint_fields_raw{j});
                raw_data_double = [raw_data, raw_data(2:end)];
                filtered_data = processed_data.left_leg_cycles(i).(joint_fields_filtered{j});
                filtered_data_double = [filtered_data, filtered_data(2:end)];

                p_filtered = plot(ax_pos, time_double, filtered_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
                p_raw = plot(ax_pos, time_double, raw_data_double, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
                if i == 1, h_for_legend(1) = p_raw; h_for_legend(2) = p_filtered; end
            end
            hold(ax_pos, 'off');
            title(ax_pos, joint_names{j});
            xlabel(ax_pos, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_pos, 'Angle (degrees)');
            grid(ax_pos, 'on');
            xlim(ax_pos, [0, 2]);
            if j == 1 && num_cycles > 0, legend(ax_pos, h_for_legend, {'Raw', 'Filtered'}, 'Location', 'best'); end

            % --- VELOCITIES SUBPLOT ---
            ax_vel = subplot(2, 2, j, 'Parent', tab_vel);
            hold(ax_vel, 'on');
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                velocity_data = processed_data.left_leg_cycles(i).(joint_fields_velocity{j});
                velocity_data_double = [velocity_data, velocity_data(2:end)];
                plot(ax_vel, time_double, velocity_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
            end
            hold(ax_vel, 'off');
            title(ax_vel, joint_names{j});
            xlabel(ax_vel, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_vel, 'Angular Velocity (deg/s)');
            grid(ax_vel, 'on');
            xlim(ax_vel, [0, 2]);

            % --- ACCELERATIONS SUBPLOT ---
            ax_accel = subplot(2, 2, j, 'Parent', tab_accel);
            hold(ax_accel, 'on');
            for i = 1:num_cycles
                cycle_color = colors(i, :);
                accel_data = processed_data.left_leg_cycles(i).(joint_fields_acceleration{j});
                accel_data_double = [accel_data, accel_data(2:end)];
                plot(ax_accel, time_double, accel_data_double, '-', 'Color', cycle_color, 'LineWidth', 1.5);
            end
            hold(ax_accel, 'off');
            title(ax_accel, joint_names{j});
            xlabel(ax_accel, 'Normalized Cycle Time (2 Cycles)');
            ylabel(ax_accel, 'Angular Acceleration (deg/s^2)');
            grid(ax_accel, 'on');
            xlim(ax_accel, [0, 2]);
        end
    else
        fprintf('  No left leg cycles to plot or plotting skipped.\n');
    end
    
    fprintf('Plotting complete!\n');
end
