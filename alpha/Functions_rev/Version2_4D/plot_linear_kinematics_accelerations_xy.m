%% plot_linear_kinematics_accelerations_xy: Plots the X vs Y linear accelerations of the ankles.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function generates X vs Y plots for the linear accelerations
%   of the right and left ankles, based on the output of
%   'calculate_linear_kinematics_v2'. It creates separate figures for right
%   and left leg kinematics, showing the trajectories for all cycles.
%
% Input:
%   linear_kinematics - struct: The structure containing linear kinematics data,
%                     typically from 'calculate_linear_kinematics_v2'.
%
% Output:
%   None. Generates MATLAB figures.

function plot_linear_kinematics_accelerations_xy(linear_kinematics)
    fprintf('\n=== PLOTTING LINEAR KINEMATICS ACCELERATIONS (X vs Y) ===\n');

    output_dir = './Plots and Figs/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % --- Plot Right Leg Kinematics Accelerations (X vs Y) ---
    if isfield(linear_kinematics, 'right_leg_kinematics') && ~isempty(linear_kinematics.right_leg_kinematics)
        figure('Name', 'Right Leg Ankle Accelerations (X vs Y)');
        set(gcf, 'WindowStyle', 'docked');
        sgtitle('Right Leg Ankle Accelerations (X vs Y) - All Cycles');

        num_cycles = length(linear_kinematics.right_leg_kinematics);

        % Right Ankle X vs Y Acceleration
        subplot(1, 2, 1);
        hold on;
        for i = 1:num_cycles
            plot(linear_kinematics.right_leg_kinematics(i).right_ankle_acc(1,:), linear_kinematics.right_leg_kinematics(i).right_ankle_acc(2,:), 'Color', [0.8 0.2 0.2 0.5]);
        end
        hold off;
        title('Right Ankle Acceleration Trajectory');
        xlabel('X Acceleration (m/s^2)');
        ylabel('Y Acceleration (m/s^2)');
        axis equal; % Maintain aspect ratio
        grid on;

        % Left Ankle X vs Y Acceleration (from right-segmented cycles)
        subplot(1, 2, 2);
        hold on;
        for i = 1:num_cycles
            plot(linear_kinematics.right_leg_kinematics(i).left_ankle_acc(1,:), linear_kinematics.right_leg_kinematics(i).left_ankle_acc(2,:), 'Color', [0.2 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle Acceleration Trajectory (Right Leg Cycles)');
        xlabel('X Acceleration (m/s^2)');
        ylabel('Y Acceleration (m/s^2)');
        axis equal; % Maintain aspect ratio
        grid on;

        % Save figure
        filename = fullfile(output_dir, sprintf('linear_kinematics_accelerations_xy_right_leg_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));
        saveas(gcf, filename);
        fprintf('  Saved %s\n', filename);
    else
        fprintf('  No right leg kinematics data to plot X vs Y accelerations.\n');
    end

    % --- Plot Left Leg Kinematics Accelerations (X vs Y) ---
    if isfield(linear_kinematics, 'left_leg_kinematics') && ~isempty(linear_kinematics.left_leg_kinematics)
        figure('Name', 'Left Leg Ankle Accelerations (X vs Y)');
        set(gcf, 'WindowStyle', 'docked');
        sgtitle('Left Leg Ankle Accelerations (X vs Y) - All Cycles');

        num_cycles = length(linear_kinematics.left_leg_kinematics);

        % Right Ankle X vs Y Acceleration (from left-segmented cycles)
        subplot(1, 2, 1);
        hold on;
        for i = 1:num_cycles
            plot(linear_kinematics.left_leg_kinematics(i).right_ankle_acc(1,:), linear_kinematics.left_leg_kinematics(i).right_ankle_acc(2,:), 'Color', [0.8 0.2 0.2 0.5]);
        end
        hold off;
        title('Right Ankle Acceleration Trajectory (Left Leg Cycles)');
        xlabel('X Acceleration (m/s^2)');
        ylabel('Y Acceleration (m/s^2)');
        axis equal; % Maintain aspect ratio
        grid on;

        % Left Ankle X vs Y Acceleration
        subplot(1, 2, 2);
        hold on;
        for i = 1:num_cycles
            plot(linear_kinematics.left_leg_kinematics(i).left_ankle_acc(1,:), linear_kinematics.left_leg_kinematics(i).left_ankle_acc(2,:), 'Color', [0.2 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle Acceleration Trajectory');
        xlabel('X Acceleration (m/s^2)');
        ylabel('Y Acceleration (m/s^2)');
        axis equal; % Maintain aspect ratio
        grid on;

        % Save figure
        filename = fullfile(output_dir, sprintf('linear_kinematics_accelerations_xy_left_leg_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));
        saveas(gcf, filename);
        fprintf('  Saved %s\n', filename);
    else
        fprintf('  No left leg kinematics data to plot X vs Y accelerations.\n');
    end

    fprintf('Linear kinematics accelerations (X vs Y) plotting complete!\n');
end
