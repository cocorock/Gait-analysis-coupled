%% plot_linear_kinematics_positions: Plots the linear positions of the ankles.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function generates plots for the linear positions (x and y components)
%   of the right and left ankles, based on the output of
%   'calculate_linear_kinematics_v2'. It creates separate figures for right
%   and left leg kinematics, showing the trajectories for all cycles.
%
% Input:
%   linear_kinematics - struct: The structure containing linear kinematics data,
%                     typically from 'calculate_linear_kinematics_v2'.
%   time_standard     - vector: The normalized time vector (1x200).
%
% Output:
%   None. Generates MATLAB figures.

function plot_linear_kinematics_positionsV2(linear_kinematics, time_standard)
    fprintf('\n=== PLOTTING LINEAR KINEMATICS POSITIONS ===\n');

    output_dir = './Plots and Figs/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % --- Plot Right Leg Kinematics Positions ---
    if isfield(linear_kinematics, 'right_leg_kinematics') && ~isempty(linear_kinematics.right_leg_kinematics)
        figure('Name', 'Right Leg Ankle Positions');
        set(gcf, 'WindowStyle', 'docked');
        sgtitle('Right Leg Ankle Positions (All Cycles)');

        num_cycles = length(linear_kinematics.right_leg_kinematics);

        % Right Ankle X Position
        subplot(2, 2, 1);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.right_leg_kinematics(i).right_ankle_pos(1,:), 'Color', [0.8 0.2 0.2 0.5]);
        end
        hold off;
        title('Right Ankle X Position');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Right Ankle Y Position
        subplot(2, 2, 2);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.right_leg_kinematics(i).right_ankle_pos(2,:), 'Color', [0.2 0.8 0.2 0.5]);
        end
        hold off;
        title('Right Ankle Y Position');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Left Ankle X Position (from right-segmented cycles)
        subplot(2, 2, 3);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.right_leg_kinematics(i).left_ankle_pos(1,:), 'Color', [0.2 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle X Position (Right Leg Cycles)');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Left Ankle Y Position (from right-segmented cycles)
        subplot(2, 2, 4);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.right_leg_kinematics(i).left_ankle_pos(2,:), 'Color', [0.8 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle Y Position (Right Leg Cycles)');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Save figure
        filename = fullfile(output_dir, sprintf('linear_kinematics_positions_right_leg_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));
        saveas(gcf, filename);
        fprintf('  Saved %s\n', filename);
    else
        fprintf('  No right leg kinematics data to plot positions.\n');
    end

    % --- Plot Left Leg Kinematics Positions ---
    if isfield(linear_kinematics, 'left_leg_kinematics') && ~isempty(linear_kinematics.left_leg_kinematics)
        figure('Name', 'Left Leg Ankle Positions');
        set(gcf, 'WindowStyle', 'docked');
        sgtitle('Left Leg Ankle Positions (All Cycles)');

        num_cycles = length(linear_kinematics.left_leg_kinematics);

        % Right Ankle X Position (from left-segmented cycles)
        subplot(2, 2, 1);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.left_leg_kinematics(i).right_ankle_pos(1,:), 'Color', [0.8 0.2 0.2 0.5]);
        end
        hold off;
        title('Right Ankle X Position (Left Leg Cycles)');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Right Ankle Y Position (from left-segmented cycles)
        subplot(2, 2, 2);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.left_leg_kinematics(i).right_ankle_pos(2,:), 'Color', [0.2 0.8 0.2 0.5]);
        end
        hold off;
        title('Right Ankle Y Position (Left Leg Cycles)');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Left Ankle X Position
        subplot(2, 2, 3);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.left_leg_kinematics(i).left_ankle_pos(1,:), 'Color', [0.2 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle X Position');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Left Ankle Y Position
        subplot(2, 2, 4);
        hold on;
        for i = 1:num_cycles
            plot(time_standard, linear_kinematics.left_leg_kinematics(i).left_ankle_pos(2,:), 'Color', [0.8 0.2 0.8 0.5]);
        end
        hold off;
        title('Left Ankle Y Position');
        xlabel('Normalized Time');
        ylabel('Position (m)');
        grid on;

        % Save figure
        filename = fullfile(output_dir, sprintf('linear_kinematics_positions_left_leg_%s.png', datestr(now, 'yyyymmdd_HHMMSS')));
        saveas(gcf, filename);
        fprintf('  Saved %s\n', filename);
    else
        fprintf('  No left leg kinematics data to plot positions.\n');
    end

    fprintf('Linear kinematics positions plotting complete!\n');
end
