%% plot_linear_kinematics_positions: Plots all gait cycle linear positions.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function plots the end-effector trajectories (x vs. y) for all gait
%   cycles on the same figure, distinguishing between right and left leg cycles.
%
% Input:
%   linear_kinematics - (1 x N_cycles) cell array: Each cell contains a struct with a field:
%                         - pos: (2 x M) matrix of [x; y] positions.
%   processed_data    - struct: Used to determine the number cycles.
%
% Output:
%   None. A .png file of the plot is saved to the 'Plots and Figs' directory.

function plot_linear_kinematics_positions(linear_kinematics, processed_data, save_flag)

    figure('Name', 'Linear Kinematics Positions - All Gait Cycles', 'Color', 'w');
%     subplot(1,2,1);
    hold on;
    grid on;
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Linear Kinematics Positions');
    
    n1 = length(processed_data.filtered.right_hip_cycles(:,1));
    n_cycles = length(linear_kinematics);
    
    for i = 1:n1
        pos = linear_kinematics{i}.pos; % 2 x M matrix
        plot(pos(1, :), pos(2, :), '--');
    end
%     hold off;
%     axis equal;
    
%     subplot(1,2,2);
%     hold on;
%     grid on;
%     xlabel('X Position (m)');
%     ylabel('Y Position (m)');
%     title('Linear Kinematics Positions of Left Leg');
%     axis equal;
    
    for i = n1+1:n_cycles
        pos = linear_kinematics{i}.pos; % 2 x M matrix
        plot(pos(1, :), pos(2, :), '.');
    end
    
    hold off;
    axis equal;
%     legend_entries = arrayfun(@(x) sprintf('Cycle %d', x), 1:n_cycles, 'UniformOutput', false);
%     legend(legend_entries, 'Location', 'bestoutside');
    
    if save_flag
        phase_plot_filename = sprintf('Plots and Figs/linear_kinematics_positions_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, phase_plot_filename);
        fprintf('Linear Kinematics Positions saved as: %s\n', phase_plot_filename);
    end
%% Velocities 

    figure('Name', 'Linear Kinematics Vel - All Gait Cycles', 'Color', 'w');
%     subplot(1,2,1);
    hold on;
    grid on;
    xlabel('X Velocity (m/s)');
    ylabel('Y Velocity (m/s)');
    title('Linear Kinematics Velocities');
    
    n1 = length(processed_data.filtered.right_hip_cycles(:,1));
    n_cycles = length(linear_kinematics);
    
    for i = 1:n1
        vel = linear_kinematics{i}.vel; % 2 x M matrix
        plot(vel(1, :), vel(2, :), '--');
    end

    for i = n1+1:n_cycles
        vel = linear_kinematics{i}.vel; % 2 x M matrix
        plot(vel(1, :), vel(2, :), '.');
    end
    
    hold off;
    axis equal;
%     legend_entries = arrayfun(@(x) sprintf('Cycle %d', x), 1:n_cycles, 'UniformOutput', false);
%     legend(legend_entries, 'Location', 'bestoutside');
    
    if save_flag
        phase_plot_filename = sprintf('Plots and Figs/linear_kinematics_velocities_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, phase_plot_filename);
        fprintf('Linear Kinematics Velocities saved as: %s\n', phase_plot_filename);
    end
    
    %% Accel

    figure('Name', 'Linear Kinematics Acc - All Gait Cycles', 'Color', 'w');
%     subplot(1,2,1);
    hold on;
    grid on;
    xlabel('X Acceleration (m/s^{2})');
    ylabel('Y Acceleration (m/s^{2})');
    title('Linear Kinematics Acceleration');
    
    n1 = length(processed_data.filtered.right_hip_cycles(:,1));
    n_cycles = length(linear_kinematics);
    
    for i = 1:n1
        acc = linear_kinematics{i}.acc; % 2 x M matrix
        plot(acc(1, :), acc(2, :), '--');
    end

    for i = n1+1:n_cycles
        acc = linear_kinematics{i}.acc; % 2 x M matrix
        plot(acc(1, :), acc(2, :), '.');
    end
    
    hold off;
    axis equal;
%     legend_entries = arrayfun(@(x) sprintf('Cycle %d', x), 1:n_cycles, 'UniformOutput', false);
%     legend(legend_entries, 'Location', 'bestoutside');
    
    if save_flag
        phase_plot_filename = sprintf('Plots and Figs/linear_kinematics_Acceleration_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, phase_plot_filename);
        fprintf('Linear Kinematics Acceleration saved as: %s\n', phase_plot_filename);
    end
end