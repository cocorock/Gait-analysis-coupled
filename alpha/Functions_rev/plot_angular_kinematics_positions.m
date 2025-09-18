%% plot_angular_kinematics_positions: Plots the angular kinematics positions.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function plots the angular positions (hip vs. knee) for all gait
%   cycles on a single figure.
%
% Input:
%   demos_filtered - (1 x N_cycles) cell array: Each cell contains a struct with a field:
%                      - pos: (2 x M) matrix of joint angles [hip; knee].
%
% Output:
%   None. A .png file of the plot is saved to the 'Plots and Figs' directory.

function plot_angular_kinematics_positions(demos_filtered, save_flag)
%%
    figure('Name', 'Linear Kinematics Positions - All Gait Cycles', 'Color', 'w');
    hold on;
    grid on;
    xlabel('\theta_{1} Position (°)');
    ylabel('\theta_{2} Position(°)');
    title('Angular Kinematics Positions of All Gait Cycles');

    n_cycles = length(demos_filtered);
    
    for i = 1:n_cycles
        pos = demos_filtered{i}.pos; % 2 x M matrix
        plot(-pos(1, :), pos(2, :), 'LineWidth', 1.5);
    end
    
    hold off;
    axis equal;
%     legend_entries = arrayfun(@(x) sprintf('Cycle %d', x), 1:n_cycles, 'UniformOutput', false);
%     legend(legend_entries, 'Location', 'bestoutside');
    
    if save_flag
        phase_plot_filename = sprintf('Plots and Figs/Angular_kinematics_positions_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, phase_plot_filename);
        fprintf('Angular Kinematics Positions saved as: %s\n', phase_plot_filename);
    end
end