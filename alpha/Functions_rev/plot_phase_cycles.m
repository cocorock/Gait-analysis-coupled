%% plot_phase_cycles: Helper function to plot phase cycles.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This is a helper function that plots the phase portraits (position vs.
%   velocity) for a set of gait cycles. It uses different colors to distinguish
%   cycles from different source files.
%
% Input:
%   pos_data     - (N x 200) matrix: Joint angle data.
%   vel_data     - (N x 200) matrix: Joint velocity data.
%   file_indices - (N x 1) vector: Indices mapping cycles to source files.
%   colors       - (F x 3) matrix: Color map for the F source files.
%   label        - string: The label for the plot.
%
% Output:
%   None. Plots are added to the current figure.

function plot_phase_cycles(pos_data, vel_data, file_indices, colors, label)
    n_cycles = size(pos_data,1);
    for i = 1:n_cycles
        file_idx = file_indices(i);
        plot(pos_data(i,:), vel_data(i,:), 'Color', [colors(file_idx,:), 0.6], 'LineWidth', 1.5);
        plot(pos_data(i,1), vel_data(i,1), 'o', 'Color', colors(file_idx,:), 'MarkerSize', 4, 'MarkerFaceColor', colors(file_idx,:));
    end
end