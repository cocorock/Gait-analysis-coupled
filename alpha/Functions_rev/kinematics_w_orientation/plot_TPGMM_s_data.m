function plot_TPGMM_s_data(s, nbSamples)
    %% plot_TPGMM_s_data: Plots trajectories and their frames from the 's' structure.
    %
    % Description:
    %   This function visualizes the original position trajectories stored in the
    %   's' structure, along with their associated frames of reference.
    %
    % Input:
    %   s         - Struct array containing demonstration data (pos, vel, frames).
    %   nbSamples - Number of demonstrations in the 's' struct array.

    fprintf('\n=== PLOTTING TRAJECTORIES FROM \s\ STRUCTURE ===\n');

    if isempty(s)
        fprintf('  Input \s\ structure is empty. Nothing to plot.\n');
        return;
    end

    figure('Name', 'Original Trajectories with Frames');
    hold on;
    box on;
    title('Original Trajectories with Defined Frames');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
    axis equal;

    % Define colors for trajectories and frames
    clrmap = lines(nbSamples); % Colormap for different trajectories
    colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078]; % Colors for Frame 1 and Frame 2

    for n = 1:nbSamples
        % Plot trajectory (position components only)
        plot(s(n).Data(1,:), s(n).Data(2,:), '-', 'LineWidth', 1.5, 'Color', clrmap(n,:));
        plot(s(n).Data(1,1), s(n).Data(2,1),'.','markersize',15,'color',clrmap(n,:)); % Start point

        % Plot frames (using a simplified representation or a helper function if available)
        % For simplicity, we'll plot the origin and a small arrow for orientation
        for m = 1:length(s(n).p)
            origin = s(n).p(m).b;
            rotation_matrix = s(n).p(m).A;
            
            % Plot origin of the frame
            plot(origin(1), origin(2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', colPegs(m,:), 'MarkerEdgeColor', colPegs(m,:));
            
            % Plot orientation arrow (e.g., along the x-axis of the frame)
            arrow_length = 0.1; % Adjust as needed
            x_axis_vec = rotation_matrix * [arrow_length; 0];
            quiver(origin(1), origin(2), x_axis_vec(1), x_axis_vec(2), 0, 'Color', colPegs(m,:), 'LineWidth', 1.5);
        end
    end

    % Add a dummy plot for legend entries for frames
    plot(NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerFaceColor', colPegs(1,:), 'MarkerEdgeColor', colPegs(1,:), 'DisplayName', 'Frame 1 (Index 200) Origin');
    plot(NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerFaceColor', colPegs(2,:), 'MarkerEdgeColor', colPegs(2,:), 'DisplayName', 'Frame 2 (Index 160) Origin');
    legend('show', 'Location', 'best');

    hold off;
    fprintf('Plotting complete.\n');
end
