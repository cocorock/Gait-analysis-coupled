function plot_TPGMM_data(demos, frames, original_kinematics)
    %% plot_TPGMM_data: Visualizes all original and transformed trajectories.
    %
    % Description:
    %   Plots all original trajectories, all transformed trajectories, and adds
    %   orientation arrows for the first original and first transformed trajectories
    %   for a detailed comparison.
    %
    % Input:
    %   demos              - (1 x N_cycles) cell array of transformed trajectory data.
    %   frames             - (3 x 3 x N_cycles) array containing the transformation for each demo.
    %   original_kinematics - (1 x N_cycles) cell array of the original kinematics data.

    fprintf('\n=== PLOTTING ALL TRAJECTORIES WITH DETAILED COMPARISON ===\n');

    if isempty(demos) || isempty(original_kinematics)
        fprintf('  Input data is empty. Nothing to plot.\n');
        return;
    end

    figure('Name', 'All Trajectories with Orientation Comparison');
    hold on;

    % --- Plot all original trajectories in the background ---
    num_orig = length(original_kinematics);
    for i = 1:num_orig
        plot(original_kinematics{i}.pos(1, :), original_kinematics{i}.pos(2, :), '-', 'LineWidth', 1, 'Color', [0.6 0.8 1 0.6], 'HandleVisibility', 'off');
    end

    % --- Plot all transformed trajectories in the background ---
    num_demos = length(demos);
    for i = 1:num_demos
        transformed_traj = demos{i}(1:2, :);
        plot(transformed_traj(1, :), transformed_traj(2, :), '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5 0.6], 'HandleVisibility', 'off');
    end

    % --- Highlight the first trajectory (Original and Transformed) ---
    original_kin_1 = original_kinematics{1};
    original_traj_1 = original_kin_1.pos;
    transformed_traj_1 = demos{1}(1:2, :);
    
    % Re-plot the first original trajectory to be on top and thicker
    plot(original_traj_1(1, :), original_traj_1(2, :), 'b-', 'LineWidth', 2, 'DisplayName', 'First Original Trajectory');
    % Re-plot the first transformed trajectory to be on top and thicker
    plot(transformed_traj_1(1, :), transformed_traj_1(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'First Transformed Trajectory');

    % --- Add Orientation Arrows for the first trajectory ---
    num_points = size(original_traj_1, 2);
    num_arrows = round(num_points * 0.9);
    indices = round(linspace(1, num_points, num_arrows));
    arrow_scale = 0.3; % Adjust this value to change arrow length

    % Arrows for the original trajectory
    x_orig = original_traj_1(1, indices);
    y_orig = original_traj_1(2, indices);
    orient_orig = original_kin_1.orientation(indices);
    u_orig = cos(orient_orig);
    v_orig = sin(orient_orig);
    quiver(x_orig, y_orig, u_orig, v_orig, arrow_scale, 'Color', 'c', 'LineWidth', 1, 'DisplayName', 'Original Orientation');

    % Arrows for the first transformed trajectory
    x_trans = transformed_traj_1(1, indices);
    y_trans = transformed_traj_1(2, indices);
    orient_trans = demos{1}(5, indices); % Orientation is the 7th row
    u_trans = cos(orient_trans);
    v_trans = sin(orient_trans);
    quiver(x_trans, y_trans, u_trans, v_trans, arrow_scale, 'Color', 'r', 'LineWidth', 1, 'DisplayName', 'Transformed Orientation');

    % --- Add Reference Markers ---
    plot(0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', 'Common End-Point (New Origin)');
    frame_origins = squeeze(frames(1:2, 3, :));
    scatter(frame_origins(1, :), frame_origins(2, :), 30, 'rx', 'LineWidth', 1.5, 'DisplayName', 'Original End-Points');
    plot(0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'World Origin');

    hold off;
    title('All Trajectories with Detailed Orientation Comparison');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
    axis equal;
    legend('show', 'Location', 'best');
    
    fprintf('Plotting complete.\n');
end