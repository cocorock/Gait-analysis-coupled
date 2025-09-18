function plot_kinematics_with_orientation(linear_kinematics_pose)
    %% plot_kinematics_with_orientation: Plots kinematics with orientation arrows.
    %
    % Description:
    %   This function plots the position, velocity, and acceleration
    %   of the first gait cycle in separate figures. It includes arrows
    %   to indicate the orientation of the variable at different points
    %   along the trajectory.
    %
    % Input:
    %   linear_kinematics_pose - (1 x N_cycles) cell array, where each cell
    %                            contains a struct with kinematics data including
    %                            position, velocity, acceleration, and orientation.

    fprintf('\n=== PLOTTING KINEMATICS WITH ORIENTATION ===\n');

    if isempty(linear_kinematics_pose)
        fprintf('  Input data is empty. Nothing to plot.\n');
        return;
    end

    % Extract data for the first gait cycle
    gait_cycle = linear_kinematics_pose{1};
    
    % Determine indices for plotting arrows (approx 10%)
    num_points = size(gait_cycle.pos, 2);
    num_arrows = round(num_points * 0.1);
    indices = round(linspace(1, num_points, num_arrows));

    % --- Figure 1: Position Trajectory with End-Effector Orientation ---
    figure('Name', 'Position Trajectory with Orientation');
    plot(gait_cycle.pos(1, :), gait_cycle.pos(2, :), 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Extract points and orientation for arrows
    x = gait_cycle.pos(1, indices);
    y = gait_cycle.pos(2, indices);
    orient = gait_cycle.orientation(indices);
    
    % Calculate arrow components from orientation angle
    u = cos(orient);
    v = sin(orient);
    
    % Scale arrows for better visualization
    arrow_length = 0.5 * range(get(gca, 'XLim'));
    quiver(x, y, u, v, arrow_length, 'r', 'LineWidth', 1, 'MaxHeadSize', 0.5);
    
    hold off;
    title('Position Trajectory with End-Effector Orientation (First Gait Cycle)');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
    axis equal;
    fprintf('  Generated Position plot.\n');

    % --- Figure 2: Velocity Hodograph with Velocity Vectors ---
    figure('Name', 'Velocity Hodograph');
    plot(gait_cycle.vel(1, :), gait_cycle.vel(2, :), 'g-', 'LineWidth', 1.5);
    hold on;

    % Extract velocity vectors for arrows
    vx = gait_cycle.vel(1, indices);
    vy = gait_cycle.vel(2, indices);

    % Plot velocity vectors from the origin to the point on the hodograph
    quiver(zeros(size(vx)), zeros(size(vy)), vx, vy, 1, 'r', 'LineWidth', 1, 'AutoScale', 'off');

    hold off;
    title('Velocity Hodograph (First Gait Cycle)');
    xlabel('X Velocity (m/s)');
    ylabel('Y Velocity (m/s)');
    grid on;
    axis equal;
    fprintf('  Generated Velocity plot.\n');

    % --- Figure 3: Acceleration Hodograph with Acceleration Vectors ---
    figure('Name', 'Acceleration Hodograph');
    plot(gait_cycle.acc(1, :), gait_cycle.acc(2, :), 'm-', 'LineWidth', 1.5);
    hold on;

    % Extract acceleration vectors for arrows
    ax = gait_cycle.acc(1, indices);
    ay = gait_cycle.acc(2, indices);

    % Plot acceleration vectors from the origin to the point on the hodograph
    quiver(zeros(size(ax)), zeros(size(ay)), ax, ay, 1, 'r', 'LineWidth', 1, 'AutoScale', 'off');

    hold off;
    title('Acceleration Hodograph (First Gait Cycle)');
    xlabel('X Acceleration (m/s^2)');
    ylabel('Y Acceleration (m/s^2)');
    grid on;
    axis equal;
    fprintf('  Generated Acceleration plot.\n');
    
    fprintf('Plotting complete.\n');
end
