function [demos_relative_start, frames_relative_start] = create_TPGMM_data_FR_relative_start(linear_kinematics_pose)
    %% create_TPGMM_data_FR_relative_start: Transforms the start point of a trajectory
    %                                       into a frame that moves along the trajectory.
    %
    % Description:
    %   For each original trajectory, this function generates a new transformed
    %   trajectory. Each point in the new trajectory represents the position
    %   of the *start point* of the original trajectory, as seen from a frame
    %   that is currently located at the end-effector's position and orientation.
    %
    % Input:
    %   linear_kinematics_pose - (1 x N_original_cycles) cell array of original kinematics data.
    %
    % Output:
    %   demos_relative_start  - (1 x N_original_cycles) cell array. Each cell contains
    %                           a transformed trajectory (e.g., 6x200 for pos, vel, ori).
    %   frames_relative_start - (3 x 3 x N_original_cycles x N_data_points) array containing
    %                           the sequence of moving frames for each original trajectory.

    fprintf('Creating data structure for TP-GMM (Frame relative to current point)...\n');

    num_original_cycles = length(linear_kinematics_pose);
    demos_relative_start = cell(1, num_original_cycles);
    frames_relative_start = zeros(3, 3, num_original_cycles, size(linear_kinematics_pose{1}.pos, 2));

    for i = 1:num_original_cycles
        original_cycle_data = linear_kinematics_pose{i};
        num_data_points = size(original_cycle_data.pos, 2);

        % Get the fixed start point of the original trajectory
        start_point_world = original_cycle_data.pos(:, 1);
        start_point_homogeneous = [start_point_world; 1];

        % Initialize transformed data for this trajectory
        transformed_pos_traj = zeros(2, num_data_points);
        transformed_vel_traj = zeros(2, num_data_points);
        transformed_acc_traj = zeros(2, num_data_points);
        transformed_orientation_traj = zeros(1, num_data_points);

        for j = 1:num_data_points
            % --- Define the moving coordinate system (Frame) at the current point j ---
            origin_pos_j = original_cycle_data.pos(:, j);
            orientation_j = original_cycle_data.orientation(j);

            % Create the 2D rotation matrix for the moving frame
            R_j = [cos(orientation_j), -sin(orientation_j);
                   sin(orientation_j),  cos(orientation_j)];

            % Store the full homogeneous transformation matrix for this moving frame
            frames_relative_start(:, :, i, j) = [R_j, origin_pos_j; 0 0 1];

            % --- Transform the start point into the current moving frame ---
            T_j_inv = [R_j', -R_j'*origin_pos_j; 0 0 1];
            
            transformed_start_point_homogeneous = T_j_inv * start_point_homogeneous;
            transformed_pos_traj(:, j) = transformed_start_point_homogeneous(1:2);

            % --- Transform current velocity/acceleration/orientation into the current moving frame ---
            % Note: For velocity and acceleration, only rotation applies.
            % For orientation, it's relative to the current frame's orientation.
            transformed_vel_traj(:, j) = R_j' * original_cycle_data.vel(:, j);
            transformed_acc_traj(:, j) = R_j' * original_cycle_data.acc(:, j);
            transformed_orientation_traj(:, j) = mod(original_cycle_data.orientation(j) - orientation_j + pi, 2*pi) - pi;
        end

        % Combine all transformed data for this demonstration
        demos_relative_start{i} = [transformed_pos_traj; 
                                   transformed_vel_traj; 
                                   transformed_acc_traj;
                                   transformed_orientation_traj;
                                   original_cycle_data.orientation_vel; % Angular velocity is invariant to coordinate system rotation
                                   original_cycle_data.orientation_acc]; % Angular acceleration is invariant to coordinate system rotation
    end

    fprintf('TP-GMM data structure created successfully for %d relative trajectories.\n', num_original_cycles);
end
