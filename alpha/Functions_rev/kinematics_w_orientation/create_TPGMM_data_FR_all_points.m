function [demos_all, frames_all] = create_TPGMM_data_FR_all_points(linear_kinematics_pose)
    %% create_TPGMM_data_FR_all_points: Transforms kinematics data into local coordinate frames
    %                                  defined at every point of the trajectory.
    %
    % Description:
    %   For each original trajectory, this function generates multiple transformed
    %   trajectories. Each transformed trajectory uses a different point along the
    %   original trajectory as its local coordinate system (origin and orientation).
    %
    % Input:
    %   linear_kinematics_pose - (1 x N_original_cycles) cell array of original kinematics data.
    %
    % Output:
    %   demos_all  - (1 x N_total_transformed_demos) cell array of transformed trajectory data.
    %                N_total_transformed_demos = N_original_cycles * N_data_points_per_cycle.
    %   frames_all - (3 x 3 x N_total_transformed_demos) array containing the transformation for each demo.

    fprintf('Creating data structure for TP-GMM (Frame at ALL points of Trajectory)...\n');

    num_original_cycles = length(linear_kinematics_pose);
    
    % Pre-allocate for efficiency (approximate size)
    % Assuming each cycle has 200 data points
    estimated_total_demos = num_original_cycles * size(linear_kinematics_pose{1}.pos, 2);
    demos_all = cell(1, estimated_total_demos);
    frames_all = zeros(3, 3, estimated_total_demos);
    
    current_demo_idx = 0;

    for i = 1:num_original_cycles
        % Get the data for the current original cycle
        original_cycle_data = linear_kinematics_pose{i};
        num_data_points = size(original_cycle_data.pos, 2);

        % Iterate through every point of the current original trajectory
        for frame_index = 1:num_data_points
            current_demo_idx = current_demo_idx + 1;

            % --- Define the new coordinate system (Frame) at the current frame_index ---
            origin_pos = original_cycle_data.pos(:, frame_index);
            final_orientation = original_cycle_data.orientation(frame_index);

            % Create the 2D rotation matrix for the new frame
            R = [cos(final_orientation), -sin(final_orientation);
                 sin(final_orientation),  cos(final_orientation)];

            % Store the full homogeneous transformation matrix for this frame
            frames_all(:, :, current_demo_idx) = [R, origin_pos; 0 0 1];

            % --- Transform all data using the INVERSE of the homogeneous transformation matrix ---
            T_inv = [R', -R'*origin_pos; 0 0 1];

            % 1. Transform position data
            pos_homogeneous = [original_cycle_data.pos; ones(1, num_data_points)];
            transformed_pos_homogeneous = T_inv * pos_homogeneous;
            transformed_pos = transformed_pos_homogeneous(1:2, :);

            % 2. Transform velocity and acceleration vectors
            vel_homogeneous = [original_cycle_data.vel; zeros(1, num_data_points)];
            transformed_vel_homogeneous = T_inv * vel_homogeneous;
            transformed_vel = transformed_vel_homogeneous(1:2, :);

            acc_homogeneous = [original_cycle_data.acc; zeros(1, num_data_points)];
            transformed_acc_homogeneous = T_inv * acc_homogeneous;
            transformed_acc = transformed_acc_homogeneous(1:2, :);

            % 3. Make the orientation relative to the new frame's orientation
            transformed_orientation = mod(original_cycle_data.orientation - final_orientation + pi, 2*pi) - pi;

            % Combine all transformed data for the demonstration
            demos_all{current_demo_idx} = [transformed_pos; 
                                           transformed_vel; 
                                           transformed_acc;
                                           transformed_orientation;
                                           original_cycle_data.orientation_vel; % Unchanged
                                           original_cycle_data.orientation_acc]; % Unchanged
        end
    end

    fprintf('TP-GMM data structure created successfully for %d total transformed demos.\n', current_demo_idx);

    % Trim pre-allocated arrays if necessary
    demos_all = demos_all(1:current_demo_idx);
    frames_all = frames_all(:, :, 1:current_demo_idx);
end
