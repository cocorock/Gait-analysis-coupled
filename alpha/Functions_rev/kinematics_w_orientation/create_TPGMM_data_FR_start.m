function [demos, frames] = create_TPGMM_data_FR_start(linear_kinematics_pose)
    %% create_TPGMM_data_FR_start: Transforms kinematics data into a local coordinate frame
    %                               defined at the start of the trajectory.
    %
    % Description:
    %   This function transforms each trajectory into a local coordinate system
    %   defined at the initial point (frame_index = 1) of the trajectory.
    %   It uses homogeneous transformations to perform the translation and rotation.
    %
    % Input:
    %   linear_kinematics_pose - (1 x N_cycles) cell array of original kinematics data.
    %
    % Output:
    %   demos  - (1 x N_cycles) cell array of transformed trajectory data.
    %   frames - (3 x 3 x N_cycles) array containing the transformation for each demo.

    fprintf('Creating data structure for TP-GMM (Frame at Start of Trajectory)...\n');

    num_cycles = length(linear_kinematics_pose);
    demos = cell(1, num_cycles);
    frames = zeros(3, 3, num_cycles);

    % Define the frame_index to be the start of the trajectory
    frame_index = 1;

    for i = 1:num_cycles
        % Get the data for the current cycle
        cycle_data = linear_kinematics_pose{i};

        % --- Define the new coordinate system (Frame) based on frame_index ---
        origin_pos = cycle_data.pos(:, frame_index);
        final_orientation = cycle_data.orientation(frame_index);

        % Create the 2D rotation matrix for the new frame
        R = [cos(final_orientation), -sin(final_orientation);
             sin(final_orientation),  cos(final_orientation)];

        % Store the full homogeneous transformation matrix for this frame
        frames(:, :, i) = [R, origin_pos; 0 0 1];

        % --- Transform all data using the INVERSE of the homogeneous transformation matrix ---
        T_inv = [R', -R'*origin_pos; 0 0 1];

        % 1. Transform position data
        pos_homogeneous = [cycle_data.pos; ones(1, size(cycle_data.pos, 2))];
        transformed_pos_homogeneous = T_inv * pos_homogeneous;
        transformed_pos = transformed_pos_homogeneous(1:2, :);

        % 2. Transform velocity and acceleration vectors
        vel_homogeneous = [cycle_data.vel; zeros(1, size(cycle_data.vel, 2))];
        transformed_vel_homogeneous = T_inv * vel_homogeneous;
        transformed_vel = transformed_vel_homogeneous(1:2, :);

        acc_homogeneous = [cycle_data.acc; zeros(1, size(cycle_data.acc, 2))];
        transformed_acc_homogeneous = T_inv * acc_homogeneous;
        transformed_acc = transformed_acc_homogeneous(1:2, :);

        % 3. Make the orientation relative to the new frame's orientation
        transformed_orientation = mod(cycle_data.orientation - final_orientation + pi, 2*pi) - pi;

        % Combine all transformed data for the demonstrationlest
        demos{i} = [transformed_pos; 
                    transformed_vel; 
                    transformed_acc;
                    transformed_orientation;
                    cycle_data.orientation_vel; % Unchanged
                    cycle_data.orientation_acc]; % Unchanged
    end

    fprintf('TP-GMM data structure created successfully.\n');
end
