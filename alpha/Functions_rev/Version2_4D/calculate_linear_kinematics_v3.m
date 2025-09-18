%% calculate_linear_kinematics_v3: Calculates angular derivatives and then linear kinematics using subject-specific bone lengths.
% 
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
% 
% Description:
%   This function reads subject-specific bone lengths from an ASF file. It then 
%   calculates the angular velocity and acceleration for the hip and knee joints 
%   from filtered position data. Finally, it computes the forward kinematics 
%   (position, velocity, acceleration) of the ankle and rotates the results.
% 
% Input:
%   processed_data - struct: A structure containing the filtered gait cycle data from V3.
%   phi            - scalar: The rotation angle in degrees.
%   subject_id     - string: The ID of the subject (e.g., '35') to load the correct ASF file.
% 
% Output:
%   linear_kinematics - struct: A structure containing the linear kinematics for both legs.

function linear_kinematics = calculate_linear_kinematics_v3(processed_data, phi, bone_lengths)

    % Time step for derivative calculation
    dt = mean(diff(processed_data.time_standard));

    % Convert phi to radians for rotation matrix
    phi_rad = deg2rad(phi);
    R = [cos(phi_rad), -sin(phi_rad);
         sin(phi_rad),  cos(phi_rad)];

    % Initialize output structure
    linear_kinematics = struct();
    linear_kinematics.right_leg_kinematics = [];
    linear_kinematics.left_leg_kinematics = [];

    % Helper function for forward kinematics calculation
    function [pos_rotated, vel_rotated, acc_rotated] = calculate_ankle_kinematics(hip_flex_filtered, knee_flex_filtered, hip_flex_velocity, knee_flex_velocity, hip_flex_acceleration, knee_flex_acceleration, R, l1, l2)
        theta1 = deg2rad(hip_flex_filtered);
        theta2 = deg2rad(knee_flex_filtered);
        theta1_dot = deg2rad(hip_flex_velocity);
        theta2_dot = deg2rad(knee_flex_velocity);
        theta1_ddot = deg2rad(hip_flex_acceleration);
        theta2_ddot = deg2rad(knee_flex_acceleration);

        x = l1*cos(theta1) + l2*cos(theta1 + theta2);
        y = l1*sin(theta1) + l2*sin(theta1 + theta2);

        J11 = -l1*sin(theta1) - l2*sin(theta1 + theta2);
        J12 = -l2*sin(theta1 + theta2);
        J21 = l1*cos(theta1) + l2*cos(theta1 + theta2);
        J22 = l2*cos(theta1 + theta2);

        x_dot = J11.*theta1_dot + J12.*theta2_dot;
        y_dot = J21.*theta1_dot + J22.*theta2_dot;

        dJ11_dt = -l1*cos(theta1).*theta1_dot - l2*cos(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ12_dt = -l2*cos(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ21_dt = -l1*sin(theta1).*theta1_dot - l2*sin(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ22_dt = -l2*sin(theta1 + theta2).*(theta1_dot + theta2_dot);

        x_ddot = J11.*theta1_ddot + J12.*theta2_ddot + dJ11_dt.*theta1_dot + dJ12_dt.*theta2_dot;
        y_ddot = J21.*theta1_ddot + J22.*theta2_ddot + dJ21_dt.*theta1_dot + dJ22_dt.*theta2_dot;

        pos_rotated = R * [x; y];
        vel_rotated = R * [x_dot; y_dot];
        acc_rotated = R * [x_ddot; y_ddot];
    end

    % Process Right Leg Cycles
    if isfield(processed_data, 'right_leg_cycles') && ~isempty(processed_data.right_leg_cycles)
        num_right_cycles = length(processed_data.right_leg_cycles);
        right_leg_kinematics_temp = cell(1, num_right_cycles);

        for i = 1:num_right_cycles
            cycle = processed_data.right_leg_cycles(i);

            right_hip_flex_velocity = calc_circular_derivative(cycle.right_hip_flex_filtered, dt);
            right_hip_flex_acceleration = calc_circular_derivative(right_hip_flex_velocity, dt);
            right_knee_flex_velocity = calc_circular_derivative(cycle.right_knee_flex_filtered, dt);
            right_knee_flex_acceleration = calc_circular_derivative(right_knee_flex_velocity, dt);

            left_hip_flex_velocity = calc_circular_derivative(cycle.left_hip_flex_filtered, dt);
            left_hip_flex_acceleration = calc_circular_derivative(left_hip_flex_velocity, dt);
            left_knee_flex_velocity = calc_circular_derivative(cycle.left_knee_flex_filtered, dt);
            left_knee_flex_acceleration = calc_circular_derivative(left_knee_flex_velocity, dt);

            [right_ankle_pos, right_ankle_vel, right_ankle_acc] = calculate_ankle_kinematics(...
                cycle.right_hip_flex_filtered, cycle.right_knee_flex_filtered, ...
                right_hip_flex_velocity, right_knee_flex_velocity, ...
                right_hip_flex_acceleration, right_knee_flex_acceleration, ...
                R, bone_lengths.rfemur, bone_lengths.rtibia);

            [left_ankle_pos, left_ankle_vel, left_ankle_acc] = calculate_ankle_kinematics(...
                cycle.left_hip_flex_filtered, cycle.left_knee_flex_filtered, ...
                left_hip_flex_velocity, left_knee_flex_velocity, ...
                left_hip_flex_acceleration, left_knee_flex_acceleration, ...
                R, bone_lengths.lfemur, bone_lengths.ltibia);

            lin_struct.right_ankle_pos = right_ankle_pos;
            lin_struct.right_ankle_vel = right_ankle_vel;
            lin_struct.right_ankle_acc = right_ankle_acc;
            lin_struct.left_ankle_pos = left_ankle_pos;
            lin_struct.left_ankle_vel = left_ankle_vel;
            lin_struct.left_ankle_acc = left_ankle_acc;

            right_leg_kinematics_temp{i} = lin_struct;
        end
        linear_kinematics.right_leg_kinematics = [right_leg_kinematics_temp{:}];
    end

    % Process Left Leg Cycles
    if isfield(processed_data, 'left_leg_cycles') && ~isempty(processed_data.left_leg_cycles)
        num_left_cycles = length(processed_data.left_leg_cycles);
        left_leg_kinematics_temp = cell(1, num_left_cycles);

        for i = 1:num_left_cycles
            cycle = processed_data.left_leg_cycles(i);

            left_hip_flex_velocity = calc_circular_derivative(cycle.left_hip_flex_filtered, dt);
            left_hip_flex_acceleration = calc_circular_derivative(left_hip_flex_velocity, dt);
            left_knee_flex_velocity = calc_circular_derivative(cycle.left_knee_flex_filtered, dt);
            left_knee_flex_acceleration = calc_circular_derivative(left_knee_flex_velocity, dt);

            right_hip_flex_velocity = calc_circular_derivative(cycle.right_hip_flex_filtered, dt);
            right_hip_flex_acceleration = calc_circular_derivative(right_hip_flex_velocity, dt);
            right_knee_flex_velocity = calc_circular_derivative(cycle.right_knee_flex_filtered, dt);
            right_knee_flex_acceleration = calc_circular_derivative(right_knee_flex_velocity, dt);

            [left_ankle_pos, left_ankle_vel, left_ankle_acc] = calculate_ankle_kinematics(...
                cycle.left_hip_flex_filtered, cycle.left_knee_flex_filtered, ...
                left_hip_flex_velocity, left_knee_flex_velocity, ...
                left_hip_flex_acceleration, left_knee_flex_acceleration, ...
                R, bone_lengths.lfemur, bone_lengths.ltibia);

            [right_ankle_pos, right_ankle_vel, right_ankle_acc] = calculate_ankle_kinematics(...
                cycle.right_hip_flex_filtered, cycle.right_knee_flex_filtered, ...
                right_hip_flex_velocity, right_knee_flex_velocity, ...
                right_hip_flex_acceleration, right_knee_flex_acceleration, ...
                R, bone_lengths.rfemur, bone_lengths.rtibia);

            lin_struct.right_ankle_pos = right_ankle_pos;
            lin_struct.right_ankle_vel = right_ankle_vel;
            lin_struct.right_ankle_acc = right_ankle_acc;
            lin_struct.left_ankle_pos = left_ankle_pos;
            lin_struct.left_ankle_vel = left_ankle_vel;
            lin_struct.left_ankle_acc = left_ankle_acc;

            left_leg_kinematics_temp{i} = lin_struct;
        end
        linear_kinematics.left_leg_kinematics = [left_leg_kinematics_temp{:}];
    end
end
