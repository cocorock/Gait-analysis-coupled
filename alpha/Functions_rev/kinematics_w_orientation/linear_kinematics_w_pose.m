%% linear_kinematics_w_pose: Calculates and rotates linear kinematics for a 2-link planar arm.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function computes the forward kinematics (position, velocity, and
%   acceleration) of the end-effector of a 2-link planar arm. It then
%   rotates the resulting kinematic vectors by a specified angle phi.
%   The joint angles, velocities, and accelerations are provided for a
%   single leg (e.g., right or left).
%
% Input:
%   Data - (1 x N_cycles) cell array: Each cell contains a struct with fields:
%          - pos: (2 x M) matrix of joint angles [hip; knee] in degrees.
%          - vel: (2 x M) matrix of joint velocities in degrees/s.
%          - acc: (2 x M) matrix of joint accelerations in degrees/s^2.
%   phi  - scalar: The rotation angle in degrees.
%
% Output:
%   linear_kinematics - (1 x N_cycles) cell array: Each cell contains a struct with fields:
%                         - pos: (2 x M) matrix of rotated end-effector positions [x'; y'].
%                         - vel: (2 x M) matrix of rotated end-effector velocities [x_dot'; y_dot'].
%                         - acc: (2 x M) matrix of rotated end-effector accelerations [x_ddot'; y_ddot'].
%                         - orientation: (1 x M) vector of end-effector orientation in radians.
%                         - orientation_vel: (1 x M) vector of end-effector angular velocity in rad/s.
%                         - orientation_acc: (1 x M) vector of end-effector angular acceleration in rad/s^2.

function linear_kinematics = linear_kinematics_w_pose(Data, phi)
    
    % Constants
    l1 = 0.3874;  % length of first link (hip to knee)
    l2 = 0.4136;  % length of second link (knee to ankle)
    
    n_cycles = length(Data);
    linear_kinematics = cell(1, n_cycles);
    
    % Convert phi to radians for rotation matrix
    phi_rad = deg2rad(phi);
    R = [cos(phi_rad), -sin(phi_rad);
         sin(phi_rad),  cos(phi_rad)];

    for i = 1:n_cycles
        demo = Data{i};
        
        % Convert degrees to radians
        theta1 = deg2rad(-demo.pos(1,:)); % hip angle
        theta2 = deg2rad(demo.pos(2,:));     % knee angle
        
        % Joint velocities and accelerations (rad/s and rad/s^2)
        theta1_dot = deg2rad(-demo.vel(1,:));
        theta2_dot = deg2rad(demo.vel(2,:));
        theta1_ddot = deg2rad(-demo.acc(1,:));
        theta2_ddot = deg2rad(demo.acc(2,:));
        
        % 1. Forward Kinematics: End-effector position
        x = l1*cos(theta1) + l2*cos(theta1 + theta2);
        y = l1*sin(theta1) + l2*sin(theta1 + theta2);
        
        % 2. Jacobian Matrix (2x2 for each time step)
        % J = [dx/dtheta1, dx/dtheta2; dy/dtheta1, dy/dtheta2]
        J11 = -l1*sin(theta1) - l2*sin(theta1 + theta2);  % dx/dtheta1
        J12 = -l2*sin(theta1 + theta2);                   % dx/dtheta2
        J21 = l1*cos(theta1) + l2*cos(theta1 + theta2);   % dy/dtheta1
        J22 = l2*cos(theta1 + theta2);                    % dy/dtheta2
        
        % 3. Velocity Kinematics: v = J * theta_dot
        x_dot = J11.*theta1_dot + J12.*theta2_dot;
        y_dot = J21.*theta1_dot + J22.*theta2_dot;
        
        % 4. Acceleration Kinematics: a = J * theta_ddot + dJ/dt * theta_dot
        % Time derivative of Jacobian elements
        dJ11_dt = -l1*cos(theta1).*theta1_dot - l2*cos(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ12_dt = -l2*cos(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ21_dt = -l1*sin(theta1).*theta1_dot - l2*sin(theta1 + theta2).*(theta1_dot + theta2_dot);
        dJ22_dt = -l2*sin(theta1 + theta2).*(theta1_dot + theta2_dot);
        
        % Acceleration = J * theta_ddot + dJ/dt * theta_dot
        x_ddot = J11.*theta1_ddot + J12.*theta2_ddot + dJ11_dt.*theta1_dot + dJ12_dt.*theta2_dot;
        y_ddot = J21.*theta1_ddot + J22.*theta2_ddot + dJ21_dt.*theta1_dot + dJ22_dt.*theta2_dot;
        
        % Rotate the kinematics
        pos_rotated = R * [x; y];
        vel_rotated = R * [x_dot; y_dot];
        acc_rotated = R * [x_ddot; y_ddot];

        % --- Orientation ---
        orientation = theta1 + theta2 + phi;
        orientation_vel = theta1_dot + theta2_dot + phi;
        orientation_acc = theta1_ddot + theta2_ddot + phi;


        % Store resultsche
        lin_struct.pos = pos_rotated;
        lin_struct.vel = vel_rotated;
        lin_struct.acc = acc_rotated;
        lin_struct.orientation = orientation;
        lin_struct.orientation_vel = orientation_vel;
        lin_struct.orientation_acc = orientation_acc;
        
        linear_kinematics{i} = lin_struct;
    end
end