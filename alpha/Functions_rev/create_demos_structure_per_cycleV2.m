%% create_demos_structure_per_cycleV2: Creates a 'Data' structure from processed gait data.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function organizes processed gait cycle data into a 'Data' cell
%   array, where each cell represents a single gait cycle. Each cycle's struct
%   contains the position, velocity, and acceleration for the right and left hip
%   and knee.
%
% Input:
%   processed_data - struct: A structure containing filtered gait data and derivatives.
%                     See 'apply_filtering_and_derivatives.m' for details.
%                     It must contain fields like:
%                       - filtered.right_hip_cycles: (N x M) matrix.
%                       - derivatives.right_hip_velocity: (N x M) matrix.
%                       - ... and so on for other joints.
%
% Output:
%   Data - (1 x N_cycles) cell array: Each cell contains a struct with fields:
%             - pos: (4 x M) matrix [rh; rk; lh; lk] of joint angles.
%             - vel: (4 x M) matrix of joint velocities.
%             - acc: (4 x M) matrix of joint accelerations.

function demos = create_demos_structure_per_cycleV2(processed_data)
   % Count the leg with less gait cycles 
    total_cycles =  min(size(processed_data.right_hip_cycles,1), size(processed_data.left_hip_cycles,1));
    
    demos = cell(1, total_cycles);
    
    idx = 1;
    n1 = size(processed_data.right_hip_cycles,1);
    for i = 1:n1
        demo_struct.pos = [processed_data.filtered.right_hip_cycles(i,:); ...
            processed_data.filtered.right_knee_cycles(i,:); ...
            processed_data.filtered.left_hip_cycles(i,:); ...
            processed_data.filtered.left_knee_cycles(i,:)];

        demo_struct.vel = [processed_data.derivatives.right_hip_velocity(i,:); ...
            processed_data.derivatives.right_knee_velocity(i,:); ...
            processed_data.derivatives.left_hip_velocity(i,:); ...
            processed_data.derivatives.left_knee_velocity(i,:)];

        demo_struct.acc = [processed_data.derivatives.right_hip_acceleration(i,:); ...
            processed_data.derivatives.right_knee_acceleration(i,:); ...
            processed_data.derivatives.left_hip_acceleration(i,:); ...
            processed_data.derivatives.left_knee_acceleration(i,:)];

        demos{idx} = demo_struct;
        idx = idx + 1;
    end

end