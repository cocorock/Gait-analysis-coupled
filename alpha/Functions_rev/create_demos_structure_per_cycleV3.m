%% create_demos_structure_per_cycleV3: Creates a 'demos' structure from processed gait data.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function organizes processed gait cycle data into a 'demos' cell
%   array. It processes right and left leg cycles separately and then
%   concatenates them. Each cell in the 'demos' array contains a single
%   matrix with position and velocity data for a single gait cycle.
%
% Input:
%   processed_data - struct: A structure containing filtered gait data and derivatives.
%                     See 'apply_filtering_and_derivatives.m' for details.
%
% Output:
%   demos - (1 x M) cell array, where M is the sum of right and left leg cycles.
%           Each cell contains a (4 x N) matrix where N is the cycle length:
%             - Row 1: Hip position
%             - Row 2: Knee position
%             - Row 3: Hip velocity
%             - Row 4: Knee velocity
%           Acceleration data is discarded.

function Data = create_demos_structure_per_cycleV3(processed_data, N_Samples)
   % Count the leg with less gait cycles 
    num_right_cycles = size(processed_data.filtered.right_hip_cycles, 1);
    num_left_cycles = size(processed_data.filtered.left_hip_cycles, 1);

    total_cycles = num_left_cycles + num_right_cycles;
    Data0 = cell(1, total_cycles);

    idx = 1;
    for i = 1:num_right_cycles
        A = zeros(4 , size(processed_data.filtered.right_hip_cycles, 2));
        A(1, :) = processed_data.filtered.right_hip_cycles(i,:);
        A(2, :) = processed_data.filtered.right_knee_cycles(i,:);
        A(3, :) = processed_data.derivatives.right_hip_velocity(i,:);
        A(4, :) = processed_data.derivatives.right_knee_velocity(i,:);
        Data0{idx}=A;
        idx = idx +1;
    end

    for i = 1:num_left_cycles
            A = zeros(4 , size(processed_data.filtered.left_hip_cycles, 2));
            A(1, :) = processed_data.filtered.left_hip_cycles(i,:);
            A(2, :) = processed_data.filtered.left_knee_cycles(i,:);
            A(3, :) = processed_data.derivatives.left_hip_velocity(i,:);
            A(4, :) = processed_data.derivatives.left_knee_velocity(i,:);
            Data0{idx}=A;
            idx = idx +1;
    end

    Data = cell(1, N_Samples);
    for i=1:N_Samples
       Data{i} = Data0{i};
    end
   
end
