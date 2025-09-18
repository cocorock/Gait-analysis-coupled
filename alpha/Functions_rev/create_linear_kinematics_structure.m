%% create_linear_kinematics_structure: Creates a '''demos''' like structure from linear kinematics data.
%
% Description:
%   This function organizes processed linear kinematics data into a '''demos''' cell
%   array. Each cell in the '''demos''' array contains a single
%   matrix with position and velocity data for a single gait cycle.
%
% Input:
%   linear_kinematics - (1 x N_cycles) cell array: Each cell contains a struct with fields:
%                         - pos: (2 x M) matrix of end-effector positions [x; y].
%                         - vel: (2 x M) matrix of end-effector velocities [x_dot; y_dot].
%                         - acc: (2 x M) matrix of end-effector accelerations [x_ddot; y_ddot].
%
% Output:
%   demos_linear - (1 x M) cell array, where M is the number of cycles.
%           Each cell contains a (4 x N) matrix where N is the cycle length:
%             - Row 1: x position
%             - Row 2: y position
%             - Row 3: x velocity
%             - Row 4: y velocity
%           Acceleration data is discarded.

function Data = create_linear_kinematics_structure(linear_kinematics)
    n_cycles = length(linear_kinematics);
    Data = cell(1, n_cycles);

    for i = 1:n_cycles
        lin_kin = linear_kinematics{i};
        
        A = zeros(4, size(lin_kin.pos, 2));
        A(1, :) = lin_kin.pos(1, :); % x position
        A(2, :) = lin_kin.pos(2, :); % y position
        A(3, :) = lin_kin.vel(1, :); % x velocity
        A(4, :) = lin_kin.vel(2, :); % y velocity
        
        Data{i} = A;
    end
end
