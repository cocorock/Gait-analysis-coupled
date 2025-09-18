function reshaped_data = reshape_leg_data(Data_filteredv2)
% RESHAPE_LEG_DATA - Reshapes data structure to separate left and right leg data.
%
% This function takes a cell array of structures, where each structure
% contains data for both right and left legs (4 rows for pos, vel, acc),
% and reorganizes it into a new cell array where right and left leg data
% are in separate cells, effectively doubling the number of cells.
%
% Input:
%   Data_filteredv2 - A cell array where each cell contains a struct with:
%                     - pos: [4xM] matrix (rows 1-2 for right, 3-4 for left)
%                     - vel: [4xM] matrix (rows 1-2 for right, 3-4 for left)
%                     - acc: [4xM] matrix (rows 1-2 for right, 3-4 for left)
%
% Output:
%   reshaped_data - A new cell array of size (1 x 2*N_cycles) where the
%                   first N_cycles contain the right leg data and the next
%                   N_cycles contain the left leg data.

    n_cycles = length(Data_filteredv2);
    reshaped_data = cell(1, 2 * n_cycles);

    for i = 1:n_cycles
        % Right leg data
        right_leg_struct = struct();
        right_leg_struct.pos = Data_filteredv2{i}.pos(1:2, :);
        right_leg_struct.vel = Data_filteredv2{i}.vel(1:2, :);
        right_leg_struct.acc = Data_filteredv2{i}.acc(1:2, :);
        reshaped_data{i} = right_leg_struct;

        % Left leg data
        left_leg_struct = struct();
        left_leg_struct.pos = Data_filteredv2{i}.pos(3:4, :);
        left_leg_struct.vel = Data_filteredv2{i}.vel(3:4, :);
        left_leg_struct.acc = Data_filteredv2{i}.acc(3:4, :);
        reshaped_data{i + n_cycles} = left_leg_struct;
    end
end