function output_struct_array = create_struct_from_demos(demos)
    % Creates a cell array of structs from the output of create_demos_structure_per_cycleV3
    %
    % Input:
    %   demos - (1 x M) cell array, where M is the sum of right and left leg cycles.
    %           Each cell contains a (4 x N) matrix where N is the cycle length:
    %             - Row 1: Hip position
    %             - Row 2: Knee position
    %             - Row 3: Hip velocity
    %             - Row 4: Knee velocity
    %
    % Output:
    %   output_struct_array - (1 x M) cell array of structs with the following fields:
    %                         - hip_pos
    %                         - knee_pos
    %                         - hip_vel
    %                         - knee_vel

    num_cycles = length(demos);
    output_struct_array = cell(1, num_cycles);

    for i = 1:num_cycles
        cycle_data = demos{i};
        
        temp_struct = struct();
        temp_struct.hip_pos = cycle_data(1, :);
        temp_struct.knee_pos = cycle_data(2, :);
        temp_struct.hip_vel = cycle_data(3, :);
        temp_struct.knee_vel = cycle_data(4, :);
        
        output_struct_array{i} = temp_struct;
    end
end