%% Function: Create demos structure with one cell per gait cycle (filtered data)
function demos = create_demos_structure_per_cycleP(processed_data)
     % Count total gait cycles from all legs combined
    total_cycles = size(processed_data.right_hip_cycles,1) + size(processed_data.left_hip_cycles,1) + ...
                   size(processed_data.right_knee_cycles,1) + size(processed_data.left_knee_cycles,1);
    total_cycles = total_cycles/2;
    demos = cell(1, total_cycles);
    
    idx = 1;
    % Helper to add cycles to demos
    function add_cycles(pos_hip, vel_hip, acc_hip, pos_knee, vel_knee, acc_knee)
        n = size(pos_hip,1);
        for i = 1:n
            demo_struct.pos = zeros(2, 200);%200
            demo_struct.vel = zeros(2, 200);%200
            demo_struct.acc = zeros(2, 200);%200
            % Assign hip to row 1
            demo_struct.pos(1,:) = pos_hip(i,:);
            demo_struct.vel(1,:) = vel_hip(i,:);
            demo_struct.acc(1,:) = acc_hip(i,:);
            % Assign knee to row 2
            demo_struct.pos(2,:) = pos_knee(i,:);
            demo_struct.vel(2,:) = vel_knee(i,:);
            demo_struct.acc(2,:) = acc_knee(i,:);
            
            demos{idx} = demo_struct;
            idx = idx + 1;
        end
    end

    % Add right leg
    add_cycles(processed_data.filtered.right_hip_cycles, processed_data.derivatives.right_hip_velocity, processed_data.derivatives.right_hip_acceleration, ...
        processed_data.filtered.right_knee_cycles, processed_data.derivatives.right_knee_velocity, processed_data.derivatives.right_hip_acceleration);
    % Add left leg
    add_cycles(processed_data.filtered.left_hip_cycles, processed_data.derivatives.left_hip_velocity, processed_data.derivatives.left_hip_acceleration, ...
        processed_data.filtered.left_knee_cycles, processed_data.derivatives.left_knee_velocity, processed_data.derivatives.left_hip_acceleration);

end
