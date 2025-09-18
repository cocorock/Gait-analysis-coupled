%% reshapeProcessedData: Reshapes processed gait data for further analysis.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function reshapes the processed gait data from a matrix format (N_cycles x 200)
%   to a single row vector for each joint. It also trims the left leg data and
%   creates corresponding time vectors.
%
% Input:
%   processed_data - struct: The processed gait data structure.
%
% Output:
%   reshaped_data - struct: The reshaped data structure.

function reshaped_data = reshapeProcessedData(processed_data)
%     % Create a copy of the input structure
%     reshaped_data = processed_data;
%     
%     % Reshape main joint cycle data
%     reshaped_data.right_hip_cycles = reshape(processed_data.right_hip_cycles', 1, []);
%     reshaped_data.left_hip_cycles = reshape(processed_data.left_hip_cycles', 1, []);
%     reshaped_data.right_knee_cycles = reshape(processed_data.right_knee_cycles', 1, []);
%     reshaped_data.left_knee_cycles = reshape(processed_data.left_knee_cycles', 1, []);
%     
%     % Reshape filtered data
%     reshaped_data.filtered.right_hip_cycles = reshape(processed_data.filtered.right_hip_cycles', 1, []);
%     reshaped_data.filtered.left_hip_cycles = reshape(processed_data.filtered.left_hip_cycles', 1, []);
%     reshaped_data.filtered.right_knee_cycles = reshape(processed_data.filtered.right_knee_cycles', 1, []);
%     reshaped_data.filtered.left_knee_cycles = reshape(processed_data.filtered.left_knee_cycles', 1, []);
%     
%     % Reshape derivatives data
%     reshaped_data.derivatives.right_hip_velocity = reshape(processed_data.derivatives.right_hip_velocity', 1, []);
%     reshaped_data.derivatives.right_hip_acceleration = reshape(processed_data.derivatives.right_hip_acceleration', 1, []);
%     reshaped_data.derivatives.left_hip_velocity = reshape(processed_data.derivatives.left_hip_velocity', 1, []);
%     reshaped_data.derivatives.left_hip_acceleration = reshape(processed_data.derivatives.left_hip_acceleration', 1, []);
%     reshaped_data.derivatives.right_knee_velocity = reshape(processed_data.derivatives.right_knee_velocity', 1, []);
%     reshaped_data.derivatives.right_knee_acceleration = reshape(processed_data.derivatives.right_knee_acceleration', 1, []);
%     reshaped_data.derivatives.left_knee_velocity = reshape(processed_data.derivatives.left_knee_velocity', 1, []);
%     reshaped_data.derivatives.left_knee_acceleration = reshape(processed_data.derivatives.left_knee_acceleration', 1, []);

    % Create a copy of the input structure
    reshaped_data = processed_data;
    
    % Reshape main joint cycle data
    reshaped_data.right_hip_cycles = reshape(processed_data.right_hip_cycles', 1, []);
    reshaped_data.left_hip_cycles = reshape(processed_data.left_hip_cycles', 1, []);
    reshaped_data.right_knee_cycles = reshape(processed_data.right_knee_cycles', 1, []);
    reshaped_data.left_knee_cycles = reshape(processed_data.left_knee_cycles', 1, []);
    
    % Reshape filtered data
    reshaped_data.filtered.right_hip_cycles = reshape(processed_data.filtered.right_hip_cycles', 1, []);
    reshaped_data.filtered.left_hip_cycles = reshape(processed_data.filtered.left_hip_cycles', 1, []);
    reshaped_data.filtered.right_knee_cycles = reshape(processed_data.filtered.right_knee_cycles', 1, []);
    reshaped_data.filtered.left_knee_cycles = reshape(processed_data.filtered.left_knee_cycles', 1, []);
    
    % Reshape derivatives data
    reshaped_data.derivatives.right_hip_velocity = reshape(processed_data.derivatives.right_hip_velocity', 1, []);
    reshaped_data.derivatives.right_hip_acceleration = reshape(processed_data.derivatives.right_hip_acceleration', 1, []);
    reshaped_data.derivatives.left_hip_velocity = reshape(processed_data.derivatives.left_hip_velocity', 1, []);
    reshaped_data.derivatives.left_hip_acceleration = reshape(processed_data.derivatives.left_hip_acceleration', 1, []);
    reshaped_data.derivatives.right_knee_velocity = reshape(processed_data.derivatives.right_knee_velocity', 1, []);
    reshaped_data.derivatives.right_knee_acceleration = reshape(processed_data.derivatives.right_knee_acceleration', 1, []);
    reshaped_data.derivatives.left_knee_velocity = reshape(processed_data.derivatives.left_knee_velocity', 1, []);
    reshaped_data.derivatives.left_knee_acceleration = reshape(processed_data.derivatives.left_knee_acceleration', 1, []);


    % Trim the first 100 elements of left leg data
    reshaped_data.left_hip_cycles = reshaped_data.left_hip_cycles(101:end);
    reshaped_data.left_knee_cycles = reshaped_data.left_knee_cycles(101:end);
    reshaped_data.filtered.left_hip_cycles = reshaped_data.filtered.left_hip_cycles(101:end);
    reshaped_data.filtered.left_knee_cycles = reshaped_data.filtered.left_knee_cycles(101:end);
    reshaped_data.derivatives.left_hip_velocity = reshaped_data.derivatives.left_hip_velocity(101:end);
    reshaped_data.derivatives.left_hip_acceleration = reshaped_data.derivatives.left_hip_acceleration(101:end);
    reshaped_data.derivatives.left_knee_velocity = reshaped_data.derivatives.left_knee_velocity(101:end);
    reshaped_data.derivatives.left_knee_acceleration = reshaped_data.derivatives.left_knee_acceleration(101:end);
    
    %time 
    size_rightLeg = size(processed_data.right_hip_cycles);
    total_time_rightLeg = kron( ones(1,size_rightLeg(1)), processed_data.time_standard);
    reshaped_data.time_standard_righ_Leg =  total_time_rightLeg;
    
    size_leftLeg = size(processed_data.left_hip_cycles);
    total_time_rightLeg = kron( ones(1,size_leftLeg(1)), processed_data.time_standard);
    reshaped_data.time_standard_left_Leg =  total_time_rightLeg(101:end);
    

end