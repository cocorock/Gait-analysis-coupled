%% apply_filtering_and_derivatives_v2: Applies a 6Hz Butterworth filter and calculates derivatives for the v2 data structure.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function takes a structure containing gait cycle data from process_all_amc_files_v2, 
%   applies a low-pass 4th-order Butterworth filter with a 6Hz cutoff frequency, and then
%   calculates the first and second derivatives (velocity and acceleration) of the filtered data.
%   It is adapted to work with the struct array format where each cycle is an element.
%
% Input:
%   all_cycles_data - struct: A structure containing the raw gait cycle data from v2.
%                     It must contain the following fields:
%                       - right_leg_cycles: (1 x N_r) struct array of right leg cycles.
%                       - left_leg_cycles:  (1 x N_l) struct array of left leg cycles.
%                       - time_standard:    (1 x 200) vector of normalized time.
%                     Each cycle struct contains fields like 'right_hip_flex', 'left_knee_flex', etc.
%
% Output:
%   processed_data - struct: A structure containing the original data plus the
%                    filtered data and its derivatives, maintaining the input structure.

function processed_data = apply_filtering_and_derivatives_v2(all_cycles_data)
    fprintf('\n=== APPLYING FILTERING AND DERIVATIVES (V2) ===\n');
    
    % Filter parameters
    time_multiplier = 1;
    cutoff_freq = 6 / time_multiplier;
    
    time_standard = all_cycles_data.time_standard;
    dt = mean(diff(time_standard));
    fs = 1/dt;
    nyquist = fs/2;
    
    % Ensure cutoff frequency is valid
    cutoff_freq = min(cutoff_freq, nyquist * 0.99); % Ensure it's just under Nyquist
    
    [b, a] = butter(4, cutoff_freq/nyquist, 'low');
    
    fprintf('Filter parameters:\n  Sampling frequency: %.1f Hz\n  Cutoff frequency: %.2f Hz\n', fs, cutoff_freq);
    
    % Initialize the output structure by copying the input
    processed_data = all_cycles_data;
    
    % Define the legs and joints to be processed
    leg_fields = {'right_leg_cycles', 'left_leg_cycles'};
    joint_fields = {'right_hip_flex', 'left_hip_flex', 'right_knee_flex', 'left_knee_flex'};
    
    % Loop over each leg type (right and left)
    for l = 1:length(leg_fields)
        leg_field = leg_fields{l};
        
        if ~isfield(all_cycles_data, leg_field) || isempty(all_cycles_data.(leg_field))
            fprintf('  Skipping %s (no data).\n', leg_field);
            continue;
        end
        
        num_cycles = length(all_cycles_data.(leg_field));
        fprintf('  Processing %d cycles for %s...\n', num_cycles, leg_field);
        
        % Loop over each cycle for the current leg
        for i = 1:num_cycles
            % Loop over each joint within the cycle
            for j = 1:length(joint_fields)
                joint_field = joint_fields{j};
                
                % Original data for the joint in the current cycle
                original_cycle_data = processed_data.(leg_field)(i).(joint_field);
                
                % Apply padding and filter
                pad_len = 60 * time_multiplier;
                padded_data = [original_cycle_data(end-pad_len+1:end), original_cycle_data, original_cycle_data(1:pad_len)];
                filtered_padded = filtfilt(b, a, padded_data);
                filtered_cycle = filtered_padded(pad_len+1:end-pad_len);
                
                % Calculate derivatives
                velocity = calc_circular_derivative(filtered_cycle, dt);
                acceleration = calc_circular_derivative(velocity, dt);
                
                % Store filtered data and derivatives back into the structure
                % Filtered data
                processed_data.(leg_field)(i).([joint_field '_filtered']) = filtered_cycle;
                % Derivatives
                processed_data.(leg_field)(i).([joint_field '_velocity']) = velocity;
                processed_data.(leg_field)(i).([joint_field '_acceleration']) = acceleration;
            end
        end
    end
    
    fprintf('Filtering and derivative calculation complete!\n');
end
