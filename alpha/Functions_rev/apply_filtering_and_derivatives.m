%% apply_filtering_and_derivatives: Applies a 6Hz Butterworth filter and calculates derivatives.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function takes a structure containing gait cycle data, applies a low-pass
%   4th-order Butterworth filter with a 6Hz cutoff frequency, and then
%   calculates the first and second derivatives (velocity and acceleration) of
%   the filtered data.
%
% Input:
%   all_cycles_data - struct: A structure containing the raw gait cycle data.
%                     It must contain the following fields:
%                       - right_hip_cycles: (N_rh x 200) matrix of right hip angles.
%                       - left_hip_cycles:  (N_lh x 200) matrix of left hip angles.
%                       - right_knee_cycles:(N_rk x 200) matrix of right knee angles.
%                       - left_knee_cycles: (N_lk x 200) matrix of left knee angles.
%                       - time_standard:    (1 x 200) vector of normalized time.
%
% Output:
%   processed_data - struct: A structure containing the original data plus the
%                    filtered data and its derivatives. It includes:
%                      - filtered: struct with filtered joint angles (same dimensions as input).
%                      - derivatives: struct with joint velocities and accelerations.
%                        - <joint>_velocity: (N x 200) matrix.
%                        - <joint>_acceleration: (N x 200) matrix.

function processed_data = apply_filtering_and_derivatives(all_cycles_data)
    fprintf('\n=== APPLYING FILTERING AND CALCULATING DERIVATIVES ===\n');
    
    % Filter parameters (time_multiplier = 1)
    time_multiplier = 1;
    cutoff_freq = 6 / time_multiplier;
    
    time_standard = all_cycles_data.time_standard;
    dt = mean(diff(time_standard));
    fs = 1/dt;
    nyquist = fs/2;
    
    cutoff_freq = min(cutoff_freq, nyquist * 0.9);
    
    [b, a] = butter(4, cutoff_freq/nyquist, 'low');
    
    fprintf('Filter parameters:\n  Sampling frequency: %.1f Hz\n  Cutoff frequency: %.2f Hz\n', fs, cutoff_freq);
    
    processed_data = all_cycles_data;
    processed_data.filtered = struct();
    processed_data.derivatives = struct();
    
    joints = {'right_hip', 'left_hip', 'right_knee', 'left_knee'};
    
    for j = 1:length(joints)
        joint = joints{j};
        field = [joint '_cycles'];
        if ~isfield(all_cycles_data, field) || isempty(all_cycles_data.(field))
            continue;
        end
        
        data = all_cycles_data.(field);
        n_cycles = size(data,1);
        
        filtered = zeros(size(data));
        velocity = zeros(size(data));
        acceleration = zeros(size(data));
        
        for i = 1:n_cycles
            cycle = data(i,:);
            pad_len = 60 * time_multiplier;
            padded = [cycle(end-pad_len+1:end), cycle, cycle(1:pad_len)];
            filtered_padded = filtfilt(b,a,padded);
            filtered_cycle = filtered_padded(pad_len+1:end-pad_len);
            filtered(i,:) = filtered_cycle;
            
            velocity(i,:) = calc_circular_derivative(filtered_cycle, dt);
            acceleration(i,:) = calc_circular_derivative(velocity(i,:), dt);
        end
        
        processed_data.filtered.(field) = filtered;
        processed_data.derivatives.([joint '_velocity']) = velocity;
        processed_data.derivatives.([joint '_acceleration']) = acceleration;
    end
    
    fprintf('Filtering and derivative calculation complete!\n');
end