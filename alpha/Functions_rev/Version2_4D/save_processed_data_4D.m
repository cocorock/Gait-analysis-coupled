%% save_processed_data_4D: Saves processed gait data into 4D arrays for each leg.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function takes the processed gait data (from apply_filtering_and_derivatives_v2)
%   and saves the filtered joint positions and velocities into two separate .mat files,
%   one for right leg cycles and one for left leg cycles. Each file contains a 2D array
%   where rows represent joint data (4 positions, 4 velocities) and columns represent
%   concatenated time series across cycles, limited to a maximum of 200 cycles.
%
% Input:
%   processed_data - struct: The processed gait data structure from apply_filtering_and_derivatives_v2.
%                     It must contain 'right_leg_cycles' and 'left_leg_cycles' fields.
%   output_dir     - string: The directory where the .mat files will be saved.
%
% Output:
%   Two .mat files: 'processed_data_right_leg_4D.mat' and 'processed_data_left_leg_4D.mat'
%   saved in the specified output_dir.

function save_processed_data_4D(processed_data, output_dir)
    fprintf('\n=== SAVING PROCESSED DATA (4D) ===\n');
    
    if nargin < 2
        output_dir = 'Gait Data'; % Default output directory
    end
    
    % Ensure output directory exists
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    interp_length = 200; % Number of samples per cycle
    max_cycles_to_save = 200; % Limit for the number of cycles to save
    
    joint_fields_filtered = {'right_hip_flex_filtered', 'left_hip_flex_filtered', 'right_knee_flex_filtered', 'left_knee_flex_filtered'};
    joint_fields_velocity = {'right_hip_flex_velocity', 'left_hip_flex_velocity', 'right_knee_flex_velocity', 'left_knee_flex_velocity'};
    
    % --- Process Right Leg Cycles ---
    if isfield(processed_data, 'right_leg_cycles') && ~isempty(processed_data.right_leg_cycles)
        num_right_cycles = length(processed_data.right_leg_cycles);
        cycles_to_save_right = min(num_right_cycles, max_cycles_to_save);
        fprintf('  Processing %d of %d right leg cycles for saving...\n', cycles_to_save_right, num_right_cycles);
        
        % Initialize the 8xN_samples*N_cycles matrix
        right_leg_data_4D = zeros(9, cycles_to_save_right * interp_length);
        
        for i = 1:cycles_to_save_right
            current_cycle = processed_data.right_leg_cycles(i);
            start_col = (i - 1) * interp_length + 1;
            end_col = i * interp_length;
            
            % Add filtered positions
            right_leg_data_4D(1, start_col:end_col) = linspace(0, 1, interp_length);
            for j = 1:length(joint_fields_filtered)
                right_leg_data_4D(j + 1, start_col:end_col) = current_cycle.(joint_fields_filtered{j});
            end
            
            % Add velocities
            for j = 1:length(joint_fields_velocity)
                right_leg_data_4D(j + 4 + 1, start_col:end_col) = current_cycle.(joint_fields_velocity{j});
            end
        end
        
        save(fullfile(output_dir, 'processed_data_right_leg_4D.mat'), 'right_leg_data_4D');
        fprintf('  Saved right leg data to %s\n', fullfile(output_dir, 'processed_data_right_leg_4D.mat'));
    else
        fprintf('  No right leg cycles to save.\n');
    end
    
    % --- Process Left Leg Cycles ---
    if isfield(processed_data, 'left_leg_cycles') && ~isempty(processed_data.left_leg_cycles)
        num_left_cycles = length(processed_data.left_leg_cycles);
        cycles_to_save_left = min(num_left_cycles, max_cycles_to_save);
        fprintf('  Processing %d of %d left leg cycles for saving...\n', cycles_to_save_left, num_left_cycles);
        
        % Initialize the 8xN_samples*N_cycles matrix
        left_leg_data_4D = zeros(9, cycles_to_save_left * interp_length);
        
        for i = 1:cycles_to_save_left
            current_cycle = processed_data.left_leg_cycles(i);
            start_col = (i - 1) * interp_length + 1;
            end_col = i * interp_length;
            
            % Add filtered positions
            left_leg_data_4D(1, start_col:end_col) = linspace(0, 1, interp_length);
            for j = 1:length(joint_fields_filtered)
                left_leg_data_4D(j + 1, start_col:end_col) = current_cycle.(joint_fields_filtered{j});
            end
            
            % Add velocities
            for j = 1:length(joint_fields_velocity)
                left_leg_data_4D(j + 4 + 1, start_col:end_col) = current_cycle.(joint_fields_velocity{j});
            end
        end
        
        save(fullfile(output_dir, 'processed_data_left_leg_4D.mat'), 'left_leg_data_4D');
        fprintf('  Saved left leg data to %s\n', fullfile(output_dir, 'processed_data_left_leg_4D.mat'));
    else
        fprintf('  No left leg cycles to save.\n');
    end
    
    fprintf('Saving complete!\n');
end
