%% process_all_amc_files_v2: Processes all AMC files and collects gait cycles from the v2 extractor.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function iterates through all specified AMC files, extracts the gait
%   cycles using the robust v2 detection method, interpolates them to a standard
%   length, and collects them into a single data structure. It now also stores  
%   the duration of each cycle and the mean duration for each leg.  
%
% Input:
%   amc_files - struct array: A list of AMC files to process.
%   show_debug_plot - (optional) boolean: If true, displays a debug plot for each file. Default is false.
%   enable_printing - (optional) boolean: If true, enables detailed print statements. Default is false.
%   interp_length - (optional) scalar: The number of points for interpolation. Default is 200.    
%
% Output:
%   all_cycles_data - struct: Contains all extracted and processed gait cycle data.
%   file_info       - struct: Contains information about the processed files.

function [all_cycles_data, file_info] = process_all_amc_files_v2(amc_files, show_debug_plot, enable_printing, interp_length)
    if nargin < 4 
       interp_length = 200;  
    end
    
    fprintf('\n=== PROCESSING ALL FILES (V2) ===\n');
    
    % Initialize storage
    all_cycles_data = struct();
    all_cycles_data.right_leg_cycles = [];
    all_cycles_data.left_leg_cycles = [];
    all_cycles_data.file_indices = struct();
    all_cycles_data.file_indices.right = [];
    all_cycles_data.file_indices.left = [];
    
    file_info = struct();
    file_info.names = {amc_files.name};
    file_info.colors = lines(length(amc_files));
    file_info.total_cycles = 0;
    
    % Initialize arrays to collect all cycle durations 
    all_right_durations = [];                      
    all_left_durations = [];    
 
     % Interpolation parameters are now passed in as an argument.  
    time_standard = linspace(0, 1, interp_length);
    
    for file_idx = 1:length(amc_files)
        filename = amc_files(file_idx).name;
        fprintf('Processing file %d/%d: %s\n', file_idx, length(amc_files), filename);
        
        try
            % Extract gait cycles using the new robust method (v2)
            [right_cycles_data, left_cycles_data] = extract_gait_cycles_knee_minima_robust_v2(filename, show_debug_plot, enable_printing);

            if isempty(right_cycles_data) && isempty(left_cycles_data)
                fprintf('  WARNING: No gait cycles detected in %s\n', filename);
                continue;
            end
            
            total_cycles_in_file = length(right_cycles_data) + length(left_cycles_data);
            fprintf('  Found %d right leg cycles and %d left leg cycles\n', length(right_cycles_data), length(left_cycles_data));
            file_info.total_cycles = file_info.total_cycles + total_cycles_in_file;
            
            % Process right leg cycles
            for i = 1:length(right_cycles_data)
                cycle = right_cycles_data(i);
                time_norm = linspace(0, 1, length(cycle.right_hip_flex));

                % Create a structure for the interpolated cycle data
                interpolated_cycle = struct();
                interpolated_cycle.right_hip_flex = interp1(time_norm, cycle.right_hip_flex, time_standard, 'linear');
                interpolated_cycle.left_hip_flex = interp1(time_norm, cycle.left_hip_flex, time_standard, 'linear');
                interpolated_cycle.right_knee_flex = interp1(time_norm, cycle.right_knee_flex, time_standard, 'linear');
                interpolated_cycle.left_knee_flex = interp1(time_norm, cycle.left_knee_flex, time_standard, 'linear');
                interpolated_cycle.duration = cycle.duration; % Store the original cycle duration   
                all_right_durations = [all_right_durations; cycle.duration]; % Accumulate duration    
                 
                % Store interpolated data
                all_cycles_data.right_leg_cycles = [all_cycles_data.right_leg_cycles; interpolated_cycle];
                all_cycles_data.file_indices.right = [all_cycles_data.file_indices.right; file_idx];
            end
            
            % Process left leg cycles
            for i = 1:length(left_cycles_data)
                cycle = left_cycles_data(i);
                time_norm = linspace(0, 1, length(cycle.left_hip_flex));
                
                % Create a structure for the interpolated cycle data
                interpolated_cycle = struct();
                interpolated_cycle.right_hip_flex = interp1(time_norm, cycle.right_hip_flex, time_standard, 'linear');
                interpolated_cycle.left_hip_flex = interp1(time_norm, cycle.left_hip_flex, time_standard, 'linear');
                interpolated_cycle.right_knee_flex = interp1(time_norm, cycle.right_knee_flex, time_standard, 'linear');
                interpolated_cycle.left_knee_flex = interp1(time_norm, cycle.left_knee_flex, time_standard, 'linear');
                interpolated_cycle.duration = cycle.duration; % Store the original cycle duration  
                all_left_durations = [all_left_durations; cycle.duration]; % Accumulate duration 
                 
                % Store interpolated data
                all_cycles_data.left_leg_cycles = [all_cycles_data.left_leg_cycles; interpolated_cycle];
                all_cycles_data.file_indices.left = [all_cycles_data.file_indices.left; file_idx];
            end
            
        catch ME
            fprintf('  ERROR processing %s: %s\n', filename, ME.message);
        end
    end
    
    all_cycles_data.time_standard = time_standard;
                                                                                                                                                                    
    % Calculate and store the mean duration for each leg                                                                                                                
    all_cycles_data.mean_duration_right = mean(all_right_durations);                                                                                                    
    all_cycles_data.mean_duration_left = mean(all_left_durations);  
    
    fprintf('\nTotal cycles collected: %d\n', file_info.total_cycles);
    fprintf('Right leg cycles: %d (Mean Duration: %.3fs)\n', length(all_cycles_data.right_leg_cycles), all_cycles_data.mean_duration_right);   
    fprintf('Left leg cycles: %d (Mean Duration: %.3fs)\n', length(all_cycles_data.left_leg_cycles), all_cycles_data.mean_duration_left);
end