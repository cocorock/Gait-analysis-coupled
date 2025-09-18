%% process_all_amc_files: Processes all AMC files and collects gait cycles.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function iterates through all specified AMC files, extracts the gait
%   cycles using a robust detection method, interpolates them to a standard
%   length, and collects them into a single data structure.
%
% Input:
%   amc_files - struct array: A list of AMC files to process.
%
% Output:
%   all_cycles_data - struct: Contains all extracted and processed gait cycle data.
%   file_info       - struct: Contains information about the processed files.

function [all_cycles_data, file_info] = process_all_amc_files(amc_files, show_debug_plot, enable_printing)
    fprintf('\n=== PROCESSING ALL FILES ===\n');
    
    % Initialize storage
    all_cycles_data = struct();
    all_cycles_data.right_hip_cycles = [];
    all_cycles_data.left_hip_cycles = [];
    all_cycles_data.right_knee_cycles = [];
    all_cycles_data.left_knee_cycles = [];
    all_cycles_data.file_indices = struct();
    all_cycles_data.file_indices.right_hip = [];
    all_cycles_data.file_indices.left_hip = [];
    all_cycles_data.file_indices.right_knee = [];
    all_cycles_data.file_indices.left_knee = [];
    
    file_info = struct();
    file_info.names = {amc_files.name};
    file_info.colors = lines(length(amc_files));
    file_info.total_cycles = 0;
    
    % Interpolation parameters
    interp_length = 200;%200
    time_standard = linspace(0, 1, interp_length);
    
    for file_idx = 1:length(amc_files)
        filename = amc_files(file_idx).name;
        fprintf('Processing file %d/%d: %s\n', file_idx, length(amc_files), filename);
        
        try
            % Extract gait cycles using robust method
            gait_cycles_data = extract_gait_cycles_knee_minima_robust(filename, show_debug_plot, enable_printing); %<<-----
%             gait_cycles_data = extract_gait_cycles_knee_minima(filename);

            if isempty(gait_cycles_data)
                fprintf('  WARNING: No gait cycles detected in %s\n', filename);
                continue;
            end
            
            fprintf('  Found %d gait cycles\n', length(gait_cycles_data));
            file_info.total_cycles = file_info.total_cycles + length(gait_cycles_data);
            
            % Separate cycles by leg
            right_cycles = find(strcmp({gait_cycles_data.leg}, 'right'));
            left_cycles = find(strcmp({gait_cycles_data.leg}, 'left'));
            
            % Process right leg cycles
            for i = 1:length(right_cycles)
                cycle_idx = right_cycles(i);
                time_norm = gait_cycles_data(cycle_idx).time_normalized;
                
                % Interpolate to standard length
                right_hip_interp = interp1(time_norm, gait_cycles_data(cycle_idx).right_hip_flex, time_standard, 'linear');
                right_knee_interp = interp1(time_norm, gait_cycles_data(cycle_idx).right_knee_flex, time_standard, 'linear');
                
                % Store interpolated data
                all_cycles_data.right_hip_cycles = [all_cycles_data.right_hip_cycles; right_hip_interp];
                all_cycles_data.right_knee_cycles = [all_cycles_data.right_knee_cycles; right_knee_interp];
                all_cycles_data.file_indices.right_hip = [all_cycles_data.file_indices.right_hip; file_idx];
                all_cycles_data.file_indices.right_knee = [all_cycles_data.file_indices.right_knee; file_idx];
            end
            
            % Process left leg cycles
            for i = 1:length(left_cycles)
                cycle_idx = left_cycles(i);
                time_norm = gait_cycles_data(cycle_idx).time_normalized;
                
                % Interpolate to standard length
                left_hip_interp = interp1(time_norm, gait_cycles_data(cycle_idx).left_hip_flex, time_standard, 'linear');
                left_knee_interp = interp1(time_norm, gait_cycles_data(cycle_idx).left_knee_flex, time_standard, 'linear');
                
                % Store interpolated data
                all_cycles_data.left_hip_cycles = [all_cycles_data.left_hip_cycles; left_hip_interp];
                all_cycles_data.left_knee_cycles = [all_cycles_data.left_knee_cycles; left_knee_interp];
                all_cycles_data.file_indices.left_hip = [all_cycles_data.file_indices.left_hip; file_idx];
                all_cycles_data.file_indices.left_knee = [all_cycles_data.file_indices.left_knee; file_idx];
            end
            
        catch ME
            fprintf('  ERROR processing %s: %s\n', filename, ME.message);
        end
    end
    
    all_cycles_data.time_standard = time_standard;
    
    fprintf('\nTotal cycles collected: %d\n', file_info.total_cycles);
    fprintf('Right hip cycles: %d\n', size(all_cycles_data.right_hip_cycles, 1));
    fprintf('Left hip cycles: %d\n', size(all_cycles_data.left_hip_cycles, 1));
    fprintf('Right knee cycles: %d\n', size(all_cycles_data.right_knee_cycles, 1));
    fprintf('Left knee cycles: %d\n', size(all_cycles_data.left_knee_cycles, 1));
end