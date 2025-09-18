%% Function: Save processed data (unchanged except demos call)
function save_processed_data(processed_data, file_info, N_Samples)
    fprintf('\n=== SAVING PROCESSED DATA ===\n');
    
    mat_data = struct();
    mat_data.amc_files = file_info.names;
    mat_data.processed_data = processed_data;
    mat_data.file_info = file_info;
    mat_data.processing_time = datetime('now');
    mat_data.filter_params = struct('cutoff_freq', 6, 'filter_order', 4);
    
    mat_filename = sprintf('Gait Data/processed_gait_data_fields_angular.mat');
    save(mat_filename, 'mat_data');
    
    % Save demos with one cell per gait cycle (filtered)
    data = create_demos_structure_per_cycleV3(processed_data, N_Samples);
    
    a_format_filename = sprintf('Gait Data/processed_gait_data_angular_%s_samples.mat', string(N_Samples));%%, datestr(now, 'yyyymmdd_HHMMSS'
    save(a_format_filename, 'data');
    
    fprintf('Data saved in two formats:\n  Original format: %s\n  A.mat format: %s\n', mat_filename, a_format_filename);
end




% a_format_filename = sprintf('Gait Data/demo_gait_data_angular_%s_samples.mat', string(10));
% save(a_format_filename, 'output_struct_array');