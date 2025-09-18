%% save_linear_kinematics_structuredV3: Saves structured linear kinematics data to a single .mat file with a rich structure.
% 
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
% 
% Description:
%   This function takes the output of calculate_linear_kinematics_v3 and
%   saves the data into a single .mat file. Unlike previous versions, this
%   function preserves the rich hierarchical structure of the data, making it
%   easy to parse and convert to other formats like JSON. The output file 
%   will contain a single parent struct with descriptive fields.
% 
% Input:
%   linear_kinematics - struct: The output structure from calculate_linear_kinematics_v3.
%   time_standard     - vector: The normalized time vector (1x200).
%   file_info         - struct: The file_info struct from process_all_amc_files_v2.
% 
% Output:
%   None. Saves one .mat file: 'linear_kinematics_export.mat' in the 'Gait Data/' directory.

function save_linear_kinematics_structuredV3(linear_kinematics, time_standard, file_info)
    fprintf('\n=== SAVING RICHLY STRUCTURED LINEAR KINEMATICS (V3) ===\n');

    output_dir = './Gait Data/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Create a parent struct to hold all the data for export
    gait_data_export = struct();
    gait_data_export.creation_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    gait_data_export.description = 'Hierarchical export of linear gait kinematics data.';
    gait_data_export.source_files = file_info.names;
    gait_data_export.gait_cycle_time_vector = time_standard;

    % Process Right Leg Kinematics
    if isfield(linear_kinematics, 'right_leg_kinematics') && ~isempty(linear_kinematics.right_leg_kinematics)
        fprintf('  Structuring %d cycles segmented by the right leg...\n', length(linear_kinematics.right_leg_kinematics));
        gait_data_export.cycles_segmented_by_right_leg = linear_kinematics.right_leg_kinematics;
    else
        gait_data_export.cycles_segmented_by_right_leg = [];
    end

    % Process Left Leg Kinematics
    if isfield(linear_kinematics, 'left_leg_kinematics') && ~isempty(linear_kinematics.left_leg_kinematics)
        fprintf('  Structuring %d cycles segmented by the left leg...\n', length(linear_kinematics.left_leg_kinematics));
        gait_data_export.cycles_segmented_by_left_leg = linear_kinematics.left_leg_kinematics;
    else
        gait_data_export.cycles_segmented_by_left_leg = [];
    end

    % Save the entire parent struct to a single .mat file
    output_filename = fullfile(output_dir, 'linear_kinematics_export.mat');
    try
        save(output_filename, 'gait_data_export', '-v7.3');
        fprintf('  Successfully saved data to %s\n', output_filename);
    catch ME
        fprintf('  ERROR: Could not save data to %s\n', output_filename);
        fprintf('  MATLAB error: %s\n', ME.message);
    end

    fprintf('Linear kinematics data saving complete!\n');
end