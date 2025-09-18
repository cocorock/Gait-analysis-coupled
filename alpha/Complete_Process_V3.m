% Setup Environment
% Clear workspace, close figures, and add necessary paths.

close all;
clc;
clear;

% Add necessary paths to access functions and data
addpath('./AMC/');
addpath('./Functions_rev/'); % Use the revised functions
addpath('./Gait Data/');
addpath('./Functions_rev/Version2_4D/')
%% Create Output Directories
% This function creates the necessary output directories for saving plots and data

create_output_directorie();

%% Main Execution - Data Processing (V3)
% Start the gait analysis process.

fprintf('\n=== Process Plot and Save All AMC Files Completed V3 ===\n');

% Get all AMC files from the specified directory.
amc_files = get_amc_files();

% Determine subject ID from the first file name
if isempty(amc_files)
    error('No AMC files found in ./AMC/ directory. Aborting.');
end
subject_id = amc_files(1).name(1:2);
fprintf('Processing data for Subject: %s\n', subject_id);

% Read subject-specific bone lengths
asf_file_path = fullfile('./AMC/', [subject_id, '.asf']);
bone_lengths = read_asf_lengths(asf_file_path);

% Define the number of points for cycle interpolation.
interp_length = 200;

% Define a multiplier to scale the velocity of the movement (e.g., 0.5 = half speed).
velocity_multiplier = 0.25;

% Process all AMC files to extract and collect gait cycle data.
[all_cycles_data, file_info] = process_all_amc_files_v2(amc_files, false, true, interp_length);

% --- MODIFICATION: Use only cycles from the right leg criterion ---
% Clear the left leg data to ensure all subsequent analysis uses only the
% cycles segmented based on the right leg's heel strike.
all_cycles_data.left_leg_cycles = [];
all_cycles_data.file_indices.left = [];
fprintf('Modified: Using only %d cycles from the right leg criterion.\n', length(all_cycles_data.right_leg_cycles));

% Apply filtering to the collected gait data (derivatives are NOT calculated here).
processed_data = apply_filtering_V3(all_cycles_data);


% --- Scale Time to Adjust Velocity ---
% This section scales the time vector based on the multiplier.
% A longer duration for the same movement results in lower velocity/acceleration.
original_mean_duration =  mean([processed_data.mean_duration_right processed_data.mean_duration_left]);
scaled_duration = original_mean_duration / velocity_multiplier;
fprintf('Velocity multiplier applied: %.2f\n', velocity_multiplier);
fprintf('Original mean duration: %.3fs. New scaled duration for kinematics: %.3fs\n', original_mean_duration, scaled_duration);

% Overwrite the time_standard vector. It will now represent scaled time in seconds.
processed_data.time_standard = linspace(0, scaled_duration, interp_length);

% Ploting right leg angular kinematics (positions only).
plot_gait_kinematics_v3(processed_data, 'right');

% Plot_gait_kinematics_v3(processed_data, 'left');

% Save the processed data in multiple formats for further analysis.
% NOTE: This is commented out as processed_data in V3 only contains filtered
% positions, which may not be what the saving function expects.
% save_processed_data_4D(processed_data);

%% Kinematics Calculation and Plotting (V3)
% Calculate and visualize linear kinematics.

% Calculate linear kinematics using the new function.
% This function now calculates angular derivatives internally.
linear_kinematics = calculate_linear_kinematics_v3(processed_data, -90, bone_lengths);

% save_linear_kinematics_structuredV3(linear_kinematics, processed_data.time_standard, file_info);

%%  %Plotting Kinematics
% plot_linear_kinematics_positionsV2(linear_kinematics, processed_data.time_standard);
% plot_linear_kinematics_velocities(linear_kinematics, processed_data.time_standard);
% plot_linear_kinematics_accelerations(linear_kinematics, processed_data.time_standard);
% 
% plot_linear_kinematics_positions_xy(linear_kinematics);
% plot_linear_kinematics_velocities_xy(linear_kinematics);
% plot_linear_kinematics_accelerations_xy(linear_kinematics);

%% Frame of Reference Analysis
fprintf('\n=== FRAME OF REFERENCE ANALYSIS ===\n');
% Calculate the different frames of reference
multi_frame_kinematics = calculate_frames_of_reference(linear_kinematics);

% Plot the trajectories comparing the frames of reference
plot_multi_frame_trajectories(multi_frame_kinematics);

% Plot the combined trajectories on a single plot
plot_multi_frame_trajectories_combined(multi_frame_kinematics);

%% Save Final Export File
fprintf('\n=== SAVING FINAL EXPORT FILE ===\n');
save_multi_frame_kinematics(multi_frame_kinematics, velocity_multiplier, interp_length, scaled_duration, original_mean_duration, file_info, bone_lengths);

%% Final Instruction
subject_id = file_info.names{1}(1:2);
fprintf('\n----------------------------------------------------\n');
fprintf('To convert the .mat file to .json, run the following command in your terminal:\n');
fprintf('python convert_full_export_to_json.py %s\n', subject_id);
fprintf('----------------------------------------------------\n');
