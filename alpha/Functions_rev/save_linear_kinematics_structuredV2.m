%% save_linear_kinematics_structured: Saves structured linear kinematics data to .mat files.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%   (Modified by Gemini)
%
% Description:
%   This function takes the output of calculate_linear_kinematics_v2 and
%   saves the linear kinematics data for right and left legs into separate
%   .mat files. Each file contains a matrix of size [9 x (200 * num_gait_cycles)],
%   where the rows represent normalized time, ankle positions (x, y), and
%   ankle velocities (x, y) for both right and left ankles.
%
% Input:
%   linear_kinematics - struct: The output structure from calculate_linear_kinematics_v2.
%   time_standard     - vector: The normalized time vector (1x200).
%
% Output:
%   None. Saves two .mat files: 'right_leg_linear_kinematics.mat' and
%   'left_leg_linear_kinematics.mat' in the 'Gait Data/' directory.

function save_linear_kinematics_structured(linear_kinematics, time_standard)
    fprintf('\n=== SAVING LINEAR KINEMATICS DATA ===\n');

    output_dir = './Gait Data/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    interp_length = length(time_standard);

    % Process Right Leg Kinematics
    if isfield(linear_kinematics, 'right_leg_kinematics') && ~isempty(linear_kinematics.right_leg_kinematics)
        num_right_cycles = length(linear_kinematics.right_leg_kinematics);
        fprintf('  Processing %d right leg cycles for saving...\n', num_right_cycles);

        % Initialize matrix for right leg data
        right_leg_data_matrix = zeros(9, interp_length * num_right_cycles);

        for i = 1:num_right_cycles
            cycle_data = linear_kinematics.right_leg_kinematics(i);
            start_col = (i - 1) * interp_length + 1;
            end_col = i * interp_length;

            % Row 1: Normalized time
            right_leg_data_matrix(1, start_col:end_col) = time_standard;

            % Rows 2-9: Ankle positions and velocities
            right_leg_data_matrix(2, start_col:end_col) = cycle_data.right_ankle_pos(1,:); % right ankle x pos
            right_leg_data_matrix(3, start_col:end_col) = cycle_data.left_ankle_pos(1,:);  % left ankle x pos
            right_leg_data_matrix(4, start_col:end_col) = cycle_data.right_ankle_pos(2,:); % right ankle y pos
            right_leg_data_matrix(5, start_col:end_col) = cycle_data.left_ankle_pos(2,:);  % left ankle y pos
            right_leg_data_matrix(6, start_col:end_col) = cycle_data.right_ankle_vel(1,:); % right ankle x vel
            right_leg_data_matrix(7, start_col:end_col) = cycle_data.left_ankle_vel(1,:);  % left ankle x vel
            right_leg_data_matrix(8, start_col:end_col) = cycle_data.right_ankle_vel(2,:); % right ankle y vel
            right_leg_data_matrix(9, start_col:end_col) = cycle_data.left_ankle_vel(2,:);  % left ankle y vel
        end

        save(fullfile(output_dir, 'right_leg_linear_kinematics.mat'), 'right_leg_data_matrix');
        fprintf('  Saved right_leg_linear_kinematics.mat\n');
    else
        fprintf('  No right leg kinematics data to save.\n');
    end

    % Process Left Leg Kinematics
    if isfield(linear_kinematics, 'left_leg_kinematics') && ~isempty(linear_kinematics.left_leg_kinematics)
        num_left_cycles = length(linear_kinematics.left_leg_kinematics);
        fprintf('  Processing %d left leg cycles for saving...\n', num_left_cycles);

        % Initialize matrix for left leg data
        left_leg_data_matrix = zeros(9, interp_length * num_left_cycles);

        for i = 1:num_left_cycles
            cycle_data = linear_kinematics.left_leg_kinematics(i);
            start_col = (i - 1) * interp_length + 1;
            end_col = i * interp_length;

            % Row 1: Normalized time
            left_leg_data_matrix(1, start_col:end_col) = time_standard;

            % Rows 2-9: Ankle positions and velocities
            left_leg_data_matrix(2, start_col:end_col) = cycle_data.right_ankle_pos(1,:); % right ankle x pos
            left_leg_data_matrix(3, start_col:end_col) = cycle_data.left_ankle_pos(1,:);  % left ankle x pos
            left_leg_data_matrix(4, start_col:end_col) = cycle_data.right_ankle_pos(2,:); % right ankle y pos
            left_leg_data_matrix(5, start_col:end_col) = cycle_data.left_ankle_pos(2,:);  % left ankle y pos
            left_leg_data_matrix(6, start_col:end_col) = cycle_data.right_ankle_vel(1,:); % right ankle x vel
            left_leg_data_matrix(7, start_col:end_col) = cycle_data.left_ankle_vel(1,:);  % left ankle x vel
            left_leg_data_matrix(8, start_col:end_col) = cycle_data.right_ankle_vel(2,:); % right ankle y vel
            left_leg_data_matrix(9, start_col:end_col) = cycle_data.left_ankle_vel(2,:);  % left ankle y vel
        end

        save(fullfile(output_dir, 'left_leg_linear_kinematics.mat'), 'left_leg_data_matrix');
        fprintf('  Saved left_leg_linear_kinematics.mat\n');
    else
        fprintf('  No left leg kinematics data to save.\n');
    end

    fprintf('Linear kinematics data saving complete!\n');
end