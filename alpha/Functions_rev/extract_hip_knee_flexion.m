%% extract_hip_knee_flexion: Extracts hip and knee flexion angles from an AMC file.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function reads an AMC file, extracts the rotation data for the hip
%   (rfemur, lfemur) and knee (rtibia, ltibia) joints, and returns the flexion
%   angles over time. Hip flexion is inverted for intuitive plotting.
%
% Input:
%   filename - string: The path to the AMC file.
%
% Output:
%   time          - (N x 1) vector: Time vector in seconds.
%   right_hip_flex  - (N x 1) vector: Right hip flexion angles in degrees.
%   left_hip_flex   - (N x 1) vector: Left hip flexion angles in degrees.
%   right_knee_flex - (N x 1) vector: Right knee flexion angles in degrees.
%   left_knee_flex  - (N x 1) vector: Left knee flexion angles in degrees.

function [time, right_hip_flex, left_hip_flex, right_knee_flex, left_knee_flex] = extract_hip_knee_flexion(filename)

    if nargin < 1
        filename = '39_01.amc';
    end

    % Read the AMC file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    % Initialize storage
    frame_data = [];
    frame_count = 0;
    current_frame_angles = [];

    % Read file line by line
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            line = strtrim(line);

            % Skip header lines and comments
            if startsWith(line, ':') || startsWith(line, '#') || isempty(line)
                continue;
            end

            % Check if line starts with a number (frame number)
            if ~isempty(regexp(line, '^\d+$', 'once'))
                % This is a frame number line
                if frame_count > 0
                    % Store previous frame data
                    frame_data(frame_count, :) = current_frame_angles;
                end

                frame_count = frame_count + 1;
                current_frame = str2double(line);

                % Initialize angles for this frame [frame, rfemur_rx, lfemur_rx, rtibia_rx, ltibia_rx]
                current_frame_angles = [current_frame, NaN, NaN, NaN, NaN];

            else
                % This should be joint data
                tokens = strsplit(line);
                if length(tokens) >= 2
                    joint_name = tokens{1};

                    % Extract flexion/extension angles (first rotation component)
                    if strcmp(joint_name, 'rfemur') && length(tokens) >= 2
                        current_frame_angles(2) = str2double(tokens{2});
                    elseif strcmp(joint_name, 'lfemur') && length(tokens) >= 2
                        current_frame_angles(3) = str2double(tokens{2});
                    elseif strcmp(joint_name, 'rtibia') && length(tokens) >= 2
                        current_frame_angles(4) = str2double(tokens{2});
                    elseif strcmp(joint_name, 'ltibia') && length(tokens) >= 2
                        current_frame_angles(5) = str2double(tokens{2});
                    end
                end
            end
        end
    end

    % Don't forget the last frame
    if frame_count > 0
        frame_data(frame_count, :) = current_frame_angles;
    end

    fclose(fid);

    if isempty(frame_data)
        error('No valid frame data found in file');
    end
    
%      disp(frame_data)
    
    % Determine hip offset angle based on filename
    [~, name, ~] = fileparts(filename);
    if startsWith(name, '39_')
        hip_offset_angle = 18; %subject 39
    else
        hip_offset_angle = 0;
    end

    % Extract data
    frames = frame_data(:, 1);
    right_hip_flex = frame_data(:, 2) + hip_offset_angle;  % Flip hip flexion/extension
    left_hip_flex  = frame_data(:, 3) + hip_offset_angle;   % Flip hip flexion/extension
    right_knee_flex = frame_data(:, 4);  % Keep knee as is
    left_knee_flex  = frame_data(:, 5);   % Keep knee as is

    % Create time vector (assuming 120 fps)
    fps = 120;
    time = (frames - frames(1)) / fps;

% %     Plot the results
%     figure('Position', [100, 100, 1200, 800]);
% 
%     %Hip flexion/extension
%     subplot(2, 1, 1);
%     plot(time, right_hip_flex, 'r-', 'LineWidth', 2, 'DisplayName', 'Right Hip');
%     hold on;
%     plot(time, left_hip_flex, 'b-', 'LineWidth', 2, 'DisplayName', 'Left Hip');
%     xlabel('Time (s)');
%     ylabel('Hip Flexion/Extension (degrees)');
%     title('Hip Flexion/Extension Angles (Flipped)');
%     legend('show');
%     grid on;
% 
%     % Knee flexion/extension
%     subplot(2, 1, 2);
%     plot(time, right_knee_flex, 'r-', 'LineWidth', 2, 'DisplayName', 'Right Knee');
%     hold on;
%     plot(time, left_knee_flex, 'b-', 'LineWidth', 2, 'DisplayName', 'Left Knee');
%     xlabel('Time (s)');
%     ylabel('Knee Flexion/Extension (degrees)');
%     title('Knee Flexion/Extension Angles');
%     legend('show');
%     grid on;

    % Display summary statistics
    fprintf('\n=== SUMMARY STATISTICS ===\n');
    fprintf('Total frames: %d\n', length(frames));
    fprintf('Duration: %.2f seconds\n', max(time));
    fprintf('Right Hip Flexion/Extension: Mean=%.2f°, Range=[%.2f°, %.2f°]\n', ...
        nanmean(right_hip_flex), nanmin(right_hip_flex), nanmax(right_hip_flex));
    fprintf('Left Hip Flexion/Extension: Mean=%.2f°, Range=[%.2f°, %.2f°]\n', ...
        nanmean(left_hip_flex), nanmin(left_hip_flex), nanmax(left_hip_flex));
    fprintf('Right Knee Flexion/Extension: Mean=%.2f°, Range=[%.2f°, %.2f°]\n', ...
        nanmean(right_knee_flex), nanmin(right_knee_flex), nanmax(right_knee_flex));
    fprintf('Left Knee Flexion/Extension: Mean=%.2f°, Range=[%.2f°, %.2f°]\n', ...
        nanmean(left_knee_flex), nanmin(left_knee_flex), nanmax(left_knee_flex));

end