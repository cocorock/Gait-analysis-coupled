%% extract_gait_cycles_knee_minima_allcycles: Extracts gait cycles based on knee minima.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function extracts gait cycles from a single AMC file by identifying
%   the minima in the knee flexion angle as heel strike events. It extracts one
%   continuous cycle for each leg.
%
% Input:
%   filename - string: The path to the AMC file.
%
% Output:
%   gait_cycles_data - struct array: Contains the segmented angle data for each gait cycle.

function [gait_cycles_data] = extract_gait_cycles_knee_minima_allcycles(filename)
    % Extract all hip and knee flexion angles
    [time, right_hip_flex, left_hip_flex, right_knee_flex, left_knee_flex] = extract_hip_knee_flexion(filename);
    
%     % Create a figure with two subplots
%     figure('Name', 'Hip and Knee Flexion', 'Position', [100, 100, 800, 600]);
% 
%     % Plot hip flexion
%     subplot(2,1,1);
%     plot(time, right_hip_flex, 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(time, left_hip_flex, 'b-', 'LineWidth', 1.5);
%     xlabel('Time (s)');
%     ylabel('Hip Flexion (degrees)');
%     title('Hip Flexion Trajectories');
%     legend('Right Hip', 'Left Hip', 'Location', 'best');
%     grid on;
% 
%     % Plot knee flexion
%     subplot(2,1,2);
%     plot(time, right_knee_flex, 'r-', 'LineWidth', 1.5);
%     hold on;
%     plot(time, left_knee_flex, 'b-', 'LineWidth', 1.5);
%     xlabel('Time (s)');
%     ylabel('Knee Flexion (degrees)');
%     title('Knee Flexion Trajectories');
%     legend('Right Knee', 'Left Knee', 'Location', 'best');
%     grid on;
    
    % Smooth the signals to reduce noise for better minima detection
    window_size = 5; % Small smoothing window
    right_knee_smooth = smooth(right_knee_flex, window_size);
    left_knee_smooth = smooth(left_knee_flex, window_size);

    % Detect knee minima (potential heel strikes)
    % Use findpeaks on inverted signal to find minima
    [~, right_knee_minima] = findpeaks(-right_knee_smooth);
    [~, left_knee_minima] = findpeaks(-left_knee_smooth);

    % Initialize gait_cycles_data structure
    gait_cycles_data = [];

    % Right leg cycle
    if ~isempty(right_knee_minima)
        start_frame = right_knee_minima(1);
        end_frame = right_knee_minima(end);

        gait_cycles_data(1).leg = 'right';
        gait_cycles_data(1).cycle_number = 1;
        gait_cycles_data(1).start_frame = start_frame;
        gait_cycles_data(1).end_frame = end_frame;
        gait_cycles_data(1).start_time = time(start_frame);
        gait_cycles_data(1).end_time = time(end_frame);
        gait_cycles_data(1).duration = time(end_frame) - time(start_frame);

        cycle_length = end_frame - start_frame + 1;
        gait_cycles_data(1).time_normalized = linspace(0, 1, cycle_length);

        cycle_indices = start_frame:end_frame;
        gait_cycles_data(1).right_hip_flex = right_hip_flex(cycle_indices);
        gait_cycles_data(1).left_hip_flex = left_hip_flex(cycle_indices);
        gait_cycles_data(1).right_knee_flex = right_knee_flex(cycle_indices);
        gait_cycles_data(1).left_knee_flex = left_knee_flex(cycle_indices);
    end

    % Left leg cycle
    if ~isempty(left_knee_minima)
        start_frame = right_knee_minima(1);
        end_frame = right_knee_minima(end);

        gait_cycles_data(end+1).leg = 'left';
        gait_cycles_data(end).cycle_number = 1;
        gait_cycles_data(end).start_frame = start_frame;
        gait_cycles_data(end).end_frame = end_frame;
        gait_cycles_data(end).start_time = time(start_frame);
        gait_cycles_data(end).end_time = time(end_frame);
        gait_cycles_data(end).duration = time(end_frame) - time(start_frame);

        cycle_length = end_frame - start_frame + 1;
        gait_cycles_data(end).time_normalized = linspace(0, 1, cycle_length);

        cycle_indices = start_frame:end_frame;
        gait_cycles_data(end).right_hip_flex = right_hip_flex(cycle_indices);
        gait_cycles_data(end).left_hip_flex = left_hip_flex(cycle_indices);
        gait_cycles_data(end).right_knee_flex = right_knee_flex(cycle_indices);
        gait_cycles_data(end).left_knee_flex = left_knee_flex(cycle_indices);
    end
    
% %     Create a figure with two subplots
%     figure('Name', 'Gait Trajectories', 'Position', [100, 100, 800, 600]);
% 
%     % Plot hip trajectories
%     subplot(2,1,1);
%     for i = 1:length(gait_cycles_data)
%         if strcmp(gait_cycles_data(i).leg, 'right')
%             plot(gait_cycles_data(i).time_normalized, gait_cycles_data(i).right_hip_flex, 'r-', 'LineWidth', 1.5);
%             hold on;
%         else
%             plot(gait_cycles_data(i).time_normalized, gait_cycles_data(i).left_hip_flex, 'b-', 'LineWidth', 1.5);
%             hold on;
%         end
%     end
%     xlabel('Normalized Time');
%     ylabel('Hip Flexion (degrees)');
%     title('Hip Trajectories');
%     legend('Right Hip', 'Left Hip', 'Location', 'best');
%     grid on;
% 
%     % Plot knee trajectories
%     subplot(2,1,2);
%     for i = 1:length(gait_cycles_data)
%         if strcmp(gait_cycles_data(i).leg, 'right')
%             plot(gait_cycles_data(i).time_normalized, gait_cycles_data(i).right_knee_flex, 'r-', 'LineWidth', 1.5);
%             hold on;
%         else
%             plot(gait_cycles_data(i).time_normalized, gait_cycles_data(i).left_knee_flex, 'b-', 'LineWidth', 1.5);
%             hold on;
%         end
%     end
%     xlabel('Normalized Time');
%     ylabel('Knee Flexion (degrees)');
%     title('Knee Trajectories');
%     legend('Right Knee', 'Left Knee', 'Location', 'best');
%     grid on;
    
    % Final summary
    fprintf('\n\n=== Gait Cycles Summary ===\n');
    fprintf('Total gait cycles: %d\n', length(gait_cycles_data));
    fprintf('Right leg: %d cycle\n', sum(strcmp({gait_cycles_data.leg}, 'right')));
    fprintf('Left leg: %d cycle\n', sum(strcmp({gait_cycles_data.leg}, 'left')));
end