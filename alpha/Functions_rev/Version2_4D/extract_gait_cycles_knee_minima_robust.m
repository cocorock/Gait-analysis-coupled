%% extract_gait_cycles_knee_minima_robust: Detects gait cycles using robust knee and hip analysis.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function implements a robust method for detecting gait cycles from an
%   AMC file. It uses knee angle minima as potential heel strikes and validates
%   them by analyzing the hip movement to ensure a single, significant hip
%   minimum occurs during each cycle.
%
% Input:
%   filename - string: Path to the AMC file.
%   show_debug_plot - (optional) boolean: If true, displays a debug plot. Default is false.
%   enable_printing - (optional) boolean: If true, enables detailed print statements for debugging. Default is false.
%
% Output:
%   gait_cycles_data - struct array: Contains segmented angle data for each validated gait cycle.

function [gait_cycles_data] = extract_gait_cycles_knee_minima_robust(filename, show_debug_plot, enable_printing)
    if nargin < 2
        show_debug_plot = false;
    end
    if nargin < 3
        enable_printing = false;
    end

    % Extract all hip and knee flexion angles
    [time, right_hip_flex, left_hip_flex, right_knee_flex, left_knee_flex] = extract_hip_knee_flexion(filename);

    % Gait cycle parameters
    sample_rate = 120; % Hz
    avg_gait_duration = 1.0; % seconds
    gait_tolerance = 0.25; % 0.25 seconds
    min_gait_duration = avg_gait_duration - gait_tolerance; % 0.75s
    max_gait_duration = avg_gait_duration + gait_tolerance; % 1.25s

    % Convert to frames
    min_frames_between_cycles = round(min_gait_duration * sample_rate); % ~90 frames
    max_frames_between_cycles = round(max_gait_duration * sample_rate); % ~150 frames

    % Smooth the signals to reduce noise for better minima detection
    window_size = 5; % Small smoothing window
    right_knee_smooth = smooth(right_knee_flex, window_size);
    left_knee_smooth = smooth(left_knee_flex, window_size);
    right_hip_smooth = smooth(right_hip_flex, window_size);
    left_hip_smooth = smooth(left_hip_flex, window_size);

    % Detect knee minima (potential heel strikes)
    % Use findpeaks on inverted signal to find minima
    [~, right_knee_minima] = findpeaks(-right_knee_smooth, ...
        'MinPeakDistance', min_frames_between_cycles, ...
        'MinPeakHeight', -max(right_knee_smooth) + 5); % At least 5 degrees from max

    [~, left_knee_minima] = findpeaks(-left_knee_smooth, ...
        'MinPeakDistance', min_frames_between_cycles, ...
        'MinPeakHeight', -max(left_knee_smooth) + 5); % At least 5 degrees from max

    % Robust hip minima detection function
    function hip_minima_count = count_significant_hip_minima(hip_segment)
        % More robust approach to count significant hip minima
        % Uses multiple criteria to avoid counting small fluctuations

        if length(hip_segment) < 20  % Too short to analyze
            hip_minima_count = 0;
            return;
        end

        % Method 1: Use findpeaks with stricter criteria
        segment_range = max(hip_segment) - min(hip_segment);
        min_prominence = segment_range * 0.15; % Minimum 15% of range
        min_distance = round(0.4 * sample_rate); % At least 0.4s apart (stricter)

        [~, minima_strict] = findpeaks(-hip_segment, ...
            'MinPeakDistance', min_distance, ...
            'MinPeakProminence', min_prominence);

        % Method 2: Find global minimum and check if it's isolated
        [~, global_min_idx] = min(hip_segment);

        % Check if global minimum is well-isolated (no other significant minima nearby)
        search_radius = round(0.3 * sample_rate); % 0.3s radius
        start_search = max(1, global_min_idx - search_radius);
        end_search = min(length(hip_segment), global_min_idx + search_radius);

        % Find local minima in the search region
        local_segment = hip_segment(start_search:end_search);
        local_min_val = min(local_segment);
        threshold = local_min_val + segment_range * 0.1; % Within 10% of minimum

        % Count how many points are near the minimum
        near_min_count = sum(hip_segment <= threshold);

        % Method 3: Simple derivative-based approach
        % Smooth more aggressively for derivative
        hip_smooth = smooth(hip_segment, round(0.1 * sample_rate));
        hip_diff = diff(hip_smooth);

        % Find zero crossings (sign changes) in derivative
        sign_changes = sum(diff(sign(hip_diff)) ~= 0);
        expected_changes = 2; % One down, one up for a single minimum

        % Decision logic: Combine multiple methods
        if length(minima_strict) == 1
            % Method 1 found exactly one significant minimum
            hip_minima_count = 1;
        elseif length(minima_strict) == 0 && near_min_count < round(0.2 * sample_rate)
            % No strict minima found, but global minimum is isolated
            hip_minima_count = 1;
        elseif sign_changes <= 4  % Allow some noise, but not too much
            % Derivative suggests simple pattern
            hip_minima_count = 1;
        else
            % Multiple minima detected - likely complex pattern
            hip_minima_count = length(minima_strict);
            if hip_minima_count == 0
                hip_minima_count = 2; % Conservative estimate if methods disagree
            end
        end
    end

    % Validate knee minima with robust hip analysis
    function valid_minima = validate_knee_minima_robust(knee_minima, hip_signal, leg_name)
        valid_minima = [];

        if enable_printing
            fprintf('\n=== ROBUST VALIDATION: %s LEG ===\n', upper(leg_name));
            fprintf('Detected %d knee minima for %s leg\n', length(knee_minima), leg_name);
        end

        if length(knee_minima) < 2
            fprintf('Not enough minima detected for cycle analysis\n');
            return;
        end

        for i = 1:length(knee_minima)-1
            current_knee_min = knee_minima(i);
            next_knee_min = knee_minima(i+1);

            if enable_printing
                fprintf('\n--- Cycle %d (%s leg) ---\n', i, leg_name);
                fprintf('Frames %d--%d (%.2fs-%.2fs)\n', ...
                        current_knee_min, next_knee_min, ...
                        time(current_knee_min), time(next_knee_min));
            end

            % Check cycle duration
            cycle_duration = (next_knee_min - current_knee_min) / sample_rate;
            if enable_printing, fprintf('Duration: %.3fs ', cycle_duration); end

            duration_valid = cycle_duration >= min_gait_duration && cycle_duration <= max_gait_duration;
            if duration_valid
                if enable_printing, fprintf('[PASS]\n'); end
            else
                if enable_printing, fprintf('[FAIL]\n'); end
            end


            if duration_valid
                % Extract hip segment and analyze
                cycle_start = current_knee_min;
                cycle_end = next_knee_min;
                hip_segment = hip_signal(cycle_start:cycle_end);

                % Use robust hip minima counting
                hip_minima_count = count_significant_hip_minima(hip_segment);

                if enable_printing, fprintf('Hip analysis: %d significant minima ', hip_minima_count); end

                hip_valid = hip_minima_count == 1;
                if hip_valid
                    if enable_printing, fprintf('[PASS]\n'); end
                    valid_minima = [valid_minima, current_knee_min];
                    if enable_printing, fprintf('>>> CYCLE %d ACCEPTED <<<\n', i); end
                else
                    if enable_printing, fprintf('[FAIL]\n'); end
                    if enable_printing, fprintf('>>> CYCLE %d REJECTED <<<\n', i); end
                end
            else
                if enable_printing, fprintf('Hip analysis: SKIPPED (duration failed)\n'); end
                if enable_printing, fprintf('>>> CYCLE %d REJECTED <<<\n', i); end
            end
        end

        % Check final cycle
        if length(knee_minima) > 1 && length(valid_minima) > 0
            if enable_printing, fprintf('\n--- Final cycle (%s leg) ---\n', leg_name); end
            last_valid = valid_minima(end);
            last_knee = knee_minima(end);

            cycle_duration = (last_knee - last_valid) / sample_rate;
            if enable_printing, fprintf('Duration: %.3fs ', cycle_duration); end

            duration_valid = cycle_duration >= min_gait_duration && cycle_duration <= max_gait_duration;
            if duration_valid
                if enable_printing, fprintf('[PASS]\n'); end

                hip_segment = hip_signal(last_valid:last_knee);
                hip_minima_count = count_significant_hip_minima(hip_segment);

                if enable_printing, fprintf('Hip analysis: %d significant minima ', hip_minima_count); end

                if hip_minima_count == 1
                    if enable_printing, fprintf('[PASS]\n'); end
                    valid_minima = [valid_minima, last_knee];
                    if enable_printing, fprintf('>>> FINAL CYCLE ACCEPTED <<<\n'); end
                else
                    if enable_printing, fprintf('[FAIL]\n'); end
                    if enable_printing, fprintf('>>> FINAL CYCLE REJECTED <<<\n'); end
                end
            else
                if enable_printing, fprintf('[FAIL]\n'); end
                if enable_printing, fprintf('>>> FINAL CYCLE REJECTED <<<\n'); end
            end
        end

        if enable_printing
            fprintf('\n%s leg: %d/%d cycles validated\n', ...
                    upper(leg_name), max(0, length(valid_minima)-1), length(knee_minima)-1);
        end
    end

    % Validate both legs
    right_knee_valid = validate_knee_minima_robust(right_knee_minima, right_hip_smooth, 'right');
    left_knee_valid = validate_knee_minima_robust(left_knee_minima, left_hip_smooth, 'left');

    % CREATE DEBUG PLOT
    if show_debug_plot
        figure('Name', 'DEBUG: Robust Hip Minima Detection', 'Position', [50, 50, 1400, 800]);

        % Plot 1: Hip Flexion with validated cutting points
        subplot(3,1,1);
        plot(time, right_hip_flex, 'r-', 'LineWidth', 1.5);
        hold on;
        plot(time, left_hip_flex, 'b-', 'LineWidth', 1.5);

        % Mark validated cycles
        if ~isempty(right_knee_valid)
            for i = 1:length(right_knee_valid)
                xline(time(right_knee_valid(i)), 'r--', 'LineWidth', 2, 'Alpha', 0.8);
                text(time(right_knee_valid(i)), max([right_hip_flex; left_hip_flex]), ...
                     sprintf('R%d', i), 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 10);
            end
        end

        if ~isempty(left_knee_valid)
            for i = 1:length(left_knee_valid)
                xline(time(left_knee_valid(i)), 'b--', 'LineWidth', 2, 'Alpha', 0.8);
                text(time(left_knee_valid(i)), min([right_hip_flex; left_hip_flex]), ...
                     sprintf('L%d', i), 'Color', 'blue', 'FontWeight', 'bold', 'FontSize', 10);
            end
        end

        xlabel('Time (s)');
        ylabel('Hip Flexion (degrees)');
        title('Hip Flexion with Robust Validation');
        legend('Right Hip', 'Left Hip', 'Location', 'best');
        grid on;

        % Plot 2: Knee Flexion
        subplot(3,1,2);
        plot(time, right_knee_flex, 'r-', 'LineWidth', 1.5);
        hold on;
        plot(time, left_knee_flex, 'b-', 'LineWidth', 1.5);

        % Mark detected vs validated
        if ~isempty(right_knee_minima)
            plot(time(right_knee_minima), right_knee_flex(right_knee_minima), ...
                 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'none', 'LineWidth', 1);
        end
        if ~isempty(left_knee_minima)
            plot(time(left_knee_minima), left_knee_flex(left_knee_minima), ...
                 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'none', 'LineWidth', 1);
        end

        if ~isempty(right_knee_valid)
            plot(time(right_knee_valid), right_knee_flex(right_knee_valid), ...
                 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'red');
        end

        if ~isempty(left_knee_valid)
            plot(time(left_knee_valid), left_knee_flex(left_knee_valid), ...
                 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'blue');
        end

        xlabel('Time (s)');
        ylabel('Knee Flexion (degrees)');
        title('Knee Flexion: Detected (hollow) vs Validated (filled)');
        legend('Right Knee', 'Left Knee', 'Location', 'best');
        grid on;

        % Plot 3: Validation summary
        subplot(3,1,3);

        % Count validation results
        right_detected = length(right_knee_minima) - 1;
        right_validated = max(0, length(right_knee_valid) - 1);
        left_detected = length(left_knee_minima) - 1;
        left_validated = max(0, length(left_knee_valid) - 1);

        categories = {'Right Leg', 'Left Leg'};
        detected_counts = [right_detected, left_detected];
        validated_counts = [right_validated, left_validated];

        x = 1:length(categories);
        width = 0.35;

        bar(x - width/2, detected_counts, width, 'FaceColor', [0.8 0.8 0.8], 'DisplayName', 'Detected');
        hold on;
        bar(x + width/2, validated_counts, width, 'FaceColor', [0.2 0.6 0.2], 'DisplayName', 'Validated');

        xlabel('Leg');
        ylabel('Number of Cycles');
        title('Cycle Detection vs Validation Results');
        set(gca, 'XTickLabel', categories);
        legend('Location', 'best');
        grid on;

        % Add text annotations
        for i = 1:length(categories)
            text(i - width/2, detected_counts(i) + 0.1, num2str(detected_counts(i)), ...
                 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            text(i + width/2, validated_counts(i) + 0.1, num2str(validated_counts(i)), ...
                 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
    end

    % Segment validated cycles
    gait_cycles_data = [];
    cycle_count = 0;

    % Right leg cycles
    for i = 1:length(right_knee_valid)-1
        cycle_count = cycle_count + 1;
        start_frame = right_knee_valid(i);
        end_frame = right_knee_valid(i+1);

        gait_cycles_data(cycle_count).leg = 'right';
        gait_cycles_data(cycle_count).cycle_number = i;
        gait_cycles_data(cycle_count).start_frame = start_frame;
        gait_cycles_data(cycle_count).end_frame = end_frame;
        gait_cycles_data(cycle_count).start_time = time(start_frame);
        gait_cycles_data(cycle_count).end_time = time(end_frame);
        gait_cycles_data(cycle_count).duration = time(end_frame) - time(start_frame);

        cycle_length = end_frame - start_frame + 1;
        gait_cycles_data(cycle_count).time_normalized = linspace(0, 1, cycle_length);

        cycle_indices = start_frame:end_frame;
        gait_cycles_data(cycle_count).right_hip_flex = right_hip_flex(cycle_indices);
        gait_cycles_data(cycle_count).left_hip_flex = left_hip_flex(cycle_indices);
        gait_cycles_data(cycle_count).right_knee_flex = right_knee_flex(cycle_indices);
        gait_cycles_data(cycle_count).left_knee_flex = left_knee_flex(cycle_indices);
    end

    % Left leg cycles
    for i = 1:length(left_knee_valid)-1
        cycle_count = cycle_count + 1;
        start_frame = left_knee_valid(i);
        end_frame = left_knee_valid(i+1);

        gait_cycles_data(cycle_count).leg = 'left';
        gait_cycles_data(cycle_count).cycle_number = i;
        gait_cycles_data(cycle_count).start_frame = start_frame;
        gait_cycles_data(cycle_count).end_frame = end_frame;
        gait_cycles_data(cycle_count).start_time = time(start_frame);
        gait_cycles_data(cycle_count).end_time = time(end_frame);
        gait_cycles_data(cycle_count).duration = time(end_frame) - time(start_frame);

        cycle_length = end_frame - start_frame + 1;
        gait_cycles_data(cycle_count).time_normalized = linspace(0, 1, cycle_length);

        cycle_indices = start_frame:end_frame;
        gait_cycles_data(cycle_count).right_hip_flex = right_hip_flex(cycle_indices);
        gait_cycles_data(cycle_count).left_hip_flex = left_hip_flex(cycle_indices);
        gait_cycles_data(cycle_count).right_knee_flex = right_knee_flex(cycle_indices);
        gait_cycles_data(cycle_count).left_knee_flex = left_knee_flex(cycle_indices);
    end

    % Final summary
    if enable_printing
        fprintf('\n\n=== ROBUST VALIDATION SUMMARY ===\n');
        fprintf('Total validated gait cycles: %d\n', length(gait_cycles_data));
        fprintf('Right leg: %d cycles\n', sum(strcmp({gait_cycles_data.leg}, 'right')));
        fprintf('Left leg: %d cycles\n', sum(strcmp({gait_cycles_data.leg}, 'left')));
    end

end