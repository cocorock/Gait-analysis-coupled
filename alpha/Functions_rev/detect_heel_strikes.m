
% Functions_rev/detect_heel_strikes.m
function heel_strike_indices = detect_heel_strikes(ankle_pos_FR2_trajectory, sample_rate)
    % DETECT_HEEL_STRIKES Detects heel strike events from ankle position trajectory
    %   in a specific frame of reference (FR2). Heel strike is identified as
    %   the maximum points in the X-axis of the trajectory.
    %
    %   Inputs:
    %     ankle_pos_FR2_trajectory: Nx2 matrix where N is the number of frames,
    %                               column 1 is X-coordinate, column 2 is Y-coordinate
    %                               in the FR2 frame of reference.
    %     sample_rate: Sampling rate of the data in Hz.
    %
    %   Output:
    %     heel_strike_indices: Column vector of indices where heel strikes occur.

    if nargin < 2
        error('Not enough input arguments. Usage: detect_heel_strikes(ankle_pos_FR2_trajectory, sample_rate)');
    end

    % --- Debugging Flag ---
    show_debug_plot = false; % Set to true to show plots for debugging

    % --- Parameters for robust detection ---
    window_size = 5; % Smoothing window size (frames)
    
    avg_gait_duration = 1.0; % Average expected gait cycle duration (seconds)
    gait_tolerance_percent = 0.25; % 25% tolerance
    
    min_gait_duration = avg_gait_duration * (1 - gait_tolerance_percent); % 0.75s
    max_gait_duration = avg_gait_duration * (1 + gait_tolerance_percent); % 1.25s

    min_frames_between_cycles = round(min_gait_duration * sample_rate);
    
    % 1. Extract X-coordinate
    x_trajectory = ankle_pos_FR2_trajectory(:, 1); % Get the X-coordinate

    % 2. Smooth the X-coordinate
    x_smooth = smooth(x_trajectory, window_size);

    % 3. Detect peaks (maximum points) in the smoothed X-coordinate
    % Calculate prominence threshold based on data range
    x_range = max(x_smooth) - min(x_smooth);
    min_prominence_threshold = x_range * 0.10; % 10% of the X-range as minimum prominence

    [pks, heel_strike_indices] = findpeaks(x_smooth, ...
        'MinPeakDistance', min_frames_between_cycles, ...
        'MinPeakProminence', min_prominence_threshold);

    % Ensure indices are column vector
    heel_strike_indices = heel_strike_indices(:);

    % --- Debugging Plots ---
    if show_debug_plot
%         figure(round(rand(1)*100))
        figure()

        % Plot 1: Original Ankle X-trajectory
        subplot(3,1,1);
        plot(x_trajectory, 'b');
        title('Original Ankle X-trajectory (FR2)');
        xlabel('Frame');
        ylabel('X-position');
        grid on;

        % Plot 2: Smoothed X-trajectory
        subplot(3,1,2);
        plot(x_smooth, 'r', 'LineWidth', 1.5);
        title('Smoothed Ankle X-trajectory');
        xlabel('Frame');
        ylabel('X-position');
        grid on;
        legend('Smoothed X-trajectory', 'Location', 'best');

        % Plot 3: Smoothed X-trajectory with Detected Heel Strikes
        subplot(3,1,3);
        plot(x_smooth, 'r', 'LineWidth', 1.5);
        hold on;
        if ~isempty(heel_strike_indices)
            plot(heel_strike_indices, x_smooth(heel_strike_indices), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
            text(heel_strike_indices, x_smooth(heel_strike_indices), num2str(heel_strike_indices), ...
                 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'g');
        end
        title(sprintf('Smoothed X-trajectory with Detected Heel Strikes (%d found)', length(heel_strike_indices)));
        xlabel('Frame');
        ylabel('X-position');
        grid on;
        if ~isempty(heel_strike_indices)
            legend('Smoothed X-trajectory', 'Detected Heel Strikes', 'Location', 'best');
        else
            legend('Smoothed X-trajectory', 'Location', 'best');
        end
        
        % Add a line for the prominence threshold
        yline(min(x_smooth) + min_prominence_threshold, ':', 'Prominence Threshold');
    end
end
