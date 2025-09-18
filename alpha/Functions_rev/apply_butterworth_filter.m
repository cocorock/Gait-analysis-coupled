% Functions_rev/apply_butterworth_filter.m
function filtered_data = apply_butterworth_filter(data, sampling_freq, cutoff_freq, plot_flag)
    % APPLY_BUTTERWORTH_FILTER Applies a zero-lag Butterworth low-pass filter.
    %   filtered_data = APPLY_BUTTERWORTH_FILTER(data, sampling_freq, cutoff_freq, plot_flag)
    %   applies a 4th order Butterworth low-pass filter to the input data.
    %   This version includes circular padding with verification and adjustment.
    %
    %   Inputs:
    %     data: Input data (can be a vector or a matrix where filtering is
    %           applied column-wise).
    %     sampling_freq: Sampling frequency of the data (Hz).
    %     cutoff_freq: Cutoff frequency for the low-pass filter (Hz).
    %     plot_flag: Optional. If true, plots the original, padded, and filtered signals.
    %
    %   Output:
    %     filtered_data: The filtered data.

    if nargin < 3
        error('Not enough input arguments. Usage: apply_butterworth_filter(data, sampling_freq, cutoff_freq)');
    end
    if nargin < 4
        plot_flag = false;
    end

    if cutoff_freq >= sampling_freq / 2
        warning('Cutoff frequency is at or above Nyquist frequency. No filtering applied.');
        filtered_data = data;
        return;
    end

    % Design a 4th order Butterworth filter
    order = 4;
    [b, a] = butter(order, cutoff_freq / (sampling_freq / 2), 'low');

    % --- Padding to reduce edge effects ---
    pad_len = 60; % Length of padding on each side
    [N, num_dims] = size(data);

    if N < pad_len * 2 + 1
        warning('Data length is too short for specified padding. Skipping padding.');
        padded_data = data;
    else
        % Start with circular padding
        start_pad = data(end-pad_len+1:end, :);
        end_pad = data(1:pad_len, :);

        % --- Verify and Adjust Padding ---
        for i = 1:num_dims
            % Check start junction
            start_orig_val = data(1, i);
            start_pad_val = start_pad(end, i); % Last point of the start padding
            
            % Use relative tolerance, but fallback to absolute for small numbers
            if abs(start_orig_val) > 1e-6
                start_diff_is_large = abs(start_pad_val - start_orig_val) / abs(start_orig_val) > 0.03;
            else
                start_diff_is_large = abs(start_pad_val - start_orig_val) > 0.03; % Fallback to absolute difference
            end

            if start_diff_is_large
                shift = start_orig_val - start_pad_val;
                start_pad(:, i) = start_pad(:, i) + shift;
%                 fprintf('Adjusting start padding for dimension %d by %f\n', i, shift);
            end

            % Check end junction
            end_orig_val = data(end, i);
            end_pad_val = end_pad(1, i); % First point of the end padding

            if abs(end_orig_val) > 1e-6
                end_diff_is_large = abs(end_pad_val - end_orig_val) / abs(end_orig_val) > 0.03;
            else
                end_diff_is_large = abs(end_pad_val - end_orig_val) > 0.03;
            end

            if end_diff_is_large
                shift = end_orig_val - end_pad_val;
                end_pad(:, i) = end_pad(:, i) + shift;
%                 fprintf('Adjusting end padding for dimension %d by %f\n', i, shift);
            end
        end
        
        padded_data = [start_pad; data; end_pad];
    end

    % Apply the filter using filtfilt for zero phase distortion
    filtered_padded = filtfilt(b, a, padded_data);

    % Extract the original portion of the filtered data
    if N < pad_len * 2 + 1
        filtered_data = filtered_padded; % No padding was applied, return as is
    else
        filtered_data = filtered_padded(pad_len+1 : pad_len+N, :);
    end

    %% Plotting if requested
    if plot_flag
        t = (0:N-1) / sampling_freq; % Time vector for original data

        figure;
        hold on;

        if N < pad_len * 2 + 1
            % No padding was applied, just plot original and filtered
            plot(t, data, 'b', 'DisplayName', 'Original Data');
            plot(t, filtered_data, 'r', 'LineWidth', 1.5, 'DisplayName', 'Filtered Data');
        else
            % Padding was applied, create a time vector for the padded data
            t_padded = (-pad_len : N + pad_len - 1) / sampling_freq;
            
            % Plot padded data (now adjusted), correctly aligned in time
            plot(t_padded, padded_data, 'k--', 'DisplayName', 'Adjusted Padded Data');

            % Plot original data
            plot(t, data, 'b', 'DisplayName', 'Original Data');
            
            % Plot filtered data
            plot(t, filtered_data, 'r', 'LineWidth', 1.5, 'DisplayName', 'Filtered Data');
        end
        
        title('Butterworth Filter Results');
        xlabel('Time (s)');
        ylabel('Amplitude');
        legend;
        grid on;
        hold off;
    end
end