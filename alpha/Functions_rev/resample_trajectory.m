
% Functions_rev/resample_trajectory.m
function resampled_data = resample_trajectory(original_data, original_time, target_points)
    % RESAMPLE_TRAJECTORY Resamples a trajectory to a target number of points
    %   with time normalized from 0 to 1.
    %   resampled_data = RESAMPLE_TRAJECTORY(original_data, original_time, target_points)
    %   uses linear interpolation to resample the original_data.
    %
    %   Inputs:
    %     original_data: NxM matrix of original data, where N is the number
    %                    of samples and M is the number of dimensions.
    %     original_time: Nx1 vector of original time points.
    %     target_points: Desired number of points for the resampled data.
    %
    %   Output:
    %     resampled_data: target_points x M matrix of resampled data.

    if nargin < 3
        error('Not enough input arguments. Usage: resample_trajectory(original_data, original_time, target_points)');
    end

    % Normalize original time to [0, 1]
    normalized_original_time = (original_time - min(original_time)) / (max(original_time) - min(original_time));

    % Create new normalized time vector for resampling
    new_normalized_time = linspace(0, 1, target_points)';

    % Resample each column (dimension) of the data
    num_dimensions = size(original_data, 2);
    resampled_data = zeros(target_points, num_dimensions);

    for dim = 1:num_dimensions
        resampled_data(:, dim) = interp1(normalized_original_time, original_data(:, dim), new_normalized_time, 'linear');
    end
end
