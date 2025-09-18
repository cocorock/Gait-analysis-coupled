
% Functions_rev/calculate_velocity.m
function velocity = calculate_velocity(position_data, dt, reduction_factor)
    % CALCULATE_VELOCITY Calculates velocity from position data.
    %   velocity = CALCULATE_VELOCITY(position_data, dt, reduction_factor) calculates the
    %   velocity. It uses a 5-point stencil for interior points for better
    %   accuracy and noise reduction, and simpler differences for the boundaries.
    %
    %   Inputs:
    %     position_data: NxM matrix of position data, where N is the number
    %                    of samples and M is the number of dimensions.
    %     dt: Time step between samples (1/sampling_frequency).
    %     reduction_factor: Optional. A factor to deliberately reduce the
    %                       calculated velocity. Defaults to 1 (no reduction).
    %
    %   Output:
    %     velocity: NxM matrix of calculated velocities.

    if nargin < 2
        error('Not enough input arguments. Usage: calculate_velocity(position_data, dt)');
    end
    if nargin < 3
        reduction_factor = 1.0;
    end

    num_samples = size(position_data, 1);
    num_dimensions = size(position_data, 2);
    velocity = zeros(num_samples, num_dimensions);

    if num_samples < 5
        % Fallback to simpler method for short signals
        warning('Signal has fewer than 5 samples, falling back to simpler differentiation.');
        for i = 1:num_samples
            if i == 1
                % Forward difference for first element
                velocity(i, :) = (position_data(i+1, :) - position_data(i, :)) / dt;
            elseif i == num_samples
                % Backward difference for last element
                velocity(i, :) = (position_data(i, :) - position_data(i-1, :)) / dt;
            else
                % Central difference for interior points
                velocity(i, :) = (position_data(i+1, :) - position_data(i-1, :)) / (2*dt);
            end
        end
        velocity = velocity * reduction_factor; % Apply reduction factor
        return;
    end

    % Forward difference for the first point
    velocity(1, :) = (position_data(2, :) - position_data(1, :)) / dt;

    % Central difference for the second point
    velocity(2, :) = (position_data(3, :) - position_data(1, :)) / (2*dt);

    % 5-point stencil for interior points
    for i = 3:num_samples-2
        velocity(i, :) = (position_data(i-2, :) - 8*position_data(i-1, :) + 8*position_data(i+1, :) - position_data(i+2, :)) / (12*dt);
    end

    % Central difference for the second to last point
    velocity(num_samples-1, :) = (position_data(num_samples, :) - position_data(num_samples-2, :)) / (2*dt);

    % Backward difference for the last point
    velocity(num_samples, :) = (position_data(num_samples, :) - position_data(num_samples-1, :)) / dt;

    % Apply the reduction factor to the final velocity calculation
    velocity = velocity * reduction_factor;
end
