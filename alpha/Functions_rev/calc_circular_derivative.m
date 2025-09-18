
%% calc_circular_derivative: Calculates the circular derivative of a vector.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function computes the derivative of a periodic (circular) signal using
%   a central difference scheme. It handles the boundary conditions by wrapping
%   around the ends of the data vector.
%
% Input:
%   data - (1 x N) vector: The input data for which to calculate the derivative.
%   dt   - scalar: The time step between data points.
%
% Output:
%   d    - (1 x N) vector: The calculated derivative of the input data.

function d = calc_circular_derivative(data, dt)
    n = length(data);
    d = zeros(size(data));
    for i = 1:n
        if i == 1
            d(i) = (data(2) - data(end)) / (2*dt);
        elseif i == n
            d(i) = (data(1) - data(end-1)) / (2*dt);
        else
            d(i) = (data(i+1) - data(i-1)) / (2*dt);
        end
    end
end