%% save_linear_kinematics: Saves the calculated linear kinematics data.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function saves the linear kinematics data (end-effector position,
%   velocity, and acceleration) to a .mat file in the 'Gait Data' directory.
%
% Input:
%   linear_kinematics - (1 x N_cycles) cell array: The linear kinematics data.
%
% Output:
%   None. A .mat file is saved.

function save_linear_kinematics(linear_kinematics, N_Samples)

    data = cell(1, N_Samples);
    for i = 1:N_Samples
        data{i} = linear_kinematics{i};
    end
    % Save in same structure as demos (cell array of structs with pos, vel, acc)
    filename = sprintf('Gait Data/linear_kinematics_gait_cycles_%s_samples.mat', string(N_Samples)); % , datestr(now, 'yyyymmdd_HHMMSS')
    save(filename, 'data');
    fprintf('Linear kinematics gait cycles saved as: %s\n', filename);
end
