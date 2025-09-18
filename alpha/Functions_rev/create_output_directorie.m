%% create_output_directories: Creates necessary output directories.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function checks for the existence of 'Plots and Figs' and 'Gait Data'
%   directories. If they do not exist, it creates them.
%
% Input:
%   None.
%
% Output:
%   None. Directories are created in the file system.

function create_output_directorie()
    if ~exist('Plots and Figs', 'dir')
        mkdir('Plots and Figs');
        fprintf('Created directory: Plots and Figs\n');
    end
    
    if ~exist('Gait Data', 'dir')
        mkdir('Gait Data');
        fprintf('Created directory: Gait Data\n');
    end
end