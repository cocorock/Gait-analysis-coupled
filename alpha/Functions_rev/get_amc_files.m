%% get_amc_files: Finds all AMC files in the specified directory.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function searches the specified subdirectory for all files with the .amc
%   extension and returns a structure array of the found files.
%
% Input:
%   directory - (optional) string: The directory to search in. Defaults to 'AMC/'.
%
% Output:
%   amc_files - struct array: A structure array containing information about the AMC files.

function amc_files = get_amc_files(directory)
    if nargin < 1
        directory = 'AMC/';
    end
    fprintf('Searching for AMC files in %s folder...\n', directory);
    amc_files = dir([directory '*.amc']);
    
    if isempty(amc_files)
        error('No AMC files found in %s directory!', directory);
    end
    
    fprintf('Found %d AMC files:\n', length(amc_files));
    for i = 1:length(amc_files)
        fprintf('  %d. %s\n', i, amc_files(i).name);
    end
end