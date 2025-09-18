function print_variable_info(var, var_name, max_depth)
    % PRINT_VARIABLE_INFO - Recursively prints all fields and sizes of a variable
    % 
    % Usage: print_variable_info(variable, 'variable_name', max_depth)
    % 
    % Inputs:
    %   var       - The variable to analyze
    %   var_name  - String name of the variable (for display purposes)
    %   max_depth - Maximum recursion depth (optional, default: 5)
    
    if nargin < 2
        var_name = 'variable';
    end
    if nargin < 3
        max_depth = 5;  % Default maximum depth to prevent infinite recursion
    end
    
    fprintf('\n=== Analysis of %s ===\n', var_name);
    fprintf('Class: %s\n', class(var));
    fprintf('Size: [%s]\n', num2str(size(var)));
    fprintf('Number of elements: %d\n', numel(var));
    
    % Start recursive analysis
    analyze_variable_recursive(var, var_name, 0, max_depth);
end

function analyze_variable_recursive(var, var_name, current_depth, max_depth)
    % Recursive function to analyze variable structure
    
    indent = repmat('  ', 1, current_depth);
    
    if current_depth >= max_depth
        fprintf('%s[Max depth reached]\n', indent);
        return;
    end
    
    if isstruct(var)
        analyze_struct_recursive(var, var_name, current_depth, max_depth);
    elseif iscell(var)
        analyze_cell_recursive(var, var_name, current_depth, max_depth);
    elseif isobject(var)
        analyze_object_recursive(var, var_name, current_depth, max_depth);
    else
        % Base case: simple data types
        if current_depth > 0
            fprintf('%s%s: %s, size [%s]', indent, var_name, class(var), num2str(size(var)));
            if ischar(var) || isstring(var)
                if length(var) <= 50
                    fprintf(' = "%s"', char(var));
                else
                    fprintf(' = "%s..."', char(var(1:47)));
                end
            elseif isnumeric(var) && numel(var) <= 5
                fprintf(' = %s', mat2str(var));
            elseif islogical(var) && numel(var) <= 5
                fprintf(' = %s', mat2str(var));
            end
            fprintf('\n');
        end
    end
end

function analyze_struct_recursive(s, var_name, current_depth, max_depth)
    % Recursively analyze structure fields
    
    indent = repmat('  ', 1, current_depth);
    
    if current_depth == 0
        fprintf('\nStructure fields:\n');
    end
    
    field_names = fieldnames(s);
    
    % Handle struct arrays
    if numel(s) > 1
        fprintf('%sStruct array with %d elements, each containing fields:\n', indent, numel(s));
        % Analyze first element as representative
        s = s(1);
        var_name = sprintf('%s(1)', var_name);
        indent = [indent '  '];
    end
    
    for i = 1:length(field_names)
        field_name = field_names{i};
        field_value = s.(field_name);
        full_field_name = sprintf('%s.%s', var_name, field_name);
        
        fprintf('%s%s: %s, size [%s]', indent, full_field_name, ...
            class(field_value), num2str(size(field_value)));
        
        % Add preview for simple types
        if ischar(field_value) || isstring(field_value)
            if length(field_value) <= 30
                fprintf(' = "%s"', char(field_value));
            else
                fprintf(' = "%s..."', char(field_value(1:27)));
            end
        elseif isnumeric(field_value) && numel(field_value) <= 3
            fprintf(' = %s', mat2str(field_value));
        elseif islogical(field_value) && numel(field_value) <= 3
            fprintf(' = %s', mat2str(field_value));
        end
        fprintf('\n');
        
        % Recursively analyze complex fields
        if isstruct(field_value) || iscell(field_value) || isobject(field_value)
            analyze_variable_recursive(field_value, full_field_name, current_depth + 1, max_depth);
        end
    end
end

function analyze_cell_recursive(c, var_name, current_depth, max_depth)
    % Recursively analyze cell array contents
    
    indent = repmat('  ', 1, current_depth);
    
    if current_depth == 0
        fprintf('\nCell array contents:\n');
    end
    
    max_elements_to_show = 10;  % Limit display for large cell arrays
    elements_to_show = min(numel(c), max_elements_to_show);
    
    for i = 1:elements_to_show
        if numel(c) > 1
            [row, col] = ind2sub(size(c), i);
            cell_name = sprintf('%s{%d,%d}', var_name, row, col);
        else
            cell_name = sprintf('%s{1}', var_name);
        end
        
        cell_value = c{i};
        fprintf('%s%s: %s, size [%s]', indent, cell_name, ...
            class(cell_value), num2str(size(cell_value)));
        
        % Add preview for simple types
        if ischar(cell_value) || isstring(cell_value)
            if length(cell_value) <= 30
                fprintf(' = "%s"', char(cell_value));
            else
                fprintf(' = "%s..."', char(cell_value(1:27)));
            end
        elseif isnumeric(cell_value) && numel(cell_value) <= 3
            fprintf(' = %s', mat2str(cell_value));
        end
        fprintf('\n');
        
        % Recursively analyze complex cell contents
        if isstruct(cell_value) || iscell(cell_value) || isobject(cell_value)
            analyze_variable_recursive(cell_value, cell_name, current_depth + 1, max_depth);
        end
    end
    
    if numel(c) > max_elements_to_show
        fprintf('%s... and %d more elements\n', indent, numel(c) - max_elements_to_show);
    end
end

function analyze_object_recursive(obj, var_name, current_depth, max_depth)
    % Recursively analyze object properties
    
    indent = repmat('  ', 1, current_depth);
    
    if current_depth == 0
        fprintf('\nObject properties:\n');
    end
    
    try
        props = properties(obj);
        for i = 1:length(props)
            prop_name = props{i};
            prop_full_name = sprintf('%s.%s', var_name, prop_name);
            
            try
                prop_value = obj.(prop_name);
                fprintf('%s%s: %s, size [%s]', indent, prop_full_name, ...
                    class(prop_value), num2str(size(prop_value)));
                
                % Add preview for simple types
                if ischar(prop_value) || isstring(prop_value)
                    if length(prop_value) <= 30
                        fprintf(' = "%s"', char(prop_value));
                    end
                elseif isnumeric(prop_value) && numel(prop_value) <= 3
                    fprintf(' = %s', mat2str(prop_value));
                end
                fprintf('\n');
                
                % Recursively analyze complex properties
                if isstruct(prop_value) || iscell(prop_value) || isobject(prop_value)
                    analyze_variable_recursive(prop_value, prop_full_name, current_depth + 1, max_depth);
                end
                
            catch
                fprintf('%s%s: (inaccessible)\n', indent, prop_full_name);
            end
        end
    catch
        fprintf('%sProperties not accessible\n', indent);
    end
end

% Example usage with nested structures:
% 
% % Create a complex nested structure
% data.personal.name = 'John';
% data.personal.age = 30;
% data.personal.address.street = '123 Main St';
% data.personal.address.city = 'Boston';
% data.personal.address.coordinates.lat = 42.3601;
% data.personal.address.coordinates.lon = -71.0589;
% data.scores = [85, 92, 78];
% data.metadata.created = datetime('now');
% data.metadata.tags = {'important', 'personal'};
% 
% % Analyze with default depth (5 levels)
% print_variable_info(data, 'data');
% 
% % Analyze with custom depth (3 levels)
% print_variable_info(data, 'data', 3);