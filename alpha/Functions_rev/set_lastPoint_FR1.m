
function processed_gait_data = set_lastPoint_FR1(processed_gait_data)

    field_names = fieldnames(processed_gait_data{1});
    for i = 1:size(processed_gait_data, 2)
        pgd = processed_gait_data{i};
        for j = 1:length(field_names)
             field = field_names{j};
            if contains(field, 'pos', 'IgnoreCase', true) ...
                    && ~contains(field, 'velocity', 'IgnoreCase', true) ...
                    && ~contains(field, 'FR2', 'IgnoreCase', true)
                trajectory = pgd.(field);
                last_pos_xy = trajectory(end, :);
%                 fprintf('Prev-%d: field:%s \t [%2.2f,%2.2f] \n', i, field, trajectory(end, :));
                trajectory = trajectory - last_pos_xy;
%                 fprintf('New-%d: field:%s \t [%2.2f,%2.2f] \n', i, field, trajectory(end, :));
                 pgd.(field) = trajectory;
%                 fprintf('PGD-%d: field:%s \t [%2.2f,%2.2f] \n', i, field, pgd.(field)(end, :));
            end
         end
        processed_gait_data{i} = pgd;
    end
end