function lengths = read_asf_lengths(asf_file)
% Parses an ASF file to extract the lengths of the leg and hip bones.
% Input: asf_file - path to the .asf file
% Output: lengths - a struct with fields lfemur, ltibia, rfemur, rtibia, lhipjoint, rhipjoint

    lengths = struct('lfemur', 0, 'ltibia', 0, 'rfemur', 0, 'rtibia', 0, 'lhipjoint', 0, 'rhipjoint', 0);
    fid = fopen(asf_file, 'r');
    if fid == -1
        error('Cannot open ASF file: %s', asf_file);
    end

    in_bonedata_section = false;
    current_bone_name = '';
    scale_factor = 1/0.45 * 25.4 / 1000; % Scale to meters

    line = fgetl(fid);
    while ischar(line)
        % Check for section boundaries
        if contains(line, ':bonedata')
            in_bonedata_section = true;
            line = fgetl(fid);
            continue;
        end
        if contains(line, ':hierarchy')
            in_bonedata_section = false;
            break; % We are done with the bonedata section
        end

        if in_bonedata_section
            % Trim leading whitespace for easier parsing
            trimmed_line = strtrim(line);
            
            % Split line into parts
            parts = strsplit(trimmed_line);

            if strcmp(parts{1}, 'name')
                current_bone_name = parts{2};
            elseif strcmp(parts{1}, 'length')
                len = str2double(parts{2});
                scaled_len = len * scale_factor;
                switch current_bone_name
                    case 'lfemur'
                        lengths.lfemur = scaled_len;
                    case 'ltibia'
                        lengths.ltibia = scaled_len;
                    case 'rfemur'
                        lengths.rfemur = scaled_len;
                    case 'rtibia'
                        lengths.rtibia = scaled_len;
                    case 'lhipjoint'
                        lengths.lhipjoint = scaled_len;
                    case 'rhipjoint'
                        lengths.rhipjoint = scaled_len;
                end
            end
        end
        
        line = fgetl(fid);
    end

    fclose(fid);
end