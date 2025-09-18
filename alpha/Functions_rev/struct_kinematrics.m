function Data = struct_kinematrics(linear_kinematics, N_Samples)
   % Count the leg with less gait cycles 
    num_cycles = size(linear_kinematics, 2);
    Data = cell(1, num_cycles);

    idx = 1;
    for i = 1:N_Samples
        A = zeros(4 , size(linear_kinematics{1}.pos, 2));
        A(1:2, :) = linear_kinematics{1}.pos;
        A(3:4, :) = linear_kinematics{1}.vel;
        Data{idx}=A;
        idx = idx +1;
    end
   
end