 %% plot_joint_cycles: Plots the joint cycles before and after filtering.
%
% Credits:
%   Victor Ferman, Adrolab FEEC/UNICAMP
%
% Description:
%   This function creates a plot with two subplots to show the hip and knee
%   joint angle trajectories. It displays both the original and filtered
%   data for comparison.
%
% Input:
%   reshaped_data - struct: A structure containing the reshaped gait cycle data,
%                    including both raw and filtered trajectories.
%
% Output:
%   None. A figure is generated.

function plot_joint_cycles(reshaped_data)                                                
figure;                                                                           
    subplot(2, 1, 1);
    hold on;
    plot(reshaped_data.right_hip_cycles);
    plot(reshaped_data.filtered.right_hip_cycles);
    plot(reshaped_data.left_hip_cycles);
    plot(reshaped_data.filtered.left_hip_cycles);
    title('Hip Cycles'); 
    legend('Right Hip', 'Filtered Right Hip', 'Left Hip', 'Filtered Left Hip'); 
    hold off;      
    
    subplot(2, 1, 2);
    hold on;
    plot(reshaped_data.right_knee_cycles);
    plot(reshaped_data.filtered.right_knee_cycles);
    plot(reshaped_data.left_knee_cycles);
    plot(reshaped_data.filtered.left_knee_cycles);
    title('Knee Cycles');  
    legend('Right Knee', 'Filtered Right Knee', 'Left Knee', 'Filtered Left Knee'); 
    hold off;       
end