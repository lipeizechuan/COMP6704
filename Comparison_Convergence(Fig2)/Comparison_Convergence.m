%% Comparison of Algorithms on Convergence Speed

clc;
cla;
clear;

%% 1. Load csv
obj_logbarrier = load('LogBarrier.csv');
obj_pcd = load('PCD.csv');
obj_ista = load('ISTA.csv');
obj_fista = load('FISTA.csv');
obj_saga = load('ProxSAGA.csv');
obj_admm = load('ADMM.csv');


optval = -0.00579500763990634;

raw_data = {obj_logbarrier, obj_pcd, obj_ista, obj_fista, obj_saga, obj_admm};
Algo_Names = {'Log-Barrier', 'PCD', 'ISTA', 'FISTA', 'Prox-SAGA', 'ADMM'};

% color setting
colors = [
    0.00, 0.45, 0.74; % Blue
    0.85, 0.33, 0.10; % Orange
    0.93, 0.69, 0.13; % Yellow
    0.49, 0.18, 0.56; % Purple
    0.47, 0.67, 0.19; % Green
    0.30, 0.75, 0.93  % Cyan
];

%% 2. Padding to target_len
target_len = 100;
num_algos = length(raw_data);
padded_data = zeros(target_len, num_algos); 

for i = 1:num_algos
    vec = raw_data{i};
    current_len = length(vec);
    
    if current_len < target_len
        last_val = vec(end);
        padding = repmat(last_val, target_len - current_len, 1);
        new_vec = [vec; padding];
    elseif current_len > target_len
        new_vec = vec(1:target_len);
    else
        new_vec = vec;
    end
    
    padded_data(:, i) = new_vec;
end


%% 3. Plot
figure('Position', [150, 150, 800, 500]);

% CVX baseline
x_axis = 1:target_len;

for i = 1:num_algos
    % optimality gap
    error_vec = abs(padded_data(:, i) - optval);
    error_vec = max(error_vec, 1e-16); 
    
    semilogy(x_axis, error_vec, ...
        'LineWidth', 2, ...
        'Color', colors(i, :), ...
        'DisplayName', Algo_Names{i});
    hold on;
end

xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$|Obj - Obj^{\star}|$ (Optimality gap)', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of Algorithms on Convergence Rate', 'FontSize', 14);
legend('show', 'Location', 'northeast'); 
grid on; grid minor;

set(gca, 'YScale', 'log'); 
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter','tex');
box off;
xlim([1 target_len]);

set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'FontWeight', 'normal', ...
         'LineWidth', 1.2, ...
         'TickLength', [0 0], ...
         'TickLabelInterpreter','tex');