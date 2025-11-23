%% Comparison of Algorithms on Solution Quality
clc;
cla;
clear;

clear; clc; close all;

%% 1. Load csv
w_logbarrier = load('LogBarrier_solution.csv');
w_pcd = load('PCD_solution.csv');
w_ista = load('ISTA_solution.csv');
w_fista = load('FISTA_solution.csv');
w_saga = load('ProxSAGA_solution.csv');
w_admm = load('ADMM_solution.csv');

%% 2. Plot
figure('Position', [150, 150, 800, 500]);
N = 50; % Number of assets
x_axis = 1:N;
total_width = 0.8;       
num_algos = 6;           
bar_width = total_width / num_algos; 
% color setting
colors = [
    0.00, 0.45, 0.74; % Blue
    0.85, 0.33, 0.10; % Orange
    0.93, 0.69, 0.13; % Yellow
    0.49, 0.18, 0.56; % Purple
    0.47, 0.67, 0.19; % Green
    0.30, 0.75, 0.93  % Cyan
];

bar(x_axis - 2.5*bar_width, w_logbarrier,  bar_width, 'FaceColor', colors(1,:), 'EdgeColor','none'); hold on;
bar(x_axis - 1.5*bar_width, w_pcd,  bar_width, 'FaceColor', colors(2,:), 'EdgeColor','none');
bar(x_axis - 0.5*bar_width, w_ista, bar_width, 'FaceColor', colors(3,:), 'EdgeColor','none');
bar(x_axis + 0.5*bar_width, w_fista,   bar_width, 'FaceColor', colors(4,:), 'EdgeColor','none');
bar(x_axis + 1.5*bar_width, w_saga,   bar_width, 'FaceColor', colors(5,:), 'EdgeColor','none');
bar(x_axis + 2.5*bar_width, w_admm, bar_width, 'FaceColor', colors(6,:), 'EdgeColor','none');

xlabel('Asset Index', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Position', 'Interpreter', 'latex', 'FontSize', 12);
title('Comparison of Algorithms on Solution Quality', 'FontSize', 14);
legend('Log-Barrier', 'PCD', 'ISTA', 'FISTA', 'SAGA', 'ADMM',  'Location', 'northwest'); 
grid on;
xlim([0.5, N + 0.5]);
box off;
hold off;

set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'FontWeight', 'normal', ...
         'LineWidth', 1.2, ...
         'TickLength', [0 0], ...
         'TickLabelInterpreter','tex');
