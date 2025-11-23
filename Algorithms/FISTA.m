clear;
clc;
close all;
rng(42); 
fprintf('Setting up Mean-Downside-Risk + L1 (FISTA) problem...\n');

if isfile('return.csv')
    R = readmatrix('return.csv'); 
else
    fprintf('Warning: return.csv not found. Generating dummy data.\n');
    R = randn(1000, 50) * 0.01 + 0.0005; 
end
[T, N] = size(R); % T observations, N assets
mu = mean(R, 1)'; % expected return (N x 1)
fprintf('Data loaded: %d observations, %d assets.\n', T, N);

params.lambda = 1.0;       % weight for downside risk
params.b = 0.0;            % target return
params.alpha_L1 = 5*1e-4;  % L1 penalty coefficient
params.R = R;
params.mu = mu;
params.T = T;
params.N = N;
params.max_epochs = 100;   % Number of iterations
params.tol = 1e-8;         % Convergence tolerance
fprintf('Calculating Global Lipschitz Constant (Spectral Norm)...\n');
L_global = (2 * params.lambda / params.T) * norm(R)^2; 
params.step_size = 1.0 / L_global; % FISTA step size
fprintf('L1 penalty alpha_L1: %e\n', params.alpha_L1);
fprintf('FISTA Global Lipschitz L: %e\n', L_global);
fprintf('FISTA Step Size eta: %e\n', params.step_size);

calc_objective = @(w, p) -w'*p.mu + ...
    (p.lambda/p.T) * sum(max(0, p.b - p.R*w).^2) + ...
    p.alpha_L1 * norm(w, 1);
% --- Proximal Operator: prox_g(z, eta)
prox_operator = @(z, eta, p) sign(z) .* max(0, abs(z) - eta * p.alpha_L1);

w = zeros(N, 1);     % Current solution x_k
w_old = w;           % Previous solution x_{k-1}
y = w;               % Search point (extrapolated) y_k
tk = 1;              % Momentum scalar t_k
eta = params.step_size; % Step size
obj_history = zeros(params.max_epochs, 1);
fprintf('Initialization complete. Starting FISTA Loop...\n');

time = tic;
for k = 1:params.max_epochs
    
    w_prev_iter = w; 
    
    loss_vec = params.b - params.R * y;
    
    active_mask = loss_vec > 0; 
    
    grad_risk = - (2 * params.lambda / params.T) * (params.R' * (loss_vec .* active_mask));
    grad_smooth_y = -params.mu + grad_risk;
    
    w = prox_operator(y - eta * grad_smooth_y, eta, params);
    
    obj_history(k) = calc_objective(w, params);
    
    tk_next = (1 + sqrt(1 + 4 * tk^2)) / 2;
        % y_{k+1} = w_k + ((t_k - 1) / t_{k+1}) * (w_k - w_{k-1})
    momentum_factor = (tk - 1) / tk_next;
    y = w + momentum_factor * (w - w_prev_iter);
    
    tk = tk_next;
    
    if k > 1
        fprintf('Iter: %4d, Obj: %e, Step Norm: %e\n', ...
            k-1, obj_history(k-1), norm(w - w_prev_iter));
    end
    % --- Convergence Check ---
    if norm(w - w_prev_iter) < params.tol * (1 + norm(w_prev_iter))
        fprintf('Convergence reached! Step size tolerance met.\n');
        obj_history = obj_history(1:k);
        break;
    end
    
    if k == params.max_epochs
        fprintf('Warning: Max iterations reached.\n');
    end
end
runtime = toc(time);
fprintf('...FISTA finished.\n');

w_opt = w;
final_obj_fista = obj_history(end);
fprintf('\n--- FISTA Optimization Results ---\n');
fprintf('Total Iterations: %d\n', length(obj_history));
fprintf('Runtime: %.6f s\n', runtime);
fprintf('Final Objective Value: %e\n', final_obj_fista);
fprintf('Non-zero elements in w: %d / %d\n', sum(abs(w_opt) > 1e-12), N);

final_obj_cvx = -0.00579500763990634;

fprintf('\n--- Comparison FISTA vs. CVX ---\n');
fprintf('FISTA Final Obj: %e\n', final_obj_fista);
fprintf('CVX Final Obj:   %e\n', final_obj_cvx);


% --- Plot 1: Convergence of the Objective Function ---
figure('Position',[200 200 800 450]);
plot(obj_history, '->', 'LineWidth', 1.8, 'Color',[0.2 0.3 0.8]);
xlabel('Iteration (k)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Objective Function Value $F(w)$ ', ...
       'Interpreter', 'latex', 'FontSize', 14);
title('Convergence of the FISTA Algorithm', 'FontSize', 16);
grid on;
grid minor;
set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'FontWeight', 'normal', ...
         'LineWidth', 1.2, ...
         'TickLength', [0 0], ...
         'TickLabelInterpreter','tex');
box off;

figure('Position', [150, 150, 1150, 750]);
% w_opt is already defined from FISTA
pos = w_opt;
neg = w_opt;
pos(pos < 0) = 0;
neg(neg > 0) = 0;

hold on;
bar(pos, 'FaceColor',[0.2 0.3 0.8], 'EdgeColor','none'); % Blue for positive
bar(neg, 'FaceColor',[0.9 0.3 0.3], 'EdgeColor','none'); % Red for negative
hold off;

ax = gca;
for k = 1:length(w_opt)
    if abs(w_opt(k)) > 1e-3
        % --- ① Convert data coordinates (bar top) to pixel coordinates ---
        pt = ax.Position;
        % data → normalized
        y_norm = (w_opt(k) - ax.YLim(1)) / diff(ax.YLim);
        x_norm = (k - ax.XLim(1)) / diff(ax.XLim);
        % normalized → pixel
        y_pix = pt(2)*ax.Parent.Position(4) + y_norm*pt(4)*ax.Parent.Position(4);
        
        % --- ② Add a fixed offset in pixel coordinates ---
        y_pix_new = y_pix + 12 * sign(w_opt(k)); % 12 pixels above/below
        
        % --- ③ Convert new pixel coordinate back to data coordinates ---
        y_norm_new = (y_pix_new - pt(2)*ax.Parent.Position(4)) / (pt(4)*ax.Parent.Position(4));
        y_data_new = ax.YLim(1) + y_norm_new * diff(ax.YLim);
        
        % --- ④ Draw the text label ---
        text(k, y_data_new, sprintf('%.2f', w_opt(k)), ...
            'HorizontalAlignment','center', ...
            'FontSize', 9);
    end
end
xlabel('Asset Index');
ylabel('Position');
title('Final Portfolio of the FISTA Algorithm');
grid on;
grid minor;
set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'LineWidth', 1.2, ...
         'FontWeight', 'normal', ...
         'TickLabelInterpreter','tex');
box off;

fprintf('\n--- Final Portfolio Vector (w_opt) ---\n');
if sum(abs(w_opt) > 1e-6) > 0
    idx = find(abs(w_opt) > 1e-6);
    vals = w_opt(idx);
    fprintf('Displaying %d non-zero elements:\n', length(idx));
    T_w_opt = table(idx, vals);
    disp(T_w_opt);
else
    fprintf('The final w_opt vector is all zeros.\n');
end
writematrix(obj_history, 'fista.csv');