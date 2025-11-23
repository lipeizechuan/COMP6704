%% PCD
clear;
clc;
close all;

%% Dataset
R = readmatrix('return.csv'); % historical data
[T, N] = size(R); % T observations, N assets
mu = mean(R, 1)'; % expected return (N x 1)

%% Parameters
params.lambda = 1.0;   % weight for downside risk
params.gamma  = 5e-4;  % weight for l1-norm
params.tau    = 0.0;   % target return
params.R = R;
params.mu = mu;
params.T = T;
params.N = N;

params.max_iter = 200;   % PCD usually converges fast
params.tol = 1e-4;       % convergence tolerance

% Lipschitz constants L_i
% L_i is an upper bound on the Lipschitz constant of grad_i(f)
% H_ii = (2*lambda/m) * sum(R(:,i).^2 * I(loss>0))
% L_i = (2*lambda/m) * sum(R(:,i).^2) is a safe and constant upper bound
L_vec = (2 * params.lambda / params.T) * sum(params.R.^2, 1)'; % N x 1 vector

%% History
w = zeros(params.N, 1); % initial portfolio positions
history_w = zeros(params.N, params.max_iter);
history_leverage = zeros(params.max_iter, 1); % L1 norm part
history_return = zeros(params.max_iter, 1);   % Return part
history_risk = zeros(params.max_iter, 1);     % Risk part
obj_history = zeros(params.max_iter, 1);

%% Functions
calc_return = @(w, p) -w'*p.mu;
calc_risk = @(w, p) (p.lambda/p.T) * sum(max(0, p.b - p.R*w).^2);

%% Main
res = params.tau - params.R * w;

iter = 0;
for epoch = 1:params.max_iter % epoch is one full pass over all N coordinates
    
    res = params.tau - params.R * w;
    iter = epoch; 
    w_old = w;
    obj_history_old = 0;
    
    % Inner loop
    for i = 1:params.N 
        w_i_old = w(i);
        
        % Calculate partial derivative: 
        %    grad_i = -mu(i) - (2*lambda/m) * (R_i' * max(0, res))
        grad_i = -params.mu(i) - (2*params.lambda/params.T) * (params.R(:,i)' * max(0, res));
        L_i = L_vec(i);
        
        % Calculate u_i (gradient step)
        u_i = w_i_old - (1/L_i) * grad_i;
        
        % Calculate T_i (threshold)
        T_i = params.gamma / L_i;
        
        % Calculate w_i (proximal step / soft-thresholding)
        w(i) = sign(u_i) .* max(0, abs(u_i) - T_i);
        
        % Update the residual term
        w_i_change = w(i) - w_i_old;
        if w_i_change ~= 0
            res = res - params.R(:,i) * w_i_change;
        end
    end
    % Inner loop ends

    % Record history (after each epoch)
    history_w(:, epoch) = w;
    history_leverage(epoch) = sum(abs(w));
    
    current_loss_vec = params.tau - params.R*w;
    history_return(epoch) = -w'*params.mu;
    history_risk(epoch) = 1/params.T * sum(max(0, current_loss_vec).^2);
    obj_history(epoch) = history_return(epoch) + params.lambda * history_risk(epoch) + params.gamma * history_leverage(epoch);
    
    % Print progress
    if mod(epoch, 20) == 0 || epoch == 1
        fprintf('Epoch: %4d, Total Leverage: %f, Return Part: %f, Risk Part: %f\n', ...
                epoch, history_leverage(epoch), history_return(epoch), history_risk(epoch));
    end
    
    % Check for convergence
    if norm(w - w_old) < params.tol
        break;
    end
end

if iter < params.max_iter
    history_w = history_w(:, 1:iter);
    history_leverage = history_leverage(1:iter);
    history_return = history_return(1:iter);
    history_risk = history_risk(1:iter);
end


%% Plot
fprintf('\n--- PCD Optimization Results ---\n');
fprintf('Total Epochs: %d\n', iter);
fprintf('Final Total Leverage: %f\n', history_leverage(end));
fprintf('Final Return Part: %f\n', history_return(end));
fprintf('Final Risk Part: %f\n', history_risk(end));
fprintf('Non-zero elements in w: %d / %d\n', sum(abs(w) > 1e-4), params.N);

obj_history = history_return + params.lambda * history_risk + params.gamma * history_leverage;

figure('Position',[200 200 800 450]);
plot(obj_history, '->', 'LineWidth', 1.8, 'Color',[0.2 0.3 0.8]);
hold on;
plot([1, length(obj_history)], [-0.00579500763990634, -0.00579500763990634], '--r', 'LineWidth', 1.8);
xlabel('Index of Epoch (Outer Loop)', 'FontSize', 14);
ylabel('Objective Function Value', 'FontSize', 14);
title('Convergence of the PCD Algorithm', 'FontSize', 16);
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
w_opt = w;
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
        % Convert data coordinates (bar top) to pixel coordinates
        pt = ax.Position;
        % data → normalized
        y_norm = (w_opt(k) - ax.YLim(1)) / diff(ax.YLim);
        x_norm = (k - ax.XLim(1)) / diff(ax.XLim);
        % normalized → pixel
        y_pix = pt(2)*ax.Parent.Position(4) + y_norm*pt(4)*ax.Parent.Position(4);
        
        % Add a fixed offset in pixel coordinates
        y_pix_new = y_pix + 12 * sign(w_opt(k)); % 12 pixels above/below
        
        % Convert new pixel coordinate back to data coordinates
        y_norm_new = (y_pix_new - pt(2)*ax.Parent.Position(4)) / (pt(4)*ax.Parent.Position(4));
        y_data_new = ax.YLim(1) + y_norm_new * diff(ax.YLim);
        
        % Draw the text label
        text(k, y_data_new, sprintf('%.2f', w_opt(k)), ...
            'HorizontalAlignment','center', ...
            'FontSize', 9);
    end
end
xlabel('Asset Index');
ylabel('Position');
title('Final Portfolio of the PCD Algorithm');
grid on;
grid minor;
set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'LineWidth', 1.2, ...
         'FontWeight', 'normal', ...
         'TickLabelInterpreter','tex');
box off;


