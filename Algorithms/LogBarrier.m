%% Log-Barrier

clc;
clear;
close all;

%% Dataset
R = readmatrix('return.csv'); % historical data
T = size(R,1);   % T observations
N = size(R,2);   % N assets
mu = mean(R,1)'; % expected return

%% Parameters
params.lambda = 1.0;   % weight for downside risk
params.gamma  = 5e-4;  % weight for l1-norm
params.tau    = 0.0;   % target return
params.R = R;
params.mu = mu;
params.T = T;
params.N = N;

%% Log-barrier setup
t0 = 1;              
mu_barrier = 10;     % increasing rate of barrier parameter
eps_outer = 1e-8;    % tolerence for outer loop
eps_inner = 1e-8;    % tolerence for inner loop (Newton's method)

%% Strict initial point z0 = [w0; s0; u0]
w0 = zeros(N, 1);
s0 = ones(T, 1); 
u0 = ones(N, 1);
z0 = [w0; s0; u0];

%% Starting log-barrier algorithm
fprintf('Starting log-barrier algorithm...\n');

time = tic;
[w_opt, s_opt, u_opt, obj_history] = solve_log_barrier(params, z0, t0, mu_barrier, eps_outer, eps_inner);
runtime = toc(time);

fprintf('Log-barrier algorithm is ended.\n');

%% Results
fprintf('Non-zero element of w: %d / %d\n', sum(abs(w_opt) > 1e-4), N);

% figure;
% plot(obj_history, '-k', 'LineWidth', 1);
% xlabel('Index of Outer Loop');
% ylabel('Objective function value $f_0$', 'Interpreter', 'latex');
% title('Convergence of the Log-Barrier Algorithm');
% grid on;
% set(gca, 'FontName', 'Times New Roman', ...
%          'FontSize', 12, ...
%          'FontWeight', 'normal', ...
%          'TickLabelInterpreter','tex');

figure('Position',[200 200 800 450]);

plot(obj_history, '->', 'LineWidth', 1.8, 'Color',[0.2 0.3 0.8]); 
hold on;
plot([1, length(obj_history)], [-0.00579500763990634, -0.00579500763990634], '--r', 'LineWidth', 1.8);
xlabel('Index of Outer Loop', 'FontSize', 14);
ylabel('Objective Function Value', 'FontSize', 14);

title('Convergence of the Log-Barrier Algorithm', 'FontSize', 16);

grid on;
grid minor;             

set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'FontWeight', 'normal', ...
         'LineWidth', 1.2, ...       
         'TickLength', [0 0], ...  
         'TickLabelInterpreter','tex');

box off;                 


fprintf('Runtime: %.6f s\n', runtime);


figure('Position',[200 200 800 450]);
semilogy(abs(obj_history - (-0.00579500763990634)), '->', 'LineWidth', 1.8, 'Color',[0.2 0.3 0.8]);
xlabel('Index of Epoch (Outer Loop)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$|Obj - Obj^{\star}|$ (Optimality gap)', 'Interpreter', 'latex', 'FontSize', 14);
title('Convergence of the Log-Barrier Algorithm', 'FontSize', 16);
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

pos = w_opt;
neg = w_opt;

pos(pos < 0) = 0;
neg(neg > 0) = 0;

hold on;
bar(pos, 'FaceColor',[0.2 0.3 0.8], 'EdgeColor','none'); 
bar(neg, 'FaceColor',[0.9 0.3 0.3], 'EdgeColor','none'); 
hold off;

ax = gca;
for k = 1:length(w_opt)
    if abs(w_opt(k)) > 1e-4
        pt = ax.Position;
        y_norm = (w_opt(k) - ax.YLim(1)) / diff(ax.YLim);
        x_norm = (k - ax.XLim(1)) / diff(ax.XLim);
        x_pix = pt(1)*ax.Parent.Position(3) + x_norm*pt(3)*ax.Parent.Position(3);
        y_pix = pt(2)*ax.Parent.Position(4) + y_norm*pt(4)*ax.Parent.Position(4);
        y_pix_new = y_pix + 12 * sign(w_opt(k));
        y_norm_new = (y_pix_new - pt(2)*ax.Parent.Position(4)) / (pt(4)*ax.Parent.Position(4));
        y_data_new = ax.YLim(1) + y_norm_new * diff(ax.YLim);
        text(k, y_data_new, sprintf('%.2f', w_opt(k)), ...
            'HorizontalAlignment','center', ...
            'FontSize', 9);
    end
end


xlabel('Asset Index');
ylabel('Position');
title('Final Portfolio of the Log-Barrier Algorithm');
grid on;
grid minor;                                    

set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 14, ...
         'LineWidth', 1.2, ...                 
         'FontWeight', 'normal', ...
         'TickLabelInterpreter','tex');

box off;                                      


% filename = 'LogBarrier.csv';
% writematrix(obj_history, filename);


%% Log-barrier solver
function [w_opt, s_opt, u_opt, obj_history] = solve_log_barrier(params, z0, t0, mu_barrier, eps_outer, eps_inner)
    
    R = params.R;
    mu = params.mu;
    tau = params.tau;
    lambda = params.lambda;
    gamma = params.gamma;
    N = params.N;
    T = params.T;

    z = z0;
    t = t0;
    
    obj_history = []; 
    m = 2*N + 2*T; 

    % Outer loop
    j = 0; 
    while m / t > eps_outer 
        
        % Inner loop (Newton's method)
        k = 0; 
        max_inner_iter = 100;
        
        while k < max_inner_iter 
            k = k + 1; 
            
            % Gradient and Hessian
            w = z(1 : N);
            s = z(N + 1 : N + T);
            u = z(N + T + 1 : 2*N + T);
            
            v1 = s - tau + R*w;
            v2 = s;
            v3 = u - w;
            v4 = u + w;
            
            if any(v1 <= 0) || any(v2 <= 0) || any(v3 <= 0) || any(v4 <= 0)
                 fprintf('Warning: Infeasible initial point\n');
                 break;
            end
            
            inv_v1 = 1 ./ v1;
            inv_v2 = 1 ./ v2;
            inv_v3 = 1 ./ v3;
            inv_v4 = 1 ./ v4;
            
            inv_v1_sq = 1 ./ (v1 .^ 2);
            inv_v2_sq = 1 ./ (v2 .^ 2);
            inv_v3_sq = 1 ./ (v3 .^ 2);
            inv_v4_sq = 1 ./ (v4 .^ 2);

            grad_w_f0 = -mu;
            grad_s_f0 = (2 * lambda / T) * s;
            grad_u_f0 = gamma * ones(N, 1);
            
            grad_w_barrier = -R' * inv_v1 + inv_v3 - inv_v4;
            grad_s_barrier = -inv_v1 - inv_v2;
            grad_u_barrier = -inv_v3 - inv_v4;
            
            grad_w = t * grad_w_f0 + grad_w_barrier;
            grad_s = t * grad_s_f0 + grad_s_barrier;
            grad_u = t * grad_u_f0 + grad_u_barrier;
            grad_f = [grad_w; grad_s; grad_u]; 

            D1 = spdiags(inv_v1_sq, 0, T, T);
            D2 = spdiags(inv_v2_sq, 0, T, T);
            D3 = spdiags(inv_v3_sq, 0, N, N);
            D4 = spdiags(inv_v4_sq, 0, N, N);
            
            H_ww_b = R' * D1 * R + D3 + D4;
            H_ss_b = D1 + D2;             
            H_uu_b = D3 + D4;             
            H_ws_b = R' * D1;             
            H_sw_b = H_ws_b'; % D1*R
            
            H_wu_b = -D3 + D4;
            H_uw_b = -D3 + D4; % (\nabla_w(-inv_v3) = -D3, \nabla_w(-inv_v4) = D4)
            
            H_ss_total = t * spdiags((2*lambda/T)*ones(T,1), 0, T, T) + H_ss_b;
            H_uu_total = H_uu_b; % H_f0_u = 0
            
            inv_diag_Hss = 1 ./ diag(H_ss_total);
            inv_diag_Huu = 1 ./ diag(H_uu_total);
            
            H_ww_total = H_ww_b; % H_f0_w = 0
            S_block1 = H_ws_b * (inv_diag_Hss .* H_sw_b);
            S_block2 = H_wu_b * (inv_diag_Huu .* H_uw_b);
            
            H_schur = H_ww_total - S_block1 - S_block2;
            
            g_schur_rhs1 = H_ws_b * (inv_diag_Hss .* grad_s);
            g_schur_rhs2 = H_wu_b * (inv_diag_Huu .* grad_u);
            g_schur = -grad_w + g_schur_rhs1 + g_schur_rhs2;
            
            % Update step in Newton's method
            % regularizer to stablize
            regularizer = 1e-9; 
            delta_w = (H_schur + regularizer*speye(N)) \ g_schur; 
            
            delta_s = -(inv_diag_Hss .* (grad_s + H_sw_b * delta_w));
            delta_u = -(inv_diag_Huu .* (grad_u + H_uw_b * delta_w));
            delta_z = [delta_w; delta_s; delta_u];
            
            decrement = - grad_f' * delta_z;
            
            if (decrement / 2 <= eps_inner)
                break;
            end

            % Backtracing line search
            alpha = 1;      
            alpha_ls = 0.01; 
            beta_ls = 0.5;  
            
            f0 = -w'*mu + (lambda/T)*sum(s.^2) + gamma*sum(u);
            barrier = -sum(log(v1)) - sum(log(v2)) - sum(log(v3)) - sum(log(v4));
            f_z = t * f0 + barrier;
            
            while true
                z_new = z + alpha * delta_z;
                
                w_new = z_new(1 : N);
                s_new = z_new(N + 1 : N + T);
                u_new = z_new(N + T + 1 : 2*N + T);
                
                v1_new = s_new - tau + R*w_new;
                v2_new = s_new;
                v3_new = u_new - w_new;
                v4_new = u_new + w_new;
                
                if any(v1_new <= 0) || any(v2_new <= 0) || any(v3_new <= 0) || any(v4_new <= 0)
                    alpha = beta_ls * alpha;
                    if alpha < 1e-16
                        break; 
                    end
                    continue;
                end

                f0_new = -w_new'*mu + (lambda/T)*sum(s_new.^2) + gamma*sum(u_new);
                barrier_new = -sum(log(v1_new)) - sum(log(v2_new)) - sum(log(v3_new)) - sum(log(v4_new));
                f_z_new = t * f0_new + barrier_new;
                
                % Armijo
                if (f_z_new <= f_z + alpha_ls * alpha * grad_f' * delta_z)
                    break; 
                end
                alpha = beta_ls * alpha;
                
                if alpha < 1e-16
                    break; 
                end
            end
            
            z = z + alpha * delta_z;
            
        end % End inner loop
        
        if k == max_inner_iter
             fprintf('Warning: Outer loop at iteration %d, inner loop stops at iteration %d \n', j+1, max_inner_iter);
        end

    
        beta_final = z(1 : N);
        s_final = z(N + 1 : N + T);
        u_final = z(N + T + 1 : 2*N + T);
        current_obj = -beta_final'*mu + (lambda/T)*sum(s_final.^2) + gamma*sum(u_final);
        obj_history = [obj_history; current_obj];

        t = mu_barrier * t;
        j = j + 1;
       
    end
    
    % Final solution
    w_opt = z(1:N);
    s_opt = z(N+1 : N+T);
    u_opt = z(N+T+1 : 2*N+T);
end