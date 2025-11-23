import numpy as np
import matplotlib.pyplot as plt
import time
import yfinance as yf

# Soft thresholding (L1 proximal operato
def soft_thresh(z, T):
    return np.sign(z) * np.maximum(np.abs(z) - T, 0.0)



# Gradient of smooth part f(w)
# f(w) = -mu^T w + (λ/m) * Σ max(0, b - Rw)^2
# grad f(w) = -mu - (2 λ/m) * R' * max(0, b - Rw)

def calc_grad(w, mu, R, lambda_, b):
    m = R.shape[0]
    Rw = R @ w
    diff = b - Rw
    mask = (diff > 0)
    grad = -mu - (2 * lambda_ / m) * (R[mask].T @ diff[mask])
    return grad


# Smooth w-subproblem: minimize F(w) + (ρ/2)||w - v||²
# Gradient = grad_F(w) + ρ(w - v)

def calc_w_grad_subproblem(w, v, mu, R, lambda_, b, rho):
    return calc_grad(w, mu, R, lambda_, b) + rho * (w - v)


# Return part: -mu^T w
# Risk part:   (λ/m) Σ max(0, b - Rw)^2
# L1 penalty:  α_L1 * ||w||₁

def calc_return(w, mu):
    return -w @ mu


def calc_risk(w, R, lambda_, b):
    m = R.shape[0]
    diff = b - R @ w
    diff = np.maximum(diff, 0)
    return (lambda_ / m) * np.sum(diff ** 2)


def calc_l1_penalty(w, alpha_L1):
    return alpha_L1 * np.sum(np.abs(w))



# Total_objective = Return + Risk + L1 Penalty

def calc_total_objective(w, mu, R, lambda_, b, alpha_L1):
    return_part = calc_return(w, mu)
    risk_part = calc_risk(w, R, lambda_, b)
    l1_part = calc_l1_penalty(w, alpha_L1)
    return return_part + risk_part + l1_part



# Main ADMM function

def admm_l1(mu, R, lambda_=1.0, b=0.0, alpha_L1=5e-4,
            rho=0.01, max_iter=1000, tol_primal=1e-7, tol_dual=1e-7,
            w_inner_steps=10, verbose=True):
    T, n = R.shape

    # Initialize variables
    w = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # Compute Lipschitz constant
    L_F = (2 * lambda_ / T) * np.linalg.norm(R.T @ R)
    alpha_inner = 1.0 / (L_F + rho)

    # History arrays
    history_w = np.zeros((max_iter, n))
    history_leverage = np.zeros(max_iter)
    history_return = np.zeros(max_iter)
    history_risk = np.zeros(max_iter)
    history_l1_penalty = np.zeros(max_iter)
    history_total_objective = np.zeros(max_iter)
    history_primal_residual = np.zeros(max_iter)
    history_dual_residual = np.zeros(max_iter)

    if verbose:
        print("Starting ADMM optimization...")
        print(f"L1 penalty alpha_L1 = {alpha_L1:e}")
        print(f"Expected return |mu| mean = {np.mean(np.abs(mu))}")
        print(f"ADMM rho = {rho}")

    # ADMM main loop
    for it in range(max_iter):

        z_old = z.copy()

        # w-update
        v_w = z - u
        for _ in range(w_inner_steps):
            grad_w = calc_w_grad_subproblem(w, v_w, mu, R, lambda_, b, rho)
            w -= alpha_inner * grad_w

        #z-update
        v_z = w + u
        threshold = alpha_L1 / rho
        z = soft_thresh(v_z, threshold)

        #u-update
        u = u + (w - z)

        #History
        history_w[it] = w
        history_leverage[it] = np.sum(np.abs(w))
        history_return[it] = calc_return(w, mu)
        history_risk[it] = calc_risk(w, R, lambda_, b)
        history_l1_penalty[it] = calc_l1_penalty(w, alpha_L1) 
        history_total_objective[it] = calc_total_objective(w, mu, R, lambda_, b, alpha_L1)  

        #Progress print
        r_primal = np.linalg.norm(w - z)
        r_dual = np.linalg.norm(rho * (z - z_old))

        history_primal_residual[it] = r_primal
        history_dual_residual[it] = r_dual

        if verbose:
            total_obj = history_total_objective[it]
            print(f"Iter {it:4d}, "
                  f"Objective={total_obj:.6f}, "
                  f"Leverage={history_leverage[it]:.4f}, "
                  f"Return={history_return[it]:.6f}, "
                  f"Risk={history_risk[it]:.6f}, "
                  f"L1={history_l1_penalty[it]:.6f}, "
                  f"Res_P={r_primal:.3e}, "
                  f"Res_D={r_dual:.3e}")

        #check convergence
        if r_primal < tol_primal and r_dual < tol_dual:
            print("ADMM converged!")
            break

    # Trim histories
    history_w = history_w[:it + 1]
    history_leverage = history_leverage[:it + 1]
    history_return = history_return[:it + 1]
    history_risk = history_risk[:it + 1]
    history_l1_penalty = history_l1_penalty[:it + 1]
    history_total_objective = history_total_objective[:it + 1]
    history_primal_residual = history_primal_residual[:it + 1]
    history_dual_residual = history_dual_residual[:it + 1]

    print("Optimization finished.")
    print(f"Total iterations: {it + 1}")
    print(f"Final objective value: {history_total_objective[-1]:.12f}")
    print(f"Final leverage: {history_leverage[-1]}")
    print(f"Final return part: {abs(history_return[-1])}")
    print(f"Final risk part: {history_risk[-1]}")
    print(f"Final L1 penalty: {history_l1_penalty[-1]}")
    print(f"Non-zero w: {np.sum(w != 0)} / {n}")
    print(f"Final primal residual: {history_primal_residual[-1]:.3e}")
    print(f"Final dual residual: {history_dual_residual[-1]:.3e}")
    print(w)
    # CSV
    np.savetxt('history_total_objective.csv', history_total_objective, delimiter=',', fmt='%.8f')

    print(f"Saved to history_total_objective.csv, {len(history_total_objective)}")

    return (w, history_w, history_leverage, history_return, history_risk,
            history_l1_penalty, history_total_objective,
            history_primal_residual, history_dual_residual)


def plot_admm_results(w, history_leverage, history_return, history_risk,
                      history_l1_penalty, history_total_objective,
                      history_primal_residual, history_dual_residual):
    n = len(w)

    #Final w
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(n), w)
    plt.title("Final Portfolio Weights (w)")
    plt.xlabel("Asset Index")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()

    eps = 1e-12

    plt.figure(figsize=(15, 10))

    # plt.subplot(3, 1, 1)
    # plt.semilogy(history_total_objective - -0.00579500335344598 + eps,
    #              label='Total Objective Convergence', linewidth=2, color='red')
    # plt.ylim(-1e-5, 1e-2)
    # plt.title("Total Objective Function Convergence")
    # plt.xlabel("Iteration")
    # plt.ylabel("|Objective - Objective*|")
    # plt.legend()
    # plt.grid(True)
    plt.subplot(3, 1, 1)
    plt.plot((history_total_objective - -0.00579500335344598) * 1000, 
             label='Total Objective Convergence', linewidth=2, color='red')
    plt.ylim(0, 6)  
    plt.title("Total Objective Function Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Objective - Objective* (×10⁻³)")  
    plt.legend()
    plt.grid(True)


    plt.subplot(3, 1, 2)
    plt.semilogy(np.abs(history_leverage - history_leverage[-1]) + eps,
                 label='Leverage Convergence', linewidth=2)
    plt.semilogy(np.abs(history_return - history_return[-1]) + eps,
                 label='Return Convergence', linewidth=2)
    plt.semilogy(np.abs(history_risk - history_risk[-1]) + eps,
                 label='Risk Convergence', linewidth=2)
    plt.semilogy(np.abs(history_l1_penalty - history_l1_penalty[-1]) + eps,
                 label='L1 Penalty Convergence', linewidth=2)
    plt.title("Component Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True)


    plt.subplot(3, 1, 3)
    plt.semilogy(history_primal_residual + eps, label='Primal Residual', linewidth=2)
    plt.semilogy(history_dual_residual + eps, label='Dual Residual', linewidth=2)
    plt.axhline(y=1e-7, color='r', linestyle='--', label='Tolerance (1e-7)', alpha=0.7)
    plt.title("ADMM Residual Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12, 8))


    plt.subplot(2, 2, 1)
    plt.plot(history_total_objective, linewidth=2)
    plt.title("Total Objective Function Value")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.grid(True)


    plt.subplot(2, 2, 2)
    plt.plot(history_return, label='Return Part', linewidth=2)
    plt.plot(history_risk, label='Risk Part', linewidth=2)
    plt.plot(history_l1_penalty, label='L1 Penalty', linewidth=2)
    plt.title("Objective Function Components")
    plt.xlabel("Iteration")
    plt.ylabel("Component Value")
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 2, 3)
    total = np.abs(history_return) + history_risk + history_l1_penalty
    plt.plot(np.abs(history_return) / total, label='Return Ratio', linewidth=2)
    plt.plot(history_risk / total, label='Risk Ratio', linewidth=2)
    plt.plot(history_l1_penalty / total, label='L1 Ratio', linewidth=2)
    plt.title("Component Ratios")
    plt.xlabel("Iteration")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 2, 4)
    plt.plot(history_leverage, linewidth=2, color='purple')
    plt.title("Total Leverage")
    plt.xlabel("Iteration")
    plt.ylabel("Leverage")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    #print w
    print("\n=== Convergence Residuals ===")
    print(f"Primal residual vector (length {len(history_primal_residual)}):")
    print(history_primal_residual)
    print(f"\nDual residual vector (length {len(history_dual_residual)}):")
    print(history_dual_residual)

    print(f"\nTotal objective function values:")
    print(history_total_objective)


tickers = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS',
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'INTC', 'AMD',
    'WMT', 'COST', 'TGT', 'HD', 'MCD', 'SBUX', 'NKE', 'KO', 'PEP', 'PG',
    'XOM', 'CVX', 'GE', 'BA', 'CAT', 'F', 'GM', 'HON', 'LMT', 'MMM',
    'JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'UNH', 'MDT', 'AMGN', 'BMY', 'CVS',
    'SPY', 'QQQ', 'DIA', 'IWM'
]
data = yf.download(tickers, start="2022-01-01", end="2025-11-01")
data = data.xs('Close', axis=1, level='Price')
prices = data.values
returns = (prices[1:] - prices[:-1]) / prices[:-1]

R = returns  # shape (T-1, N)

mu = R.mean(axis=0)

(w, history_w, history_leverage, history_return, history_risk,
 history_l1_penalty, history_total_objective,
 history_primal_residual, history_dual_residual) = admm_l1(
    mu=mu,
    R=R,
    lambda_=1.0,
    b=0.0,
    alpha_L1=5e-4,
    rho=0.01,
    max_iter=1000
)

plot_admm_results(w, history_leverage, history_return, history_risk,
                  history_l1_penalty, history_total_objective,
                  history_primal_residual, history_dual_residual)