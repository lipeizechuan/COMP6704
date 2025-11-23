# Portfolio Optimization under the Regularized Mean–Downside Risk Framework

This repository supports the group project from **Group 11** for **COMP6704 Advanced Topics in Optimization**. Six optimization algorithms are implemented to solve an unconstrained portfolio optimization problem under the regularized mean–downside risk framework. The performance of different algorithms on solution quality and convergence is compared.

## 🚀 Algorithms Implemented

The following six algorithms have been implemented:

1. **Log-Barrier Method** (MATLAB)
2. **Proximal Coordinate Descent (PCD)** (MATLAB)
3. **Iterative Shrinkage-Thresholding Algorithm (ISTA)** (MATLAB)
4. **Fast ISTA (FISTA)** (MATLAB)
5. **Proximal Stochastic Average Gradient Augmented (Prox-SAGA)** (Python)
6. **Alternating Direction Method of Multipliers (ADMM)** (Python)

## 💻 Environment & Dependencies

### System Configuration

- **Operating System:** Windows 11
- **IDE:** PyCharm (for Python)

### 1. MATLAB Environment

**Used for:** `Log-Barrier`, `PCD`, `ISTA`, `FISTA`, and comparing algorithm performance.

- **Version:** MATLAB R2024b
- **Toolbox:** CVX toolbox for performance baseline

### 2. Python Environment

**Used for:** Fetching historical stock and ETF data from Yahoo Finance,  `Prox-SAGA`, `ADMM`

- **Python Version:** 3.11
- **Required Libraries:** `numpy`,  `matplotlib`, etc.

## 📂 Project Structure

The repository is organized into specific directories for dataset, algorithms, and comparisons:

```
.
├── Algorithms/                     # Source code for the six optimization algorithms
├── Comparison_Convergence(Fig2)/   # Code and results to generate Figure 2 in the report
├── Comparison_Solution(Fig1)/      # Code and results to generate Figure 1 in the report
├── Dataset/                        # Real-world financial data
└── README.md                       # Project documentation
```

## 📊 Dataset Details

The financial data used for this analysis is stored in the `Dataset/` directory.

- **Source:** Data was extracted using the **Yahoo Finance API** (via the `yfinance` library).
- **Scope:** A diverse portfolio of **50 assets** (including Stocks and ETFs).
- **Time Horizon:** January 1, 2022 – November 1, 2025.

### File Descriptions

1. **`prices.csv`**:
   - Contains the daily **Closing Prices** for the selected 50 assets over the specified timeframe.
   - Generated directly via Python script using `yfinance`.
2. **`return.csv`**:
   - Contains the **Simple Returns** calculated based on the daily prices found in `prices.csv`.
   - This dataset serves as the input for the optimization algorithms.

## 📈 Usage & Reproduction

To reproduce the results and figures presented in the final report, please refer to the following guides:

### 1. Running the Algorithms

Please navigate to the `Algorithms/` folder.

### 2. Generating Report Figures

To reproduce Figure 1 in the report, please navigate to the `Comparison_Solution(Fig1)/` folder.

To reproduce Figure 2 in the report, please navigate to the `Comparison_Convergence(Fig2)/` folder.