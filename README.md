### **`README.md`**

# STRAPSim: A Portfolio Similarity Metric for ETF Alignment and Portfolio Trades

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.24151-b31b1b.svg)](https://arxiv.org/abs/2509.24151)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Discipline](https://img.shields.io/badge/Discipline-Quantitative%20Finance-00529B)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Primary Data](https://img.shields.io/badge/Data-Corporate%20Bond%20ETF%20Holdings-lightgrey)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Ground Truth](https://img.shields.io/badge/Ground%20Truth-Monthly%20Total%20Returns-lightgrey)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Core Algorithm](https://img.shields.io/badge/Algorithm-STRAPSim-orange)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Similarity Method](https://img.shields.io/badge/Similarity-Random%20Forest%20Proximity-red)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Evaluation Metric](https://img.shields.io/badge/Evaluation-Spearman%20Rank%20Correlation-yellow)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Baselines](https://img.shields.io/badge/Baselines-Jaccard%20%7C%20BERTScore-blueviolet)](https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Numba](https://img.shields.io/badge/Numba-00A3E0.svg?style=flat&logo=Numba&logoColor=white)](https://numba.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
--

**Repository:** `https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"STRAPSim: A Portfolio Similarity Metric for ETF Alignment and Portfolio Trades"** by:

*   Mingshu Li
*   Dhruv Desai
*   Jerinsh Jeyapaulraj
*   Philip Sommer
*   Riya Jain
*   Peter Chu
*   Dhagash Mehta

The project provides a complete, end-to-end computational framework for replicating the paper's novel portfolio similarity metric, STRAPSim. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and feature engineering, through the training of a supervised similarity model and the implementation of the core STRAPSim algorithm, to the final statistical evaluation and a comprehensive suite of robustness checks.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callables](#key-callables)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "STRAPSim: A Portfolio Similarity Metric for ETF Alignment and Portfolio Trades." The core of this repository is the iPython Notebook `strapsim_portfolio_similarity_metric_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation of the summary evaluation table (Table 4) and a full suite of sensitivity analyses.

The paper introduces **STRAPSim (Semantic, Two-level, Residual-Aware Portfolio Similarity)**, a novel method for measuring the similarity between structured asset baskets like ETFs. It addresses the key limitations of traditional metrics by incorporating learned, constituent-level similarity and a dynamic, weight-aware matching process. This codebase operationalizes the paper's advanced approach, allowing users to:
-   Rigorously validate and cleanse institutional-grade fixed-income portfolio data.
-   Train a Random Forest model to learn a "semantic" similarity metric between individual bonds based on their financial characteristics.
-   Generate a pairwise proximity matrix for all bonds in the universe.
-   Compute the STRAPSim score between any two portfolios using the novel greedy, residual-aware matching algorithm.
-   Benchmark STRAPSim's performance against standard metrics (Jaccard, Weighted Jaccard) and an adapted BERTScore.
-   Evaluate all metrics against a market-based ground truth (return correlation) using Spearman rank correlation.
-   Systematically test the stability of the findings across a wide array of robustness checks.

## Theoretical Background

The implemented methods are grounded in machine learning, algorithm design, and quantitative finance.

**1. Supervised Similarity Learning:**
The foundation of the method is a constituent-level similarity score, $S_{ij}$, between any two bonds, $i$ and $j$. Instead of relying on simple distance metrics, the paper learns this similarity in a supervised fashion. A Random Forest model is trained to predict key financial characteristics of bonds (OAS and Yield) from their features (issuer, maturity, rating, etc.). The "proximity" between two bonds is then defined as the fraction of trees in the forest where they fall into the same terminal leaf node. This creates a nuanced, non-linear similarity measure that is tailored to the financial properties of the assets.
$$
\text{Proximity}(x_i, x_j) = \frac{1}{T} \sum_{t=1}^{T} I[\text{Leaf}_t(x_i) = \text{Leaf}_t(x_j)]
$$

**2. The STRAPSim Algorithm:**
STRAPSim is a greedy, bipartite matching algorithm that computes the similarity between two portfolios, X and Y, defined by their constituents and weights. It iteratively matches the pair of constituents $(x_i, y_j)$ with the highest remaining similarity score $S_{ij}$. The key innovation is its **residual-aware** dynamic: after each match, the weights of the involved constituents are decremented by the amount of weight transferred (the minimum of the two). This prevents a single, high-weight constituent from being "re-used" in multiple matches and ensures the algorithm correctly models the one-to-one nature of portfolio alignment.
$$
\text{STRAPSim}(x, y) = \sum_{i,j=\text{argsort}(S)} S_{ij} \min(w_x^{(t)}(i), w_y^{(t)}(j))
$$
where weights $w^{(t)}$ are updated at each step $t$.

**3. Evaluation via Rank Correlation:**
The performance of STRAPSim and the baseline metrics is evaluated by comparing their ability to rank portfolio pairs in a way that aligns with a market-based measure of similarity. The paper uses the historical monthly return correlation between ETFs as this "ground truth." The primary evaluation metric is the **Spearman Rank Correlation Coefficient ($\rho_s$)**, which measures the monotonic relationship between the ranking produced by a similarity metric and the ranking produced by the return correlations. A higher $\rho_s$ indicates a better alignment with market behavior.

## Features

The provided iPython Notebook (`strapsim_portfolio_similarity_metric_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Phase Architecture:** The entire pipeline is broken down into 28 distinct, modular tasks, each with its own orchestrator function, covering validation, cleansing, feature engineering, modeling, computation, evaluation, and robustness testing.
-   **Configuration-Driven Design:** All methodological and computational parameters are managed in an external `config.yaml` file, allowing for easy customization and scenario testing without code changes.
-   **Advanced Similarity Learning:** A complete pipeline for training a multi-output Random Forest Regressor, including cross-validated hyperparameter tuning, to serve as the engine for the similarity metric.
-   **Optimized Algorithm Implementation:** A high-performance, memory-efficient implementation of the core STRAPSim algorithm and all baseline metrics (Jaccard, Weighted Jaccard, adapted BERTScore with residuals).
-   **Rigorous Evaluation Framework:** A systematic process for transforming similarity and correlation matrices into ranks and computing the Spearman rank correlation and p-values for every method and every reference ETF.
-   **Comprehensive Robustness Suite:** A powerful, extensible set of orchestrators for running a full suite of sensitivity analyses on model hyperparameters, data partitioning, and metric-specific parameters.
-   **Automated Reporting:** Programmatic generation of the final summary table (replicating Table 4 from the paper) and detailed logs for all analysis and robustness check results.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-8):** Ingests and rigorously validates all raw data (holdings, features, returns) and the `config.yaml` file, performs a deep data quality audit, and standardizes all data into an analysis-ready format.
2.  **Feature Engineering (Tasks 9-11):** Prepares the bond feature data for machine learning, including one-hot encoding of categorical variables and max-scaling of numerical variables, and partitions the data into training and testing sets.
3.  **Model Training & Proximity Generation (Tasks 12-14):** Performs a grid search to find optimal Random Forest hyperparameters, trains the final model, validates its performance against the paper's benchmarks, and uses it to generate the crucial N x N bond proximity matrix.
4.  **Similarity Computation (Tasks 15-21):** Prepares the portfolio data structures and then computes the full N x N similarity matrices for STRAPSim and all baseline methods, as well as the ground-truth return correlation matrix.
5.  **Evaluation & Reporting (Tasks 22-24):** Prepares the ranked data, computes the Spearman rank correlation for all methods, and aggregates the results into the final summary table.
6.  **Robustness Analysis (Tasks 25-28):** Orchestrates the entire suite of sensitivity checks.

## Core Components (Notebook Structure)

The `strapsim_portfolio_similarity_metric_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callables

The project is designed around two primary user-facing interface functions:

1.  **`run_strapsim_pipeline` (or its variant `run_strapsim_pipeline_for_robustness`):** This function executes the core research pipeline from end-to-end, producing the main findings of the study and all the necessary data artifacts (e.g., the trained model, the proximity matrix) required for deeper analysis.

2.  **`run_full_study`:** This is the top-level orchestrator. It first calls the main pipeline function to generate the core results and artifacts, and then passes these artifacts to the robustness analysis suite (`run_robustness_analysis_suite`) to perform a complete validation of the study's conclusions. A single call to this function reproduces the entire project.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `tqdm`, `numba`, `jsonschema`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric.git
    cd strapsim_portfolio_similarity_metric
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires three `pandas.DataFrame`s and a `config.yaml` file. The schemas for these DataFrames are rigorously defined and validated by the pipeline.
-   **ETF Holdings DataFrame:** Contains portfolio composition data, including `etf_id`, `cusip`, and `weight`.
-   **Bond Features DataFrame:** The security master file, containing financial characteristics for every bond, indexed by `cusip`.
-   **Monthly Returns DataFrame:** A time-series DataFrame with a `date` column and one column for each ETF's monthly total returns.

## Usage

The `strapsim_portfolio_similarity_metric_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to call the top-level orchestrator:

```python
# This single call runs the entire project, including the main analysis
# and all robustness checks.
full_results = run_full_study(
    etf_holdings_df=my_holdings_data,
    bond_features_df=my_features_data,
    monthly_returns_df=my_returns_data,
    config=my_config_dict
)
```

## Output Structure

The `run_full_study` function returns a comprehensive nested dictionary containing all results and artifacts:

```
{
    "main_analysis_artifacts": {
        "config": {...},
        "trained_model": <RandomForestRegressor object>,
        "proximity_matrix_df": <DataFrame>,
        "similarity_matrices": {
            "STRAPSim": <DataFrame>,
            "Jaccard": <DataFrame>,
            ...
        },
        "final_summary_table": <DataFrame>,
        ...
    },
    "robustness_analysis_results": {
        "hyperparameter_sensitivity": <DataFrame>,
        "data_split_sensitivity": <DataFrame>,
        "metric_component_sensitivity": <DataFrame>
    }
}
```

## Project Structure

```
strapsim_portfolio_similarity_metric/
│
├── strapsim_portfolio_similarity_metric_draft.ipynb   # Main implementation notebook
├── config.yaml                                        # Master configuration file
├── requirements.txt                                   # Python package dependencies
├── LICENSE                                            # MIT license file
└── README.md                                          # This documentation file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all methodological parameters, such as data paths, feature definitions, model hyperparameters, and evaluation settings, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Similarity Models:** Implementing other supervised models (e.g., Gradient Boosting, Neural Networks) to generate the constituent-level proximity matrix.
-   **Optimal Transport Baselines:** Adding Wasserstein distance (Optimal Transport) as a more advanced baseline for comparing weighted distributions.
-   **Performance Optimization:** For extremely large universes, the STRAPSim algorithm could be further accelerated using more advanced data structures or a compiled language extension.
-   **Application Modules:** Building specific application modules on top of the STRAPSim score, such as a portfolio recommendation engine or a tool for optimizing portfolio trades.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{li2025strapsim,
  title={{STRAPSim: A Portfolio Similarity Metric for ETF Alignment and Portfolio Trades}},
  author={Li, Mingshu and Desai, Dhruv and Jeyapaulraj, Jerinsh and Sommer, Philip and Jain, Riya and Chu, Peter and Mehta, Dhagash},
  journal={arXiv preprint arXiv:2509.24151},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Professional-Grade Implementation of the STRAPSim Framework.
GitHub repository: https://github.com/chirindaopensource/strapsim_portfolio_similarity_metric
```

## Acknowledgments

-   Credit to **Mingshu Li, Dhruv Desai, Jerinsh Jeyapaulraj, Philip Sommer, Riya Jain, Peter Chu, and Dhagash Mehta** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Scikit-learn, Numba, and Jupyter**, whose work makes complex computational analysis accessible and robust.

--

*This README was generated based on the structure and content of `strapsim_portfolio_similarity_metric_draft.ipynb` and follows best practices for research software documentation.*
