"""
Centralized plot generation for the Option Pricing Engine.

All plotting logic is contained here - no plots elsewhere.
Returns base64-encoded images for web display.
"""

import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Centralized plot styling
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
}

# Color palette for models
MODEL_COLORS = {
    'Black-Scholes': '#2E86AB',
    'Binomial (CRR)': '#A23B72',
    'Monte Carlo': '#F18F01',
}


def apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(PLOT_STYLE)


def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 string for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def plot_price_vs_volatility(
    volatilities: List[float],
    prices: Dict[str, List[float]],
    title: str = "Option Price vs Volatility",
) -> str:
    """
    Plot option price against volatility for multiple models.
    
    Args:
        volatilities: List of volatility values
        prices: Dict mapping model name to list of prices
        title: Plot title
        
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, price_list in prices.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        ax.plot(volatilities, price_list, label=model_name, color=color, marker='o')
    
    ax.set_xlabel('Volatility (σ)')
    ax.set_ylabel('Option Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def plot_price_vs_maturity(
    maturities: List[float],
    prices: Dict[str, List[float]],
    title: str = "Option Price vs Maturity",
) -> str:
    """Plot option price against time to maturity."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, price_list in prices.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        ax.plot(maturities, price_list, label=model_name, color=color, marker='o')
    
    ax.set_xlabel('Time to Maturity (years)')
    ax.set_ylabel('Option Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def plot_convergence(
    parameter_values: List[int],
    errors: List[float],
    model_name: str,
    parameter_name: str,
    baseline_price: float,
) -> str:
    """
    Plot convergence of numerical model to baseline.
    
    Args:
        parameter_values: X-axis values (steps or paths)
        errors: Absolute errors at each point
        model_name: Name of the model
        parameter_name: Name of varying parameter
        baseline_price: Reference price
        
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    color = MODEL_COLORS.get(model_name, '#333333')
    
    # Linear scale
    ax1.plot(parameter_values, errors, color=color, marker='o')
    ax1.set_xlabel(parameter_name.capitalize())
    ax1.set_ylabel('Absolute Error')
    ax1.set_title(f'{model_name} Convergence (Linear)')
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale for convergence rate
    ax2.loglog(parameter_values, errors, color=color, marker='o')
    ax2.set_xlabel(f'{parameter_name.capitalize()} (log scale)')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.set_title(f'{model_name} Convergence (Log-Log)')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Convergence Analysis: {model_name} (Baseline = {baseline_price:.4f})', 
                 fontsize=14, y=1.02)
    fig.tight_layout()
    
    return fig_to_base64(fig)


def plot_model_comparison(
    model_names: List[str],
    prices: List[float],
    errors: List[float],
    baseline_name: str,
) -> str:
    """
    Bar chart comparing model prices and errors.
    
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = [MODEL_COLORS.get(name, '#333333') for name in model_names]
    x = np.arange(len(model_names))
    
    # Prices
    bars1 = ax1.bar(x, prices, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.set_ylabel('Price')
    ax1.set_title('Option Prices by Model')
    ax1.axhline(y=prices[0], color='gray', linestyle='--', alpha=0.5, label=f'{baseline_name} baseline')
    
    # Add value labels
    for bar, price in zip(bars1, prices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(prices),
                f'{price:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Errors
    bars2 = ax2.bar(x, errors, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha='right')
    ax2.set_ylabel('Absolute Error vs Baseline')
    ax2.set_title('Error vs Black-Scholes')
    
    for bar, error in zip(bars2, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(errors) if max(errors) > 0 else 0.001,
                f'{error:.6f}', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    
    return fig_to_base64(fig)


def plot_greeks_comparison(
    greeks_data: Dict[str, Dict[str, float]],
) -> str:
    """
    Compare Greeks across models.
    
    Args:
        greeks_data: Dict mapping model name to dict of Greek values
        
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    
    greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
    model_names = list(greeks_data.keys())
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    
    for i, greek in enumerate(greek_names):
        ax = axes[i]
        values = [greeks_data[model].get(greek, 0) for model in model_names]
        colors = [MODEL_COLORS.get(model, '#333333') for model in model_names]
        
        bars = ax.bar(range(len(model_names)), values, color=colors)
        ax.set_title(greek.capitalize())
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.split()[0] for m in model_names], rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                   fontsize=8)
    
    fig.suptitle('Greeks Comparison Across Models', fontsize=14)
    fig.tight_layout()
    
    return fig_to_base64(fig)


def plot_moneyness_analysis(
    moneyness_values: List[float],
    prices: Dict[str, List[float]],
    option_type: str,
) -> str:
    """
    Plot option price vs moneyness (S/K ratio).
    
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, price_list in prices.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        ax.plot(moneyness_values, price_list, label=model_name, color=color, marker='o')
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM (S=K)')
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Option Price')
    ax.set_title(f'{option_type} Price vs Moneyness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add ITM/OTM annotations
    ax.text(0.75, ax.get_ylim()[1]*0.9, 'OTM' if option_type == 'Call' else 'ITM',
           fontsize=10, alpha=0.5)
    ax.text(1.2, ax.get_ylim()[1]*0.9, 'ITM' if option_type == 'Call' else 'OTM',
           fontsize=10, alpha=0.5)
    
    return fig_to_base64(fig)


def plot_error_heatmap(
    x_values: List[float],
    y_values: List[float],
    errors: List[List[float]],
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    """
    Create heatmap of errors across two parameters.
    
    Returns:
        Base64-encoded PNG image
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(errors, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in x_values])
    ax.set_yticklabels([f'{v:.2f}' for v in y_values])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Error')
    
    fig.tight_layout()
    
    return fig_to_base64(fig)
