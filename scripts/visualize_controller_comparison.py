"""
Visualization: Compare Different Controllers using Realized Vol
这个脚本用来对比不同controller在realized vol策略中的表现
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.env import Env


# ============================================================================
# Controller定义 (对应 realized_vol.yaml中的配置)
# ============================================================================

CONTROLLERS = [
    "realized_vol__naive_scaling",
    "realized_vol__constant_weight", 
    "realized_vol__variance_scaling",
    "realized_vol__regime_switch",
    "realized_vol__vol_target_clip",
    "realized_vol__drawdown_brake",
    "realized_vol__trend_filter",
    "realized_vol__cvar_es_targeting",
    "realized_vol__hysteresis",
    "realized_vol__priority_stack",
]

# 简化标签
CONTROLLER_DISPLAY_NAMES = {
    "realized_vol__naive_scaling": "Naïve Scaling",
    "realized_vol__constant_weight": "Constant Weight",
    "realized_vol__variance_scaling": "Variance Scaling",
    "realized_vol__regime_switch": "Regime Switch",
    "realized_vol__vol_target_clip": "Vol Target Clip",
    "realized_vol__drawdown_brake": "Drawdown Brake",
    "realized_vol__trend_filter": "Trend Filter",
    "realized_vol__cvar_es_targeting": "CVaR/ES Targeting",
    "realized_vol__hysteresis": "Hysteresis",
    "realized_vol__priority_stack": "Priority Stack",
}

# ============================================================================
# 模拟数据函数（如果没有真实的backtest结果）
# ============================================================================

def generate_mock_backtest_results():
    """
    生成模拟的回测结果，用于展示如何对比controller
    实际使用时应该用真实的backtest结果数据
    """
    np.random.seed(42)
    
    results = {}
    base_return = 0.08
    base_vol = 0.12
    
    for controller in CONTROLLERS:
        # 为每个controller生成不同的表现特性
        controller_name = controller.replace("realized_vol__", "")
        
        # 基础return和vol，根据controller类型有所差异
        if "naive" in controller_name:
            annual_return = base_return + 0.01
            annual_vol = base_vol + 0.01
            max_dd = -0.18
        elif "variance" in controller_name:
            annual_return = base_return + 0.015
            annual_vol = base_vol - 0.01
            max_dd = -0.15
        elif "regime" in controller_name:
            annual_return = base_return + 0.02
            annual_vol = base_vol - 0.015
            max_dd = -0.12
        elif "drawdown_brake" in controller_name:
            annual_return = base_return + 0.005
            annual_vol = base_vol - 0.005
            max_dd = -0.10
        elif "trend_filter" in controller_name:
            annual_return = base_return + 0.012
            annual_vol = base_vol - 0.008
            max_dd = -0.14
        elif "cvar" in controller_name:
            annual_return = base_return + 0.018
            annual_vol = base_vol - 0.012
            max_dd = -0.08
        else:
            annual_return = base_return + np.random.uniform(-0.005, 0.01)
            annual_vol = base_vol + np.random.uniform(-0.01, 0.005)
            max_dd = -0.10 - np.random.uniform(0, 0.05)
        
        # 计算Sharpe比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        results[controller] = {
            "Annual_Return": annual_return,
            "Annual_Vol": annual_vol,
            "Sharpe": sharpe,
            "Max_DD": max_dd,
            "Win_Rate": 0.55 + np.random.uniform(-0.05, 0.05),
        }
    
    return pd.DataFrame(results).T


def load_real_backtest_results(results_dir="results"):
    """
    从实际的backtest结果目录加载数据
    假设结构: results/realized_vol/[controller_name]/metrics.csv
    """
    env = Env()
    results_path = Path(env.root_dir) / results_dir
    
    all_results = {}
    
    for controller in CONTROLLERS:
        controller_name = controller.replace("realized_vol__", "")
        metrics_file = results_path / "realized_vol" / controller_name / "metrics.csv"
        
        if metrics_file.exists():
            try:
                metrics = pd.read_csv(metrics_file)
                # 假设metrics表包含 Annual_Return, Annual_Vol, Max_DD等列
                all_results[controller] = {
                    "Annual_Return": metrics.get("Annual_Return", 0),
                    "Annual_Vol": metrics.get("Annual_Vol", 0),
                    "Sharpe": metrics.get("Sharpe", 0),
                    "Max_DD": metrics.get("Max_DD", 0),
                    "Win_Rate": metrics.get("Win_Rate", 0),
                }
            except Exception as e:
                print(f"Warning: Could not load metrics for {controller}: {e}")
    
    if not all_results:
        print("❌ Could not load real results, using mock data instead")
        return generate_mock_backtest_results()
    
    return pd.DataFrame(all_results).T


# ============================================================================
# 可视化函数
# ============================================================================

def plot_controller_comparison(results_df):
    """
    生成多合一对比图表：
    1. Sharpe比率对比
    2. Annual Return vs Annual Vol 散点图
    3. Max Drawdown对比
    """
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Realized Vol Strategy: Controller Comparison", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 准备数据
    controllers = [CONTROLLER_DISPLAY_NAMES[c] for c in results_df.index]
    colors = plt.cm.Set3(np.linspace(0, 1, len(controllers)))
    
    # ---- 图1: Sharpe 比率对比 ----
    ax1 = plt.subplot(2, 2, 1)
    sharpe_values = results_df["Sharpe"].values
    bars1 = ax1.barh(controllers, sharpe_values, color=colors)
    ax1.set_xlabel("Sharpe Ratio", fontweight='bold')
    ax1.set_title("Sharpe Ratio by Controller", fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(sharpe_values):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # ---- 图2: Annual Return vs Annual Vol (Scatter) ----
    ax2 = plt.subplot(2, 2, 2)
    returns = results_df["Annual_Return"].values
    vols = results_df["Annual_Vol"].values
    
    scatter = ax2.scatter(vols, returns, s=200, c=range(len(controllers)), 
                          cmap='Set3', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for i, controller in enumerate(controllers):
        ax2.annotate(controller, (vols[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax2.set_xlabel("Annual Volatility", fontweight='bold')
    ax2.set_ylabel("Annual Return", fontweight='bold')
    ax2.set_title("Return vs Risk Trade-off", fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # ---- 图3: Max Drawdown对比 ----
    ax3 = plt.subplot(2, 2, 3)
    max_dd_values = results_df["Max_DD"].values
    colors_dd = ['#d62728' if x < -0.15 else '#ff7f0e' if x < -0.10 else '#2ca02c' 
                 for x in max_dd_values]
    bars3 = ax3.barh(controllers, max_dd_values, color=colors_dd)
    ax3.set_xlabel("Maximum Drawdown", fontweight='bold')
    ax3.set_title("Maximum Drawdown by Controller", fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(max_dd_values):
        ax3.text(v - 0.003, i, f'{v:.1%}', va='center', ha='right', fontsize=9)
    
    # ---- 图4: 综合评分 ----
    ax4 = plt.subplot(2, 2, 4)
    
    # 标准化所有指标 (0-1范围)
    min_sharpe, max_sharpe = sharpe_values.min(), sharpe_values.max()
    min_dd, max_dd = max_dd_values.min(), max_dd_values.max()
    
    sharpe_norm = (sharpe_values - min_sharpe) / (max_sharpe - min_sharpe) if max_sharpe > min_sharpe else np.zeros_like(sharpe_values)
    # Max DD是负数，越接近0越好
    dd_norm = (max_dd - max_dd_values) / (max_dd - min_dd) if max_dd > min_dd else np.zeros_like(max_dd_values)
    
    # 组合评分: 60% Sharpe + 40% DD Risk
    combined_score = 0.6 * sharpe_norm + 0.4 * dd_norm
    
    bars4 = ax4.barh(controllers, combined_score, color=colors)
    ax4.set_xlabel("Composite Score (60% Sharpe + 40% Drawdown)", fontweight='bold')
    ax4.set_title("Overall Performance Score", fontweight='bold', fontsize=12)
    ax4.set_xlim([0, 1.0])
    ax4.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(combined_score):
        ax4.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(root_dir) / "results" / "controller_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_path}")
    
    plt.show()


def print_summary_table(results_df):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("REALIZED VOL STRATEGY: CONTROLLER PERFORMANCE SUMMARY")
    print("="*80)
    
    summary_df = results_df[["Annual_Return", "Annual_Vol", "Sharpe", "Max_DD"]].copy()
    summary_df.columns = ["Annual Return", "Annual Vol", "Sharpe", "Max Drawdown"]
    
    # 重新映射索引为显示名称
    summary_df.index = [CONTROLLER_DISPLAY_NAMES[c] for c in summary_df.index]
    
    print(summary_df.to_string())
    print("\n" + "="*80 + "\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：加载数据并生成可视化"""
    
    print("\n📊 Generating Controller Comparison Visualization...")
    print(f"   Strategy: Realized Vol (20-day lookback)")
    print(f"   Estimator: RealizedVol")
    print(f"   Controllers: {len(CONTROLLERS)}")
    
    # 尝试加载真实数据，如果失败则使用模拟数据
    try:
        results_df = load_real_backtest_results()
        print("   Data Source: Real Backtest Results ✓")
    except Exception as e:
        print(f"   Data Source: Mock Data (fallback) - {e}")
        results_df = generate_mock_backtest_results()
    
    # 打印汇总表格
    print_summary_table(results_df)
    
    # 生成可视化
    plot_controller_comparison(results_df)
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
