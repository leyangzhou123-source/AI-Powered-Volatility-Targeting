# 📋 JPM-Project Controllers 快速参考

## 🎯 检查结果总结

```
✅ 已配置 Controller
├── constant_weight.py
├── cvar_es_targeting.py
├── drawdown_brake.py
├── hysteresis_controller.py
├── naive_scaling.py
├── priority_stack_controller.py
├── regime_controller.py
├── trend_filter.py
├── variance_scaling.py
└── vol_target_clip.py

❌ 未配置 Controller
└── drawdown_modulated.py (新增，待配置)
```

---

## 📊 Controllers 功能分类

### 基础控制器
- **Naïve Scaling**: 基础波动率缩放（Baseline）
- **Constant Weight**: 固定权重分配

### 风险调整
- **Variance Scaling**: 方差加权缩放
- **Vol Target Clip**: 波动率上下界控制
- **CVaR/ES Targeting**: 尾部风险感知

### 自适应策略  
- **Regime Switch**: 制度转换自适应
- **Hysteresis**: 迟滞阻尼重平衡
- **Priority Stack**: 多优先级堆栈

### 风险管理
- **Drawdown Brake**: 回撤制动保护
- **Trend Filter**: 趋势过滤器
- **Drawdown Modulated** (CPPI-Lite): CPPI轻量级动态杠杆

---

## 🔍 realized_vol.yaml 配置概览

```yaml
# 配置位置: configs/strategies/realized_vol.yaml

name: realized_vol
description: 20-day realized volatility strategy

数据配置:
├── symbol: SPY
├── source: databento
├── lookback: 20
└── vol_ann: 252

风险参数:
├── target_vol: 0.1 (10% 年化波动率目标)
├── weight_min: 0.0 (最小权重)
├── weight_max: 1.5 (最大权重)
└── cost_bps: 5.0 (交易成本)

Router配置:
├── enabled: true
├── default_pair: realized_vol__naive_scaling
└── 10个策略对组合
```

---

## 🚀 下一步行动

### 1. 为 drawdown_modulated 添加配置

编辑 `configs/strategies/realized_vol.yaml`，在 `router.pairs` 末尾添加：

```yaml
  - name: realized_vol__drawdown_modulated
    estimator:
      class: src.estimators.RealizedVol
      params:
        lookback: 20
        vol_ann: 252
    controller:
      class: src.controllers.drawdown_modulated.DrawdownModulatedController
      params:
        no_trade_band: 0.05
        eps_vol: 1.0e-08
        w_min: 0.0
        w_max: 1.5
        start_cut_dd: 0.05
        max_dd: 0.20
```

### 2. 更新 __init__.py

在 `src/controllers/__init__.py` 添加：

```python
from src.controllers.drawdown_modulated import DrawdownModulatedController

# 在 __all__ 中添加:
__all__ = [
    # ... existing ...
    "DrawdownModulatedController",
]
```

### 3. 生成对比图表

运行对比可视化脚本：

```bash
python scripts/visualize_controller_comparison.py
```

这将生成：
- 📊 Sharpe 比率对比 (柱状图)
- 🎯 Return vs Risk (散点图)  
- 📉 最大回撤对比 (柱状图)
- 🏆 综合评分排名 (柱状图)

输出文件: `results/controller_comparison.png`

---

## 📈 Performance Benchmarking (示例)

### 按 Sharpe 比率排名

| 排名 | Controller | Sharpe | 特点 |
|------|-----------|--------|------|
| 🥇 | Regime Switch | 0.92 | 最佳风险调整收益 |
| 🥈 | CVaR/ES | 0.88 | 最佳回撤保护 |
| 🥉 | Priority Stack | 0.86 | 多策略组合 |
| 4️⃣ | Variance Scaling | 0.85 | 稳定风险调整 |
| 5️⃣ | Hysteresis | 0.82 | 减少交易次数 |

### 按最大回撤排名 (最佳风险管理)

| 排名 | Controller | Max DD | 特点 |
|------|-----------|--------|------|
| 🥇 | CVaR/ES | -8% | 尾部风险极低 |
| 🥈 | Drawdown Brake | -10% | 触发式回撤制动 |
| 🥉 | Priority Stack | -11% | 多层保护 |

---

## 💡 Controller 选择建议

**保守型** (重视回撤保护)
→ CVaR/ES Targeting + Drawdown Brake

**平衡型** (风险收益均衡)
→ Regime Switch + Variance Scaling

**激进型** (追求超额收益)
→ Trend Filter + Priority Stack

**新手推荐**
→ Constant Weight (简单) 或 Naïve Scaling (标准)

---

## 📁 相关文件位置

```
jpm-project/
├── src/
│   └── controllers/
│       ├── __init__.py ⭐ (需更新)
│       ├── naive_scaling.py
│       ├── constant_weight.py
│       └── ... (其他10个)
│
├── configs/
│   └── strategies/
│       └── realized_vol.yaml ⭐ (需更新)
│
├── scripts/
│   ├── visualize_controller_comparison.py ✨ (刚创建)
│   ├── compare_table.py
│   └── analysus_result.py
│
└── CONFIG_AUDIT_REPORT.md ✨ (检查报告)
```

---

## 🔗 相关命令

```bash
# 了解 drawdown_modulated 详情
cat src/controllers/drawdown_modulated.py | head -30

# 查看 realized_vol 配置
cat configs/strategies/realized_vol.yaml

# 运行对比分析
python scripts/visualize_controller_comparison.py

# 运行完整backtest
python scripts/run_backtests.py --config realized_vol.yaml
```

---

**Last Updated**: 2026-04-03  
**Status**: ✅ Analysis Complete | ⏳ Config Pending
