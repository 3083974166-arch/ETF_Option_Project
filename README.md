# ETF_Option_Project

**上证50ETF期权隐含波动率、波动率曲面与Delta中性策略研究。**

这是一个可学习、可审计的 Python 研究工程，不是已验证盈利的交易产品。核心是自主定价、数值反解、真实数据工程、因果时序和交易成本。没有使用机器学习堆模型。

求职展示版研究报告：[ETF_Option_Project_研究报告.pdf](docs/ETF_Option_Project_研究报告.pdf)。报告集中说明数值方法、数据工程、时序设计、验证证据和当前数据边界，不包含合成数据收益宣传。

## 交付状态：先看这一段

- 已实现定价、清洗、隐含远期/持有收益、IV与Greeks、曲面、RV预测研究、跨式与ETF对冲回测、样本外与稳健性分析。
- 2026-09-30 实际联网检查：新浪单合约历史日线返回23行；当前月份接口返回4个月份；东方财富ETF日线返回183行。原始文件与 SHA256 在 `data/raw/probe/`。
- **尚未取得经核验、覆盖已到期合约的连续历史期权链、历史盘口及完整分红/利率输入。因此本交付不含真实市场策略收益、Sharpe或盈利结论。** 接口可访问不代表数据足以回测。
- synthetic fixture 仅位于 `tests/fixture_factory.py`，即时生成、只用于单元及集成测试。正式命令拒绝 `is_synthetic=True`。测试产生的图表和虚构绩效留在临时目录，不作为展示业绩。
- 阅读 `docs/完成度与局限.md`，了解“代码完成”和“实证研究完成”的区别。不要声称自己独立完成了尚未理解的代码或尚未运行的真实研究。
- `docs/overleaf/main.tex` 是宝宝教程的 Overleaf 版本；上传 `宝宝教程_Overleaf.zip` 并选择 XeLaTeX 即可编译。

## 安装

Python 3.11–3.12 为推荐环境；本项目已在 Windows / Python 3.12 验证。

```powershell
cd ETF_Option_Project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
python -m pytest -q
```

PowerShell 阻止激活脚本时，不必修改系统策略，直接使用 `.venv\Scripts\python.exe` 代替 `python`。macOS/Linux 使用 `source .venv/bin/activate`。依赖范围见 `pyproject.toml`；本次实测版本见 `results/environment_verified.txt`。

## 第一次运行：不需要准备市场数据

```powershell
python -m pytest -q
python run_pipeline.py --help
python run_pipeline.py probe --out data/raw/probe_new
```

第一条验证数值方法和整条测试流程；第三条联网检查并下载真实小样本。它们都不会凭空生成真实收益表。`probe` 每个接口最多等待35秒；无网络时仍可运行所有离线测试。

## 完整真实研究的运行顺序

1. 阅读 `docs/数据规范.md`。收集带生效起止日期的完整历史合约表，必须包含已退市/到期合约，不能仅下载当前挂牌列表。
2. 将核实后的映射保存为 `data/raw/ready/contracts.csv`，用交易所导出的规范化快照进行交叉验证。空白字段模板在 `data/raw/templates/`，**模板没有伪造示例观测**。
3. 下载已知合约历史价格及ETF价格，保存原始响应。

```powershell
python run_pipeline.py validate-official --contracts data/raw/ready/contracts.csv --official data/raw/ready/official_contracts.csv --out results/official_validation.csv
python run_pipeline.py download --contracts data/raw/ready/contracts.csv --out data/raw/download
python run_pipeline.py download-etf --start 20230101 --end 20251231 --dividends data/raw/ready/dividends.csv --out data/raw/download
```

日期只是命令参数示例，不表示本项目已获得对应时期数据。分红事件文件需要按公告核对，空文件表示研究者已确认该区间没有分红，不得为省事填空。

4. 将下载的 `options.csv`、`underlying.csv` 与已核验的 `contracts.csv`、`rates.csv` 放在 `data/raw/ready/`；清理超出映射生效区间的行情。每日利率必须有可得日期及来源，不能默填0。
5. 有同步历史买卖报价时运行严格报价模式。样本外分割日期必须位于你的实际样本内。

```powershell
python run_pipeline.py run --input data/raw/ready --out results/real_run_01 --split 2025-01-01
```

公开历史日线通常只有收盘价。此时可以**显式**运行假设价差的研究情景，不能把它当作可成交验证：

```powershell
python run_pipeline.py run --input data/raw/ready --out results/proxy_run_01 --split 2025-01-01 --proxy-costs --assume-etf-borrow
```

`--assume-etf-borrow` 表示假设ETF可借券，并按配置计费，**不是验证了历史券源**。默认禁用ETF卖空：负对冲目标会受到限制，不能保证持续中性；完整中性研究需券源假设或真实可借数据。关闭卖空可用于保守可执行性对照。

## 输出与时序

每次输出到新的目录；拒绝覆盖旧实验。包括：

- `manifest.json`：输入文件散列、软件版本、参数、是否代理成本、成功/失败状态。
- `iv_panel.csv`、`forwards.csv`、`rejections.csv`、`atm_term.csv`、`features.csv`。
- `ledger_is.csv`、`ledger_oos.csv`：每日期权腿、ETF腿、成本、现金、权益、Delta残差和换手。
- `performance.json`、`forecast_study.json`、`oos_forecasts.csv`、`robustness_oos.csv`。
- `figures/01_smile_term.png` 至 `05_pnl_attribution.png`：Smile/期限、2D、3D、IV/RV、净值归因。

收盘 t 计算信号并选择合约，用 t 的Delta产生目标，收盘 t+1 执行；成交检查只决定是否拒单，不改成“今天最佳合约”。因此价格会变、Delta会漂移，中性是目标而非完美事实。现有仓位缺报价会停止实验，禁止前填抹平亏损。

30自然日固定期限IV按总方差插值，禁止外推；过去20交易日RV是信号输入，未来20交易日RV只是标签。预测模型训练标签结束日必须早于样本外起点。策略参数预先固定，IS/OOS分别从空仓开始、各自结尾平仓。OOS敏感性表是诊断，不能在查看后把最佳参数再宣称为未见样本验证。

## 模块导航

| 文件 | 职责 |
|---|---|
| `src/etf_option/pricing.py` | BSM、五个Greeks、平价、边界、Brent和保守Newton |
| `data.py` | AKShare隔离下载、超时、来源记录、官方CSV核对 |
| `cleaning.py` | 按日期映射条款、筛选日志、配对远期、IV反解 |
| `research.py` | ATM/期限、RV、未来标签、purged切分、HAC预测研究 |
| `backtest.py` | 事件顺序、跨式持仓、ETF对冲、现金账本、指标 |
| `pipeline.py` | 串联流程、IS/OOS、9组稳健性情景、运行清单 |
| `plots.py` | 非套利约束的描述性曲面与结果图 |
| `cli.py` / `run_pipeline.py` | 命令行主入口 |
| `tests/` | 数学、数据、时序、账本和集成检查 |
| `notebooks/` | 定价练习与真实结果阅读，不复制另一套核心算法 |

根目录 `figures/` 和 `data/processed/` 保留规范位置，实际一次实验的派生数据和图集中在该次 `results/<run>/`，避免混用版本。

## 常见错误

| 现象 | 含义与处理 |
|---|---|
| `ModuleNotFoundError` | 激活项目环境后重新安装 `-e ".[dev,data]"` |
| 网络超时/返回空表 | 查 `availability.json` / `download_audit.json`，稍后重试；不替换成随机数据 |
| `Missing or overlapping effective contract mapping` | 行情日期没有唯一生效条款；补历史映射或明确裁剪区间 |
| `No valid IV rows` | 检查C/P配对、条款、单位、时间、缺失利率及过滤日志 |
| `Insufficient 30-day bracketed ATM history` | 缺近月或次月，不能外推伪造30日IV |
| `Historical bid/ask missing` | 提供历史盘口或显式选择代理成本情景 |
| `Missing held contract` | 持仓不可估值或被过滤，修复数据；不能直接跳过亏损日 |
| `Output run directory must be empty` | 换新的输出目录，保留原实验 |
| 没有交易 / Sharpe为null | 阈值、券源或数据约束可能确实阻止交易；不是程序必须“修成赚钱” |
| 合约因分红调整 | 当前版本记录并剔除调整合约，不假设其仍是10000单位 |

## 推荐学习顺序

先读 `docs/宝宝教程.md` 第1–3章 → 定价 notebook 和定价测试 → 数据规范与清洗 → 平价远期和IV → RV及时序测试 → 回测账本 → 稳健性 → 研究报告、简历、面试问答。练习必须能自己修改和解释，不能只记结论。

来源链接和方法参考见 `docs/参考来源.md`；完整交付清单见 `docs/文件清单.md`。
