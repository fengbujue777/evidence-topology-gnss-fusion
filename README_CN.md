# 多接收机 GNSS 证据拓扑重构代码

这是论文《面向多接收机 GNSS 分歧的证据拓扑重构与保守因子构造方法》的实验代码与冻结结果仓库。

核心问题不是简单判断“哪个接收机坏了”，而是在观测进入因子图之前判断有多少份独立物理证据：同一物理接收机的重复解算流先归一，接收机分歧时再依据 KISS-ICP 运动一致性与接收机间分离度选择单接收机或一致接收机对，并为接收机对构造一个保守因子。

## 快速检查

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python examples/synthetic_topology_demo.py
python -m pytest -q
```

## 从公开原始数据完整复现

仓库已经补齐以下链路：

```text
UrbanNav/LOCSP 原始数据
  → KISS-ICP 轨迹与固定版本 RTKLIB 单点定位
  → 真值隔离的多接收机案例
  → 本文方法、基线、消融和参数敏感性实验
  → 统计检验、运行时间与冻结结果核验
```

- `prepare_urbannav_panel.py`：从 UrbanNav ROS bag 与 RINEX 构建一个面板；
- `reproduce_paper.py`：一次运行论文的 UrbanNav 完整实验矩阵；
- `reproduce_locsp.py`：运行 LOCSP 同源消息流外部验证；
- `verify_reproduction.py`：核验历元数和论文主结果。

具体下载链接、目录要求与可直接复制的命令见
`docs/DATASETS.md` 和 `docs/REPRODUCTION.md`。

## 重要说明

- 原始 UrbanNav、LOCSP 数据不上传 GitHub，只提供官方公开下载来源、读取器、
  固定参数、外参、RTKLIB 配置和完整处理代码。
- `results/` 保存论文表格对应的小型 JSON/CSV、运行时间记录和外部复现实验摘要。
- 主算法入口是 `run_evidence_topology.py`。
- 冻结参数见 `configs/frozen_parameters.json`。
- 当前仓库应保持 Private；投稿政策与作者信息确定前不要直接公开。

完整复现流程见 `docs/REPRODUCTION.md`，方法公式与代码位置对应关系见 `docs/METHOD_TO_CODE.md`。
