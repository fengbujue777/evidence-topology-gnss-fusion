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

## 重要说明

- 原始 UrbanNav、LOCSP 数据不上传 GitHub，只提供公开下载来源与处理代码。
- `results/` 保存论文表格对应的小型 JSON/CSV、运行时间记录和外部复现实验摘要。
- 主算法入口是 `run_evidence_topology.py`。
- 冻结参数见 `configs/frozen_parameters.json`。
- 当前仓库应保持 Private；投稿政策与作者信息确定前不要直接公开。

完整复现流程见 `docs/REPRODUCTION.md`，方法公式与代码位置对应关系见 `docs/METHOD_TO_CODE.md`。
