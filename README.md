# Constellation Clustering

基于恒星星表数据的新星座聚类、全天区划分、可视化与命名实验项目。

本项目对应 IMMC 2026 Problem A 的建模与编程工作，尝试将恒星按球面空间位置和可见星等划分为若干新星座，并生成可交互的三维天球图、星座图片以及后续命名所需的数据。

## 项目内容

- **全天区分区**：使用恒星赤经、赤纬，将天球坐标转换为单位球面上的三维向量。
- **球面聚类**：以较亮恒星作为星座骨架，再将可见范围内的恒星分配到最近的聚类中心。
- **聚类评价**：使用球面距离和 Silhouette Score 选择较合适的星座数量。
- **星座连线**：对每个星座的亮星计算最小生成树（MST），生成更接近传统星座连线的图案。
- **交互式可视化**：使用 Plotly 输出三维天球和可筛选的 HTML 页面。
- **语义命名实验**：使用视觉模型分析星座图案，并生成中文名、拉丁名和具象程度评分。

## 目录结构

```text
.
├── asu.tsv                              # 恒星主星表
├── asu_names.tsv                        # 著名恒星名称
├── asu_constellations.tsv               # 原有星座边界数据
├── constellation_clustering_q2.py       # 根目录中的聚类脚本副本
├── MST_k-means/                         # 主要的 Q2 聚类与可视化实现
├── MST_k-means(q2_q3_choice)/            # Q2/Q3 组合实验版本
├── constellation/                       # Q1、Q2、Q3 及三维展示脚本
├── constellation_maps/                  # 生成的星座图像
├── new_constellations_results.json       # 新星座聚类结果
├── constellations_for_3d.json            # 三维展示使用的数据
├── optimized_constellations_interactive_03.html
│                                        # 已生成的交互式结果示例
└── 算法说明.md                           # 算法思路与设计说明
```

## 环境要求

- Python 3.10 或更高版本
- NumPy、pandas、SciPy、scikit-learn
- Matplotlib、Plotly
- `astropy`（三维天球展示脚本需要）
- `Pillow`、`scikit-image`、`requests`（部分 Q3 图像处理脚本需要）
- `openai`（调用视觉模型的 Q3 脚本需要）

项目当前未提供锁定版本的 `requirements.txt`。可以使用虚拟环境安装常用依赖：

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib plotly astropy pillow scikit-image requests openai
```

## 快速运行

### Q2：聚类与交互式可视化

推荐从仓库根目录运行主要实现：

```bash
python MST_k-means/constellation_clustering_q2.py
```

脚本读取根目录的 `asu.tsv` 和 `asu_names.tsv`，并在 `MST_k-means/` 下生成：

- `optimized_constellations.html`
- `optimized_constellations_interactive.html`
- `constellation_images/` 中的各星座 PNG 图像

主要参数位于脚本开头，包括亮星骨架阈值 `VMAG_ANCHOR_LIMIT` 和可见星等阈值 `VMAG_VISIBLE_LIMIT`。

### 三维天球展示

```bash
python constellation/3d_constellation_viewer.py
```

也可以运行 `constellation/3d_constellation_viewer_2.py` 查看另一种展示版本。脚本使用 Plotly 打开交互式三维图，并读取根目录中的星表和聚类结果文件。

### Q3：图案命名

Q3 脚本会读取星座 PNG 图像，并调用视觉模型生成结构化命名结果。例如：

```bash
python "MST_k-means(q2_q3_choice)/q3.py"
```

运行前请根据脚本中的 `IMAGE_DIR` 和 `OUTPUT_FILE` 调整输入输出路径。不同目录中的 Q3 脚本属于不同实验版本，使用前应先检查对应脚本的配置区域。

## API 配置

请勿把 API Key 写入源代码。当前相关脚本从环境变量读取凭据：

```powershell
$env:DASHSCOPE_API_KEY = "your-api-key"
$env:BAIDU_APP_ID = "your-app-id"
$env:BAIDU_API_KEY = "your-api-key"
$env:BAIDU_SECRET_KEY = "your-secret-key"
$env:DOUBAO_API_KEY = "your-api-key"
$env:DOUBAO_SECRET_KEY = "your-secret-key"
$env:AI_API_KEY = "your-api-key"
```

如果没有配置对应变量，部分脚本会跳过 API 调用或直接提示配置错误。`.env` 文件和虚拟环境目录已加入 `.gitignore`。

## 数据说明

- `asu.tsv`、`asu_names.tsv` 和 `asu_constellations.tsv` 为项目使用的恒星及星座数据文件。
- `Vmag` 表示视星等；数值越小通常表示恒星越亮。
- 根目录下的 JSON、TSV、CSV、PNG 和 HTML 文件包含运行过程中保存的中间结果或可视化结果。
- `code-full.zip`、`code-h-full.zip` 和 `code.rar` 是历史代码归档，不是当前运行入口。
- 部分文件名和注释保留中文，Windows 环境运行时建议使用 UTF-8 编码。

## 方法限制

- 聚类结果依赖亮星阈值、可见星等阈值、随机种子和候选聚类数量范围。
- 球面聚类得到的是数学意义上的分区，不等同于经过天文学界确认的正式星座边界。
- MST 主要用于增强图案可读性，不代表唯一或标准的星座连线方案。
- AI 命名结果具有主观性，应作为辅助结果而不是科学结论。
- 各实验目录中存在重复或历史版本脚本，复现实验时应记录具体脚本路径和参数。

## 复现建议

1. 从仓库根目录创建并激活 Python 虚拟环境。
2. 先运行 `MST_k-means/constellation_clustering_q2.py` 生成或更新聚类结果。
3. 检查输出的 HTML 和 PNG 图像，再运行三维展示或 Q3 命名脚本。
4. 记录 Python 版本、依赖版本、脚本路径和关键阈值，便于比较不同实验结果。

## 许可证

当前仓库未声明正式开源许可证。如需再分发、商用或引用，请先联系仓库维护者确认授权方式。
