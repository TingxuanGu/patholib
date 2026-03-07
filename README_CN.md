# patholib - 病理图像分析库

[English](README.md) | [中文](README_CN.md)

![Status](https://img.shields.io/badge/status-experimental-orange) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue)

**⚠️ 实验性 / 测试版**: 本库正在积极开发中。API 可能会发生变化。测试覆盖不完整。在生产环境中请谨慎使用。

自动化定量分析 IHC（免疫组织化学）和 H&E 染色组织切片。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/TingxuanGu/patholib.git
cd patholib

# 使用 pip 安装
pip install -e .

# GPU 支持（推荐用于 H&E 分析）
pip install cellpose torch --index-url https://download.pytorch.org/whl/cu121

# 全切片图像支持
pip install openslide-python
```

### 系统依赖

**Ubuntu/Debian:**
```bash
sudo apt-get install openslide-tools
```

**macOS:**
```bash
brew install openslide
```

## 使用方法

### IHC 分析
```bash
# Ki-67（核染色）
python analyze_ihc.py --input ki67_slide.tif --stain-type nuclear --marker Ki67 \
    --output-dir ./results --save-overlay --save-csv

# HER2（膜染色）
python analyze_ihc.py --input her2_slide.tif --stain-type membrane --marker HER2 \
    --output-dir ./results --save-overlay

# ER/PR（核染色，包含 Allred 评分）
python analyze_ihc.py --input er_slide.tif --stain-type nuclear --marker ER \
    --output-dir ./results --save-overlay --save-csv
```

### H&E 分析
```bash
# 炎症评分
python analyze_he.py --input tissue.tif --mode inflammation --use-gpu \
    --output-dir ./results --save-overlay --save-heatmap --save-csv

# 肿瘤/坏死面积比
python analyze_he.py --input tissue.tif --mode area-ratio \
    --output-dir ./results --save-overlay

# 两种分析
python analyze_he.py --input tissue.tif --mode both --use-gpu \
    --output-dir ./results --save-overlay --save-heatmap --save-csv
```

### 关键参数
- `--fail-fast`: 如果专用分析器不可用则中止（而不是静默回退）
- `--use-gpu`: 为 Cellpose 启用 GPU 加速（H&E 分析）
- `--grid-size`: 设置了 `--mpp` 时按微米解释，否则按像素解释
- `--normalize-stain`: 分析前应用染色归一化

## 输出文件
- `*_report.json`: 包含所有指标的完整分析报告
- `*_cells.csv`: 单细胞数据（质心、面积、等级、强度）
- `*_overlay.png`: 带细胞检测标注的图像
- `*_heatmap.png`: 密度热图（炎症模式）

## 包结构
```
patholib_deploy/
├── analyze_ihc.py          # IHC 命令行工具
├── analyze_he.py           # H&E 命令行工具
├── setup.sh                # 一键部署脚本
├── setup.py                # pip install -e .
├── requirements.txt        # 依赖项
├── patholib/
│   ├── io/                 # 图像加载 + WSI 切片
│   ├── stain/              # 颜色反卷积 + 归一化
│   ├── detection/          # 组织/细胞检测（CV + Cellpose）
│   ├── analysis/           # IHC + H&E 分析模块
│   ├── scoring/            # H-score、Allred、百分比
│   └── viz/                # 叠加图、热图、报告生成
└── references/             # 文档
```

## 功能特性

- **IHC 分析**: 核、膜和胞浆染色定量
- **H&E 分析**: 炎症评分和肿瘤/坏死面积比
- **评分方法**: H-score、Allred 评分、阳性百分比
- **检测方法**: 经典 CV（分水岭）和深度学习（Cellpose）
- **可视化**: 标注叠加图、密度热图、综合报告

## 示例

查看 `examples/` 目录获取批处理脚本和参数调优示例。

## Benchmark

第一阶段公开 benchmark 计划见 [benchmarks/phase1_plan.md](benchmarks/phase1_plan.md)，当前优先覆盖 `PanNuke`、`BCSS`、`BCData` 和 `HER2-IHC-40x`。

## 许可证

MIT License - 详见 LICENSE 文件。

## 引用

如果您在研究中使用本库，请引用：

```
@software{patholib2026,
  author = {Gu, Tingxuan},
  title = {patholib: Pathology Image Analysis Library},
  year = {2026},
  url = {https://github.com/TingxuanGu/patholib}
}
```

## 贡献

这是一个实验性项目。欢迎通过 GitHub issues 提交贡献、错误报告和功能请求。
