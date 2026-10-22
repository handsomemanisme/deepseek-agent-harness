# DeepSeek Agent Harness 复现

> A DeepSeek-powered reproduction and extension of the Stanford CS329Z Agent Harness.

本项目复现了 Stanford CS329Z Assignment 1 中的 Agent Harness，并将原有
OpenRouter/GPT 调用替换为 DeepSeek API。

系统能够让大语言模型自主选择和调用邮件搜索、文档检索、受限终端、持久记忆等工具，
并实现多轮任务循环、上下文压缩、人工审批和提示注入防护。

本项目仅用于学习。

## 项目来源

本项目基于以下公开项目开发：

- 原始仓库：https://github.com/cs329z/assignment1-harness
- 课程：Stanford CS329Z — Engineering AI Agents

## 测试方法

### 环境要求
Python 3.11或更高版本
uv
DeepSeek API Key
macOS、Linux或Windows

### 配置DeepSeek
复制环境变量文件：
```
cp .env.example .env
```
编辑 .env：
```
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

可选配置
```
DEEPSEEK_THINKING=false
DEEPSEEK_PRICE_TIER=peak
DEEPSEEK_BUDGET_CNY=5
```
### 准备邮件数据
```
uv run python data/download.py
```
该命令会把压缩邮件数据解压到：
data/emails/emails.jsonl
### 运行确定性测试
```
uv run pytest
```
这些测试默认不调用真实模型，因此不会产生 API 费用。
### 启动Agent
推荐使用需要人工审批的模式：
```
uv run python -m cs329z_hw1.chat --mode confirm
```
会产生DeepSeek API费用。
### Agent综合评估
```
uv run python evaluate_agent_deepseek.py --only all --limit 3
```
该命令会调用DeepSeek API，可能产生少量费用。


License
本项目沿用原项目的MIT License，详情见 LICENSE。
原始项目：
- Copyright (c) 2026 cs329z
- https://github.com/cs329z/assignment1-harness
