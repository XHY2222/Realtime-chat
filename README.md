# Realtime-chat（中文说明）

一个基于 `RealtimeSTT` 的实时语音对话项目，包含：

- 实时语音识别（ASR）
- 大模型对话（支持 OpenAI 兼容接口）
- 语音合成（TTS，支持 Edge-TTS、阿里云 CosyVoice 等）
- WebUI 配置与运行

适合做本地语音助手、实时语音问答、语音交互 Demo。

## 项目结构

- `RealtimeSTT_server2.py`：主后端服务（推荐）
- `webui.py`：Web 配置界面
- `config.json`：主配置文件
- `RealtimeSTT/`：语音识别核心模块
- `utils/`：LLM、TTS、日志、通用工具
- `tests/`：脚本式测试示例

## 环境要求

- Python 3.10+
- 建议 Linux/Windows
- 可选 GPU（CUDA）用于更低延迟识别

## 安装

在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows 可用：

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

如果你使用 GPU，请按需安装匹配版本的 CUDA/cuDNN 与 PyTorch。

## 快速启动

### 1) 启动 WebUI（推荐）

```bash
python webui.py
```

启动后在浏览器打开界面，按需修改配置并运行。

### 2) 直接启动后端

```bash
python RealtimeSTT_server2.py
```

再打开 `index.html` 或前端页面连接 WebSocket 服务（默认端口见配置）。

## 关键配置说明

编辑 `config.json`：

- `chat_type`：对话后端类型（如 `chatgpt`）
- `openai.api`：OpenAI 兼容接口地址（可填 DashScope 兼容地址）
- `openai.api_key`：API Key（建议不要写入仓库）
- `audio_synthesis_type`：TTS 类型（如 `edge-tts`、`aliyun_cosyvoice`）
- `talk.faster_whisper.model_size`：Whisper 模型路径或模型名

## 测试

项目测试以脚本形式提供，可直接运行：

```bash
python tests/simple_test.py
python tests/realtimestt_test.py
python tests/wakeword_test.py
```

## 常见问题

1. **麦克风无输入**
   - 检查系统麦克风权限与设备索引 `device_index`。

2. **TTS 无声音或报错**
   - 检查对应 TTS 服务配置与 API Key。

3. **GPU 未生效**
   - 检查 CUDA 与 PyTorch 版本是否匹配。

4. **WebUI 获取鼠标坐标失败**
   - 无图形界面环境（无 DISPLAY）下无法使用该功能。

## 安全建议

- 不要把真实 API Key、Token、账号密码提交到 GitHub。
- 推荐使用本地环境变量或本地私有文件管理密钥。

## 许可证

本项目沿用仓库中的 `LICENSE`（MIT）。
