# Local Qwen Multimodal Service

本目录用于规划本地部署的多模态大模型服务。业务主服务不直接加载 Qwen 模型，而是通过通用 OpenAI-compatible API 调用本服务。这样后续可以在不改业务识别流程的情况下，切换到其他本地模型、vLLM/SGLang 服务，或第三方多模态接口。

## 目标

本地 Qwen 服务负责接收：

- 一张代表场景帧或候选片段关键帧。
- YOLO 检测统计。
- pose 摘要。
- 人物轨迹。
- 登高候选片段。
- 交底候选片段。
- 指定任务类型和 JSON 输出要求。

并返回结构化 JSON 判断结果：

- 场景：`machine_room`、`near_tower`、`other`
- 登高作业：是否存在、开始时间、结束时间、置信度、原因
- 交底行为：是否存在、开始时间、结束时间、参与人数、是否有文档、是否有签字场景、原因

## 推荐技术方案

推荐优先实现 OpenAI-compatible 的 `/v1/chat/completions` 接口。

```text
Supervisor Agent
  -> POST /v1/chat/completions
  -> Local Qwen Service
  -> Qwen3-VL inference
  -> JSON result
```

服务内部可以采用以下方案之一：

- `transformers` + `Qwen/Qwen3-VL-8B-Instruct`
- ModelScope 下载模型 + `transformers` 加载
- vLLM / SGLang 等推理框架，如果已支持目标 Qwen-VL 模型
- 后续可替换为其他多模态模型，只要保持 API 兼容

## API 设计

本服务使用 FastAPI 实现 OpenAI-compatible API。FastAPI 只负责 HTTP 服务、鉴权、请求解析和响应包装；模型推理逻辑放在独立模块中，避免 API 层和模型层耦合。

### Endpoint

```http
POST /v1/chat/completions
Authorization: Bearer <api_key>
Content-Type: application/json
```

建议同时提供健康检查接口：

```http
GET /health
GET /v1/models
```

`/health` 用于进程和模型状态检查，`/v1/models` 用于返回当前服务支持的模型名称，方便业务服务或运维系统确认模型是否可用。

### Request

采用 OpenAI-compatible vision chat 格式：

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": "你是工地施工安全视频审核模型。只输出合法JSON，不要输出解释性文本。"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "任务类型：briefing。请根据图片和结构化摘要判断是否存在交底行为。结构化摘要：..."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "response_format": {
    "type": "json_object"
  }
}
```

请求约束：

- `messages` 至少包含一个 user 消息。
- user 消息中必须包含一个 `type=image_url` 的图片。
- 图片优先支持 `data:image/jpeg;base64,...`。
- 文本中必须包含任务类型，例如 `scene`、`height_work`、`briefing`。
- `response_format.type=json_object` 表示服务应该返回 JSON 字符串。

### Response

保持 OpenAI-compatible 响应结构，JSON 放在 `choices[0].message.content` 中：

```json
{
  "id": "chatcmpl-local-qwen",
  "object": "chat.completion",
  "created": 1782450000,
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"briefing\":true,\"briefing_confidence\":0.82,\"briefing_start_ms\":3000,\"briefing_end_ms\":26000,\"participant_count\":5,\"has_document\":true,\"has_signature_scene\":true,\"reason\":\"一名人员面向多人进行说明，其他人员集中站立听取\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

错误响应也尽量保持通用格式：

```json
{
  "error": {
    "message": "missing image_url in user message",
    "type": "invalid_request_error",
    "code": "missing_image"
  }
}
```

## 任务类型

建议业务服务通过 prompt 中的 `任务类型` 字段区分任务。

### scene

输入：

- 代表场景帧
- 场景目标统计
- YOLO top labels

输出：

```json
{
  "scene": "near_tower",
  "scene_confidence": 0.91,
  "reason": "画面中存在铁塔和室外基站环境"
}
```

### height_work

输入：

- 代表帧或候选片段关键帧
- 登高候选片段
- 人物轨迹
- pose 摘要
- 梯子、脚手架、铁塔、高处平台等目标统计

输出：

```json
{
  "height_work": true,
  "height_work_confidence": 0.86,
  "height_work_start_ms": 12000,
  "height_work_end_ms": 48000,
  "reason": "人员位于梯子上并持续进行高处作业"
}
```

### briefing

输入：

- 代表帧或候选片段关键帧
- 交底候选片段
- 人员数量
- 人物轨迹
- pose 摘要
- 文档类物品检测结果
- 签字候选动作

输出：

```json
{
  "briefing": true,
  "briefing_confidence": 0.76,
  "briefing_start_ms": 5000,
  "briefing_end_ms": 34000,
  "participant_count": 2,
  "has_document": true,
  "has_signature_scene": true,
  "signature_start_ms": 28000,
  "signature_end_ms": 34000,
  "reason": "两人持续停留，一人持文档讲解，末尾出现签字确认动作"
}
```

## 实现方案

### 1. 服务框架

推荐使用 FastAPI：

```text
qwen/
  README.md
  requirements.txt
  service/
    main.py
    config.py
    schemas.py
    model.py
    openai_compat.py
    prompts.py
```

模块职责：

- `main.py`：FastAPI 应用入口，暴露 `/v1/chat/completions`
- `config.py`：模型名称、缓存目录、设备配置、API key、超时配置
- `schemas.py`：OpenAI-compatible 请求/响应结构
- `model.py`：Qwen 模型加载和推理封装
- `openai_compat.py`：把请求转换成模型输入，把模型输出包装成 OpenAI-compatible 响应
- `prompts.py`：场景、登高、交底任务 prompt 模板

### FastAPI 最小骨架

`service/main.py` 建议结构：

```python
from fastapi import Depends, FastAPI, HTTPException

from service.config import Settings, get_settings
from service.model import QwenVLModel
from service.openai_compat import (
    build_chat_completion_response,
    parse_chat_completion_request,
)
from service.schemas import ChatCompletionRequest

app = FastAPI(title="Local Qwen Multimodal Service")
model: QwenVLModel | None = None


@app.on_event("startup")
def startup() -> None:
    global model
    settings = get_settings()
    model = QwenVLModel(settings)
    model.load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None and model.ready,
    }


@app.get("/v1/models")
def list_models(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    if model is None or not model.ready:
        raise HTTPException(status_code=503, detail="model is not ready")
    parsed = parse_chat_completion_request(request)
    result = model.generate_json(parsed.image, parsed.prompt)
    return build_chat_completion_response(
        model=request.model,
        content=result,
    )
```

这个骨架只说明分层方式，真实实现时需要补充鉴权、异常处理、日志和模型输出重试。

### API 鉴权

建议使用简单 Bearer Token：

```http
Authorization: Bearer change-me
```

校验逻辑：

- 如果 `QWEN_API_KEY` 为空，可以允许无鉴权，适合本地调试。
- 如果配置了 `QWEN_API_KEY`，请求必须携带 `Authorization`。
- 鉴权失败返回 `401 unauthorized`。

生产环境建议：

- API 服务只暴露在内网。
- 通过 Nginx / API Gateway / Service Mesh 做 TLS 和访问控制。
- 不在日志中打印完整 base64 图片。

### 2. 模型加载

推荐支持两种模型来源：

- 本地路径：适合生产环境提前下载模型。
- ModelScope：适合国内环境自动下载模型。

建议配置：

```bash
export QWEN_MODEL_NAME='Qwen/Qwen3-VL-8B-Instruct'
export QWEN_MODEL_PATH=''
export QWEN_USE_MODELSCOPE=true
export QWEN_MODELSCOPE_CACHE_DIR='data/modelscope'
export QWEN_DEVICE_MAP='auto'
export QWEN_MAX_NEW_TOKENS=512
export QWEN_API_KEY='change-me'
```

加载顺序：

1. 如果 `QWEN_MODEL_PATH` 存在，优先从本地路径加载。
2. 如果启用 ModelScope，使用 `snapshot_download` 下载模型。
3. 否则交给 `transformers.from_pretrained` 自动加载。

### 依赖建议

`qwen/requirements.txt` 建议包含：

```text
fastapi
uvicorn[standard]
pydantic-settings
httpx
Pillow
torch
transformers
accelerate
modelscope
```

如果后续改用 vLLM / SGLang，依赖和启动方式可以放到单独 profile 中，不影响 API 契约。

### 3. 请求处理

处理步骤：

1. 校验 `Authorization`。
2. 解析 OpenAI-compatible 请求。
3. 提取文本 prompt。
4. 提取 `image_url`，支持 `data:image/jpeg;base64,...`。
5. 将图片和文本交给 Qwen-VL。
6. 强制解析模型输出 JSON。
7. 将 JSON 包装到 `choices[0].message.content`。

请求解析建议：

- 从 `messages` 中合并 system prompt 和 user text。
- 只取第一张 `image_url` 作为主图。
- 如果请求包含多张图，第一版可以拒绝或只取第一张。
- 将 `data:image/jpeg;base64,...` 解码为 PIL Image。
- 如果是 HTTP 图片 URL，第一版可以禁用；后续再按安全策略开启。

prompt 构造建议：

- 不要让业务服务直接写很长的自由 prompt。
- 业务服务传结构化摘要。
- Qwen 服务根据 `task` 使用固定 prompt 模板。
- 模板中明确输出 JSON schema。

### 4. JSON 输出约束

模型服务应该做两层约束：

- prompt 约束：明确要求只输出 JSON。
- 服务端约束：对模型输出进行 JSON 解析，失败时可重试一次或返回错误。

失败响应示例：

```json
{
  "error": {
    "message": "model output is not valid json",
    "type": "invalid_model_output"
  }
}
```

建议模型输出处理流程：

1. 尝试直接 `json.loads(output)`。
2. 如果失败，从文本中提取第一个 `{...}` JSON 块。
3. 如果仍失败，追加“只输出 JSON”修正提示重试一次。
4. 仍失败则返回 `invalid_model_output`。

### 5. 日志与观测

建议记录：

- 请求 ID。
- 模型名称。
- 任务类型。
- 图片尺寸。
- prompt token 近似长度。
- 推理耗时。
- JSON 解析是否成功。
- 错误类型。

不要记录：

- 完整 base64 图片。
- 过长的结构化摘要全文。
- 敏感环境变量。

建议指标：

- `qwen_requests_total`
- `qwen_request_duration_seconds`
- `qwen_inference_errors_total`
- `qwen_json_parse_errors_total`
- `qwen_model_loaded`

## 部署方式

### 部署前置条件

GPU 部署建议环境：

- Ubuntu 22.04 或兼容 Linux 发行版。
- NVIDIA Driver 已安装。
- Docker 已安装。
- Docker Compose 已安装。
- NVIDIA Container Toolkit 已安装。
- 服务器可以访问 ModelScope，或已经提前下载好模型权重。

检查 GPU：

```bash
nvidia-smi
```

检查 Docker GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

如果上述命令无法看到 GPU，需要先修复 NVIDIA Driver 或 NVIDIA Container Toolkit。

### Docker Compose 快速启动

推荐使用 Docker Compose 启动本地 Qwen 服务。

```bash
cd qwen
docker compose up --build
```

服务启动后检查：

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/v1/models
```

默认端口：

```text
http://127.0.0.1:9001/v1/chat/completions
```

默认配置见 `docker-compose.yml`：

```yaml
QWEN_MODEL_NAME: Qwen/Qwen3-VL-8B-Instruct
QWEN_USE_MODELSCOPE: true
QWEN_MODELSCOPE_CACHE_DIR: /models/modelscope
QWEN_HF_CACHE_DIR: /models/huggingface
QWEN_DEVICE_MAP: auto
QWEN_API_KEY: change-me
QWEN_MOCK_MODE: false
```

模型缓存挂载：

```text
qwen/data/modelscope   -> /models/modelscope
qwen/data/huggingface  -> /models/huggingface
```

第一次启动会下载模型，耗时较长。后续重启会复用本地缓存。

后台启动：

```bash
cd qwen
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f qwen
```

停止服务：

```bash
docker compose down
```

重启服务：

```bash
docker compose restart qwen
```

查看容器状态：

```bash
docker compose ps
```

### Mock 模式快速验证

如果只是验证 API 是否可用，不想下载模型，可以开启 mock 模式：

```bash
cd qwen
QWEN_MOCK_MODE=true docker compose up --build
```

或修改 `docker-compose.yml`：

```yaml
QWEN_MOCK_MODE: "true"
QWEN_LOAD_ON_STARTUP: "false"
```

mock 模式不会加载模型，会返回固定 JSON，适合联调主业务服务、鉴权、网络和响应格式。

mock 模式启动后，可以直接验证 OpenAI-compatible 响应结构：

```bash
curl -X POST http://127.0.0.1:9001/v1/chat/completions \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "任务类型：scene。结构化摘要：{}"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2w=="}}
        ]
      }
    ],
    "response_format": {"type": "json_object"}
  }'
```

上面的图片只是极小占位图，真实推理请传有效 JPEG base64。

### 生产部署建议

生产环境建议修改 `docker-compose.yml` 中的配置：

```yaml
environment:
  QWEN_MODEL_NAME: "Qwen/Qwen3-VL-8B-Instruct"
  QWEN_MODEL_PATH: ""
  QWEN_USE_MODELSCOPE: "true"
  QWEN_MODELSCOPE_CACHE_DIR: "/models/modelscope"
  QWEN_HF_CACHE_DIR: "/models/huggingface"
  QWEN_DEVICE_MAP: "auto"
  QWEN_MAX_NEW_TOKENS: "512"
  QWEN_API_KEY: "replace-with-strong-token"
  QWEN_MOCK_MODE: "false"
  QWEN_LOAD_ON_STARTUP: "true"
  QWEN_LOG_LEVEL: "INFO"
```

建议：

- `QWEN_API_KEY` 必须改成强随机 token。
- Qwen 服务只暴露在内网，不建议直接暴露公网。
- 通过 Nginx、API Gateway 或服务网格增加 TLS、访问控制和限流。
- 模型缓存目录使用宿主机持久化磁盘，避免容器重建后重复下载。
- 生产环境建议 `QWEN_LOAD_ON_STARTUP=true`，服务启动时完成模型加载，避免首个请求超时。

### 使用本地模型路径部署

如果模型已经提前下载到服务器，例如：

```text
/data/models/Qwen3-VL-8B-Instruct
```

可以在 `docker-compose.yml` 中挂载并设置：

```yaml
volumes:
  - /data/models/Qwen3-VL-8B-Instruct:/models/qwen:ro

environment:
  QWEN_MODEL_PATH: "/models/qwen"
  QWEN_USE_MODELSCOPE: "false"
```

这种方式适合生产环境，启动更稳定，也避免容器启动时访问外网。

### 单机部署

```bash
cd qwen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn service.main:app --host 0.0.0.0 --port 9001
```

本地验证：

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/v1/models
```

调用示例：

```bash
curl -X POST http://127.0.0.1:9001/v1/chat/completions \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "你是工地施工安全视频审核模型。只输出合法JSON。"
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "任务类型：scene。结构化摘要：{\"scene_labels\":[[\"tower\",12]]}"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,..."
            }
          }
        ]
      }
    ],
    "temperature": 0,
    "response_format": {"type": "json_object"}
  }'
```

### Docker 文件说明

本目录提供：

```text
qwen/Dockerfile          # GPU 运行镜像，基于 nvidia/cuda runtime
qwen/docker-compose.yml  # 本地快速启动和 GPU 设备声明
qwen/.dockerignore       # 排除缓存和临时文件
qwen/requirements.txt    # Qwen 服务 Python 依赖
qwen/service/            # FastAPI 服务实现
```

如果机器没有 NVIDIA Container Toolkit，GPU 容器无法正常使用。需要先安装：

```text
nvidia-driver
nvidia-container-toolkit
docker compose
```

可以用以下命令确认 Docker 是否能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

业务服务配置：

```bash
export SUPERVISOR_MULTIMODAL_PROVIDER='remote'
export SUPERVISOR_MULTIMODAL_API_URL='http://127.0.0.1:9001/v1/chat/completions'
export SUPERVISOR_MULTIMODAL_API_FORMAT='openai_compatible'
export SUPERVISOR_MULTIMODAL_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_MULTIMODAL_API_KEY='change-me'
```

如果 Qwen 服务部署在另一台机器，例如 `10.0.0.12`：

```bash
export SUPERVISOR_MULTIMODAL_API_URL='http://10.0.0.12:9001/v1/chat/completions'
```

如果主服务和 Qwen 服务都通过 Docker Compose 部署在同一个 Docker 网络里，可以使用服务名：

```bash
export SUPERVISOR_MULTIMODAL_API_URL='http://qwen:9001/v1/chat/completions'
```

### GPU 建议

`Qwen3-VL-8B-Instruct` 建议部署在独立 GPU 机器上。实际显存需求取决于：

- 模型精度
- 上下文长度
- 图片分辨率
- batch size
- 推理框架

建议第一版：

- batch size 设置为 1。
- 图片压缩到合理分辨率，例如长边 1280 或更低。
- 每个请求只提交关键帧和结构化摘要，不提交完整视频。
- 对登高和交底采用候选片段方式，减少无效调用。

### 常见问题

#### 1. 容器启动后一直下载模型

检查缓存目录是否正确挂载：

```bash
docker compose exec qwen ls -lah /models/modelscope
```

如果目录为空，说明模型没有成功缓存，或挂载路径不正确。

#### 2. `torch.cuda.is_available()` 为 false

通常是 Docker 没有拿到 GPU。检查：

```bash
docker compose exec qwen nvidia-smi
```

如果命令不存在或失败，检查 NVIDIA Container Toolkit。

#### 3. 首次请求超时

如果 `QWEN_LOAD_ON_STARTUP=false`，模型会在首次请求时加载，容易超时。生产建议设置：

```yaml
QWEN_LOAD_ON_STARTUP: "true"
```

#### 4. 返回 `401 unauthorized`

检查请求头：

```http
Authorization: Bearer <QWEN_API_KEY>
```

如果只是本地调试，可以临时把 `QWEN_API_KEY` 设置为空，但生产不建议这样做。

#### 5. 模型输出不是合法 JSON

服务会尝试解析 JSON，并在失败时重试一次。仍失败时会返回错误。可以从以下方向优化：

- 缩短结构化摘要。
- 在 prompt 中更明确 JSON schema。
- 降低生成温度。
- 减少一次请求中的候选片段数量。

## 与业务服务的关系

业务服务负责：

- 视频接入。
- YOLO 检测。
- pose 分析。
- 跟踪。
- 候选片段生成。
- 告警入库和上报。

Qwen 服务负责：

- 读取一张图片。
- 读取结构化摘要。
- 判断场景、登高或交底。
- 返回 JSON。

两者之间只通过通用 API 交互，避免模型部署细节污染业务服务。

## 后续扩展

- 支持流式输出，但业务当前只需要最终 JSON。
- 支持多个模型名称，例如 `qwen-vl-local`、`qwen-vl-large`。
- 支持第三方 OpenAI-compatible API 代理。
- 支持请求日志和图片脱敏。
- 支持模型输出缓存，避免重复分析同一候选片段。
- 支持 Prometheus 指标：请求数、耗时、失败率、GPU 显存占用。
