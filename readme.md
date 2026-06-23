# Supervisor Agent

面向工地施工视频的智能识别服务。服务接收离线视频或视频流，抽帧后调用 YOLO / 视觉语言模型适配器进行识别，将识别事件写入数据库，并对安全帽/反光衣、抽烟、动火等实时风险生成告警。

## 机器配置

[1/5] 操作系统信息:
  系统名称: Ubuntu 22.04.4 LTS
  内核版本: 5.15.0-179-generic
  系统架构: x86_64

[2/5] CPU 硬件配置:
  CPU 型号: Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz
  物理核心: 48 核

[3/5] 内存配置 (RAM):
  总内存大小: 125Gi
  当前可用内存: 106Gi

[4/5] GPU / 显卡配置 (NVIDIA):
  NVIDIA 驱动及 CUDA 检测成功！
  显卡型号: NVIDIA GeForce RTX 4090
  显存总量: 23.98 GB (24564 MiB)
  驱动版本:  575.64.03
  CUDA (nvcc) 版本: 12.1

## 已选模型

- `Qwen/Qwen3-VL-8B-Instruct`
- `yoloe-26l-seg.pt`
- `yolo26n-pose.pt`
- `MobileNetV3`

当前代码已实现模型适配层。YOLO 权重文件存在时会自动加载；没有权重时，服务仍可启动、创建任务、写数据库，方便先对接前后端与任务流程。多模态大模型默认作为独立 HTTP 服务调用，业务服务只发送一张代表场景帧和 YOLO/pose 汇总结果；后续可通过配置切换到自建 Qwen 服务、OpenAI-compatible 第三方接口，或本进程 local provider。

## 识别目标

1. 施工场景：机房内、铁塔附近、其他
2. 是否有登高作业
3. 是否有交底行为：一名人员对其他人员进行任务交代及安全事项强调
4. 是否都佩戴安全帽并穿反光衣
5. 是否有抽烟行为
6. 是否有动火行为：切割、电焊、火星、火花、明火等

其中第 4、5、6 项会生成实时告警记录；配置 `SUPERVISOR_ALERT_WEBHOOK_URL` 后会同步 POST 到外部告警接口。

## 实现架构

当前流程调整为“YOLO 先完整处理视频，远程多模态服务后置综合判断”：

1. 使用 YOLO 检测视频中的场景相关目标，生成场景标签签名；处理完整个视频后，只选取一张代表场景帧。
2. 使用 YOLO 检测人物、安全帽、反光衣、香烟、动火目标。
3. 使用 YOLO pose 提取人物关键点，和人物 track_id 一起写入轨迹缓存。
4. PPE 判断使用人物、安全帽、反光衣的数量和位置关系，并结合多帧 track 状态；如果人物头部/身体位于画面边缘、头部或身体出镜、身体被遮挡、目标过小等，会记录为例外，不直接判违规。
5. 抽烟判断需要同时出现香烟/烟雾目标和“手靠近面部”等 pose 动作候选。
6. 动火判断需要同时出现火焰/火星/焊接/切割等目标和对应作业动作候选。
7. YOLO 全部处理完成后，将一张代表场景帧、筛选后的检测统计、PPE 异常候选、抽烟/动火候选、人物轨迹和 pose 摘要一次性提交给多模态 API，判断场景、是否有登高作业、是否有交底场景。

离线视频和视频流共用同一条处理链路；视频流任务保留实时告警能力，离线视频任务处理完成后标记为 `completed`。

主要模块：

- `app/api.py`：FastAPI 接口，提交离线视频/视频流任务，查询任务、事件和告警。
- `app/processor.py`：打开视频源、抽帧、调用模型、入库。
- `app/inference/yolo.py`：Ultralytics YOLO 适配器，自动加载 `weights/yoloe-26l-seg.pt` 和 `weights/yolo26n-pose.pt`。
- `app/scene.py`：根据 YOLO 检测结果判断场景签名是否变化，为最终代表帧和摘要提供依据。
- `app/activity.py`：维护人物轨迹和 pose 缓存，生成抽烟/动火动作候选，并为最终多模态分析提供轨迹摘要。
- `app/tracking.py`：人员目标跟踪和 PPE 关联，降低安全帽/反光衣误报。
- `app/inference/qwen.py`：多模态确认接口，默认调用远程 HTTP API；也保留 `local` provider 用于本进程加载 Qwen3-VL。
- `app/rules.py`：将检测结果归并为六类业务事件，并对 PPE、抽烟、动火生成告警。
- `app/models.py`：数据库表结构。

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

默认数据库为 `sqlite:///./data/supervisor.db`。

## 权重与配置

把模型权重放到默认路径：

```text
weights/yoloe-26l-seg.pt
weights/yolo26n-pose.pt
weights/mobilenetv3.pth
```

也可以通过环境变量覆盖：

```bash
export SUPERVISOR_DATABASE_URL='sqlite:///./data/supervisor.db'
export SUPERVISOR_FRAME_SAMPLE_INTERVAL=15
export SUPERVISOR_SNAPSHOT_DIR='data/snapshots'
export SUPERVISOR_YOLO_SEG_MODEL='weights/yoloe-26l-seg.pt'
export SUPERVISOR_YOLO_POSE_MODEL='weights/yolo26n-pose.pt'
export SUPERVISOR_QWEN_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_QWEN_ENABLED=true
export SUPERVISOR_MULTIMODAL_PROVIDER='remote'
export SUPERVISOR_MULTIMODAL_API_URL='http://127.0.0.1:9001/v1/analyze'
export SUPERVISOR_MULTIMODAL_API_FORMAT='custom'
export SUPERVISOR_MULTIMODAL_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_QWEN_USE_MODELSCOPE=true
export SUPERVISOR_QWEN_MODELSCOPE_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_QWEN_CACHE_DIR='data/modelscope'
export SUPERVISOR_TRACKER_BACKEND='bytetrack'
export SUPERVISOR_PPE_REQUIRED_HITS=2
export SUPERVISOR_PPE_MISSING_TOLERANCE=2
export SUPERVISOR_PPE_EDGE_MARGIN_RATIO=0.03
export SUPERVISOR_SCENE_MIN_CHANGE_INTERVAL_MS=10000
export SUPERVISOR_ACTIVITY_ANALYSIS_INTERVAL_MS=5000
export SUPERVISOR_TRAJECTORY_WINDOW_MS=30000
export SUPERVISOR_LOG_LEVEL='INFO'
export SUPERVISOR_ALERT_WEBHOOK_URL='http://127.0.0.1:9000/alerts'
```

多模态服务配置：

- `SUPERVISOR_MULTIMODAL_PROVIDER=remote`：默认值，业务服务通过 HTTP API 调用独立多模态服务。
- `SUPERVISOR_MULTIMODAL_API_FORMAT=custom`：发送简单 JSON 请求，适合自建 Qwen 服务。
- `SUPERVISOR_MULTIMODAL_API_FORMAT=openai_compatible`：发送 OpenAI-compatible vision chat/completions 请求，适合切换第三方接口。
- `SUPERVISOR_MULTIMODAL_PROVIDER=local`：保留本进程加载 Qwen 的方式；此时才会使用 `SUPERVISOR_QWEN_USE_MODELSCOPE`、`SUPERVISOR_QWEN_CACHE_DIR` 等 ModelScope 配置。

自建多模态服务 `custom` 请求体：

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "prompt": "只输出JSON...",
  "image": "data:image/jpeg;base64,...",
  "image_base64": "...",
  "image_mime_type": "image/jpeg",
  "response_format": "json"
}
```

服务响应可以直接返回判断 JSON，也可以包在 `json`、`result`、`output`、`content`、`text` 字段里。期望字段：

```json
{
  "scene": "other",
  "scene_confidence": 0.8,
  "height_work": false,
  "height_work_confidence": 0.7,
  "briefing": false,
  "briefing_confidence": 0.6,
  "reason": "简短原因"
}
```

`SUPERVISOR_TRACKER_BACKEND` 可选：

- `bytetrack`：默认值，使用 Ultralytics 内置 ByteTrack。
- `deepsort`：使用 `deep-sort-realtime` 给人员框补充 track_id。
- 其他值或跟踪器不可用时：使用 `app/tracking.py` 内置 IoU 跟踪回退。

## API 示例

提交离线视频：

```bash
curl -X POST http://127.0.0.1:8000/videos/offline \
  -H 'Content-Type: application/json' \
  -d '{"path":"/data/videos/site.mp4","camera_id":"camera-001"}'
```

提交视频流：

```bash
curl -X POST http://127.0.0.1:8000/streams \
  -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://user:pass@host/stream","camera_id":"camera-rtsp-001"}'
```

查询任务、事件、告警：

```bash
curl http://127.0.0.1:8000/jobs
curl http://127.0.0.1:8000/events?job_id=1
curl http://127.0.0.1:8000/alerts?job_id=1
```

命令行处理单个视频：

```bash
python -m app.cli /data/videos/site.mp4 --camera-id camera-001
```

## 数据库表

- `video_jobs`：视频任务，包含来源、任务状态、摄像头编号、错误信息等。
- `detection_events`：识别事件，包含事件类型、值、置信度、视频时间戳、帧号和详细信息。
- `alert_events`：实时告警，包含告警类型、等级、消息和上下文。

截图默认保存到 `data/snapshots/job_{job_id}/`：

- `scene/`：YOLO 全量处理后选出的代表场景帧，也是提交给多模态 API 的图片。

截图路径会写入 `detection_events.details.snapshot_path`，便于前端回放和人工复核。

## 后续可增强项

- 接入 MobileNetV3 做抽烟/明火小目标二分类复核。
- 将视频流任务改为独立 worker 或队列消费，支持多路摄像头长期运行。
- 将多模态 API 返回结果缓存到帧级中间表，便于复核和调参。
- 将 PPE 遮挡例外从几何规则升级为分割 mask / 深度估计辅助判断。
