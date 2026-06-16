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

当前代码已实现模型适配层。YOLO 权重文件存在时会自动加载；没有权重时，服务仍可启动、创建任务、写数据库，方便先对接前后端与任务流程。Qwen3-VL 已接入 Transformers 推理，并默认通过 ModelScope 下载/缓存模型；依赖或模型不可用时会打印日志并回退到保守规则。

## 识别目标

1. 施工场景：机房内、铁塔附近、其他
2. 是否有登高作业
3. 是否有交底行为：一名人员对其他人员进行任务交代及安全事项强调
4. 是否都佩戴安全帽并穿反光衣
5. 是否有抽烟行为
6. 是否有动火行为：切割、电焊、火星、火花、明火等

其中第 4、5、6 项会生成实时告警记录；配置 `SUPERVISOR_ALERT_WEBHOOK_URL` 后会同步 POST 到外部告警接口。

## 实现架构

原始思路是“YOLO 实时识别 -> 跟踪筛选 -> 大模型确认 -> 重点项直接上报告警”。实现时做了两点优化：

- 将“检测、人员/PPE 跟踪关联、视觉语言确认、规则归并、告警推送、数据库写入”拆成独立模块。
- 离线视频和视频流共用同一条处理链路；视频流任务保留实时告警能力，离线视频任务处理完成后标记为 `completed`。
- PPE 检测不再只按单帧数量判断，而是优先使用 ByteTrack / DeepSORT track_id，并用内置 IoU 跟踪回退，把安全帽、反光衣与人员目标进行多帧稳定关联。

主要模块：

- `app/api.py`：FastAPI 接口，提交离线视频/视频流任务，查询任务、事件和告警。
- `app/processor.py`：打开视频源、抽帧、调用模型、入库。
- `app/inference/yolo.py`：Ultralytics YOLO 适配器，自动加载 `weights/yoloe-26l-seg.pt` 和 `weights/yolo26n-pose.pt`。
- `app/tracking.py`：人员目标跟踪和 PPE 关联，降低安全帽/反光衣误报。
- `app/inference/qwen.py`：Qwen3-VL 确认接口，用于场景、交底行为、登高作业二次确认。
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
export SUPERVISOR_YOLO_SEG_MODEL='weights/yoloe-26l-seg.pt'
export SUPERVISOR_YOLO_POSE_MODEL='weights/yolo26n-pose.pt'
export SUPERVISOR_QWEN_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_QWEN_ENABLED=true
export SUPERVISOR_QWEN_USE_MODELSCOPE=true
export SUPERVISOR_QWEN_MODELSCOPE_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_QWEN_CACHE_DIR='data/modelscope'
export SUPERVISOR_TRACKER_BACKEND='bytetrack'
export SUPERVISOR_PPE_REQUIRED_HITS=2
export SUPERVISOR_PPE_MISSING_TOLERANCE=2
export SUPERVISOR_LOG_LEVEL='INFO'
export SUPERVISOR_ALERT_WEBHOOK_URL='http://127.0.0.1:9000/alerts'
```

Qwen 模型加载顺序：

1. 如果 `SUPERVISOR_QWEN_MODEL` 是本地目录，直接从该目录加载。
2. 如果 `SUPERVISOR_QWEN_USE_MODELSCOPE=true`，使用 ModelScope `snapshot_download` 下载 `SUPERVISOR_QWEN_MODELSCOPE_MODEL`，并从本地缓存目录加载。
3. 如果 ModelScope 不可用或下载失败，回退为直接把 `SUPERVISOR_QWEN_MODEL` 交给 Transformers 加载；仍失败则使用规则回退。

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

## 后续可增强项

- 接入 MobileNetV3 做抽烟/明火小目标二分类复核。
- 将视频流任务改为独立 worker 或队列消费，支持多路摄像头长期运行。
- 将 Qwen3-VL 推理结果缓存到帧级中间表，便于复核和调参。
