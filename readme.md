# Supervisor Agent

面向工地施工视频的安全监督识别服务。服务接收离线视频或持续视频流，先使用 YOLO 系列模型完成目标检测、人员跟踪和 pose 结构化分析，再将筛选后的关键结果与一张代表场景帧提交给独立部署的多模态大模型服务，由大模型完成场景、登高作业、交底行为等综合判断。

## 目标

接收一个视频源，支持两种形式：

- 离线视频：例如 `.mp4`、`.avi`、`.mov` 等文件。
- 持续视频流：例如 RTSP、RTMP、HTTP-FLV、摄像头实时流。

需要识别并记录以下内容：

1. 是否存在人员未佩戴安全帽或未穿反光衣，一旦确认触发告警上报。
2. 是否存在抽烟行为，一旦确认触发告警上报。
3. 是否存在动火行为，包括切割、焊接、火星、火花、明火等，一旦确认触发告警上报。
4. 视频所处施工场景，例如机房、铁塔附近、其他场景。
5. 是否存在登高作业，记录开始时间、结束时间和截图。
6. 是否存在交底行为，记录开始时间、结束时间、参与人数和截图。

其中 1、2、3 属于实时安全告警类事件，应尽量在 YOLO 阶段快速触发；4、5、6 属于场景和复杂行为判断，先由 YOLO/pose 完成结构化筛选，再提交给多模态大模型综合判断。

## 模型选择

- `Qwen/Qwen3-VL-8B-Instruct`：多模态大模型，用于场景、登高作业、交底行为的综合判断。
- `yoloe-26l-seg.pt`：目标检测/分割，用于人物、安全帽、反光衣、香烟、火焰、火星、焊接、切割、场景相关目标识别。
- `yolo26n-pose.pt`：人体姿态识别，用于动作分析，例如手靠近面部、作业姿态、登高姿态等。
- `MobileNetV3`：可作为轻量二分类复核模型，用于抽烟、动火、小目标误报过滤。

## 总体方案

整体采用“YOLO 先完整处理视频，多模态大模型后置综合判断”的方案。

核心原则：

- YOLO 负责高频、实时、结构化检测。
- 跟踪器负责把跨帧人物、PPE 和 pose 关联起来。
- 规则层负责对明确安全风险进行快速告警。
- 多模态大模型单独部署，通过 API 调用，便于后续替换为第三方模型服务。
- 大模型不逐帧调用，只接收一张代表场景帧和筛选后的结构化摘要，降低成本和延迟。

推荐架构：

```text
视频文件/视频流
  -> 抽帧
  -> YOLO 检测：人物、安全帽、反光衣、香烟、动火、场景目标
  -> YOLO Pose：人体关键点
  -> 人员跟踪：ByteTrack / DeepSORT / IoU fallback
  -> 规则筛选：PPE、抽烟、动火候选
  -> 聚合摘要：场景签名、轨迹、pose、候选事件、关键时间点
  -> 代表场景帧 + 聚合摘要
  -> 多模态大模型 API
  -> 场景/登高/交底最终判断
  -> 数据库存储 + 告警上报
```

## 代码架构说明

推荐采用分层架构，避免模型推理、业务规则、数据库和 API 混在一起。

```text
app/
  api.py                 # HTTP API：提交任务、查询任务、查询事件、查询告警
  cli.py                 # 命令行入口：处理单个离线视频
  config.py              # 配置：模型路径、抽帧间隔、告警地址、多模态API配置
  db.py                  # 数据库连接和 Session 管理
  models.py              # 数据库表结构：任务、事件、告警
  schemas.py             # API 请求/响应 schema
  processor.py           # 主处理流程：读视频、抽帧、调度检测、聚合结果、入库
  labels.py              # 标签集合：人员、安全帽、反光衣、抽烟、动火等
  rules.py               # 规则引擎：PPE、抽烟、动火告警和事件生成
  tracking.py            # 人员跟踪与 PPE 空间关联
  activity.py            # 人物轨迹、pose 缓存、候选片段生成
  scene.py               # 场景签名、代表场景帧选择
  alerts.py              # 告警 webhook 上报
  logging_config.py      # 日志配置
  inference/
    base.py              # 推理通用数据结构：Detection、FrameContext、PoseObservation
    yolo.py              # YOLO/YOLO-pose 适配器
    qwen.py              # 多模态 API 适配器，支持 OpenAI-compatible 接口
tests/
  test_rules.py          # 规则层和跟踪关联的轻量测试
```

### 核心分层

### API 层

职责：

- 接收离线视频任务和视频流任务。
- 返回任务状态。
- 查询识别事件和告警事件。

建议文件：

- `app/api.py`
- `app/schemas.py`

API 层只负责请求校验和任务创建，不直接写视频处理逻辑。

### 任务处理层

职责：

- 打开视频文件或视频流。
- 按配置间隔抽帧。
- 调用 YOLO 检测和 pose 推理。
- 调用跟踪、规则、候选片段生成、聚合摘要。
- 调用多模态 API。
- 写入数据库。

建议文件：

- `app/processor.py`

`processor.py` 是编排层，不应包含复杂规则细节；规则细节应放到 `rules.py`、`tracking.py`、`activity.py`。

### 推理适配层

职责：

- 隔离模型调用细节。
- 统一输出结构化结果。
- 后续替换模型时不影响业务流程。

建议文件：

- `app/inference/yolo.py`
- `app/inference/qwen.py`
- `app/inference/base.py`

YOLO 适配器输出统一 `Detection`：

```json
{
  "label": "person",
  "confidence": 0.92,
  "bbox": [120, 80, 360, 720],
  "track_id": 12,
  "source": "yolo_seg",
  "metadata": {}
}
```

多模态适配器建议只暴露业务方法：

- `analyze_scene(...)`
- `analyze_height_work_candidates(...)`
- `analyze_briefing_candidates(...)`

内部再转换成 OpenAI-compatible API 请求。

### 跟踪与候选层

职责：

- 根据 `track_id` 或 IoU 关联跨帧人物。
- 将安全帽、反光衣和人员位置关联。
- 缓存人物轨迹和 pose。
- 根据目标、pose、轨迹生成登高和交底候选片段。

建议文件：

- `app/tracking.py`
- `app/activity.py`
- `app/scene.py`

候选层输出的是结构化片段，不直接做最终复杂行为结论。

### 规则层

职责：

- PPE 违规判断。
- 抽烟候选确认。
- 动火候选确认。
- 生成事件对象和告警对象。

建议文件：

- `app/rules.py`
- `app/labels.py`

规则层要尽量保持可测试，不依赖数据库、HTTP、模型实例。

### 数据层

职责：

- 管理任务、识别事件、告警事件。
- 保存多模态结果、候选片段详情、截图路径。

建议文件：

- `app/db.py`
- `app/models.py`

数据库表不应依赖具体模型版本，模型输出细节可放在 `details` JSON 字段中。

### 告警层

职责：

- 将 PPE、抽烟、动火等实时风险推送给外部系统。
- 与识别入库解耦，避免 webhook 失败影响主流程。

建议文件：

- `app/alerts.py`

### 数据流

离线视频的数据流：

```text
POST /videos/offline
  -> VideoJob
  -> VideoProcessor.process_job
  -> YoloDetector.detect
  -> PPETracker.update
  -> TrajectoryBuffer.update
  -> Rules.generate_findings
  -> CandidateBuilder.build_height_work / build_briefing
  -> MultimodalClient.chat_completions
  -> DetectionEvent / AlertEvent
```

视频流的数据流：

```text
POST /streams
  -> VideoJob
  -> 持续抽帧
  -> YOLO + pose + tracking
  -> 实时规则告警
  -> 滑动窗口候选片段
  -> 周期性多模态判断
  -> 窗口级事件入库
```

### 推荐接口边界

YOLO 检测接口：

```python
class ObjectDetector:
    def detect(self, frame) -> list[Detection]:
        ...
```

候选片段生成接口：

```python
class CandidateBuilder:
    def update(self, frame_context, detections, pose_observations) -> None:
        ...

    def flush(self) -> list[CandidateSegment]:
        ...
```

多模态接口：

```python
class MultimodalClient:
    def analyze_scene(self, image, summary) -> dict:
        ...

    def analyze_height_work(self, image, candidates) -> dict:
        ...

    def analyze_briefing(self, image, candidates) -> dict:
        ...
```

事件写入接口：

```python
class EventWriter:
    def write_detection_event(self, event) -> None:
        ...

    def write_alert_event(self, alert) -> None:
        ...
```

### 代码实现原则

- 模型适配和业务规则分离。
- 多模态 API 适配和 prompt 构造分离。
- 候选片段生成和大模型确认分离。
- 实时告警和最终分析分离。
- 数据库写入和 webhook 上报分离。
- 所有中间结果尽量结构化，便于调试和人工复核。

## 模块职责

### 视频接入层

负责接收离线视频路径或视频流地址，并创建处理任务。

离线视频处理特点：

- 任务有明确开始和结束。
- 可以在视频处理完成后一次性生成最终分析结果。
- 适合完整输出场景、登高、交底的开始和结束时间。

视频流处理特点：

- 任务长期运行。
- PPE、抽烟、动火需要实时上报。
- 场景、登高、交底可以按滑动时间窗口周期性分析。

### YOLO 检测层

负责对抽样帧进行目标检测，输出结构化结果：

- 人物框：`person`
- 安全帽：`helmet`、`hardhat`、`safety_helmet`
- 反光衣：`vest`、`reflective_vest`、`safety_vest`
- 香烟/烟雾：`cigarette`、`smoking`、`smoke`
- 动火目标：`fire`、`flame`、`spark`、`welding`、`cutting`
- 场景目标：机柜、设备柜、铁塔、脚手架、梯子、升降设备等

每条检测结果建议包含：

```json
{
  "label": "person",
  "confidence": 0.92,
  "bbox": [120, 80, 360, 720],
  "track_id": 12,
  "frame_index": 150,
  "timestamp_ms": 6000
}
```

### Pose 分析层

使用 `yolo26n-pose.pt` 提取人体关键点，重点关注：

- 手腕、手肘、肩膀、头部位置。
- 手是否靠近口鼻区域。
- 双手是否处于作业姿态。
- 人体是否处于梯子、脚手架、平台等高处区域附近。
- 多人是否面向同一人或形成交底队形。

pose 数据不直接作为最终复杂行为结论，而是作为大模型输入摘要和规则候选依据。

### 跟踪与关联层

推荐优先使用 ByteTrack 或 DeepSORT，将跨帧人物稳定关联。

跟踪目标：

- 人物轨迹。
- 人物与安全帽的空间关联。
- 人物与反光衣的空间关联。
- 人物与香烟、火星、焊接点等目标的空间关联。
- 人物 pose 的时间序列。

PPE 判断不能只比较数量，例如 “3 个人、2 个安全帽” 不能直接判定一定有人未戴安全帽。应结合位置关系和遮挡例外。

### PPE 判断规则

安全帽判断：

- 安全帽框应位于人物头部区域附近。
- 如果头部区域在画面上边缘，可能是头部出镜，不直接判未戴。
- 如果头部被遮挡或目标过小，不直接判未戴。
- 多帧连续缺失时再确认违规。

反光衣判断：

- 反光衣框应位于人物躯干区域。
- 如果躯干区域被遮挡，不直接判未穿。
- 如果人物身体部分在镜头外，不直接判未穿。
- 多帧连续缺失时再确认违规。

建议记录例外原因：

- `head_out_of_frame`
- `body_out_of_frame`
- `head_occluded`
- `body_occluded`
- `person_too_small`
- `low_confidence`

### 抽烟判断规则

抽烟不应只依赖香烟目标出现，应结合人物动作。

候选条件：

- YOLO 检测到香烟、烟雾或类似小目标。
- 目标位于人物手部或面部附近。
- pose 显示手腕靠近口鼻区域。
- 多帧连续出现或在短时间窗口内重复出现。

满足以上条件后生成抽烟告警。

### 动火判断规则

动火也不应只依赖火星或火焰目标出现，应结合人物作业动作。

候选条件：

- YOLO 检测到火焰、火星、焊接、切割等目标。
- 目标位于人物手部、工具或作业面附近。
- pose 显示人员处于作业姿态。
- 目标在连续帧中出现，或亮点区域随作业动作变化。

满足以上条件后生成动火告警。

### 场景判断

场景判断由 YOLO 先生成场景签名，再由多模态大模型最终判断。

YOLO 场景签名可包含：

- 机房：机柜、服务器柜、设备柜、线缆、室内设备等。
- 铁塔附近：铁塔、塔身、抱杆、室外基站、天线等。
- 其他：无法归入上述类别的施工环境。

处理方式：

1. YOLO 完整处理视频。
2. 根据检测结果选取一张最具代表性的场景帧。
3. 将代表帧和场景目标统计提交给多模态 API。
4. 多模态模型返回 `machine_room`、`near_tower` 或 `other`。

### 登高作业判断

登高作业属于复杂行为，建议由 YOLO 先筛选候选，再由大模型判断。登高场景不一定存在，因此不建议把登高和交底强行放在同一次大模型判断里；如果 YOLO 没有发现梯子、脚手架、铁塔、高处平台、升降设备、明显垂直位移等候选，应直接跳过登高大模型调用，或只记录为 `height_work=false`。

登高的开始和结束时间不应由大模型从完整视频中自由判断，而应先由目标识别、pose 和轨迹生成候选时间段。大模型只对候选时间段进行确认、剔除误报，并在候选边界附近做微调。

YOLO/pose 候选依据：

- 人物靠近梯子、脚手架、铁塔、高处平台、升降设备。
- 人物 bbox 的垂直位置异常，处于画面高处区域。
- 人物轨迹存在明显上升或下降。
- pose 显示攀爬、站立高处、身体伸展作业等姿态。

提交给大模型的信息：

- 代表场景帧。
- 候选时间段列表。
- 每个候选时间段内的人物轨迹。
- 每个候选时间段内的关键 pose 样本。
- 梯子、脚手架、铁塔等相关目标统计。
- YOLO 初选的候选开始时间和结束时间。
- 候选片段截图，例如开始帧、中间帧、结束帧。

大模型输出：

```json
{
  "height_work": true,
  "height_work_confidence": 0.86,
  "height_work_start_ms": 12000,
  "height_work_end_ms": 48000,
  "reason": "人员位于梯子上并持续进行高处作业"
}
```

### 交底行为判断

交底行为需要结合人数、站位、姿态、手持物品和场景上下文。交底不一定是多人围站的强特征场景；当画面中总共只有 2 个人时，不能只依赖“一人面向多人”的模式，需要重点观察是否存在本子、手稿、交底单、签字板等物品，以及交底结束后的签字动作。

交底的开始和结束时间同样应先由 YOLO、pose 和轨迹初选。大模型不直接在整段视频上寻找交底，而是接收候选片段的结构化摘要，判断该片段是否为交底，并修正开始/结束时间。

候选依据：

- 同一时间窗口内出现多人。
- 一人面向多人，或多人围绕一人。
- 人员相对静止，姿态不像普通施工动作。
- 可能出现手势讲解、指向、围站等动作。
- 人员手上出现本子、手稿、纸张、夹板、签字板、平板等适合交底记录的物品。
- 交底结束阶段可能出现签字动作，例如一人持笔、另一人持纸/本子，或多人依次靠近同一份材料。
- 当总人数只有 2 人时，重点结合“讲解动作 + 文档类物品 + 签字/确认动作 + 停留时间”综合判断。
- 如果只有两人短暂同框、无文档类物品、无签字动作、无持续讲解姿态，应降低交底置信度。

提交给大模型的信息：

- 代表场景帧。
- 人员数量统计。
- 候选时间段列表。
- 每个候选时间段内的主要人物轨迹。
- 每个候选时间段内的 pose 摘要。
- YOLO 初选的候选开始时间和结束时间。
- 文档类物品检测结果，例如 `notebook`、`paper`、`clipboard`、`document`、`pen`、`tablet`。
- 签字候选片段，例如人员手部靠近纸张/本子、多人围绕同一文档、交底末尾出现书写动作。
- 候选阶段划分，例如 `explain_phase`、`confirm_phase`、`signature_phase`。

大模型输出：

```json
{
  "briefing": true,
  "briefing_confidence": 0.82,
  "briefing_start_ms": 3000,
  "briefing_end_ms": 26000,
  "participant_count": 5,
  "has_document": true,
  "has_signature_scene": true,
  "signature_start_ms": 22000,
  "signature_end_ms": 26000,
  "reason": "一名人员面向多人进行说明，其他人员集中站立听取"
}
```

两人交底场景的示例判断：

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
  "reason": "画面中两人持续停留，一人持文档讲解，末尾出现手部靠近纸张的签字确认动作"
}
```

## 多模态调用拆分策略

场景、登高、交底建议拆成不同的多模态任务，而不是统一放进一个大 prompt 中判断。

拆分原因：

- 登高场景可能不存在，强行让大模型判断登高会增加无效调用和误判。
- 交底关注人员关系、文档类物品和签字动作，和登高关注的空间高度、梯子、脚手架、垂直轨迹不同。
- 拆分 prompt 后，每个任务的输入更聚焦，输出字段更稳定，后续切换第三方模型也更容易。

推荐调用顺序：

1. 场景识别任务：始终调用一次，输入代表场景帧和场景目标统计。
2. 登高判断任务：只有存在登高候选时调用，例如梯子、脚手架、铁塔、高处平台、升降设备、明显垂直轨迹。
3. 交底判断任务：只有存在交底候选时调用，例如多人聚集、两人长时间停留、文档类物品、签字候选动作。

如果 YOLO 阶段没有筛出候选：

- 登高：直接输出 `height_work=false`，不调用大模型。
- 交底：如果人数不足 2 人，直接输出 `briefing=false`；如果刚好 2 人但检测到文档或签字候选，可以调用大模型进一步判断。

## 第三方模型判断规则契约

接入第三方多模态模型时，必须把登高和交底的判断规则写入 system prompt 或 developer prompt，不能只让模型根据图片自由判断。第三方模型只负责在候选片段内做确认和边界微调，不负责从完整视频中寻找候选。

通用约束：

- 只判断业务服务提交的候选片段。
- 不得臆测候选片段外发生的事情。
- 如果图片证据和结构化证据冲突，应以结构化证据为主，并在 `reason` 中说明。
- 如果证据不足，应返回 `confirmed=false` 或对应布尔字段为 `false`。
- 开始时间和结束时间只能在候选片段边界附近微调，不得生成超出候选片段太多的时间。
- 输出必须是合法 JSON，不要输出解释性自然语言。

### 登高判断规则

第三方模型判断登高时，应遵守以下规则。

可以判定为登高的情况：

- 人物位于梯子、脚手架、铁塔、高处平台、升降设备等高处结构上或附近。
- 人物轨迹显示明显上升、下降或长时间停留在高处区域。
- pose 显示攀爬、站立高处、身体伸展作业、双手在高处作业等动作。
- 目标识别结果中存在梯子、脚手架、铁塔、高处平台、升降设备等支撑证据。
- 候选片段持续时间足够，不是单帧偶然误检。

不应判定为登高的情况：

- 只有人物出现在画面上方，但没有高处结构或垂直轨迹证据。
- 只是摄像头角度导致人物看起来较高。
- 只有铁塔或梯子目标，但没有人员靠近或攀爬。
- 只有单帧检测到疑似高处动作，缺少连续轨迹。
- 人物只是路过梯子、脚手架或铁塔附近，没有作业动作。

登高 prompt 规则示例：

```text
你正在判断工地视频候选片段是否存在登高作业。
只允许根据输入的候选片段、图片、YOLO目标、人物轨迹和pose摘要判断。
如果没有人员靠近高处结构、没有垂直移动、没有攀爬或高处作业pose，不得判定为登高。
如果只有高处结构但没有人员作业，也不得判定为登高。
开始时间和结束时间只能在候选片段 start_ms/end_ms 附近微调。
只输出合法JSON。
```

登高输出建议：

```json
{
  "confirmed_candidates": [
    {
      "candidate_id": "height_work_001",
      "confirmed": true,
      "confidence": 0.86,
      "start_ms": 12000,
      "end_ms": 48000,
      "evidence": [
        "person_near_ladder",
        "vertical_movement",
        "climb_pose"
      ],
      "reject_reason": null,
      "reason": "人员靠近梯子并出现持续攀爬和高处作业姿态"
    }
  ]
}
```

### 交底判断规则

第三方模型判断交底时，应遵守以下规则。

可以判定为交底的情况：

- 多人持续聚集，形成一人讲解、多人听取或围站确认关系。
- 人员相对静止，持续时间较长，不像普通路过或施工动作。
- 存在讲解手势、指向动作、多人注视同一人或同一材料。
- 出现本子、纸张、手稿、交底单、夹板、签字板、笔、平板等文档类物品。
- 片段后段出现签字、确认、查看文档等动作。
- 当只有 2 人时，必须重点依赖文档类物品、讲解姿态、签字动作和持续停留时间。

不应判定为交底的情况：

- 只有多人同框，但没有持续停留、讲解、文档或签字证据。
- 只有两人短暂同框或普通交谈，未见交底材料。
- 人员正在明显施工、搬运、焊接、切割等，不应误判为交底。
- 只有文档类物品，但没有人员围绕、讲解或签字动作。
- 签字动作不明确，且没有前序讲解或确认过程。

交底 prompt 规则示例：

```text
你正在判断工地视频候选片段是否存在交底行为。
交底通常包括任务说明、安全事项强调、人员确认，末尾可能出现签字。
如果人数较多，应重点判断是否存在一人讲解、多人听取或围站确认。
如果总人数只有2人，不能依赖“一人对多人”，必须结合文档类物品、讲解姿态、持续停留和签字/确认动作。
只有短暂同框、普通交谈、普通施工动作，不得判定为交底。
开始时间和结束时间只能在候选片段 start_ms/end_ms 附近微调。
只输出合法JSON。
```

交底输出建议：

```json
{
  "confirmed_candidates": [
    {
      "candidate_id": "briefing_001",
      "confirmed": true,
      "confidence": 0.82,
      "start_ms": 5000,
      "end_ms": 34000,
      "participant_count": 2,
      "has_document": true,
      "has_signature_scene": true,
      "signature_start_ms": 28000,
      "signature_end_ms": 34000,
      "evidence": [
        "document_visible",
        "explain_gesture",
        "signature_motion"
      ],
      "reject_reason": null,
      "reason": "两人持续停留，一人持文档讲解，末尾出现签字确认动作"
    }
  ]
}
```

### 置信度规则

建议统一置信度含义：

- `0.9-1.0`：图片和结构化证据都很充分。
- `0.7-0.9`：主要证据充分，但局部证据不完整。
- `0.5-0.7`：存在候选迹象，但仍有不确定性。
- `<0.5`：证据不足，不建议确认为事件。

业务侧建议只在 `confirmed=true` 且 `confidence >= 0.7` 时写入正式登高/交底事件；低于阈值可写入候选记录或人工复核队列。

## 候选时间段初选

登高和交底都需要先做候选时间段初选。初选由 YOLO、pose、跟踪和规则完成，大模型只处理候选片段。

这样做的原因：

- 大模型不适合直接从长视频中定位开始和结束。
- 候选片段能显著降低大模型输入量和调用成本。
- 候选片段携带结构化证据，判断更稳定。
- 开始和结束时间可由轨迹状态变化、目标出现/消失和动作阶段变化初步确定。

### 候选片段生成方式

对每个抽样帧生成一组状态：

```json
{
  "timestamp_ms": 12000,
  "frame_index": 300,
  "people": [],
  "objects": [],
  "poses": [],
  "signals": {
    "near_ladder": true,
    "vertical_movement": true,
    "group_static": false,
    "document_visible": false,
    "signature_motion": false
  }
}
```

再按时间合并连续状态，形成候选片段：

```json
{
  "candidate_id": "height_work_001",
  "task": "height_work",
  "start_ms": 12000,
  "end_ms": 48000,
  "duration_ms": 36000,
  "key_frame_ms": 26000,
  "evidence": {
    "tracks": [12],
    "objects": ["ladder"],
    "pose_signals": ["climb_pose", "arms_extended"],
    "trajectory_signals": ["vertical_movement"]
  },
  "snapshots": {
    "start": "data/snapshots/job_1/height_work/12000.jpg",
    "middle": "data/snapshots/job_1/height_work/26000.jpg",
    "end": "data/snapshots/job_1/height_work/48000.jpg"
  }
}
```

### 登高候选片段

生成登高候选的典型信号：

- 人物靠近梯子、脚手架、铁塔、高处平台、升降设备。
- 同一 `track_id` 的人物 bbox 出现明显垂直移动。
- 人物长期位于画面高处区域。
- pose 出现攀爬、伸手作业、站立高处等动作。
- 人物与高处结构之间的空间关系持续存在。

开始时间：

- 第一次出现“人物 + 高处结构 + 登高 pose/垂直移动”组合信号的时间。

结束时间：

- 人物离开高处结构。
- 垂直移动结束并回到地面区域。
- 登高 pose 信号连续消失超过阈值。

### 交底候选片段

生成交底候选的典型信号：

- 同一时间窗口内至少 2 人持续出现。
- 多人相对静止，形成面对面、围站或一对多关系。
- 出现手势讲解、指向、注视同一目标等 pose 信号。
- 出现本子、纸张、手稿、夹板、签字板、笔、平板等文档类物品。
- 末尾出现签字候选动作，例如手靠近纸张、多人依次靠近同一文档。

开始时间：

- 人员聚集并进入稳定讲解/确认状态的时间。
- 如果有文档类物品，则以文档出现且人员停留的时间作为候选起点。

结束时间：

- 人员散开。
- 文档类物品消失。
- 签字动作结束。
- 交底相关 pose 信号连续消失超过阈值。

### 提交给大模型的候选数据

大模型请求中不只提交一张图，还应提交候选片段结构化数据：

```json
{
  "task": "briefing",
  "image": "data:image/jpeg;base64,...",
  "candidates": [
    {
      "candidate_id": "briefing_001",
      "start_ms": 5000,
      "end_ms": 34000,
      "key_frame_ms": 28000,
      "participant_count": 2,
      "tracks": [3, 7],
      "document_labels": ["paper", "pen"],
      "pose_signals": ["explain_gesture", "hand_near_document"],
      "signature_candidates": [
        {
          "start_ms": 28000,
          "end_ms": 34000,
          "description": "hand near paper, possible signing"
        }
      ]
    }
  ],
  "response_schema": {
    "confirmed_candidates": [
      {
        "candidate_id": "briefing_001",
        "confirmed": true,
        "start_ms": 5000,
        "end_ms": 34000,
        "confidence": 0.0,
        "reason": "string"
      }
    ]
  }
}
```

### 场景识别请求

```json
{
  "task": "scene",
  "image": "data:image/jpeg;base64,...",
  "summary": {
    "scene_labels": [["tower", 12], ["cabinet", 3]],
    "top_labels": [["person", 80], ["helmet", 70]]
  },
  "response_schema": {
    "scene": "machine_room | near_tower | other",
    "scene_confidence": 0.0,
    "reason": "string"
  }
}
```

### 登高判断请求

```json
{
  "task": "height_work",
  "image": "data:image/jpeg;base64,...",
  "summary": {
    "height_work_candidates": [
      {
        "track_id": 12,
        "start_ms": 12000,
        "end_ms": 48000,
        "near_labels": ["ladder"],
        "vertical_delta": -180,
        "pose_samples": []
      }
    ]
  },
  "response_schema": {
    "height_work": false,
    "height_work_confidence": 0.0,
    "height_work_start_ms": null,
    "height_work_end_ms": null,
    "reason": "string"
  }
}
```

### 交底判断请求

```json
{
  "task": "briefing",
  "image": "data:image/jpeg;base64,...",
  "summary": {
    "participant_candidates": 2,
    "document_labels": ["notebook", "paper", "pen"],
    "signature_candidates": [
      {
        "start_ms": 28000,
        "end_ms": 34000,
        "description": "hand near document"
      }
    ],
    "pose_samples": []
  },
  "response_schema": {
    "briefing": false,
    "briefing_confidence": 0.0,
    "briefing_start_ms": null,
    "briefing_end_ms": null,
    "participant_count": 0,
    "has_document": false,
    "has_signature_scene": false,
    "signature_start_ms": null,
    "signature_end_ms": null,
    "reason": "string"
  }
}
```

## 多模态大模型服务

大模型建议单独部署为服务，主业务服务通过 HTTP API 调用。

这样做的好处：

- YOLO 服务和大模型服务可以独立扩缩容。
- 大模型可以部署在单独 GPU 机器上。
- 后续可以切换为第三方多模态接口。
- 主业务服务不需要直接安装和加载大模型权重。

### 请求方式

业务服务对外调用多模态大模型时，建议优先采用通用的 OpenAI-compatible Chat Completions / Vision API 形态，而不是自定义 `/analyze` 请求体。这样可以适配自建 Qwen 服务、vLLM、SGLang、Ollama、DashScope 兼容层，以及其他第三方多模态接口。

推荐分两层设计：

- 内部任务对象：业务系统内部保留结构化字段，例如 `task`、候选片段、轨迹、pose、候选截图。
- 外部通用 API：调用第三方时，将内部任务对象序列化到 `messages[].content[].text`，图片放到 `image_url`，请求发送到 `/v1/chat/completions`。

内部任务对象示例：

```json
{
  "task": "briefing",
  "summary": {
    "sampled_frames": 320,
    "top_labels": [["person", 180], ["helmet", 150], ["tower", 30]],
    "participant_candidates": 2,
    "document_labels": ["paper", "pen"],
    "signature_candidates": [],
    "trajectory_pose_summary": {}
  },
  "response_format": "json"
}
```

外部通用请求示例：

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
          "text": "任务类型：briefing。请根据图片和结构化摘要判断是否存在交底行为，并返回指定JSON字段。\n结构化摘要：{\"sampled_frames\":320,\"participant_candidates\":2,\"document_labels\":[\"paper\",\"pen\"],\"signature_candidates\":[],\"trajectory_pose_summary\":{}}"
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

接口约定：

- URL 推荐：`POST /v1/chat/completions`
- 鉴权推荐：`Authorization: Bearer <api_key>`
- 图片推荐：`data:image/jpeg;base64,...`；如果第三方支持公网 URL，也可以传图片 URL。
- 输出推荐：使用 `response_format={"type":"json_object"}`；不支持该字段的服务，则通过 prompt 强约束只输出 JSON。
- 超时建议：离线视频任务 60-180 秒；视频流窗口任务 10-30 秒。

### 响应格式

通用 Chat Completions 通常会把 JSON 放在 `choices[0].message.content` 中：

```json
{
  "choices": [
    {
      "message": {
        "content": "{\"briefing\":false,\"briefing_confidence\":0.73,\"reason\":\"未发现持续讲解或签字确认动作\"}"
      }
    }
  ]
}
```

业务服务需要从 `content` 中解析 JSON。推荐多模态服务按任务返回对应 JSON。场景任务返回：

```json
{
  "scene": "near_tower",
  "scene_confidence": 0.91,
  "reason": "画面中存在铁塔和室外基站环境"
}
```

登高任务返回：

```json
{
  "height_work": true,
  "height_work_confidence": 0.84,
  "height_work_start_ms": 12000,
  "height_work_end_ms": 48000,
  "reason": "人员在塔体附近高处作业"
}
```

交底任务返回：

```json
{
  "briefing": false,
  "briefing_confidence": 0.73,
  "briefing_start_ms": null,
  "briefing_end_ms": null,
  "participant_count": 0,
  "has_document": false,
  "has_signature_scene": false,
  "signature_start_ms": null,
  "signature_end_ms": null,
  "reason": "未发现持续讲解或签字确认动作"
}
```

## 事件与告警

### PPE 告警

事件类型：`ppe`

触发条件：

- 人员稳定跟踪成功。
- 安全帽或反光衣在多帧内持续缺失。
- 未命中遮挡、出镜、目标过小等例外条件。

建议详情：

```json
{
  "person_count": 3,
  "helmet_count": 2,
  "vest_count": 3,
  "missing_helmet": true,
  "missing_vest": false,
  "tracked_people": [],
  "exempt_people": []
}
```

### 抽烟告警

事件类型：`smoking`

触发条件：

- 检测到香烟或烟雾目标。
- pose 显示手靠近口鼻。
- 目标与对应人物位置关联。

### 动火告警

事件类型：`hot_work`

触发条件：

- 检测到火焰、火星、焊接或切割目标。
- pose 显示人员处于作业动作。
- 目标与人物、工具或作业面位置关联。

### 场景事件

事件类型：`scene`

由多模态服务返回最终场景：

- `machine_room`
- `near_tower`
- `other`

### 登高作业事件

事件类型：`height_work`

记录：

- 是否存在登高作业。
- 开始时间。
- 结束时间。
- 关键截图。
- 大模型判断原因。

### 交底事件

事件类型：`briefing`

记录：

- 是否存在交底行为。
- 开始时间。
- 结束时间。
- 参与人数。
- 关键截图。
- 大模型判断原因。

## 数据库存储建议

建议至少包含三类表。

### video_jobs

记录视频处理任务：

- `id`
- `source`
- `source_type`
- `camera_id`
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `error`

### detection_events

记录识别事件：

- `id`
- `job_id`
- `camera_id`
- `event_type`
- `value`
- `confidence`
- `timestamp_ms`
- `frame_index`
- `details`
- `created_at`

### alert_events

记录实时告警：

- `id`
- `job_id`
- `camera_id`
- `alert_type`
- `severity`
- `message`
- `timestamp_ms`
- `frame_index`
- `details`
- `created_at`

## API 设计建议

### 提交离线视频

```http
POST /videos/offline
Content-Type: application/json

{
  "path": "/data/videos/site.mp4",
  "camera_id": "camera-001"
}
```

### 提交视频流

```http
POST /streams
Content-Type: application/json

{
  "url": "rtsp://user:pass@host/stream",
  "camera_id": "camera-rtsp-001"
}
```

### 查询任务

```http
GET /jobs/{job_id}
```

### 查询事件

```http
GET /events?job_id=1
```

### 查询告警

```http
GET /alerts?job_id=1
```

## 配置建议

```bash
export SUPERVISOR_DATABASE_URL='sqlite:///./data/supervisor.db'
export SUPERVISOR_FRAME_SAMPLE_INTERVAL=15
export SUPERVISOR_SNAPSHOT_DIR='data/snapshots'

export SUPERVISOR_YOLO_SEG_MODEL='weights/yoloe-26l-seg.pt'
export SUPERVISOR_YOLO_POSE_MODEL='weights/yolo26n-pose.pt'
export SUPERVISOR_DETECTION_CONFIDENCE=0.35

export SUPERVISOR_TRACKER_BACKEND='bytetrack'
export SUPERVISOR_PPE_REQUIRED_HITS=2
export SUPERVISOR_PPE_MISSING_TOLERANCE=2
export SUPERVISOR_PPE_EDGE_MARGIN_RATIO=0.03

export SUPERVISOR_MULTIMODAL_PROVIDER='remote'
export SUPERVISOR_MULTIMODAL_API_URL='http://127.0.0.1:9001/v1/chat/completions'
export SUPERVISOR_MULTIMODAL_API_FORMAT='openai_compatible'
export SUPERVISOR_MULTIMODAL_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export SUPERVISOR_MULTIMODAL_API_TIMEOUT=60

export SUPERVISOR_ALERT_WEBHOOK_URL='http://127.0.0.1:9000/alerts'
export SUPERVISOR_LOG_LEVEL='INFO'
```

## 当前实现与启动方式

当前代码已经实现主服务骨架：

- FastAPI API：提交离线视频、提交视频流、查询任务、查询事件、查询告警。
- SQLite 数据库：任务表、识别事件表、告警事件表。
- YOLO 适配器：加载 `yoloe-26l-seg.pt` 和 `yolo26n-pose.pt`，输出统一检测结构。
- PPE 跟踪关联：人员、安全帽、反光衣空间关联，支持出画/遮挡例外。
- 实时规则：PPE、抽烟、动火告警。
- 候选片段：登高和交底候选片段初选。
- 场景聚合：选择代表场景帧。
- 多模态 API：通过 OpenAI-compatible `/v1/chat/completions` 调用第三方或本地 Qwen 服务。

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

启动 API 服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

命令行处理单个视频：

```bash
python -m app.cli /data/videos/site.mp4 --camera-id camera-001
```

如果 YOLO 权重不存在，服务仍可启动，但检测结果为空，适合先调试 API、数据库和任务流程。

如果未配置 `SUPERVISOR_MULTIMODAL_API_URL`，场景、登高、交底会使用空结果或保守降级结果，不影响 PPE、抽烟、动火规则链路。

## 推荐处理流程

### 离线视频流程

1. 创建视频任务。
2. 打开视频文件。
3. 按固定间隔抽帧。
4. YOLO 检测目标。
5. YOLO pose 提取人体关键点。
6. 跟踪人物并关联安全帽、反光衣。
7. 根据规则实时生成 PPE、抽烟、动火告警。
8. 聚合检测统计、人物轨迹和 pose 摘要。
9. 根据目标、pose、轨迹生成登高候选片段和交底候选片段。
10. 为候选片段保存关键截图，例如开始帧、中间帧、结束帧。
11. 选取一张代表场景帧并保存截图。
12. 调用场景识别多模态任务。
13. 如果存在登高候选片段，按候选片段调用登高判断多模态任务；否则直接记录无登高。
14. 如果存在交底候选片段，按候选片段调用交底判断多模态任务；否则直接记录无交底。
15. 根据大模型结果修正候选片段的开始和结束时间。
16. 写入场景、登高、交底最终事件。
17. 标记任务完成。

### 视频流流程

1. 创建视频流任务。
2. 持续读取视频流。
3. 按固定间隔抽帧。
4. YOLO 检测和 pose 分析。
5. 持续生成 PPE、抽烟、动火告警。
6. 按滑动窗口聚合摘要。
7. 在滑动窗口内生成登高候选片段和交底候选片段。
8. 周期性选择代表帧调用场景任务。
9. 有登高候选片段时调用登高任务。
10. 有交底候选片段时调用交底任务。
11. 根据大模型结果修正候选片段的开始和结束时间。
12. 写入窗口级场景、登高、交底事件。

## 截图策略

建议保存以下截图：

- 代表场景帧：提交给多模态服务。
- PPE 违规帧：用于人工复核。
- 抽烟告警帧：用于人工复核。
- 动火告警帧：用于人工复核。
- 登高作业关键帧：由多模态服务确认后保存。
- 交底关键帧：由多模态服务确认后保存。

推荐路径：

```text
data/snapshots/job_{job_id}/scene/
data/snapshots/job_{job_id}/ppe/
data/snapshots/job_{job_id}/smoking/
data/snapshots/job_{job_id}/hot_work/
data/snapshots/job_{job_id}/height_work/
data/snapshots/job_{job_id}/briefing/
```

## 告警上报

PPE、抽烟、动火属于实时告警，建议通过 webhook 上报给现有后端：

```json
{
  "job_id": 1,
  "camera_id": "camera-001",
  "alert_type": "ppe",
  "severity": "warning",
  "timestamp_ms": 12000,
  "frame_index": 300,
  "message": "Detected worker without safety helmet or reflective vest.",
  "details": {}
}
```

## 后续优化方向

- 使用 MobileNetV3 对抽烟、动火小目标进行二次复核。
- 使用分割 mask 优化安全帽、反光衣和人体部位的空间关联。
- 使用更稳定的轨迹分析判断登高开始和结束时间。
- 为多模态服务增加缓存，避免同一视频重复调用。
- 支持 OpenAI-compatible、DashScope、ModelScope Serving 等多种第三方接口。
- 将视频流处理拆分为 worker 队列，支持多路摄像头并发。
