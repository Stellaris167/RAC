## 这轮已经确认并重铺的关键补丁

### 1. `img_context_token_id` 必须手动绑定

- 现象：InternVL 前向在真正训练前就崩，或者在带图像 token 的路径上直接报 `img_context_token_id` 未初始化。
- 根因：上游 InternVL 只在 chat / batch_chat 里初始化这个字段，不在 `__init__` 里初始化。
- 修复：训练侧要从 tokenizer / processor 解析 `<IMG_CONTEXT>` 对应 id，并绑定到模型上。
- 结果：修复后，padded 与 remove-padding 两条 response logprob 路径已经做过单样本一致性验证。

### 2. repetition penalty 之前没有真正透传到 rollout

- 现象：脚本里改了 `actor_rollout_ref.rollout.repetition_penalty`，生成行为却几乎不变。
- 根因：agent loop、sync agent loop、vLLM server 曾经把 repetition penalty 实际写死成了 `1.0`。
- 修复：SamplingConfig、agent loop、sync agent loop、vLLM server 统一透传 `repetition_penalty`。

### 3. InternVL 的 stop / eos 链必须显式补齐

- 现象：生成容易拖尾、重复，EOS 停不下来。
- 根因：agent loop 请求里之前没有把 stop 语义完整传给 rollout。
- 修复：从 tokenizer 解析 EOS，显式传 `stop_token_ids` 和 `ignore_eos`。
- 当前已确认：InternVL3.5_8B 的 EOS 是 `<|im_end|>`，id 为 `151645`。

### 4. `pixel_values is None while image context token exists in input_ids`

- 现象：训练中出现 warning：

```text
[InternVL] Warning: pixel_values is None while image context token exists in input_ids; falling back to text-only forward.
```

- 直接后果：模型退化成 text-only forward，输出空模板、重复 `<think>` 样式文本。
- 根因：`agent_loop._count_retained_multi_modal_items()` 对 InternVL 错误套用了 Qwen 的 `<|vision_start|>` 计数逻辑。InternVL tokenizer 词表里虽然有这个 token，但 prompt 实际没有，结果图片数被误判成 0，后续对齐时把 images 丢了。
- 修复：只有在 `vision_start_token_id in prompt_ids` 时，才走 vision-start 分支。
- 验证：同一样本修复前 `image_count_from_prompt=0 / aligned_images=None`，修复后恢复为 `1 / 1`。

### 5. InternVL + vLLM 不能同时发送“已展开 token prompt”与原始图片数据

- 现象：vLLM 断言失败：

```text
AssertionError: Failed to apply prompt replacement for mm_items['image'][0]
```

- 根因：InternVL 本地 processor 已经把 `<image>` 展开成 `<img><IMG_CONTEXT>...` token 序列；如果再把这串 `prompt_token_ids` 连同 `multi_modal_data.image` 一起发给 vLLM，vLLM 会再做一次 prompt replacement，于是断言失败。
- 修复：默认 `single_turn_agent` 路径下，InternVL 图像请求改成给 vLLM 发送 TextPrompt，也就是“原始 prompt 文本 + image multi_modal_data”，让 vLLM 自己只做一次 HF processor 更新。

### 6. `Token indices sequence length is longer than the specified maximum sequence length`

- 现象：AgentLoopWorker 日志里出现：

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (16827 > 14588). Running this sequence through the model will result in indexing errors
```

- 判断：在当前 agent-loop 链路里，这通常是 InternVL 预处理阶段的“误导性早期警告”，不是最终已经把 16827 token 直接送进模型。
- 根因：InternVLProcessor 会先把 `<image>` 展开成大量 `<IMG_CONTEXT>`，然后立刻调 tokenizer；而真正的 visual-run-aware prompt 截断发生在后面。于是 tokenizer 先按自身 `model_max_length=14588` 报警，但 agent loop 随后还会按 rollout budget 再截断。
- 修复：InternVLProcessor 内部 tokenizer 调用显式传 `verbose=False`，不再输出这条误导性警告；真正的长度控制仍然由后续 prompt budget 截断负责。