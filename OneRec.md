# 配置环境变量

```bash
# 设置 HTTP 代理
export http_proxy="http://star-proxy.oa.com:3128"

# 设置 HTTPS 代理
export https_proxy="http://star-proxy.oa.com:3128"
```

# 下载项目文件
```bash
git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
cd OpenOneRec
```

# 下载模型到本地

```bash
mkdir -p model
cd model

# 下载 Qwen3-1.7B 基础模型 (复现训练)
git clone https://huggingface.co/Qwen/Qwen3-1.7B

# 下载 OneRec-1.7B-pro (用于推理测试，若仅训练可跳过)
git clone https://huggingface.co/OpenOneRec/OneRec-1.7B-pro
```

# 安装依赖

```bash
# 安装 Python 基础依赖
pip install accelerate huggingface_hub
pip install torch numpy pandas pyarrow faiss-cpu tqdm
conda install main::faiss

# 安装 MPI 依赖
sudo apt-get update && sudo apt-get install -y openmpi-bin libopenmpi-dev

# 安装 OpenOneRec Pretrain 模块
cd benchmarks
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
cd ..
```

# 🚀 Quick Start

- Code release and detailed usage instructions are coming soon.
- Currently, you can load our models using `transformers>=4.51.0`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./model/OneRec-1.7B-pro"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
# case - prompt with itemic tokens
prompt = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
# Note: In our experience, default decoding settings may be unstable for small models.
# For 1.7B, we suggest: top_p=0.95, top_k=20, temperature=0.75 (during 0.6 to 0.8)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

# 下载辅助模型与数据集

### 1. 下载 Qwen3-Embedding-8B
```bash
cd model
git clone https://huggingface.co/Qwen/Qwen3-Embedding-8B
cd ..
```

### 2. 下载 RecIF-Bench 官方数据集
从项目根目录执行，确保数据进入 `raw_data` 文件夹。
```bash
pip install huggingface_hub

export HF_TOKEN=<Create your token in https://huggingface.co/settings/tokens>

# 1. 下载通用预训练 & SFT 数据 (用于协同训练)
hf download OpenOneRec/OpenOneRec-General-Pretrain \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/general_text/pretrain

hf download OpenOneRec/OpenOneRec-General-SFT \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/general_text/sft

# 2. 下载 OneRec 核心推荐业务数据
hf download OpenOneRec/OpenOneRec-RecIF \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/onerec_data
```


# OneRec Stage 1：语义对齐训练准备

## 1. 词表扩充 (Model Expansion)

基于纯净的 **Qwen3-1.7B** 基础模型添加 Itemic Tokens。

### 1.1 运行扩充脚本
修改 `pretrain/scripts/expand_qwen3_vocab.sh`：

```python
#!/bin/bash
set -e

HF_MODEL_DIR=./model/Qwen3-1.7B
OUTPUT_MODEL_DIR=./model/Qwen3-1.7B_itemic
ITEMIC_LAYER_N=3
VOCAB_SIZE_PER_LAYER=8192

python3 pretrain/tools/model_converter/expand_qwen3_vocab.py \
    --hf_model_dir $HF_MODEL_DIR \
    --output_model_dir $OUTPUT_MODEL_DIR \
    --itemic_layer_n $ITEMIC_LAYER_N \
    --vocab_size_per_layer $VOCAB_SIZE_PER_LAYER
```
在项目根目录下运行脚本 `pretrain/scripts/expand_qwen3_vocab.sh`：
```bash
bash pretrain/scripts/expand_qwen3_vocab.sh
```
控制台出现以下结果，表明模型已成功扩充了 24578 个 Item Token，并将总词表对齐至 176384 位：
```bash
➜  OpenOneRec git:(main) ✗ bash pretrain/scripts/expand_qwen3_vocab.sh
2026-02-19 10:55:13,629 - INFO - Generating itemic tokens dynamically...
2026-02-19 10:55:13,635 - INFO - Generated 24578 itemic tokens in strict order:
2026-02-19 10:55:13,635 - INFO -   Layers: 3 (s_a, s_b, s_c)
2026-02-19 10:55:13,635 - INFO -   Vocab size per layer: 8192
2026-02-19 10:55:13,635 - INFO -   Special tokens: <|sid_begin|>, <|sid_end|>
2026-02-19 10:55:13,635 - INFO - Expanding vocabulary for pretraining
2026-02-19 10:55:13,635 - INFO -   Input model: ./model/Qwen3-1.7B
2026-02-19 10:55:13,635 - INFO -   Output model: ./model/Qwen3-1.7B_itemic
2026-02-19 10:55:13,635 - INFO -   New tokens: 24578
2026-02-19 10:55:13,635 - INFO - Loading original model components...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████| 311/311 [00:01<00:00, 198.03it/s, Materializing param=model.norm.weight]
The tied weights mapping and config for this model specifies to tie model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
2026-02-19 10:55:17,519 - INFO - Original vocabulary size: 151669
2026-02-19 10:55:17,519 - INFO - Adding 24578 new tokens...
2026-02-19 10:55:17,585 - INFO - Successfully added 24578 tokens
2026-02-19 10:55:17,636 - INFO - New vocabulary size: 176247
2026-02-19 10:55:17,636 - INFO - Target vocabulary size (aligned to 256): 176384
2026-02-19 10:55:17,637 - INFO - Resizing model token embeddings...
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
2026-02-19 10:56:05,788 - INFO - Updated config vocab_size to 176384
2026-02-19 10:56:05,788 - INFO - Saving expanded model components...
Writing model shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.72s/it]
2026-02-19 10:56:27,802 - INFO - Model components saved successfully
2026-02-19 10:56:27,802 - INFO - Fixing chat template...
2026-02-19 10:56:27,802 - INFO - Chat template copied from original model
2026-02-19 10:56:27,802 - INFO - Testing expanded vocabulary...
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
2026-02-19 10:56:35,339 - INFO - Vocabulary expansion test:
2026-02-19 10:56:35,340 - INFO -   Input text: <s_c_6438> <s_a_727> <s_a_416> Hello world
2026-02-19 10:56:35,340 - INFO -   Decoded input: <s_c_6438> <s_a_727> <s_a_416> Hello world
2026-02-19 10:56:35,340 - INFO -   Input IDs shape: torch.Size([1, 7])
2026-02-19 10:56:35,340 - INFO -   Generated: <s_c_6438> <s_a_727> <s_a_416> Hello world! I'm a new user here. I'm
2026-02-19 10:56:35,340 - INFO - ✓ Vocabulary expansion completed! Final vocab size: 176384
2026-02-19 10:56:35,818 - INFO - All operations completed successfully!
```

## 2. 映射文件准备 (Data Mapping)

在官方 `RecIF` 数据集中，物品的语义 ID（SID）和标题（Caption）通常分布在各领域的元数据文件中（如 `video_metadata.parquet`）。我们需要从下载的 `raw_data` 中提取出 `run.sh` 能够识别的 `pid2sid` 和 `caption` 文件。

### 2.1 创建转换脚本
在项目根目录下创建文件`prepare_mappings.py`，运行`python prepare_mappings.py`。
```python
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

RAW_DATA_DIR = './raw_data/onerec_data'
OUTPUT_DIR = './data/onerec_data/pretrain'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fast_vectorized_sid(df):
    """
    使用 NumPy 矩阵操作将 [s_a, s_b, s_c] 列表向量化转换为 OneRec 字符串
    """
    if df.empty:
        return df
    
    # 1. 将 'sid' 列 (list[int]) 转换为二维 NumPy 数组 [N, 3]
    sid_matrix = np.array(df['sid'].tolist())
    
    # 2. 提取三层索引并转为字符串序列
    a = sid_matrix[:, 0].astype(str)
    b = sid_matrix[:, 1].astype(str)
    c = sid_matrix[:, 2].astype(str)
    
    # 3. 向量化拼接字符串
    df['sid'] = (
        "<|sid_begin|><s_a_" + a + 
        "><s_b_" + b + 
        "><s_c_" + c + 
        "><|sid_end|>"
    )
    return df[['pid', 'sid']]

def process_single_sid_file(file_name):
    """并行任务单元：处理单个 SID 文件"""
    path = os.path.join(RAW_DATA_DIR, file_name)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return fast_vectorized_sid(df)
    return None

def main():

    cap_file = os.path.join(RAW_DATA_DIR, 'pid2caption.parquet')
    if os.path.exists(cap_file):
        df_cap = pd.read_parquet(cap_file)
        df_cap = df_cap[['pid', 'dense_caption']]
        df_cap.to_parquet(os.path.join(OUTPUT_DIR, 'behavior_caption.parquet'), index=False)

    sid_files = ['video_ad_pid2sid.parquet', 'product_pid2sid.parquet']
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_sid_file, sid_files))
    
    final_dfs = [r for r in results if r is not None]
    if final_dfs:
        full_sid_df = pd.concat(final_dfs, ignore_index=True)
        full_sid_df.to_parquet(
            os.path.join(OUTPUT_DIR, 'behavior_pid2sid.parquet'), 
            index=False,
            engine='pyarrow'
        )

    print(f"数据输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

```python
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

RAW_DATA_DIR = './raw_data/onerec_data'
OUTPUT_DIR = './data/onerec_data/pretrain'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fast_vectorized_sid(df):
    """
    使用 NumPy 矩阵操作将 [s_a, s_b, s_c] 列表向量化转换为 OneRec 字符串
    """
    if df.empty:
        return df[['pid', 'sid']]
    
    sid_matrix = np.stack(df['sid'].values)
    
    df['sid'] = [
        f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>" 
        for a, b, c in sid_matrix
    ]
    
    return df[['pid', 'sid']]


def process_single_sid_file(file_name):
    """并行任务单元：处理单个 SID 文件"""
    path = os.path.join(RAW_DATA_DIR, file_name)
    if os.path.exists(path):
        df = pd.read_parquet(path, columns=['pid', 'sid'])
        return fast_vectorized_sid(df)
    return None

def main():

    cap_file = os.path.join(RAW_DATA_DIR, 'pid2caption.parquet')
    if os.path.exists(cap_file):
        df_cap = pd.read_parquet(cap_file, columns=['pid', 'dense_caption'])
        df_cap.to_parquet(os.path.join(OUTPUT_DIR, 'behavior_caption.parquet'), index=False)

    sid_files = ['video_ad_pid2sid.parquet', 'product_pid2sid.parquet']
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_sid_file, sid_files))
    
    final_dfs = [r for r in results if r is not None]
    if final_dfs:
        full_sid_df = pd.concat(final_dfs, ignore_index=True)
        full_sid_df.to_parquet(
            os.path.join(OUTPUT_DIR, 'behavior_pid2sid.parquet'), 
            index=False,
            engine='pyarrow'
        )

    print(f"数据输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```


串行化版本，防止出现`raise self._exception concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.`
```python
import pandas as pd
import numpy as np
import os
import gc

RAW_DATA_DIR = './raw_data/onerec_data'
OUTPUT_DIR = './data/onerec_data/pretrain'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def vectorized_convert_sid(df):
    if df.empty:
        return df

    df = df.dropna(subset=['sid'])
    if df.empty:
        return df

    df = df[df['sid'].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 3)]

    sid_matrix = np.array(df['sid'].tolist())

    a = sid_matrix[:, 0].astype(str)
    b = sid_matrix[:, 1].astype(str)
    c = sid_matrix[:, 2].astype(str)

    df = df.drop(columns=['sid'])

    df['sid'] = (
        "<|sid_begin|><s_a_" + a + 
        "><s_b_" + b + 
        "><s_c_" + c + 
        "><|sid_end|>"
    )
    return df[['pid', 'sid']]

def main():
    df_cap = pd.DataFrame()
    cap_file = os.path.join(RAW_DATA_DIR, 'pid2caption.parquet')
    
    if os.path.exists(cap_file):
        df_cap = pd.read_parquet(cap_file)
        df_cap = df_cap[['pid', 'dense_caption']]

        df_cap = df_cap.dropna(subset=['pid', 'dense_caption'])
        df_cap = df_cap[df_cap['dense_caption'].str.strip() != ""]

        df_cap = df_cap.drop_duplicates(subset=['pid'])

    sid_files = ['video_ad_pid2sid.parquet', 'product_pid2sid.parquet']
    all_dfs = []

    for f_name in sid_files:
        path = os.path.join(RAW_DATA_DIR, f_name)
        if os.path.exists(path):
            df = pd.read_parquet(path)

            df_processed = vectorized_convert_sid(df)
            
            if not df_processed.empty:
                all_dfs.append(df_processed)

            del df
            gc.collect()

    full_sid_df = pd.DataFrame()
    if all_dfs:
        full_sid_df = pd.concat(all_dfs, ignore_index=True)
        full_sid_df = full_sid_df.drop_duplicates(subset=['pid'])

    if not df_cap.empty and not full_sid_df.empty:
        valid_pids = np.intersect1d(df_cap['pid'].unique(), full_sid_df['pid'].unique())
        
        print(f"Caption 原始数量: {len(df_cap)}")
        print(f"SID 原始数量: {len(full_sid_df)}")
        print(f"对齐后有效数量: {len(valid_pids)}")

        df_cap = df_cap[df_cap['pid'].isin(valid_pids)]
        full_sid_df = full_sid_df[full_sid_df['pid'].isin(valid_pids)]

        df_cap.to_parquet(os.path.join(OUTPUT_DIR, 'behavior_caption.parquet'), index=False)
        full_sid_df.to_parquet(
            os.path.join(OUTPUT_DIR, 'behavior_pid2sid.parquet'), 
            index=False,
            engine='pyarrow'
        )
    else:
        print("错误：无法生成有效数据，Caption 或 SID 数据为空。")

    del df_cap
    del full_sid_df
    gc.collect()

    print(f"\n 数据输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```
读取输出数据文件并查看：
```python
import pandas as pd

pd.set_option('display.max_colwidth', None)

df_cap = pd.read_parquet('./data/onerec_data/pretrain/behavior_caption.parquet')
print("--- behavior_caption.parquet ---")
print(df_cap.head(1))
"""
      pid                                            caption
0  241922  这是一个关于展示化妆品和美妆产品使用效果的视频。视频中一位女性展示了她的化妆技巧，她用一款名...
"""

df_sid = pd.read_parquet('./data/onerec_data/pretrain/behavior_pid2sid.parquet')
print("\n--- behavior_pid2sid.parquet ---")
print(df_sid.head(1))
"""
        pid                                                    sid
0  13508074  <|sid_begin|><s_a_1636><s_b_1470><s_c_676><|sid_end|>
"""
```


```python
import pyarrow.parquet as pq

# --- behavior_caption.parquet ---
print("--- behavior_caption.parquet ---")
pf_cap = pq.ParquetFile('./data/onerec_data/pretrain/behavior_caption.parquet')
print(f"总行数: {pf_cap.metadata.num_rows}, 列: {pf_cap.schema.names}")
df_cap = next(pf_cap.iter_batches(batch_size=1)).to_pandas()
print(df_cap)

# --- behavior_pid2sid.parquet ---
print("\n--- behavior_pid2sid.parquet ---")
pf_sid = pq.ParquetFile('./data/onerec_data/pretrain/behavior_pid2sid.parquet')
print(f"总行数: {pf_sid.metadata.num_rows}, 列: {pf_sid.schema.names}")
df_sid = next(pf_sid.iter_batches(batch_size=1)).to_pandas()
print(df_sid)

```

输出内容如下：
```bash
--- behavior_caption.parquet ---
总行数: 12660465, 列: ['pid', 'dense_caption']
    pid                 dense_caption
0  241922  这是一个关于展示化妆品和美妆产品使用效果的视频。视频中一位女性展示了她的化妆技巧，她用一款名...

--- behavior_pid2sid.parquet ---
总行数: 17951318, 列: ['pid', 'sid']
    pid                    sid
0  13508074  <|sid_begin|><s_a_1636><s_b_1470><s_c_676><|si...
```


## 3. 生成语义对齐预训练数据

### 3.1 配置 `data/onerec_data/run.sh`
确保脚本指向刚才生成的官方映射文件。

```bash
# ============== Task Selection ==============
# Comment out tasks you don't want to run

# Pretrain tasks
RUN_PRETRAIN_VIDEO_REC=0
RUN_PRETRAIN_USER_PROFILE=0
RUN_PRETRAIN_ITEM_UNDERSTAND=1

# SFT tasks
RUN_SFT_VIDEO_REC=0
RUN_SFT_INTERACTIVE_REC=0
RUN_SFT_LABEL_COND_REC=0
RUN_SFT_LABEL_PRED=0
RUN_SFT_AD_REC=0
RUN_SFT_PRODUCT_REC=0
RUN_SFT_ITEM_UNDERSTAND=0
RUN_SFT_REC_REASON=0

# ============== Configuration ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_METADATA="../../raw_data/onerec_data/onerec_bench_release.parquet"
PID2SID_MAPPING="./pretrain/behavior_pid2sid.parquet" 
PRODUCT_PID2SID_MAPPING="./pretrain/behavior_pid2sid.parquet" 
CAPTION_INPUT="./pretrain/behavior_caption.parquet"
OUTPUT_BASE_DIR="../../data/pretrain_data/output"

SEED=42
```

### 3.2 运行生成
这个过程可能耗时很长。
```bash
cd data/onerec_data
bash run.sh
cd ../..
```

## 4. 预训练数据分片 (Data Sharding)

配置 `prepare_pretrain.sh`，在 data/prepare_pretrain.sh 里进行以下修改：

```bash
GENERAL_TEXT_PATH="../data/pretrain_data/output/pretrain_item_understand.parquet"
REC_DATA_PATH="../raw_data/general_text/empty_general"
OUTPUT_DIR="../output/split_data_pretrain"
MAX_ROWS=1000
ENGINE="pyarrow"
```

在项目根下执行，这个过程可能耗时很长。

```bash
mkdir -p raw_data/general_text/empty_general
cd data
bash prepare_pretrain.sh
```

运行脚本`python3 -c "import pandas as pd; df = pd.read_parquet('../output/split_data_pretrain/part-00000-of-12661.parquet'); print(df.head(10))"`，在控制台查看数据格式：
```
                          source  ...                                           metadata
0  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 241922, "sid": "<|sid_begin|><s_a_3953...
1  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 407941, "sid": "<|sid_begin|><s_a_7397...
2  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 1134774, "sid": "<|sid_begin|><s_a_350...
3  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 1524945, "sid": "<|sid_begin|><s_a_742...
4  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 7269848, "sid": "<|sid_begin|><s_a_414...
5  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 6861621, "sid": "<|sid_begin|><s_a_207...
6  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 10315063, "sid": "<|sid_begin|><s_a_38...
7  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 356228, "sid": "<|sid_begin|><s_a_7144...
8  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 2713150, "sid": "<|sid_begin|><s_a_608...
9  RecIF_ItemUnderstand_Pretrain  ...  {"pid": 5413404, "sid": "<|sid_begin|><s_a_328...

[10 rows x 4 columns]
```
把 file_list 写进 JSON 配置，训练时用的是 pretrain/examples/dataset_config/pretrain.json 里的 sources，需要指向上一步生成的file_list.json。冻结 LLM 主干参数，仅训练新扩展的 Token Embedding，使token能够映射到对应的文本语义。

```json
{
    "name": "chat_completion_parquet",
    "sources": "../output/split_data_pretrain/file_list.json",
    "only_assistant_loss": false,
    "max_length": 4096,
    "base_model_dir": "../model/Qwen3-1.7B_itemic",
    "num_workers": 2,
    "num_epochs": 100,
    "cut_to_pad": 1,
    "model_class": "Qwen3ForCausalLM",
    "full_attention": false,
    "local_shuffle_buffer_size": 10000,
    "max_sample_length": 4096,
    "local_shuffle_random_fetch": 0.0001,
    "itemic_id_range": [151669, 176246]
}
```
### 2.1 修改 pretrain_stg1.sh
编辑 `/workspace/OpenOneRec/pretrain/examples/pretrain_stg1.sh`，同时注意删除72行的`with_nccl_local_env \`，注意不要出现换行符 \ 后面多了空行导致命令被截断的情况，所以72行命令应该整行删除。

```bash
# 修改 1：指向扩充词表后的模型路径
MODEL_DIR=../model/Qwen3-1.7B_itemic

# 修改 2：指向保存训练结果的目录
OUTPUT_DIR=./output/stg1_training_results

# 修改 3：使用 hostname 获取本机 IP
MASTER_ADDR=$(hostname -i)
```

### 2.2 启动训练
确保你位于 `pretrain` 目录下，并已配置好 `/etc/mpi/hostfile`。如果运行`set_env.sh`时终端闪退，就在先进入`bash`再执行`source`。
```bash
# bash
# source set_env.sh
bash examples/pretrain_stg1.sh
```
启动训练后在`OpenOneRec/pretrain/output/stg1_training_results`路径下查看实验结果。

> 对于LD_PRELOAD 路径错误，几乎每个命令都弹出 ERROR: ld.so: object '...' cannot be preloaded，这是因为环境变量里强行指定了一个不存在的 NCCL 库文件，因此运行脚本前取消该设置： 打开脚本同目录下的 .env 文件，查找 LD_PRELOAD 这一行，直接删除或注释掉该行让系统使用默认的 libnccl.so。

> 遇到Open RTE was unable to open the hostfile: /etc/mpi/hostfile问题，MPI 执行时需要一个 hostfile 来确定在哪些机器上运行，脚本里指定了路径为 /etc/mpi/hostfile，但系统找不到这个文件。在当前服务器（以及集群所有服务器）上创建该目录和文件。如果是单机环境，执行以下命令：
> 
> ```bash
> GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
> echo "检测到 ${GPU_COUNT} 张显卡"
> sudo mkdir -p /etc/mpi/
> echo "localhost slots=${GPU_COUNT}" | sudo tee /etc/mpi/hostfile
> cat /etc/mpi/hostfile
> ```

> 如果遇到`mpirun: command not found`首先查找OpenMPI在什么位置：
>  ```bash
>  which mpirun
>  # 如果没结果，试试：
>  find /usr -name mpirun 2>/dev/null
>  find /opt -name mpirun 2>/dev/null
>  ```
>  如果没有，可以使用conda安装下载。
>  ```bash
>  conda install conda-forge::openmpi
>  ```

> 如果找到了（例如在 `/usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun`），将其加入环境变量：
> ```bash
> export PATH=/usr/mpi/gcc/openmpi-4.1.7a1/bin/:$PATH
> ```

> 验证一下：
> ```bash
> (torch-base) openmpi-3.1.0$ mpirun --version
> mpirun (Open MPI) 4.1.7a1
> ```

 > 如果遇到`ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2.`
> ```bash
> pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
> ```

> 如果遇到 ``ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found (required by /opt/conda/envs/torch-base/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so)``，系统的 **glibc 版本太低**，预编译的 flash-attention wheel 需要 GLIBC 2.32，但系统只有更低版本。
> ```bash
> # 卸载预编译版本
> pip uninstall flash-attn -y
> # 从源码编译安装（需要较长时间，约10-30分钟）
> pip install flash-attn --no-build-isolation
> ```

# OneRec 模型转换与验证

2000个step后Stage 1 训练完成，此时训练损失收敛到1.90附近。训练生成的 Checkpoint 为分布式分片格式，需转换为标准的 HuggingFace 格式才能进行推理或 Stage 2 训练。

## 1. 模型格式转换 (Checkpoint to HF)

使用项目提供的转换脚本，将指定的训练步数（如 step 2000）转换为 HF 格式：

```bash
cd pretrain

# 执行转换脚本
# 参数：<扩充词表后的初始模型路径> <训练输出目录> <转换步数>
bash scripts/convert_checkpoint_to_hf.sh \
    ../model/Qwen3-1.7B_itemic \
    ./output/stg1_training_results \
    2000

# 转换后的模型将保存在：
# ../output/stg1_checkpoints/step2000/global_step2000/converted/
```

## 2. 验证对齐效果

在`pretrain`目录下运行以下代码，验证模型是否能根据语义 ID 理解物品内容：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指向转换后的 HF 模型目录
model_path = "output/stg1_training_results/step2000/global_step2000/converted/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 测试 Prompt：输入训练中的一个 SID
test_sid = "<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>"
# prompt = f"视频{test_sid} 的内容完整描述如下："
prompt = f" 视频{test_sid} 展示了以下内容："
# 视频{test_sid}展示了以下内容：
# 视频{test_sid}的内容完整描述如下：

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
经过测试模型此时还不具备语义对齐的能力：
```bash
 视频<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|> 展示了以下内容： 
 
 1. 1987年，美国电影《星球大战》（Star Wars）首次上映，这是第一部由乔治·卢卡斯执导的科幻电影，开启了科幻电影的黄金时代。 
 2. 1997年，美国电影《黑客帝国》（The Matrix）上映，这部电影以其独特的视觉效果和哲学主题，成为科幻电影的经典之作。 
 3. 1999年，美国电影《终结者2》（Terminator 2）上映，这部电影以其对末日场景的描绘和对人工智能的探讨，成为科幻电影的又一里程碑。 
 4. 2001年，美国电影《阿凡达》（Avatar）上映，这部电影以其宏大的视觉效果和对环境保护的探讨，成为科幻电影的又一里程碑。 
 5. 2003年，美国电影《盗梦空间》（Inception）上映，这部电影以其复杂的剧情和对梦境的探讨，成为科幻电影的又一里程碑。 
 6. 2006年，美国电影《阿凡达》（Avatar）上映，这部电影以其宏大的视觉效果和对环境保护的探讨，成为科幻电影的又一里程碑。 
 7. 2008年，美国电影《钢铁侠》（Iron Man）上映，这部电影以其对科技与英雄的探讨，成为科幻电影的又一里程碑。 
 8. 2010年，美国电影《复仇者联盟》（Avengers）上映，这部电影以其对超级英雄的探讨，成为科幻电影的又一里程碑。 
 9. 2011年，美国电影《星际穿越》（Interstellar）上映，这部电影以其对宇宙和时间的探讨，成为科幻电影的又一里程碑。 
 10. 2015年，美国电影《阿凡达》（Avatar）上映，这部电影以其宏大的视觉效果和对环境保护的探讨，成为科幻电影的又一里程碑。 
 11. 2017年，美国电影《复仇者联盟3》（Avengers: Age of Ultron）上映，这部电影以其对超级英雄的探讨，成为科幻电影的又一里程碑。 
 12. 2019年，美国电影《阿凡达》（Avatar）上映，这部电影以其宏大的视觉效果和对环境保护的探讨，成为科幻电影的又
```
