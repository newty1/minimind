import os
import sys
import json
import subprocess
import threading
import time
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
#TODO 
#1.添加多卡训练支持
#2.日志bug修复

# 添加项目根目录到路径
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="mini大模型训练云平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式设置
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .task-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-completed {
        color: #007bff;
        font-weight: bold;
    }
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    .status-pending {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 线程安全的数据存储（用于后台线程）
_thread_safe_data = {
    'task_logs': defaultdict(list),
    'task_metrics': defaultdict(dict),
}
_data_lock = threading.Lock()

# 初始化session state
if 'tasks' not in st.session_state:#任务
    st.session_state.tasks = {}
if 'task_processes' not in st.session_state:#任务进程
    st.session_state.task_processes = {}
if 'task_logs' not in st.session_state:#任务日志
    st.session_state.task_logs = {}
if 'task_metrics' not in st.session_state:#任务指标
    st.session_state.task_metrics = {}


def load_saved_tasks():
    """从文件加载已保存的任务"""
    tasks_file = Path("../tasks.json")
    if tasks_file.exists():
        with open(tasks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_tasks(tasks):
    """保存任务到文件"""
    tasks_file = Path("../tasks.json")
    with open(tasks_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False, default=str)


def load_task_logs_from_file(task_id: str, max_lines: int = None):
    """从文件加载任务日志"""
    log_file = get_log_file_path(task_id)
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 去除换行符
        logs = [line.rstrip('\n\r') for line in lines]
        
        # 如果指定了最大行数，只返回最后N行
        if max_lines and len(logs) > max_lines:
            return logs[-max_lines:]
        
        return logs
    except Exception as e:
        # 静默处理错误，避免影响程序运行
        return []


def extract_metrics_from_logs(logs: List[str]) -> Dict:
    """从日志列表中提取所有指标（用于图表绘制）"""
    metrics = {
        'loss': [],
        'lr': [],
        'step': [],
        'epoch': [],
        'timestamp': []
    }
    
    for log_line in logs:
        parsed = parse_training_log(log_line)
        if parsed and 'loss' in parsed:
            metrics['loss'].append(parsed['loss'])
            metrics['lr'].append(parsed.get('lr', 0))
            metrics['step'].append(parsed.get('step', 0))
            metrics['epoch'].append(parsed.get('epoch', 0))
            # 使用步数作为时间戳（因为日志文件中没有实际时间戳）
            metrics['timestamp'].append(parsed.get('step', len(metrics['step'])))
    
    return metrics


def plot_training_metrics(metrics: Dict):
    """绘制训练指标图表（Loss、Learning Rate、Steps、Epochs）
    
    Args:
        metrics: 包含 'loss', 'lr', 'step', 'epoch' 的字典
    """
    if not metrics.get('loss'):
        st.info("暂无训练指标数据，请等待训练开始...")
        return
    
    # 创建图表布局，增加子图间距
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss 曲线', 'Learning Rate', 'Training Steps', 'Training Epochs'),
        vertical_spacing=0.18,  # 增加垂直间距
        horizontal_spacing=0.15  # 增加水平间距
    )
    
    # 使用步数作为 X 轴（更符合训练可视化习惯）
    x_axis = metrics['step'] if metrics['step'] and len(metrics['step']) == len(metrics['loss']) else list(range(len(metrics['loss'])))
    
    # Loss 曲线（左上）- 添加填充效果
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=metrics['loss'],
            mode='lines',
            name='Loss',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            hovertemplate='<b>Loss</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Step", row=1, col=1, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Loss", row=1, col=1, showgrid=True, gridcolor='lightgray')
    
    # Learning Rate（右上）
    if metrics['lr']:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=metrics['lr'],
                mode='lines',
                name='Learning Rate',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Learning Rate</b><br>Step: %{x}<br>LR: %{y:.2e}<extra></extra>'
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Step", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log", showgrid=True, gridcolor='lightgray')
    
    # Steps（左下）- 显示步数进度
    if metrics['step']:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(metrics['step']))),
                y=metrics['step'],
                mode='lines+markers',
                name='Steps',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=4, color='#2ca02c'),
                hovertemplate='<b>Steps</b><br>Index: %{x}<br>Step: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Log Index", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Step", row=2, col=1, showgrid=True, gridcolor='lightgray')
    
    # Epochs（右下）- 显示轮数进度
    if metrics['epoch']:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(metrics['epoch']))),
                y=metrics['epoch'],
                mode='lines+markers',
                name='Epochs',
                line=dict(color='#d62728', width=2),
                marker=dict(size=4, color='#d62728'),
                hovertemplate='<b>Epochs</b><br>Index: %{x}<br>Epoch: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Log Index", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Epoch", row=2, col=2, showgrid=True, gridcolor='lightgray')
    
    # 更新整体布局
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="📊 训练指标监控",
        title_x=0.5,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示统计信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前 Loss", f"{metrics['loss'][-1]:.4f}" if metrics['loss'] else "N/A")
    with col2:
        min_loss = min(metrics['loss']) if metrics['loss'] else 0
        st.metric("最低 Loss", f"{min_loss:.4f}")
    with col3:
        st.metric("当前 LR", f"{metrics['lr'][-1]:.2e}" if metrics['lr'] and metrics['lr'][-1] > 0 else "N/A")
    with col4:
        st.metric("数据点数", len(metrics['loss']))


def get_training_scripts():
    """获取可用的训练脚本"""
    trainer_dir = Path("../trainer")
    scripts = {}
    if trainer_dir.exists():
        for script in trainer_dir.glob("train_*.py"):#匹配
            script_name = script.stem
            display_name = {
                "train_pretrain": "预训练 (Pretrain)",
                "train_full_sft": "监督微调 (SFT)",
                "train_lora": "LoRA微调",
                "train_dpo": "DPO强化学习",
                "train_ppo": "PPO强化学习",
                "train_grpo": "GRPO强化学习",
                "train_spo": "SPO强化学习",
                "train_distill_reason": "推理模型蒸馏",
                "train_distillation": "模型蒸馏",
            }.get(script_name, script_name)
            scripts[script_name] = {
                "display": display_name,
                "path": str(script)
            }
    return scripts


def get_datasets():
    """获取可用的数据集"""
    dataset_dir = Path("../dataset")
    datasets = {}
    if dataset_dir.exists():
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            datasets[jsonl_file.name] = str(jsonl_file)
    return datasets


def get_available_weights(save_dir="../out"):
    """获取可用的模型权重前缀列表"""
    weight_dir = Path(save_dir)
    weight_prefixes = set()
    
    if weight_dir.exists():
        # 扫描所有 .pth 文件
        for pth_file in weight_dir.glob("*.pth"):
            filename = pth_file.stem  # 去掉 .pth 扩展名
            
            # 匹配格式：{prefix}_{hidden_size} 或 {prefix}_{hidden_size}_moe
            # 例如：pretrain_512.pth -> pretrain
            #      full_sft_768_moe.pth -> full_sft
            match = re.match(r'^(.+?)_(\d+)(?:_moe)?$', filename)
            if match:
                prefix = match.group(1)
                weight_prefixes.add(prefix)
    
    # 添加 "none" 选项（从头开始训练）
    weight_prefixes.add("none")
    
    # 排序并返回列表
    sorted_prefixes = sorted(weight_prefixes)
    # 将 "none" 放在最后
    if "none" in sorted_prefixes:
        sorted_prefixes.remove("none")
        sorted_prefixes.append("none")
    
    return sorted_prefixes


def get_available_models():
    """获取可用的模型列表（仅Transformers格式）"""
    models = {}
    
    # 扫描项目根目录下所有包含config.json的目录
    root_dir = Path("..")
    
    # 预定义的模型目录
    predefined_dirs = [
        Path("../MiniMind2"),
        Path("../MiniMind2-Small"),
        Path("../MiniMind2-MoE"),
        Path("../MiniMind2-R1"),
        Path("../MiniMind2-Small-R1"),
    ]
    
    # 扫描预定义目录
    predefined_paths = set()
    for model_dir in predefined_dirs:
        if model_dir.exists() and (model_dir / "config.json").exists():
            resolved_path = str(model_dir.resolve())
            predefined_paths.add(resolved_path)
            models[resolved_path] = {
                "name": model_dir.name,
                "path": resolved_path,
                "type": "transformers"
            }
    
    # 动态扫描项目根目录下所有包含config.json的目录
    if root_dir.exists():
        for item in root_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                resolved_path = str(item.resolve())
                # 跳过已经添加的预定义目录
                if resolved_path in predefined_paths:
                    continue
                
                config_file = item / "config.json"
                if config_file.exists():
                    # 确保是有效的Transformers模型目录（至少包含config.json）
                    try:
                        models[resolved_path] = {
                            "name": item.name,
                            "path": resolved_path,
                            "type": "transformers"
                        }
                    except Exception:
                        # 如果检查过程中出错，跳过
                        continue
    
    return models


def load_model_for_inference(model_path, model_info, device="cuda:0"):
    """加载模型用于推理（仅支持Transformers格式）"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 只支持Transformers格式模型
        if model_info["type"] != "transformers":
            st.warning("仅支持Transformers格式模型，请先在模型管理页面将PyTorch模型转换为Transformers格式")
            return None, None
        
        # 加载Transformers格式模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model = model.eval().to(device)
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        import traceback
        st.error(f"错误详情: {traceback.format_exc()}")
        return None, None


def release_model_from_gpu():
    """释放模型占用的GPU内存"""
    if 'current_model' in st.session_state and st.session_state.current_model is not None:
        try:
            # 将模型移到CPU
            st.session_state.current_model = st.session_state.current_model.cpu()
            # 删除模型引用
            del st.session_state.current_model
            # 清空CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 清理session state
            st.session_state.current_model = None
            st.session_state.current_tokenizer = None
            st.session_state.current_model_path = None
            return True
        except Exception as e:
            st.error(f"释放GPU失败: {str(e)}")
            return False
    return True


def process_assistant_content(content):
    """处理助手回复内容（处理推理标签等）"""
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
            content,
            flags=re.DOTALL
        )
    return content


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_model_config_from_filename(filename: str):
    """从文件名解析模型配置"""
    import re
    filename_stem = Path(filename).stem
    
    # 解析hidden_size
    hidden_size_match = re.search(r'_(\d+)(?:_moe)?(?:\.pth)?$', filename_stem)
    hidden_size = int(hidden_size_match.group(1)) if hidden_size_match else 512
    
    # 解析是否使用MoE
    use_moe = '_moe' in filename_stem
    
    # 根据hidden_size推断num_hidden_layers
    # 512 -> 8层, 768 -> 16层, 640 -> 8层(MoE)
    if hidden_size == 512:
        num_hidden_layers = 8
    elif hidden_size == 768:
        num_hidden_layers = 16
    elif hidden_size == 640:
        num_hidden_layers = 8  # MoE通常8层
    else:
        # 默认根据hidden_size估算
        num_hidden_layers = 8 if hidden_size <= 512 else 16
    
    return {
        'hidden_size': hidden_size,
        'num_hidden_layers': num_hidden_layers,
        'use_moe': use_moe
    }


def convert_torch_to_transformers(
    torch_path: str, 
    output_path: str, 
    config: Dict,
    convert_type: str = "llama",  # "llama" 或 "minimind"
    dtype: str = "float16"
):
    """转换PyTorch模型到Transformers格式"""
    try:
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        
        # 加载PyTorch权重
        state_dict = torch.load(torch_path, map_location=device)
        
        # 创建配置（使用默认的max_position_embeddings=32768）
        lm_config = MiniMindConfig(
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers'],
            use_moe=config['use_moe']
        )
        
        model_params = 0
        
        if convert_type == "minimind":
            # 转换为MiniMind格式
            MiniMindConfig.register_for_auto_class()
            MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
            model = MiniMindForCausalLM(lm_config)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(torch_dtype)
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.save_pretrained(output_path, safe_serialization=False)
            
        else:  # convert_type == "llama"
            # 转换为Llama兼容格式
            llama_config = LlamaConfig(
                vocab_size=lm_config.vocab_size,
                hidden_size=lm_config.hidden_size,
                intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
                num_hidden_layers=lm_config.num_hidden_layers,
                num_attention_heads=lm_config.num_attention_heads,
                num_key_value_heads=lm_config.num_key_value_heads,
                max_position_embeddings=lm_config.max_position_embeddings,
                rms_norm_eps=lm_config.rms_norm_eps,
                rope_theta=lm_config.rope_theta,
                tie_word_embeddings=True
            )
            llama_model = LlamaForCausalLM(llama_config)
            llama_model.load_state_dict(state_dict, strict=False)
            llama_model = llama_model.to(torch_dtype)
            model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
            llama_model.save_pretrained(output_path)
        
        # 保存tokenizer（从模型目录加载）
        tokenizer = AutoTokenizer.from_pretrained('../model/')
        tokenizer.save_pretrained(output_path)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True, f"✅ 模型已成功转换为Transformers-{convert_type.upper()}格式\n📊 参数量: {model_params / 1e6:.2f}M ({model_params / 1e9:.3f}B)\n📁 保存路径: {output_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 转换失败: {str(e)}\n{traceback.format_exc()}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, error_msg


def convert_transformers_to_torch(
    transformers_path: str,
    output_path: str
):
    """转换Transformers模型到PyTorch格式"""
    try:
        from transformers import AutoModelForCausalLM
        
        # 加载Transformers模型
        model = AutoModelForCausalLM.from_pretrained(
            transformers_path,
            trust_remote_code=True,
            torch_dtype=torch.float32  # 保存为float32以确保兼容性
        )
        
        # 保存为PyTorch格式
        torch.save(model.state_dict(), output_path)
        
        # 计算参数量
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 清理GPU缓存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True, f"✅ 模型已成功转换为PyTorch格式\n📊 参数量: {model_params / 1e6:.2f}M ({model_params / 1e9:.3f}B)\n📁 保存路径: {output_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 转换失败: {str(e)}\n{traceback.format_exc()}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, error_msg


def generate_task_id():
    """生成任务ID"""
    return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(st.session_state.tasks)}"


def parse_training_log(log_line: str) -> Optional[Dict]:
    """解析训练日志，提取指标"""
    import re
    metrics = {}
    
    # 匹配Loss
    loss_match = re.search(r'loss[:\s]+([\d.]+)', log_line, re.I)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    
    # 匹配学习率
    lr_match = re.search(r'lr[:\s]+([\d.e-]+)', log_line, re.I)
    if lr_match:
        metrics['lr'] = float(lr_match.group(1))
    
    # 匹配步数
    step_match = re.search(r'\((\d+)/(\d+)\)', log_line)
    if step_match:
        metrics['step'] = int(step_match.group(1))
        metrics['total_steps'] = int(step_match.group(2))
    
    # 匹配Epoch
    epoch_match = re.search(r'Epoch\[(\d+)/(\d+)\]', log_line)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
        metrics['total_epochs'] = int(epoch_match.group(2))
    
    return metrics if metrics else None


def get_log_file_path(task_id: str):
    """获取任务日志文件路径"""
    # 使用与代码第25行相同的路径获取方式，确保一致性
    # 使用绝对路径，避免工作目录变化导致的问题
    script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts 目录（绝对路径）
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # 项目根目录（绝对路径）
    logs_dir = Path(project_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{task_id}.log"


def process_log_line(line: str, task_id: str, task_logs: list, task_metrics: dict):
    """处理单行日志：添加日志、解析指标、更新线程安全存储"""
    line = line.strip()
    if not line:  # 忽略空行
        return
    
    # 1. 添加日志到本地列表
    task_logs.append(line)
    
    # 2. 写入日志文件（追加模式）
    try:
        log_file = get_log_file_path(task_id)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception as e:
        # 文件写入失败不影响程序运行
        pass
    
    # 3. 使用线程安全的方式更新日志（只保留最近100行）
    with _data_lock:
        _thread_safe_data['task_logs'][task_id] = task_logs[-100:]
    
    # 4. 解析指标
    metrics = parse_training_log(line)
    if not metrics:
        return
    
    # 5. 添加指标到本地字典
    timestamp = time.time()
    if 'loss' in metrics:
        task_metrics['loss'].append(metrics['loss'])
        task_metrics['lr'].append(metrics.get('lr', 0))
        task_metrics['step'].append(metrics.get('step', 0))
        task_metrics['epoch'].append(metrics.get('epoch', 0))
        task_metrics['timestamp'].append(timestamp)
        
        # 6. 更新任务指标（只保留最近500个点）
        for key in task_metrics:
            if len(task_metrics[key]) > 500:
                task_metrics[key] = task_metrics[key][-500:]
        
        # 7. 使用线程安全的方式更新指标
        with _data_lock:
            _thread_safe_data['task_metrics'][task_id] = {
                k: v.copy() if isinstance(v, list) else v 
                for k, v in task_metrics.items()
            }


def monitor_training_task(task_id: str, process: subprocess.Popen, task_config: Dict):
    """监控训练任务进程（在后台线程中运行）"""
    task_logs = []
    task_metrics = {
        'loss': [],
        'lr': [],
        'step': [],
        'epoch': [],
        'timestamp': []
    }
    
    if process.stdout:
        try:
            while True:
                # 检查进程是否还在运行
                if process.poll() is not None:
                    # 进程已结束，读取剩余输出
                    remaining = process.stdout.read()
                    if remaining:
                        for line in remaining.splitlines():
                            process_log_line(line, task_id, task_logs, task_metrics)
                    break
                
                # 尝试读取一行（非阻塞）
                line = process.stdout.readline()
                if line:
                    process_log_line(line, task_id, task_logs, task_metrics)
                else:
                    # 没有新输出，短暂休眠避免CPU占用过高
                    time.sleep(0.1)
        except Exception as e:
            # 记录错误但不中断
            with _data_lock:
                _thread_safe_data['task_logs'][task_id] = task_logs[-100:] + [f"[错误] 日志读取异常: {str(e)}"]
    
    # 等待进程结束
    process.wait()
    
    # 更新任务状态（需要同步到主线程，这里先保存到文件）
    # 注意：不能在这里直接修改 st.session_state，需要在主线程中处理
    final_status = {
        'status': 'completed' if process.returncode == 0 else 'failed',
        'end_time': datetime.now().isoformat(),
        'returncode': process.returncode
    }
    
    # 保存最终状态到线程安全存储
    with _data_lock:
        _thread_safe_data['task_final_status'] = _thread_safe_data.get('task_final_status', {})
        _thread_safe_data['task_final_status'][task_id] = final_status
        if process.returncode != 0:
            _thread_safe_data['task_final_status'][task_id]['error'] = "训练进程异常退出"


def start_training_task(task_config: Dict) -> str:
    """启动训练任务"""
    task_id = generate_task_id()
    
    # 构建训练命令
    script_path = task_config['training_script']
    cmd = ['python', script_path]
    
    # 添加参数
    param_mapping = {
        'epochs': '--epochs',
        'batch_size': '--batch_size',
        'learning_rate': '--learning_rate',
        'hidden_size': '--hidden_size',
        'num_hidden_layers': '--num_hidden_layers',
        'max_seq_len': '--max_seq_len',
        'use_moe': '--use_moe',
        'data_path': '--data_path',
        'from_weight': '--from_weight',
        'from_resume': '--from_resume',
        'save_dir': '--save_dir',
        'save_weight': '--save_weight',
        'device': '--device',
        'dtype': '--dtype',
    }
    
    for key, arg_name in param_mapping.items():
        if key in task_config and task_config[key] is not None:
            if key == 'use_moe':#后续拓展
                cmd.extend([arg_name, str(task_config[key])])
            elif key == 'from_resume':
                cmd.extend([arg_name, str(task_config[key])])
            elif key == 'data_path':
                cmd.extend([arg_name, str(task_config[key])])
            else:
                cmd.extend([arg_name, str(task_config[key])])
    
    # 处理 use_wandb 参数（action="store_true" 类型，只需要添加参数名，不需要值）
    if task_config.get('use_wandb', False):
        cmd.append('--use_wandb')
    
    # 启动训练进程
    try:
        # 设置环境变量强制Python输出不缓冲（确保实时输出）
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path("../trainer").resolve()),
            bufsize=1,  # 行缓冲
            env=env  # 使用修改后的环境变量
        )
        
        # 保存任务信息
        st.session_state.tasks[task_id] = {
            'id': task_id,
            'config': task_config,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'pid': process.pid
        }
        st.session_state.task_processes[task_id] = process
        
        # 启动监控线程
        monitor_thread = threading.Thread(
            target=monitor_training_task,
            args=(task_id, process, task_config),
            daemon=True
        )
        monitor_thread.start()
        
        save_tasks(st.session_state.tasks)
        return task_id
        
    except Exception as e:
        st.error(f"启动训练任务失败: {str(e)}")
        return None


def stop_training_task(task_id: str):
    """停止训练任务"""
    if task_id in st.session_state.task_processes:
        process = st.session_state.task_processes[task_id]
        try:
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            
            if task_id in st.session_state.tasks:
                st.session_state.tasks[task_id]['status'] = 'stopped'
                st.session_state.tasks[task_id]['end_time'] = datetime.now().isoformat()
                save_tasks(st.session_state.tasks)
            
            del st.session_state.task_processes[task_id]
            return True
        except Exception as e:
            st.error(f"停止任务失败: {str(e)}")
            return False
    return False


def sync_thread_data_to_session():
    """从线程安全存储同步数据到session_state"""
    with _data_lock:
        # 同步日志（优先从文件读取完整日志，如果没有则使用内存中的）
        for task_id, logs in _thread_safe_data['task_logs'].items():
            # 尝试从文件读取完整日志
            file_logs = load_task_logs_from_file(task_id)
            if file_logs:
                # 如果文件中有日志，使用文件中的（更完整）
                st.session_state.task_logs[task_id] = file_logs
            else:
                # 如果没有文件，使用内存中的（最近100行）
                st.session_state.task_logs[task_id] = logs.copy()
        
        # 同步指标
        for task_id, metrics in _thread_safe_data['task_metrics'].items():
            st.session_state.task_metrics[task_id] = {
                k: v.copy() if isinstance(v, list) else v 
                for k, v in metrics.items()
            }
        
        # 同步最终状态
        if 'task_final_status' in _thread_safe_data:
            for task_id, final_status in _thread_safe_data['task_final_status'].items():
                if task_id in st.session_state.tasks:
                    st.session_state.tasks[task_id].update(final_status)
                    if final_status['status'] in ['completed', 'failed']:
                        save_tasks(st.session_state.tasks)
            # 清理已处理的状态
            _thread_safe_data['task_final_status'] = {}


def main():
    
    # 页面标题
    st.markdown('<h1 class="main-header">🚀 Mini大模型训练云平台</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("导航")
        page = st.radio(
            "选择页面",
            ["创建训练任务", "任务监控", "任务管理", "模型管理", "模型使用"],
            index=0
        )
        
        # 如果切换到其他页面，自动释放GPU（如果模型已加载）
        if page != "模型使用" and 'current_model' in st.session_state and st.session_state.current_model is not None:
            if st.session_state.get('auto_release_gpu', True):
                try:
                    release_model_from_gpu()
                except:
                    pass  # 静默处理错误，避免影响页面切换
        
        st.markdown("---")
        st.header("系统信息")
        st.info("Mini大模型训练云平台\n\n简化大语言模型训练流程")
    
    # 加载已保存的任务
    if not st.session_state.tasks:
        saved_tasks = load_saved_tasks()
        st.session_state.tasks.update(saved_tasks)
    
    # 同步线程数据到session_state（每次页面刷新时）
    sync_thread_data_to_session()
    
    # 创建训练任务页面
    if page == "创建训练任务":
        st.header("📝 创建训练任务")
        
        # 将文件上传移到表单外部，使其可以立即处理
        st.subheader("数据集配置")
        
        # 文件上传功能
        uploaded_file = st.file_uploader(
            "📤 上传训练数据集文件",
            type=['jsonl'],
            help="支持 .jsonl 格式的数据集文件，上传后将保存到 dataset 目录"
        )
        
        if uploaded_file is not None:
            # 保存上传的文件到 dataset 目录
            dataset_dir = Path("../dataset")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查文件是否已存在
            file_path = dataset_dir / uploaded_file.name
            if file_path.exists():
                st.warning(f"⚠️ 文件 `{uploaded_file.name}` 已存在，将被覆盖。")
            
            # 保存文件
            try:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                st.success(f"✅ 文件已成功上传到 `{file_path}` (大小: {file_size_mb:.2f} MB)")
                # 自动刷新页面，使新文件立即显示在下拉列表中
                st.rerun()
            except Exception as e:
                st.error(f"❌ 文件上传失败: {str(e)}")
        
        with st.form("training_task_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("基础配置")
                
                # 训练类型
                training_scripts = get_training_scripts()
                if not training_scripts:
                    st.error("未找到训练脚本，请确保trainer目录存在")
                    st.stop()
                
                training_type = st.selectbox(
                    "训练类型",
                    options=list(training_scripts.keys()),
                    format_func=lambda x: training_scripts[x]['display']
                )
                training_script = training_scripts[training_type]['path']
                
                # 数据集选择（重新获取，确保包含新上传的文件）
                st.subheader("选择数据集")
                datasets = get_datasets()
                
                if not datasets:
                    st.warning("⚠️ 未找到数据集文件！")
                    st.info("📁 请上传数据集文件（.jsonl格式）或将其放置在 `dataset` 目录下。")
                    data_path = None  # 如果没有数据集，设置为 None
                else:
                    selected_dataset = st.selectbox("选择数据集", options=list(datasets.keys()))
                    data_path = datasets[selected_dataset]
                
                # 模型配置
                st.subheader("模型配置")
                hidden_size = st.number_input("隐藏层维度 (hidden_size)", min_value=256, max_value=2048, value=512, step=64)
                num_hidden_layers = st.number_input("隐藏层数量 (num_hidden_layers)", min_value=4, max_value=32, value=8, step=2)
                max_seq_len = st.number_input("最大序列长度 (max_seq_len)", min_value=128, max_value=8192, value=340, step=64)
                use_moe = st.checkbox("使用MoE架构", value=False)
                
            with col2:
                st.subheader("训练配置")
                
                epochs = st.number_input("训练轮数 (epochs)", min_value=1, max_value=100, value=2)
                batch_size = st.number_input("批次大小 (batch_size)", min_value=1, max_value=128, value=16, step=4)
                learning_rate = st.number_input("学习率 (learning_rate)", min_value=1e-8, max_value=1e-3, value=5e-7, format="%.2e", step=1e-7)
                
                accumulation_steps = st.number_input("梯度累积步数", min_value=1, max_value=32, value=1)
                grad_clip = st.number_input("梯度裁剪阈值", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                
                st.subheader("其他配置")
                device = st.selectbox("训练设备", options=["cuda:0", "cuda:1", "cpu"], index=0)
                dtype = st.selectbox("数据类型", options=["bfloat16", "float16", "float32"], index=0)
                
                # 获取可用的权重前缀
                available_weights = get_available_weights(save_dir="../out")
                if not available_weights or available_weights == ["none"]:
                    st.warning("⚠️ 未找到可用的模型权重文件，将从头开始训练")
                    from_weight = "none"
                else:
                    # 设置默认值：如果有 "pretrain" 则优先选择，否则选择第一个非 "none" 的
                    default_index = 0
                    if "pretrain" in available_weights:
                        default_index = available_weights.index("pretrain")
                    elif "none" in available_weights:
                        # 如果只有 "none"，选择它
                        default_index = available_weights.index("none")
                    
                    from_weight = st.selectbox(
                        "基础权重 (from_weight)",
                        options=available_weights,
                        index=default_index,
                        help="选择要加载的模型权重前缀。选择 'none' 表示从头开始训练。"
                    )
                
                from_resume = st.checkbox("启用断点续训 (from_resume)", value=False)
                use_wandb = st.checkbox("启用WandB/SwanLab记录 (use_wandb)", value=False, help="启用后将使用WandB或SwanLab记录训练过程")
                
                save_dir = st.text_input("保存目录", value="../out")
                save_weight = st.text_input("权重名称前缀", value="full_sft")
            
            # 提交按钮
            submitted = st.form_submit_button("🚀 提交训练任务", use_container_width=True)
            
            if submitted:
                # 验证数据集路径
                if 'data_path' not in locals() or data_path is None:
                    st.error("❌ 请先上传或选择数据集文件！")
                    st.stop()
                
                task_config = {
                    'training_script': training_script,
                    'training_type': training_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_size': hidden_size,
                    'num_hidden_layers': num_hidden_layers,
                    'max_seq_len': max_seq_len,
                    'use_moe': 1 if use_moe else 0,
                    'data_path': data_path,
                    'from_weight': from_weight,
                    'from_resume': 1 if from_resume else 0,
                    'use_wandb': use_wandb,
                    'save_dir': save_dir,
                    'save_weight': save_weight,
                    'device': device,
                    'dtype': dtype,
                    'accumulation_steps': accumulation_steps,
                    'grad_clip': grad_clip,
                }
                
                task_id = start_training_task(task_config)
                if task_id:
                    st.success(f"✅ 训练任务已提交！任务ID: {task_id}")
                    st.balloons()
                else:
                    st.error("❌ 提交失败，请检查配置")
    
    # 任务监控页面
    elif page == "任务监控":
        st.header("📊 任务监控")
        
        if not st.session_state.tasks:
            st.info("暂无训练任务")
        else:
            # 选择要监控的任务
            running_tasks = {k: v for k, v in st.session_state.tasks.items() if v['status'] == 'running'}
            
            if not running_tasks:
                st.info("当前没有运行中的任务")
            else:
                selected_task_id = st.selectbox(
                    "选择任务",
                    options=list(running_tasks.keys()),
                    format_func=lambda x: f"{x} - {running_tasks[x]['config'].get('training_type', 'N/A')}"
                )
                
                if selected_task_id:
                    task = st.session_state.tasks[selected_task_id]
                    
                    # 任务信息卡片
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("任务状态", task['status'])
                    with col2:
                        start_time = datetime.fromisoformat(task['start_time'])
                        elapsed = datetime.now() - start_time
                        st.metric("运行时间", f"{elapsed.seconds // 60}分钟")
                    with col3:
                        if 'pid' in task:
                            st.metric("进程ID", task['pid'])
                    with col4:
                        if 'config' in task:
                            st.metric("训练类型", task['config'].get('training_type', 'N/A'))
                    
                    # 训练指标图表
                    # 优先从日志文件提取指标，如果没有则使用内存中的
                    file_logs = load_task_logs_from_file(selected_task_id)
                    if file_logs:
                        metrics = extract_metrics_from_logs(file_logs)
                    elif selected_task_id in st.session_state.task_metrics:
                        metrics = st.session_state.task_metrics[selected_task_id]
                    else:
                        metrics = {'loss': [], 'lr': [], 'step': [], 'epoch': [], 'timestamp': []}
                    
                    plot_training_metrics(metrics)
                    
                    # 实时日志
                    col_log1, col_log2 = st.columns([3, 1])
                    with col_log1:
                        st.subheader("📋 训练日志")
                    with col_log2:
                        if st.button("🔄 刷新日志", key=f"refresh_logs_{selected_task_id}", use_container_width=True):
                            # 强制同步日志
                            sync_thread_data_to_session()
                            st.rerun()
                    
                    # 优先从文件读取日志（完整日志）
                    file_logs = load_task_logs_from_file(selected_task_id)
                    if file_logs:
                        logs = file_logs
                    elif selected_task_id in st.session_state.task_logs:
                        logs = st.session_state.task_logs[selected_task_id]
                    else:
                        logs = []
                    
                    if logs:
                        # 显示最近500行日志（使用code block以便更好地显示）
                        display_logs = logs[-500:] if len(logs) > 500 else logs
                        log_text = "\n".join(display_logs)
                        st.code(log_text, language=None)
                        st.caption(f"显示最近 {len(display_logs)} 行日志（共 {len(logs)} 行）")
                        if len(logs) > 500:
                            st.info(f"💡 日志文件包含 {len(logs)} 行，仅显示最近 500 行。完整日志保存在 `logs/{selected_task_id}.log`")
                    else:
                        st.info("暂无日志输出")
                    
                    # 添加自动刷新提示
                    st.caption("💡 提示：日志会实时更新，点击「刷新日志」按钮或刷新页面查看最新日志")
    
    # 任务管理页面
    elif page == "任务管理":
        st.header("📋 任务管理")
        
        if not st.session_state.tasks:
            st.info("暂无任务记录")
        else:
            # 任务筛选
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox(
                    "筛选状态",
                    options=["全部", "running", "completed", "failed", "stopped", "pending"]
                )
            with col2:
                search_keyword = st.text_input("搜索任务ID或训练类型")
            
            # 筛选任务
            filtered_tasks = st.session_state.tasks.copy()
            if status_filter != "全部":
                filtered_tasks = {k: v for k, v in filtered_tasks.items() if v['status'] == status_filter}
            if search_keyword:
                filtered_tasks = {
                    k: v for k, v in filtered_tasks.items()
                    if search_keyword.lower() in k.lower() or
                    search_keyword.lower() in v.get('config', {}).get('training_type', '').lower()
                }
            
            # 任务列表
            for task_id, task in filtered_tasks.items():
                expander_expanded = (st.session_state.get('view_logs_task') == task_id)
                with st.expander(f"任务: {task_id} | 状态: {task['status']} | 类型: {task.get('config', {}).get('training_type', 'N/A')}", expanded=expander_expanded):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.json(task.get('config', {}))
                        
                        if 'start_time' in task:
                            st.write(f"开始时间: {task['start_time']}")
                        if 'end_time' in task:
                            st.write(f"结束时间: {task['end_time']}")
                        if 'error' in task:
                            st.error(f"错误信息: {task['error']}")
                    
                    with col2:
                        if task['status'] == 'running':
                            if st.button("停止任务", key=f"stop_{task_id}"):
                                if stop_training_task(task_id):
                                    st.success("任务已停止")
                                    st.rerun()
                        
                        if st.button("查看日志", key=f"logs_{task_id}"):
                            st.session_state['view_logs_task'] = task_id
                            st.rerun()
                        
                        if st.button("删除任务", key=f"delete_{task_id}"):
                            if task['status'] == 'running':
                                st.warning("请先停止运行中的任务")
                            else:
                                del st.session_state.tasks[task_id]
                                if task_id in st.session_state.task_processes:
                                    del st.session_state.task_processes[task_id]
                                if task_id in st.session_state.task_logs:
                                    del st.session_state.task_logs[task_id]
                                if task_id in st.session_state.task_metrics:
                                    del st.session_state.task_metrics[task_id]
                                save_tasks(st.session_state.tasks)
                                st.success("任务已删除")
                                st.rerun()
                    
                    # 显示日志区域（包含可视化图表）
                    if st.session_state.get('view_logs_task') == task_id:
                        st.markdown("---")
                        
                        # 任务信息卡片
                        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                        with col_info1:
                            st.metric("任务状态", task['status'])
                        with col_info2:
                            if 'start_time' in task:
                                start_time = datetime.fromisoformat(task['start_time'])
                                elapsed = datetime.now() - start_time
                                st.metric("运行时间", f"{elapsed.seconds // 60}分钟")
                        with col_info3:
                            if 'pid' in task:
                                st.metric("进程ID", task['pid'])
                        with col_info4:
                            if 'config' in task:
                                st.metric("训练类型", task['config'].get('training_type', 'N/A'))
                        
                        # 训练指标图表
                        # 优先从日志文件提取指标，如果没有则使用内存中的
                        file_logs = load_task_logs_from_file(task_id)
                        if file_logs:
                            metrics = extract_metrics_from_logs(file_logs)
                        elif task_id in st.session_state.task_metrics:
                            metrics = st.session_state.task_metrics[task_id]
                        else:
                            metrics = {'loss': [], 'lr': [], 'step': [], 'epoch': [], 'timestamp': []}
                        
                        plot_training_metrics(metrics)
                        
                        # 日志文本显示区域
                        st.markdown("---")
                        st.subheader(f"📋 任务日志: {task_id}")
                        
                        # 从文件读取日志
                        if file_logs:
                            logs = file_logs
                        elif task_id in st.session_state.task_logs:
                            logs = st.session_state.task_logs[task_id]
                        else:
                            logs = []
                        
                        if logs:
                            # 显示最近500行日志
                            display_logs = logs[-500:] if len(logs) > 500 else logs
                            log_text = "\n".join(display_logs)
                            st.code(log_text, language=None)
                            st.caption(f"显示最近 {len(display_logs)} 行日志（共 {len(logs)} 行）")
                            if len(logs) > 500:
                                st.info(f"💡 完整日志保存在 `logs/{task_id}.log`")
                        else:
                            st.info("暂无日志输出。日志文件可能尚未创建或任务刚刚启动。")
                        
                        if st.button("关闭日志", key=f"close_logs_{task_id}", use_container_width=True):
                            st.session_state['view_logs_task'] = None
                            st.rerun()
    
    # 模型管理页面
    elif page == "模型管理":
        st.header("📦 模型管理")
        
        # 创建两个标签页：PyTorch模型和Transformers模型
        tab1, tab2 = st.tabs(["PyTorch格式模型 (.pth)", "Transformers格式模型"])
        
        # ========== PyTorch格式模型 ==========
        with tab1:
            out_dir = Path("../out")
            if out_dir.exists():
                model_files = list(out_dir.glob("*.pth"))
                
                if not model_files:
                    st.info("未找到PyTorch格式的模型文件")
                else:
                    st.write(f"找到 {len(model_files)} 个PyTorch模型文件")
                    
                    for model_file in model_files:
                        with st.expander(f"模型: {model_file.name}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                                st.metric("文件大小", f"{file_size:.2f} MB")
                                st.write(f"路径: {model_file}")
                            
                            with col2:
                                mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                                st.metric("修改时间", mtime.strftime("%Y-%m-%d %H:%M:%S"))
                            
                            with col3:
                                if st.button("下载", key=f"download_{model_file.name}"):
                                    with open(model_file, 'rb') as f:
                                        st.download_button(
                                            "点击下载",
                                            f.read(),
                                            file_name=model_file.name,
                                            mime="application/octet-stream",
                                            key=f"dl_{model_file.name}"
                                        )
                                
                                # 转换格式按钮
                                if st.button("🔄 转换为Transformers格式", key=f"convert_{model_file.name}"):
                                    st.session_state[f'show_convert_{model_file.name}'] = True
                            
                            # 转换格式对话框
                            if st.session_state.get(f'show_convert_{model_file.name}', False):
                                with st.expander("模型格式转换 (PyTorch → Transformers)", expanded=True):
                                    st.write(f"**源文件**: {model_file.name}")
                                    
                                    # 解析模型配置
                                    try:
                                        model_config = parse_model_config_from_filename(str(model_file))
                                        st.info(f"检测到的配置: hidden_size={model_config['hidden_size']}, "
                                              f"num_hidden_layers={model_config['num_hidden_layers']}, "
                                              f"use_moe={model_config['use_moe']}")
                                    except Exception as e:
                                        st.error(f"解析模型配置失败: {str(e)}")
                                        st.session_state[f'show_convert_{model_file.name}'] = False
                                        st.stop()
                                    
                                    # 转换类型选择
                                    convert_type = st.radio(
                                        "转换格式",
                                        options=["llama", "minimind"],
                                        format_func=lambda x: "Llama兼容格式（推荐）" if x == "llama" else "MiniMind原生格式",
                                        index=0,
                                        help="Llama格式兼容更多第三方工具（vllm、ollama等）"
                                    )
                                    
                                    # 数据类型选择
                                    dtype = st.selectbox(
                                        "数据类型",
                                        options=["float16", "bfloat16"],
                                        index=0,
                                        help="float16兼容性更好，bfloat16精度更高"
                                    )
                                    
                                    # 输出路径
                                    default_output_name = f"{model_file.stem}_transformers"
                                    output_name = st.text_input(
                                        "输出目录名",
                                        value=default_output_name,
                                        help="将在项目根目录下创建此目录"
                                    )
                                    output_path = str(Path(f"../{output_name}").resolve())
                                    
                                    col_conv1, col_conv2 = st.columns(2)
                                    
                                    with col_conv1:
                                        if st.button("✅ 开始转换", key=f"do_convert_{model_file.name}", use_container_width=True):
                                            if not output_name.strip():
                                                st.error("请输入输出目录名")
                                            else:
                                                # 检查输出目录是否已存在
                                                if Path(output_path).exists():
                                                    st.warning(f"输出目录 {output_path} 已存在，转换将覆盖现有文件")
                                                
                                                with st.spinner("正在转换模型，请稍候..."):
                                                    success, message = convert_torch_to_transformers(
                                                        str(model_file.resolve()),
                                                        output_path,
                                                        model_config,
                                                        convert_type=convert_type,
                                                        dtype=dtype
                                                    )
                                                
                                                if success:
                                                    st.success(message)
                                                    st.balloons()
                                                    # 清理状态
                                                    st.session_state[f'show_convert_{model_file.name}'] = False
                                                    st.rerun()
                                                else:
                                                    st.error(message)
                                    
                                    with col_conv2:
                                        if st.button("❌ 取消", key=f"cancel_convert_{model_file.name}", use_container_width=True):
                                            st.session_state[f'show_convert_{model_file.name}'] = False
                                            st.rerun()
            else:
                st.warning("输出目录不存在: ../out")
        
        # ========== Transformers格式模型 ==========
        with tab2:
            # 扫描Transformers格式模型目录
            model_dirs = [
                Path("../MiniMind2"),
                Path("../MiniMind2-Small"),
                Path("../MiniMind2-MoE"),
                Path("../MiniMind2-R1"),
                Path("../MiniMind2-Small-R1"),
            ]
            
            # 扫描项目根目录下所有包含config.json的目录
            root_dir = Path("..")
            transformers_models = []
            for model_dir in model_dirs:
                if model_dir.exists() and (model_dir / "config.json").exists():
                    transformers_models.append(model_dir)
            
            # 扫描其他可能的模型目录（名称包含transformers的）
            for item in root_dir.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    if item not in transformers_models and not item.name.startswith('.'):
                        transformers_models.append(item)
            
            if not transformers_models:
                st.info("未找到Transformers格式的模型目录")
            else:
                st.write(f"找到 {len(transformers_models)} 个Transformers模型目录")
                
                for model_dir in transformers_models:
                    with st.expander(f"模型: {model_dir.name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # 计算目录大小
                            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                            dir_size_mb = total_size / (1024 * 1024)
                            st.metric("目录大小", f"{dir_size_mb:.2f} MB")
                            st.write(f"路径: {model_dir}")
                        
                        with col2:
                            # 获取config.json的修改时间
                            config_file = model_dir / "config.json"
                            if config_file.exists():
                                mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                                st.metric("修改时间", mtime.strftime("%Y-%m-%d %H:%M:%S"))
                            else:
                                st.metric("修改时间", "未知")
                        
                        with col3:
                            # 转换格式按钮
                            model_key = f"convert_tf_{model_dir.name}"
                            if st.button("🔄 转换为PyTorch格式", key=model_key):
                                st.session_state[f'show_convert_tf_{model_dir.name}'] = True
                        
                        # 转换格式对话框
                        if st.session_state.get(f'show_convert_tf_{model_dir.name}', False):
                            with st.expander("模型格式转换 (Transformers → PyTorch)", expanded=True):
                                st.write(f"**源目录**: {model_dir.name}")
                                
                                # 输出文件名
                                default_output_name = f"{model_dir.name}.pth"
                                output_name = st.text_input(
                                    "输出文件名",
                                    value=default_output_name,
                                    help="将在 ../out/ 目录下创建此文件",
                                    key=f"output_name_tf_{model_dir.name}"
                                )
                                output_path = str((Path("../out") / output_name).resolve())
                                
                                # 确保输出目录存在
                                Path("../out").mkdir(parents=True, exist_ok=True)
                                
                                col_conv1, col_conv2 = st.columns(2)
                                
                                with col_conv1:
                                    if st.button("✅ 开始转换", key=f"do_convert_tf_{model_dir.name}", use_container_width=True):
                                        if not output_name.strip():
                                            st.error("请输入输出文件名")
                                        else:
                                            # 检查输出文件是否已存在
                                            if Path(output_path).exists():
                                                st.warning(f"输出文件 {output_path} 已存在，转换将覆盖现有文件")
                                            
                                            with st.spinner("正在转换模型，请稍候..."):
                                                success, message = convert_transformers_to_torch(
                                                    str(model_dir.resolve()),
                                                    output_path
                                                )
                                            
                                            if success:
                                                st.success(message)
                                                st.balloons()
                                                # 清理状态
                                                st.session_state[f'show_convert_tf_{model_dir.name}'] = False
                                                st.rerun()
                                            else:
                                                st.error(message)
                                
                                with col_conv2:
                                    if st.button("❌ 取消", key=f"cancel_convert_tf_{model_dir.name}", use_container_width=True):
                                        st.session_state[f'show_convert_tf_{model_dir.name}'] = False
                                        st.rerun()
    
    # 模型使用页面
    elif page == "模型使用":
        st.header("💬 模型使用")
        
        # 初始化对话历史和模型状态
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'current_model_path' not in st.session_state:
            st.session_state.current_model_path = None
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'current_tokenizer' not in st.session_state:
            st.session_state.current_tokenizer = None
        if 'model_last_used_time' not in st.session_state:
            st.session_state.model_last_used_time = None
        if 'auto_release_gpu' not in st.session_state:
            st.session_state.auto_release_gpu = True
        if 'gpu_release_timeout' not in st.session_state:
            st.session_state.gpu_release_timeout = 300  # 默认5分钟无操作自动释放
        
        # 自动释放GPU检查（如果超过指定时间未使用）
        if st.session_state.current_model is not None and st.session_state.auto_release_gpu:
            if st.session_state.model_last_used_time is not None:
                time_since_last_use = time.time() - st.session_state.model_last_used_time
                if time_since_last_use > st.session_state.gpu_release_timeout:
                    with st.spinner("检测到模型长时间未使用，正在自动释放GPU..."):
                        if release_model_from_gpu():
                            st.info("✅ GPU已自动释放（模型超过5分钟未使用）")
                            st.rerun()
        
        # 获取可用模型列表（在主区域也使用）
        available_models = get_available_models()
        
        # 初始化设备选择
        if 'inference_device' not in st.session_state:
            st.session_state.inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 侧边栏配置
        with st.sidebar:
            st.subheader("模型配置")
            
            if not available_models:
                st.warning("未找到可用模型\n\n请确保：\n1. Transformers格式模型在项目根目录下（包含config.json文件）\n2. 如需使用PyTorch模型，请先在「模型管理」页面转换为Transformers格式")
            else:
                # 模型选择（只显示Transformers格式模型）
                model_options = {info['name']: path 
                                for path, info in available_models.items()}
                selected_model_display = st.selectbox(
                    "选择模型 (Transformers格式)",
                    options=list(model_options.keys()),
                    index=0,
                    help="仅显示Transformers格式模型，如需使用PyTorch模型请先在「模型管理」页面转换"
                )
                selected_model_path = model_options[selected_model_display]
                selected_model_info = available_models[selected_model_path]
                
                # 设备选择
                device = st.selectbox(
                    "运行设备",
                    options=["cuda:0", "cuda:1", "cpu"],
                    index=0 if torch.cuda.is_available() else 2,
                    key="device_selector"
                )
                st.session_state.inference_device = device
                
                # GPU自动释放设置
                st.markdown("---")
                st.subheader("GPU管理")
                auto_release = st.checkbox(
                    "自动释放GPU",
                    value=st.session_state.auto_release_gpu,
                    help="切换页面或超过指定时间未使用时自动释放GPU"
                )
                st.session_state.auto_release_gpu = auto_release
                
                if auto_release:
                    timeout_minutes = st.number_input(
                        "自动释放时间（分钟）",
                        min_value=1,
                        max_value=60,
                        value=int(st.session_state.gpu_release_timeout / 60),
                        help="超过此时间未使用模型将自动释放GPU"
                    )
                    st.session_state.gpu_release_timeout = timeout_minutes * 60
                
                # 加载模型按钮
                if st.button("🔄 加载模型", use_container_width=True):
                    # 如果已有模型，先释放
                    if st.session_state.current_model is not None:
                        release_model_from_gpu()
                    
                    with st.spinner("正在加载模型..."):
                        model, tokenizer = load_model_for_inference(
                            selected_model_path, 
                            selected_model_info,
                            device=st.session_state.inference_device
                        )
                        if model is not None:
                            st.session_state.current_model = model
                            st.session_state.current_tokenizer = tokenizer
                            st.session_state.current_model_path = selected_model_path
                            st.session_state.model_last_used_time = time.time()
                            st.success("✅ 模型加载成功！")
                            # 切换模型时清空对话历史
                            st.session_state.chat_messages = []
                        else:
                            st.error("❌ 模型加载失败")
                
                # 手动释放GPU按钮
                if st.session_state.current_model is not None:
                    if st.button("🗑️ 释放GPU", use_container_width=True):
                        if release_model_from_gpu():
                            st.success("✅ GPU已释放")
                            st.rerun()
                
                st.markdown("---")
                st.subheader("生成参数")
                
                temperature = st.slider(
                    "Temperature (温度)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.85,
                    step=0.05,
                    help="控制生成的随机性，值越大越随机"
                )
                
                max_new_tokens = st.slider(
                    "Max New Tokens (最大生成长度)",
                    min_value=128,
                    max_value=8192,
                    value=2048,
                    step=128,
                    help="模型生成的最大token数量"
                )
                
                top_p = st.slider(
                    "Top-p (核采样)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.85,
                    step=0.05,
                    help="nucleus采样阈值"
                )
                
                history_chat_num = st.slider(
                    "历史对话轮数",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=2,
                    help="保留的历史对话轮数（0表示不使用历史）"
                )
                
                if st.button("🗑️ 清空对话", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
        
        # 主对话区域
        if st.session_state.current_model is None:
            st.info("👈 请在左侧侧边栏选择模型并点击「加载模型」按钮")
        else:
            # 显示模型信息和GPU状态
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                # 从session_state获取模型路径对应的信息
                current_path = st.session_state.current_model_path
                if current_path and current_path in available_models:
                    model_name = available_models[current_path]['name']
                else:
                    model_name = "未知模型"
                st.success(f"✅ 当前使用模型: {model_name}")
            
            with col2:
                if st.session_state.model_last_used_time:
                    last_use_ago = int(time.time() - st.session_state.model_last_used_time)
                    st.caption(f"最后使用: {last_use_ago // 60}分{last_use_ago % 60}秒前")
            
            with col3:
                if torch.cuda.is_available() and st.session_state.inference_device.startswith("cuda"):
                    try:
                        device_id = int(st.session_state.inference_device.split(":")[1])
                        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                        gpu_allocated = torch.cuda.memory_allocated(device_id) / 1e9
                        st.caption(f"GPU显存: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB")
                    except:
                        st.caption("GPU信息获取失败")
            
            # 显示对话历史
            for i, msg in enumerate(st.session_state.chat_messages):
                if msg["role"] == "user":
                    st.markdown(
                        f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
                        f'<div style="display: inline-block; padding: 10px 15px; background-color: #007bff; '
                        f'border-radius: 15px; color: white; max-width: 70%;">'
                        f'{msg["content"]}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="display: flex; justify-content: flex-start; margin: 10px 0;">'
                        f'<div style="display: inline-block; padding: 10px 15px; background-color: #f0f0f0; '
                        f'border-radius: 15px; max-width: 70%;">'
                        f'{process_assistant_content(msg["content"])}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
            
            # 输入框
            user_input = st.chat_input("输入消息...")
            
            if user_input:
                # 更新最后使用时间
                st.session_state.model_last_used_time = time.time()
                
                # 添加用户消息
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                # 显示用户消息
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
                    f'<div style="display: inline-block; padding: 10px 15px; background-color: #007bff; '
                    f'border-radius: 15px; color: white; max-width: 70%;">'
                    f'{user_input}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                
                # 生成回复
                with st.spinner("思考中..."):
                    try:
                        model = st.session_state.current_model
                        tokenizer = st.session_state.current_tokenizer
                        
                        # 准备对话历史
                        history_messages = st.session_state.chat_messages
                        if history_chat_num > 0:
                            history_messages = history_messages[-(history_chat_num + 1):]
                        
                        # 应用聊天模板
                        try:
                            prompt = tokenizer.apply_chat_template(
                                history_messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        except:
                            # 如果没有chat_template，使用简单格式
                            prompt = "\n".join([
                                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                                for m in history_messages
                            ]) + "\n助手: "
                        
                        # Tokenize
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True
                        ).to(st.session_state.inference_device)
                        
                        # 流式生成
                        from transformers import TextIteratorStreamer
                        from threading import Thread
                        
                        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        
                        generation_kwargs = {
                            "input_ids": inputs.input_ids,
                            "attention_mask": inputs.attention_mask,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": top_p,
                            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            "eos_token_id": tokenizer.eos_token_id,
                            "streamer": streamer,
                        }
                        
                        # 启动生成线程
                        placeholder = st.empty()
                        thread = Thread(target=model.generate, kwargs=generation_kwargs)
                        thread.start()
                        
                        # 流式显示
                        answer = ""
                        for text in streamer:
                            answer += text
                            placeholder.markdown(
                                f'<div style="display: flex; justify-content: flex-start; margin: 10px 0;">'
                                f'<div style="display: inline-block; padding: 10px 15px; background-color: #f0f0f0; '
                                f'border-radius: 15px; max-width: 70%;">'
                                f'{process_assistant_content(answer)}'
                                f'</div></div>',
                                unsafe_allow_html=True
                            )
                        
                        # 添加到对话历史
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                        
                        # 更新最后使用时间
                        st.session_state.model_last_used_time = time.time()
                        
                    except Exception as e:
                        st.error(f"生成失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

