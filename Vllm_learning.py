#!/usr/bin/env python3
"""
vLLM 工业级高性能部署服务
支持上千用户并发，优化延迟和吞吐量
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
import psutil
import GPUtil
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class VLLMServer:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9):
        """
        初始化vLLM服务器
        
        Args:
            model_path: 模型路径
            gpu_memory_utilization: GPU内存利用率 (0.8-0.95 推荐，留出空间给KV cache)
        """
        self.model_path = model_path
        self.engine = None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.test = 0.8
        
    async def initialize_engine(self):
        """初始化异步推理引擎"""
        # 核心性能优化配置
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            # GPU内存管理 - 关键优化参数
            gpu_memory_utilization=self.gpu_memory_utilization,  # GPU内存利用率，越高越好但要防OOM
            swap_space=4,  # CPU-GPU内存交换空间(GB)，处理长序列时有用
            
            # 并发和批处理优化 - 核心性能参数
            max_num_batched_tokens=8192,  # 单batch最大token数，影响吞吐量
            max_num_seqs=256,  # 最大并发序列数，支持更多用户
            max_model_len=4096,  # 模型最大序列长度
            
            # KV Cache优化 - 内存效率关键
            block_size=16,  # KV cache块大小，16是最优值
            enable_prefix_caching=True,  # 启用前缀缓存，复用system prompt
            
            # 调度器优化 - 延迟优化
            use_v2_block_manager=True,  # 使用v2块管理器，更高效
            preemption_mode="recompute",  # 抢占模式：recompute(低延迟) vs swap(省内存)
            
            # 推理优化
            enforce_eager=False,  # 禁用可减少延迟但降低吞吐量
            tensor_parallel_size=torch.cuda.device_count(),  # 张量并行，多GPU必须
            pipeline_parallel_size=1,  # 流水线并行，通常设为1
            
            # 量化加速 - 可选，牺牲少量精度换取速度
            # quantization="awq",  # 支持AWQ/GPTQ量化
            # load_format="auto",
            
            # 其他优化
            disable_log_stats=False,  # 启用统计日志便于监控
            trust_remote_code=True,  # 信任远程代码，某些模型需要
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM引擎初始化完成，模型: {self.model_path}")
        
    async def generate(self, prompt: str, sampling_params: SamplingParams) -> str:
        """异步生成文本"""
        request_id = random_uuid()
        
        # 提交生成请求
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id
        )
        
        # 收集生成结果
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        if final_output is None:
            raise HTTPException(status_code=500, detail="生成失败")
            
        return final_output.outputs[0].text

# 创建FastAPI应用
app = FastAPI(title="vLLM High Performance API", version="1.0.0")

# CORS中间件 - 生产环境需要限制origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局vLLM服务器实例
vllm_server = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    global vllm_server
    
    # 配置模型路径
    model_path = "Qwen/Qwen2.5-7B-Instruct"  # 修改为你的模型路径
    
    vllm_server = VLLMServer(model_path)
    await vllm_server.initialize_engine()
    
    logger.info("vLLM服务启动完成")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成接口"""
    
    if vllm_server is None or vllm_server.engine is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    # 构建prompt
    if request.messages:
        # 简单的消息格式转换，实际需要根据模型格式调整
        prompt = ""
        for msg in request.messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
    else:
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    # 采样参数优化
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        # 性能优化采样参数
        use_beam_search=False,  # 关闭束搜索提升速度
        early_stopping=True,    # 启用早停
        skip_special_tokens=True,  # 跳过特殊token
    )
    
    start_time = time.time()
    
    try:
        # 异步生成
        generated_text = await vllm_server.generate(prompt, sampling_params)
        
        # 计算用量
        prompt_tokens = len(prompt.split())  # 简化计算
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{random_uuid()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
        # 性能日志
        elapsed_time = time.time() - start_time
        logger.info(f"请求完成 - 耗时: {elapsed_time:.2f}s, Token: {total_tokens}")
        
        return response
        
    except Exception as e:
        logger.error(f"生成错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": vllm_server is not None and vllm_server.engine is not None,
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用率
    memory = psutil.virtual_memory()
    
    # GPU使用率
    gpu_stats = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_stats.append({
                "id": gpu.id,
                "name": gpu.name,
                "utilization": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
            })
    except:
        gpu_stats = []
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "gpu_stats": gpu_stats,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # 生产级别的uvicorn配置
    uvicorn.run(
        "vllm_server:app",  # 如果保存为vllm_server.py
        host="0.0.0.0",
        port=8000,
        # 性能优化配置
        workers=1,  # vLLM使用单进程多线程，不要设置多worker
        loop="uvloop",  # 使用uvloop提升性能
        http="httptools",  # 使用httptools提升HTTP解析性能
        
        # 生产环境配置
        access_log=True,
        log_level="info",
        
        # 超时配置 - 处理长文本生成
        timeout_keep_alive=300,  # Keep-alive超时
        timeout_graceful_shutdown=300,  # 优雅关闭超时
        
        # SSL配置（生产环境启用）
        # ssl_keyfile="path/to/keyfile",
        # ssl_certfile="path/to/certfile",
    )
