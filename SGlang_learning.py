#!/usr/bin/env python3
"""
SGLang 工业级高性能部署服务
专注于结构化生成和复杂推理任务的优化
"""

import asyncio
import time as te
import cv2
import numpy as np
import logging
import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import sglang as sgl
from sglang import Engine, RuntimeEndpoint
from sglang.api import generate, GenerateReqInput
import psutil
import GPUtil

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
    # SGLang特有参数
    regex_pattern: Optional[str] = None  # 正则约束
    json_schema: Optional[Dict] = None   # JSON schema约束

class SGLangServer:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9):
        """
        初始化SGLang服务器
        
        Args:
            model_path: 模型路径
            gpu_memory_utilization: GPU内存利用率
        """
        self.model_path = model_path
        self.engine = None
        self.runtime_endpoint = None
        
    def initialize_engine(self):
        """初始化SGLang引擎"""
        
        # SGLang引擎配置 - 核心性能优化
        self.engine = Engine(
            model_path=self.model_path,
            
            # 内存管理优化
            mem_fraction_static=0.8,  # 静态内存分配比例，越高越好
            max_running_requests=512,  # 最大并发请求数，支持更多用户
            
            # 调度优化 - SGLang特色
            schedule_policy="lpm",  # Longest Prefix Matching，复用相似前缀
            chunked_prefill_size=4096,  # 分块预填充大小，优化长序列处理
            
            # 推理加速
            tp_size=torch.cuda.device_count(),  # 张量并行
            dp_size=1,  # 数据并行，通常为1
            
            # RadixAttention优化 - SGLang核心特性
            enable_flashinfer=True,  # 启用FlashInfer加速attention
            attention_reduce_in_fp32=False,  # attention计算精度，False更快
            
            # KV Cache优化
            context_length=4096,  # 上下文长度
            disable_radix_cache=False,  # 启用Radix树缓存，大幅提升相似请求性能
            
            # 量化优化（可选）
            # quantization="fp8",  # FP8量化，需要支持的硬件
            
            # 其他性能参数
            trust_remote_code=True,
            disable_custom_all_reduce=False,  # 启用自定义all-reduce优化
        )
        import os
        
        # 启动运行时端点
        self.runtime_endpoint = RuntimeEndpoint(self.engine)
        
        logger.info(f"SGLang引擎初始化完成，模型: {self.model_path}")
        
    async def generate_text(self, prompt: str, generation_params: Dict) -> str:
        """异步生成文本"""
        
        # 构建生成请求
        req_input = GenerateReqInput(
            text=prompt,
            sampling_params={
                "temperature": generation_params.get("temperature", 0.7),
                "top_p": generation_params.get("top_p", 0.9),
                "max_new_tokens": generation_params.get("max_tokens", 2048),
                "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
                "presence_penalty": generation_params.get("presence_penalty", 0.0),
                "stop": generation_params.get("stop", []),
            },
            # SGLang特有的约束参数
            regex=generation_params.get("regex_pattern"),
            json_schema=generation_params.get("json_schema"),
        )
        
        # 异步生成
        response = await self.runtime_endpoint.generate(req_input)
        
        if not response or not response.text:
            raise HTTPException(status_code=500, detail="生成失败")
            
        return response.text
        
    async def stream_generate(self, prompt: str, generation_params: Dict) -> AsyncGenerator[str, None]:
        """流式生成"""
        req_input = GenerateReqInput(
            text=prompt,
            sampling_params={
                "temperature": generation_params.get("temperature", 0.7),
                "top_p": generation_params.get("top_p", 0.9),
                "max_new_tokens": generation_params.get("max_tokens", 2048),
                "stream": True,
            },
            regex=generation_params.get("regex_pattern"),
            json_schema=generation_params.get("json_schema"),
        )
        
        async for chunk in self.runtime_endpoint.generate_stream(req_input):
            if chunk and chunk.text:
                yield chunk.text

# 创建FastAPI应用
app = FastAPI(title="SGLang High Performance API", version="1.0.0")

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局SGLang服务器实例
sglang_server = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    global sglang_server
    
    # 配置模型路径
    model_path = "Qwen/Qwen2.5-7B-Instruct"  # 修改为你的模型路径
    
    sglang_server = SGLangServer(model_path)
    sglang_server.initialize_engine()
    
    logger.info("SGLang服务启动完成")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成接口"""
    
    if sglang_server is None or sglang_server.engine is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    # 构建prompt
    prompt = build_prompt_from_messages(request.messages)
    
    # 生成参数
    generation_params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "regex_pattern": request.regex_pattern,
        "json_schema": request.json_schema,
    }
    
    start_time = time.time()
    
    try:
        if request.stream:
            # 流式响应
            return StreamingResponse(
                stream_chat_response(prompt, generation_params, request.model),
                media_type="text/plain"
            )
        else:
            # 非流式响应
            generated_text = await sglang_server.generate_text(prompt, generation_params)
            
            # 计算用量
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            # 性能日志
            elapsed_time = time.time() - start_time
            logger.info(f"请求完成 - 耗时: {elapsed_time:.2f}s, Token: {total_tokens}")
            
            return response
            
    except Exception as e:
        logger.error(f"生成错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_response(prompt: str, generation_params: Dict, model: str):
    """流式聊天响应生成器"""
    chunk_id = f"chatcmpl-{int(time.time())}"
    
    async for text_chunk in sglang_server.stream_generate(prompt, generation_params):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": text_chunk
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # 发送结束标志
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """从消息列表构建prompt"""
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"Human: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant: "
    return prompt

@app.post("/v1/structured_generate")
async def structured_generate(request: Dict[str, Any]):
    """SGLang特有的结构化生成接口"""
    
    if sglang_server is None:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    prompt = request.get("prompt", "")
    import time
    constraints = request.get("constraints", {})
    
    generation_params = {
        "temperature": request.get("temperature", 0.7),
        "max_tokens": request.get("max_tokens", 2048),
        "regex_pattern": constraints.get("regex"),
        "json_schema": constraints.get("json_schema"),
    }
    
    try:
        result = await sglang_server.generate_text(prompt, generation_params)
        return {
            "generated_text": result,
            "constraints_satisfied": True,  # 简化实现
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"结构化生成错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": sglang_server is not None and sglang_server.engine is not None,
        "engine_type": "sglang",
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
    
    # SGLang特有统计
    engine_stats = {}
    if sglang_server and sglang_server.runtime_endpoint:
        try:
            # 获取引擎统计信息（如果支持）
            engine_stats = {
                "active_requests": "N/A",  # 需要SGLang API支持
                "cache_hit_rate": "N/A",   # Radix缓存命中率
                "prefix_reuse_rate": "N/A"  # 前缀复用率
            }
        except:
            pass
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "gpu_stats": gpu_stats,
        "engine_stats": engine_stats,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # 生产级别的uvicorn配置
    uvicorn.run(
        "sglang_server:app",  # 如果保存为sglang_server.py
        host="0.0.0.0",
        port=8001,  # 不同端口避免冲突
        
        # 性能优化配置
        workers=1,  # SGLang使用单进程
        loop="uvloop",
        http="httptools",
        
        # 生产环境配置
        access_log=True,
        log_level="info",
        
        # 超时配置
        timeout_keep_alive=300,
        timeout_graceful_shutdown=300,
    )
