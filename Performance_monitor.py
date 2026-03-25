#!/usr/bin/env python3
"""
LLM服务性能测试和监控工具
支持vLLM、SGLang等推理引擎的全面性能测试
测试指标包括：吞吐量、延迟、资源利用率、首token时间等
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
import os
from datetime import datetime
import threading
import queue
import cv2
import numpy as np

# 尝试导入GPU监控库
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("警告: GPUtil未安装，无法监控GPU使用情况")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """
    性能指标数据类
    包含单次请求的所有性能数据
    """
    # 时间相关指标
    latency_ms: float              # 总延迟：从发送请求到接收完整响应的时间(毫秒)
    first_token_latency_ms: float  # 首token延迟：从发送请求到接收第一个token的时间(毫秒)
    inter_token_latency_ms: float  # token间延迟：生成token之间的平均间隔时间(毫秒)
    
    # 吞吐量相关指标
    tokens_per_second: float       # 生成速度：每秒生成的token数量
    throughput_per_minute: float   # 每分钟处理的token数量
    
    # Token统计
    prompt_tokens: int             # 输入prompt的token数量
    completion_tokens: int         # 生成的completion token数量
    total_tokens: int              # 总token数量(prompt + completion)
    
    # 请求状态
    success: bool                  # 请求是否成功
    error_message: Optional[str] = None  # 错误信息
    
    # 时间戳
    timestamp: float = 0           # 请求完成时的时间戳

@dataclass
class SystemMetrics:
    """
    系统资源使用指标
    监控CPU、内存、GPU等系统资源
    """
    # CPU相关
    cpu_percent: float             # CPU使用率百分比
    cpu_count: int                 # CPU核心数
    cpu_freq_mhz: float           # CPU频率(MHz)
    
    # 内存相关
    memory_percent: float          # 内存使用率百分比
    memory_used_gb: float         # 已使用内存(GB)
    memory_total_gb: float        # 总内存(GB)
    memory_available_gb: float    # 可用内存(GB)
    
    # GPU相关(如果可用)
    gpu_count: int = 0            # GPU数量
    gpu_utilization: List[float] = None  # 各GPU使用率
    gpu_memory_used: List[float] = None  # 各GPU显存使用量(GB)
    gpu_memory_total: List[float] = None # 各GPU总显存(GB)
    gpu_temperature: List[float] = None  # 各GPU温度(℃)
    
    # 网络相关
    network_sent_mb: float = 0     # 网络发送量(MB)
    network_recv_mb: float = 0     # 网络接收量(MB)
    
    timestamp: float = 0           # 监控时间戳

class SystemMonitor:
    """
    系统资源监控器
    实时监控系统CPU、内存、GPU等资源使用情况
    """
    
    def __init__(self):
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控系统资源
        Args:
            interval: 监控间隔时间(秒)
        """
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"开始系统监控，间隔{interval}秒")
        
    def stop_monitoring(self):
        """停止监控系统资源"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("停止系统监控")
        logger.info("在干什么？")
        
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控系统资源时出错: {e}")
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0
        
        # 内存指标
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # 网络指标
        network = psutil.net_io_counters()
        network_sent_mb = network.bytes_sent / (1024**2)
        network_recv_mb = network.bytes_recv / (1024**2)
        
        # GPU指标
        gpu_count = 0
        gpu_utilization = []
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_temperature = []
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                for gpu in gpus:
                    gpu_utilization.append(gpu.load * 100)  # 转换为百分比
                    gpu_memory_used.append(gpu.memoryUsed / 1024)  # 转换为GB
                    gpu_memory_total.append(gpu.memoryTotal / 1024)  # 转换为GB
                    gpu_temperature.append(gpu.temperature)
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            gpu_count=gpu_count,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperature,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            timestamp=time.time()
        )
    
    def get_metrics_history(self) -> List[SystemMetrics]:
        """获取监控历史数据"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics

class LLMPerformanceTester:
    """
    LLM性能测试器
    支持单次测试、并发测试、吞吐量测试等多种测试模式
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.system_monitor = SystemMonitor()
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5分钟超时
            connector=aiohttp.TCPConnector(limit=1000)  # 连接池大小
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
            
    async def single_request_test(self, prompt: str, model: str = "test-model", 
                                max_tokens: int = 512) -> PerformanceMetrics:
        """
        单次请求性能测试
        测试单个请求的延迟、吞吐量等指标
        
        Args:
            prompt: 输入提示词
            model: 模型名称
            max_tokens: 最大生成token数
            
        Returns:
            PerformanceMetrics: 性能指标数据
        """
        request_data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": True  # 使用流式响应以测量首token延迟
        }
        
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        token_times = []
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return PerformanceMetrics(
                        latency_ms=0,
                        first_token_latency_ms=0,
                        inter_token_latency_ms=0,
                        tokens_per_second=0,
                        throughput_per_minute=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        success=False,
                        error_message=f"HTTP {response.status}: {error_text}",
                        timestamp=time.time()
                    )
                
                # 处理流式响应
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 去除'data: '前缀
                            if data_str == '[DONE]':
                                break
                                
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        current_time = time.time()
                                        if first_token_time is None:
                                            first_token_time = current_time
                                        else:
                                            token_times.append(current_time)
                                        completion_tokens += 1
                            except json.JSONDecodeError:
                                continue
                                
            end_time = time.time()
            
            # 计算各项指标
            total_latency_ms = (end_time - start_time) * 1000
            first_token_latency_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
            
            # 计算token间延迟
            inter_token_latency_ms = 0
            if len(token_times) > 1:
                inter_token_intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
                inter_token_latency_ms = statistics.mean(inter_token_intervals) * 1000
            
            # 计算吞吐量
            generation_time = end_time - (first_token_time or start_time)
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            throughput_per_minute = tokens_per_second * 60
            
            # 估算prompt tokens (简化计算，实际应该使用tokenizer)
            prompt_tokens = len(prompt.split()) * 1.3  # 粗略估算
            
            return PerformanceMetrics(
                latency_ms=total_latency_ms,
                first_token_latency_ms=first_token_latency_ms,
                inter_token_latency_ms=inter_token_latency_ms,
                tokens_per_second=tokens_per_second,
                throughput_per_minute=throughput_per_minute,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(prompt_tokens) + completion_tokens,
                success=True,
                timestamp=time.time()
            )
            
        except Exception as e:
            return PerformanceMetrics(
                latency_ms=0,
                first_token_latency_ms=0,
                inter_token_latency_ms=0,
                tokens_per_second=0,
                throughput_per_minute=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(e),
                timestamp=time.time()
            )
    
    async def concurrent_test(self, prompts: List[str], concurrency: int = 10, 
                            model: str = "test-model", max_tokens: int = 512) -> List[PerformanceMetrics]:
        """
        并发请求测试
        测试系统在并发负载下的性能表现
        
        Args:
            prompts: 测试提示词列表
            concurrency: 并发数
            model: 模型名称
            max_tokens: 最大生成token数
            
        Returns:
            List[PerformanceMetrics]: 所有请求的性能指标
        """
        logger.info(f"开始并发测试，并发数: {concurrency}, 请求数: {len(prompts)}")
        
        # 启动系统监控
        self.system_monitor.start_monitoring(interval=0.5)
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(prompt: str) -> PerformanceMetrics:
            async with semaphore:
                return await self.single_request_test(prompt, model, max_tokens)
        
        # 执行并发请求
        start_time = time.time()
        tasks = [limited_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 停止系统监控
        self.system_monitor.stop_monitoring()
        
        # 处理结果
        metrics = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"请求失败: {result}")
                metrics.append(PerformanceMetrics(
                    latency_ms=0,
                    first_token_latency_ms=0,
                    inter_token_latency_ms=0,
                    tokens_per_second=0,
                    throughput_per_minute=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    success=False,
                    error_message=str(result),
                    timestamp=time.time()
                ))
            else:
                metrics.append(result)
        
        total_time = end_time - start_time
        successful_requests = sum(1 for m in metrics if m.success)
        logger.info(f"并发测试完成，总耗时: {total_time:.2f}秒, 成功请求: {successful_requests}/{len(prompts)}")
        
        return metrics
    
    async def throughput_test(self, prompt: str, duration_seconds: int = 60, 
                            concurrency: int = 10, model: str = "test-model") -> Dict[str, Any]:
        """
        吞吐量测试
        在指定时间内持续发送请求，测试系统的最大吞吐能力
        
        Args:
            prompt: 测试提示词
            duration_seconds: 测试持续时间(秒)
            concurrency: 并发数
            model: 模型名称
            
        Returns:
            Dict: 吞吐量测试结果
        """
        logger.info(f"开始吞吐量测试，持续时间: {duration_seconds}秒, 并发数: {concurrency}")
        
        # 启动系统监控
        self.system_monitor.start_monitoring(interval=1.0)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        metrics = []
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def continuous_request():
            while time.time() < end_time:
                async with semaphore:
                    try:
                        metric = await self.single_request_test(prompt, model)
                        metrics.append(metric)
                    except Exception as e:
                        logger.error(f"吞吐量测试请求失败: {e}")
                        
        # 启动并发任务
        tasks = [continuous_request() for _ in range(concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 停止系统监控
        self.system_monitor.stop_monitoring()
        
        # 计算吞吐量指标
        actual_duration = time.time() - start_time
        successful_requests = [m for m in metrics if m.success]
        failed_requests = [m for m in metrics if not m.success]
        
        if successful_requests:
            avg_latency = statistics.mean([m.latency_ms for m in successful_requests])
            avg_first_token_latency = statistics.mean([m.first_token_latency_ms for m in successful_requests])
            avg_tokens_per_second = statistics.mean([m.tokens_per_second for m in successful_requests])
            total_tokens = sum([m.total_tokens for m in successful_requests])
            
            # 系统级吞吐量指标
            requests_per_second = len(successful_requests) / actual_duration
            tokens_per_second_system = total_tokens / actual_duration
            
        else:
            avg_latency = 0
            avg_first_token_latency = 0
            avg_tokens_per_second = 0
            total_tokens = 0
            requests_per_second = 0
            tokens_per_second_system = 0
        
        result = {
            'test_duration_seconds': actual_duration,
            'total_requests': len(metrics),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(metrics) if metrics else 0,
            
            # 吞吐量指标
            'requests_per_second': requests_per_second,
            'tokens_per_second_system': tokens_per_second_system,
            'tokens_per_minute_system': tokens_per_second_system * 60,
            
            # 延迟指标
            'avg_latency_ms': avg_latency,
            'avg_first_token_latency_ms': avg_first_token_latency,
            'avg_tokens_per_second_per_request': avg_tokens_per_second,
            
            # Token统计
            'total_tokens_processed': total_tokens,
            'avg_tokens_per_request': total_tokens / len(successful_requests) if successful_requests else 0,
            
            # 原始数据
            'detailed_metrics': metrics
        }
        
        logger.info(f"吞吐量测试完成:")
        logger.info(f"  请求/秒: {requests_per_second:.2f}")
        logger.info(f"  系统tokens/秒: {tokens_per_second_system:.2f}")
        logger.info(f"  成功率: {result['success_rate']*100:.1f}%")
        
        return result
    
    def analyze_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        性能分析
        对测试结果进行统计分析，计算各种性能指标
        
        Args:
            metrics: 性能指标列表
            
        Returns:
            Dict: 分析结果
        """
        successful_metrics = [m for m in metrics if m.success]
        failed_metrics = [m for m in metrics if not m.success]
        
        if not successful_metrics:
            return {
                'total_requests': len(metrics),
                'successful_requests': 0,
                'failed_requests': len(failed_metrics),
                'success_rate': 0,
                'error': 'No successful requests to analyze'
            }
        
        # 延迟分析
        latencies = [m.latency_ms for m in successful_metrics]
        first_token_latencies = [m.first_token_latency_ms for m in successful_metrics]
        inter_token_latencies = [m.inter_token_latency_ms for m in successful_metrics if m.inter_token_latency_ms > 0]
        
        # 吞吐量分析
        tokens_per_second = [m.tokens_per_second for m in successful_metrics]
        
        # Token统计
        prompt_tokens = [m.prompt_tokens for m in successful_metrics]
        completion_tokens = [m.completion_tokens for m in successful_metrics]
        total_tokens = [m.total_tokens for m in successful_metrics]
        
        def calculate_percentiles(data: List[float]) -> Dict[str, float]:
            """计算百分位数"""
            if not data:
                return {}
            return {
                'p50': np.percentile(data, 50),
                'p90': np.percentile(data, 90),
                'p95': np.percentile(data, 95),
                'p99': np.percentile(data, 99)
            }
        
        analysis = {
            # 基本统计
            'total_requests': len(metrics),
            'successful_requests': len(successful_metrics),
            'failed_requests': len(failed_metrics),
            'success_rate': len(successful_metrics) / len(metrics),
            
            # 延迟分析
            'latency_ms': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                **calculate_percentiles(latencies)
            },
            
            'first_token_latency_ms': {
                'mean': statistics.mean(first_token_latencies),
                'median': statistics.median(first_token_latencies),
                'min': min(first_token_latencies),
                'max': max(first_token_latencies),
                'stdev': statistics.stdev(first_token_latencies) if len(first_token_latencies) > 1 else 0,
                **calculate_percentiles(first_token_latencies)
            },
            
            # 吞吐量分析
            'tokens_per_second': {
                'mean': statistics.mean(tokens_per_second),
                'median': statistics.median(tokens_per_second),
                'min': min(tokens_per_second),
                'max': max(tokens_per_second),
                'stdev': statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0,
                **calculate_percentiles(tokens_per_second)
            },
            
            # Token统计
            'token_stats': {
                'avg_prompt_tokens': statistics.mean(prompt_tokens),
                'avg_completion_tokens': statistics.mean(completion_tokens),
                'avg_total_tokens': statistics.mean(total_tokens),
                'total_tokens_processed': sum(total_tokens)
            }
        }
        
        # 添加token间延迟分析(如果有数据)
        if inter_token_latencies:
            analysis['inter_token_latency_ms'] = {
                'mean': statistics.mean(inter_token_latencies),
                'median': statistics.median(inter_token_latencies),
                'min': min(inter_token_latencies),
                'max': max(inter_token_latencies),
                'stdev': statistics.stdev(inter_token_latencies) if len(inter_token_latencies) > 1 else 0,
                **calculate_percentiles(inter_token_latencies)
            }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], system_metrics: List[SystemMetrics] = None, 
                       output_file: str = None) -> str:
        """
        生成性能测试报告
        
        Args:
            analysis: 性能分析结果
            system_metrics: 系统资源监控数据
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LLM性能测试报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基本统计
        report_lines.append("## 基本统计")
        report_lines.append(f"总请求数: {analysis['total_requests']}")
        report_lines.append(f"成功请求数: {analysis['successful_requests']}")
        report_lines.append(f"失败请求数: {analysis['failed_requests']}")
        report_lines.append(f"成功率: {analysis['success_rate']*100:.2f}%")
        report_lines.append("")
        
        # 延迟分析
        if 'latency_ms' in analysis:
            lat = analysis['latency_ms']
            report_lines.append("## 延迟分析")
            report_lines.append(f"平均延迟: {lat['mean']:.2f} ms")
            report_lines.append(f"中位延迟: {lat['median']:.2f} ms")
            report_lines.append(f"最小延迟: {lat['min']:.2f} ms")
            report_lines.append(f"最大延迟: {lat['max']:.2f} ms")
            report_lines.append(f"标准差: {lat['stdev']:.2f} ms")
            if 'p95' in lat:
                report_lines.append(f"P95延迟: {lat['p95']:.2f} ms")
                report_lines.append(f"P99延迟: {lat['p99']:.2f} ms")
            report_lines.append("")
        
        # 首token延迟
        if 'first_token_latency_ms' in analysis:
            ftl = analysis['first_token_latency_ms']
            report_lines.append("## 首Token延迟")
            report_lines.append(f"平均首token延迟: {ftl['mean']:.2f} ms")
            report_lines.append(f"中位首token延迟: {ftl['median']:.2f} ms")
            if 'p95' in ftl:
                report_lines.append(f"P95首token延迟: {ftl['p95']:.2f} ms")
            report_lines.append("")
        
        # 吞吐量分析
        if 'tokens_per_second' in analysis:
            tps = analysis['tokens_per_second']
            report_lines.append("## 吞吐量分析")
            report_lines.append(f"平均tokens/秒: {tps['mean']:.2f}")
            report_lines.append(f"中位tokens/秒: {tps['median']:.2f}")
            report_lines.append(f"最大tokens/秒: {tps['max']:.2f}")
            if 'p95' in tps:
                report_lines.append(f"P95 tokens/秒: {tps['p95']:.2f}")
            report_lines.append("")
        
        # Token统计
        if 'token_stats' in analysis:
            ts = analysis['token_stats']
            report_lines.append("## Token统计")
            report_lines.append(f"平均prompt tokens: {ts['avg_prompt_tokens']:.1f}")
            report_lines.append(f"平均completion tokens: {ts['avg_completion_tokens']:.1f}")
            report_lines.append(f"平均总tokens: {ts['avg_total_tokens']:.1f}")
            report_lines.append(f"总处理tokens: {ts['total_tokens_processed']}")
            report_lines.append("")
        
        # 系统资源分析
        if system_metrics:
            report_lines.append("## 系统资源使用")
            cpu_usage = [m.cpu_percent for m in system_metrics]
            memory_usage = [m.memory_percent for m in system_metrics]
        
        a = b + c

        from test import func1,func2,func4

        e = func1(a)
        d = func2(a,b,c)
        f = func1(abc)

        y = func3(e)
