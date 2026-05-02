"""
监控模块
实现系统的实时监控和预警机制
"""

import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class Monitor:
    """
    系统监控器
    用于监控系统的运行状态和发出预警
    """
    
    def __init__(self, log_dir: str = './logs'):
        """
        初始化监控器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 监控数据
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0,
            'last_request_time': None,
            'model_accuracy': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 预警阈值
        self.thresholds = {
            'response_time': 5.0,  # 响应时间阈值（秒）
            'error_rate': 0.1,  # 错误率阈值
            'request_rate': 100  # 请求率阈值（每分钟）
        }
        
        # 预警历史
        self.alerts = []
        
        # 锁，用于线程安全
        self.lock = threading.Lock()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✅ 监控器初始化完成")
    
    def _monitor_loop(self):
        """
        监控循环
        """
        while True:
            # 每60秒记录一次监控数据
            time.sleep(60)
            self._log_metrics()
            self._check_thresholds()
    
    def _log_metrics(self):
        """
        记录监控数据
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = os.path.join(self.log_dir, f"monitor_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        with self.lock:
            metrics_copy = self.metrics.copy()
        
        log_entry = {
            'timestamp': timestamp,
            'metrics': metrics_copy
        }
        
        # 追加到日志文件
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        # 只保留最近24小时的日志
        if len(logs) > 24 * 60:
            logs = logs[-24*60:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _check_thresholds(self):
        """
        检查阈值并发出预警
        """
        with self.lock:
            metrics_copy = self.metrics.copy()
        
        # 计算错误率
        total_requests = metrics_copy['total_requests']
        if total_requests > 0:
            error_rate = metrics_copy['failed_requests'] / total_requests
        else:
            error_rate = 0
        
        # 检查响应时间
        if metrics_copy['avg_response_time'] > self.thresholds['response_time']:
            self._add_alert('高响应时间', f'平均响应时间 {metrics_copy["avg_response_time"]:.2f} 秒超过阈值 {self.thresholds["response_time"]} 秒')
        
        # 检查错误率
        if error_rate > self.thresholds['error_rate']:
            self._add_alert('高错误率', f'错误率 {error_rate:.2f} 超过阈值 {self.thresholds["error_rate"]}')
    
    def _add_alert(self, alert_type: str, message: str):
        """
        添加预警
        
        Args:
            alert_type: 预警类型
            message: 预警消息
        """
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': alert_type,
            'message': message
        }
        
        with self.lock:
            self.alerts.append(alert)
        
        # 记录预警到日志
        alert_file = os.path.join(self.log_dir, f"alerts_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        if os.path.exists(alert_file):
            with open(alert_file, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []
        
        alerts.append(alert)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        print(f"⚠️  预警: {alert_type} - {message}")
    
    def update_metrics(self, success: bool, response_time: float, cache_hit: bool = False):
        """
        更新监控指标
        
        Args:
            success: 请求是否成功
            response_time: 响应时间（秒）
            cache_hit: 是否命中缓存
        """
        with self.lock:
            self.metrics['total_requests'] += 1
            if success:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            self.metrics['total_response_time'] += response_time
            self.metrics['avg_response_time'] = self.metrics['total_response_time'] / self.metrics['total_requests']
            self.metrics['last_request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
    
    def update_accuracy(self, accuracy: float):
        """
        更新模型准确率
        
        Args:
            accuracy: 模型准确率
        """
        with self.lock:
            self.metrics['model_accuracy'] = accuracy
    
    def get_metrics(self) -> Dict:
        """
        获取监控指标
        
        Returns:
            监控指标字典
        """
        with self.lock:
            return self.metrics.copy()
    
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """
        获取预警信息
        
        Args:
            limit: 返回的预警数量限制
            
        Returns:
            预警信息列表
        """
        with self.lock:
            return self.alerts[-limit:]
    
    def get_recent_logs(self, hours: int = 1) -> List[Dict]:
        """
        获取最近的日志
        
        Args:
            hours: 最近的小时数
            
        Returns:
            日志列表
        """
        log_file = os.path.join(self.log_dir, f"monitor_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        if not os.path.exists(log_file):
            return []
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        # 过滤最近hours小时的日志
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_logs = []
        
        for log in logs:
            log_time = datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp()
            if log_time >= cutoff_time:
                recent_logs.append(log)
        
        return recent_logs


# 全局监控器实例
monitor = None

def get_monitor() -> Monitor:
    """
    获取监控器实例
    
    Returns:
        监控器实例
    """
    global monitor
    if monitor is None:
        monitor = Monitor()
    return monitor