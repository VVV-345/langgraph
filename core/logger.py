"""
统一日志系统：终端实时显示 + 文件持久化
所有模块通过 `from core.logger import logger` 获取同一个 logger 实例
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

# 终端 handler —— INFO 级别，简洁可读
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
logger.addHandler(console)

# 文件 handler —— DEBUG 级别，保留完整上下文
file = logging.FileHandler(LOG_FILE, encoding="utf-8")
file.setLevel(logging.DEBUG)
file.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
))
logger.addHandler(file)

logger.info(f"日志文件: {LOG_FILE}")
