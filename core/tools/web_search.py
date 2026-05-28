"""网络搜索工具 —— 使用 DuckDuckGo 瞬时搜索"""

import urllib.request
import urllib.parse
import json


class WebSearchTool:
    name = "web_search"
    description = "搜索网络获取信息。用于查找 API 文档、报错解决方案、技术资料等。"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        },
        "required": ["query"]
    }

    def execute(self, query: str) -> str:
        try:
            # 使用 DuckDuckGo Instant Answer API（免费，无需 API Key）
            url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            })
            req = urllib.request.Request(url, headers={"User-Agent": "AgentUpgrade/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            parts = []
            # 摘要
            if data.get("AbstractText"):
                parts.append(f"摘要：{data['AbstractText']}")
                if data.get("AbstractURL"):
                    parts.append(f"来源：{data['AbstractURL']}")

            # 相关主题
            topics = data.get("RelatedTopics", [])
            if topics:
                parts.append("\n相关结果：")
                for t in topics[:5]:
                    if isinstance(t, dict) and t.get("Text"):
                        parts.append(f"  - {t['Text'][:200]}")

            if not parts:
                return f"未找到关于 '{query}' 的搜索结果"

            return "\n".join(parts)

        except urllib.error.URLError as e:
            return f"网络搜索失败（网络连接问题）：{str(e)}"
        except Exception as e:
            return f"搜索异常：{str(e)}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
