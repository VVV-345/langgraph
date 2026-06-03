"""网络搜索工具 —— DuckDuckGo Lite 网页搜索 + Instant Answer 兜底"""

import re
import json
import urllib.request
import urllib.parse

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


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
        # 策略 1：用 requests 爬 DuckDuckGo Lite（HTML 精简版，支持中文）
        if _HAS_REQUESTS:
            result = self._search_lite(query)
            if result and "未找到" not in result:
                return result

        # 策略 2：回退到 Instant Answer API
        result = self._search_instant_answer(query)
        if result and "未找到" not in result:
            return result

        return f"未找到关于 '{query}' 的搜索结果"

    # ==================================================================
    # 策略 1：DuckDuckGo Lite HTML 搜索
    # ==================================================================

    def _search_lite(self, query: str) -> str:
        """使用 requests 爬取 DuckDuckGo Lite 页面"""
        try:
            url = "https://lite.duckduckgo.com/lite/"
            resp = requests.get(
                url,
                params={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                },
                timeout=15,
            )
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return ""

        # 解析 Lite 页面：<a rel="nofollow" href="...">标题</a> 后面跟 <span>URL</span> 和摘要
        # Lite 页面的结构每行一个结果，用 <a rel="nofollow" 标记
        results = []

        # 匹配每个结果块：标题链接 + URL span + 描述文本
        # 模式：<a rel="nofollow" href="URL">TITLE</a> ... 后面是描述
        pattern = re.compile(
            r'<a\s+rel="nofollow"\s+(?:class="[^"]*"\s+)?href="([^"]+)"[^>]*>'
            r'(.+?)'
            r'</a>\s*<br\s*/?>\s*'
            r'<span\s+class="(?:link-text|url)"[^>]*>([^<]*)</span>'
            r'(?:\s*<br\s*/?>\s*(.+?))?\s*</td>',
            re.DOTALL | re.IGNORECASE,
        )

        for m in pattern.finditer(html):
            href = m.group(1)
            title = self._strip_html(m.group(2))
            snippet = ""
            if m.lastindex and m.lastindex >= 4 and m.group(4):
                snippet = self._strip_html(m.group(4))
            if title:
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet[:300],
                })

        # 备选：如果上面的正则没匹配到，尝试更宽松的匹配
        if not results:
            results = self._parse_lite_fallback(html)

        if not results:
            return ""

        # 格式化输出
        lines = [f"搜索 '{query}' 的结果："]
        for i, r in enumerate(results[:5], 1):
            lines.append(f"\n{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet'][:200]}")
        return "\n".join(lines)

    def _parse_lite_fallback(self, html: str) -> list:
        """宽松解析：提取所有 rel=nofollow 链接"""
        results = []
        # 找到所有 <a rel="nofollow" href="URL">TITLE</a>
        link_pattern = re.compile(
            r'<a\s+[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        for m in link_pattern.finditer(html):
            href = m.group(1)
            title = self._strip_html(m.group(2))
            if title:
                results.append({"title": title, "url": href, "snippet": ""})
        return results

    @staticmethod
    def _strip_html(text: str) -> str:
        """移除 HTML 标签，保留纯文本"""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#x27;", "'")
        return text.strip()

    # ==================================================================
    # 策略 2：Instant Answer API（兜底）
    # ==================================================================

    def _search_instant_answer(self, query: str) -> str:
        """使用 DuckDuckGo Instant Answer API（原有逻辑，作为兜底）"""
        try:
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
            if data.get("AbstractText"):
                parts.append(f"摘要：{data['AbstractText']}")
                if data.get("AbstractURL"):
                    parts.append(f"来源：{data['AbstractURL']}")

            topics = data.get("RelatedTopics", [])
            if topics:
                parts.append("\n相关结果：")
                for t in topics[:5]:
                    if isinstance(t, dict) and t.get("Text"):
                        parts.append(f"  - {t['Text'][:200]}")

            if not parts:
                return ""

            return "\n".join(parts)

        except urllib.error.URLError:
            return "网络搜索失败（网络连接问题）"
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
