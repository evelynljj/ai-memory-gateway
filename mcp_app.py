"""
MCP 端点 —— 把长期记忆检索暴露为一个只读 MCP 工具 search_memory
================================================================
- 只读：仅暴露 search_memory（按 query 检索），不提供任何写/改/删工具。
- 复用：同进程内直接调用 database.search_memories(...)，共享全局连接池，
        不自己再发 HTTP 去打 /api/memories/search。
- 传输：SSE（claude.ai 自定义连接器兼容好）。
- 鉴权：密钥塞进挂载路径（见 main.py 的 MCP_SECRET_PATH 与白名单），
        本文件不读取、不保存任何密钥。
"""

import os
import re
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from database import search_memories

# 时区偏移（与主应用一致），用于把 UTC 的 created_at 本地化显示
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))

# ------------------------------------------------------------
# 传输层 Host/Origin 校验（DNS rebinding 保护）
# ------------------------------------------------------------
# mcp 1.28 的 FastMCP 默认 enable_dns_rebinding_protection=True、allowed_hosts 只含 localhost，
# 这是为「本地 localhost MCP 服务」设计的防护。但本服务是公网服务、真正的鉴权是路径里的密钥；
# claude.ai 用 Render 公网域名连入时 Host=xxx.onrender.com 会被这道校验拒绝握手。
# 设计：
#   - 默认（不设 MCP_ALLOWED_HOST）→ 关掉 Host/Origin 校验，避免因 Host/Origin 配错而静默失败。
#   - 可选加固：设了 MCP_ALLOWED_HOST（值=Render 域名，不带 https:// 和斜杠）才开校验并放行该域名。
_host = os.getenv("MCP_ALLOWED_HOST", "").strip().strip("/")
_sec = TransportSecuritySettings(
    enable_dns_rebinding_protection=bool(_host),
    allowed_hosts=[_host, f"{_host}:*"] if _host else [],
    allowed_origins=[f"https://{_host}"] if _host else [],
)

mcp_server = FastMCP("ai-memory-gateway", transport_security=_sec)


@mcp_server.tool(name="search_memory", description="检索阿临与安澈的长期记忆库。当需要回忆两人之间发生过的事、过往对话、纪念日、共同经历等具体内容时调用。传入要回忆的主题或关键词，返回最相关的记忆条目（每条带日期）。")
async def search_memory(query: str, limit: int = 10) -> str:
    """检索长期记忆（关键词 + 向量混合搜索）。

    用自然语言描述要回忆的内容/主题，返回最相关的记忆条目（每条带日期）。

    Args:
        query: 要检索的内容或主题。
        limit: 返回的记忆条数上限（1-50，默认 10）。
    """
    q = (query or "").strip()
    if not q:
        return "（未提供检索关键词）"

    try:
        n = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        n = 10

    # 同进程直接调用现有检索逻辑（复用全局连接池）
    results = await search_memories(q, n)
    if not results:
        return f"没有检索到与「{q}」相关的记忆。"

    tz = timezone(timedelta(hours=TIMEZONE_HOURS))
    lines = []
    for r in results:
        content = r.get("content", "")
        created = r.get("created_at")
        date_str = ""
        if created:
            try:
                dt = created
                if getattr(dt, "tzinfo", None) is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_str = dt.astimezone(tz).strftime("%Y-%m-%d")
            except Exception:
                date_str = str(created)[:10]
        prefix = f"[{date_str}] " if date_str else ""
        lines.append(f"- {prefix}{content}")

    return f"检索到 {len(results)} 条相关记忆：\n" + "\n".join(lines)


class _EndpointPrefixFix:
    """兜底中间件：把 SSE 握手 endpoint 事件里的 message 路径规整成恰好一个 /mcp/<secret>/ 前缀。

    幂等——缺前缀补齐、双前缀收敛、已正确不变。
    局限：按【单条】http.response.body 消息匹配 event: endpoint，若该事件被拆到多个 chunk 会漏修，
    故仅作一键救急、不当默认修复手段（默认关闭，见 mount_mcp 的 MCP_FIX_ENDPOINT_PREFIX）。
    """
    _MARKER = "/messages/"

    def __init__(self, app, base_path: str):
        self.app = app
        self.base = "/" + base_path.strip("/")  # /mcp/<secret>

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def _send(message):
            if message.get("type") == "http.response.body":
                body = message.get("body", b"")
                if body and b"event: endpoint" in body and b"/messages/" in body:
                    message = {**message, "body": self._fix(body)}
            await send(message)

        await self.app(scope, receive, _send)

    def _fix(self, body: bytes) -> bytes:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return body

        def repl(m):
            data = m.group(1)
            i = data.find(self._MARKER)
            return "data: " + self.base + data[i:] if i != -1 else m.group(0)

        return re.sub(r"data:\s*([^\r\n]*" + re.escape(self._MARKER) + r"[^\r\n]*)",
                      repl, text).encode("utf-8")


def mount_mcp(app, secret_path: str):
    """把 MCP 的 SSE 端点挂到现有 FastAPI app 上。

    返回需要加入鉴权白名单的「带密钥路径前缀」（如 /mcp/<secret>/），
    未配置密钥时返回 None。
    """
    secret = (secret_path or "").strip().strip("/")
    if not secret:
        return None

    base = f"/mcp/{secret}"
    # 关键：app.mount(base, ...) 已让 Starlette 把 root_path 设成 base，SSE 传输层会自动把
    # root_path 拼到 endpoint 前；因此这里【不能】再传 mount_path=base，否则前缀被拼两遍
    # （/mcp/<s>/mcp/<s>/messages/）→ 客户端 POST 回 404。已在 starlette 0.38.6/1.3.1 实测确认。
    sub = mcp_server.sse_app()
    # 兜底（默认关闭）：仅在握手 endpoint 前缀异常时一键救急，设 MCP_FIX_ENDPOINT_PREFIX=true 启用。
    if os.getenv("MCP_FIX_ENDPOINT_PREFIX", "false").lower() == "true":
        sub = _EndpointPrefixFix(sub, base)
    app.mount(base, sub)
    print(f"✅ MCP 端点已挂载: {base}/sse （工具: search_memory）")
    return f"{base}/"
