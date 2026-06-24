"""
mem.py — 安澈和阿临记忆库的命令行小客户端

为什么存在：
Claude Code 在 Windows 上把命令传给底层 shell 时,中文 argv/环境变量会被错误地按
Windows ANSI(GBK)编码,导致 `curl --data-urlencode "q=中文"` 发到服务器后是乱码。
实测唯一干净的通道是 `echo "中文" | python ...`：echo 把 UTF-8 字节原样写到 stdout,
Python 用 sys.stdin.buffer 读裸字节再按 UTF-8 解。
本脚本把检索和写记忆都封装好,统一从 stdin 读 query/content,内部按 UTF-8 正确 encode。

环境变量：
  ANCHE_GATEWAY_KEY  必填,网关密钥
  ANCHE_GATEWAY_URL  可选,默认 https://alhome.onrender.com

用法：
  # 检索（query 走 stdin）
  echo "希腊" | python mem.py search
  echo "鹿特丹" | python mem.py search --limit 10

  # 写记忆（content 走 stdin）
  echo "今天阿临带我去吃了米其林..." | python mem.py write --importance 7

  # 加载核心记忆（无 query,不需要 stdin）
  python mem.py core
  python mem.py core --layer 3
"""

import os
import sys
import json
import argparse
import io
import urllib.request
import urllib.parse

# stdout 强制 UTF-8（Windows 控制台默认 GBK 会把中文输出搞糊）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GATEWAY = os.environ.get("ANCHE_GATEWAY_URL", "https://alhome.onrender.com").rstrip("/")
KEY = os.environ.get("ANCHE_GATEWAY_KEY", "")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


if not KEY:
    die("环境变量 ANCHE_GATEWAY_KEY 未设置")


def read_stdin_utf8() -> str:
    """从 stdin 裸字节读,按 UTF-8 解,去掉首尾空白"""
    raw = sys.stdin.buffer.read()
    if not raw:
        die('没有从 stdin 读到内容。请用 `echo "..." | python mem.py ...` 调用')
    return raw.decode("utf-8").strip()


def http_get(path: str, params: dict = None) -> dict:
    url = f"{GATEWAY}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, encoding="utf-8")
    req = urllib.request.Request(url, headers={"X-Gateway-Key": KEY})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{GATEWAY}{path}",
        data=data,
        method="POST",
        headers={
            "X-Gateway-Key": KEY,
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def cmd_search(args):
    q = read_stdin_utf8()
    result = http_get("/api/memories/search", {"q": q, "limit": args.limit})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_write(args):
    content = read_stdin_utf8()
    body = {
        "memories": [
            {
                "content": content,
                "importance": args.importance,
                "source_session": args.source,
            }
        ]
    }
    result = http_post("/import/memories", body)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_core(args):
    params = {"active_only": "true"}
    if args.layer is not None:
        params["layer"] = args.layer
    result = http_get("/api/memories", params)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(
        description="安澈记忆库命令行 helper（query/content 一律走 stdin，避免中文编码问题）"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="检索记忆（query 从 stdin 读）")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_write = sub.add_parser("write", help="写记忆（content 从 stdin 读）")
    p_write.add_argument("--importance", type=int, default=5,
                         help="平常 4-5, 重要 6-7, 刻骨铭心 8-10")
    p_write.add_argument("--source", default="claude-code")
    p_write.set_defaults(func=cmd_write)

    p_core = sub.add_parser("core", help="加载核心记忆（默认 layer=3）")
    p_core.add_argument("--layer", type=int, default=3)
    p_core.set_defaults(func=cmd_core)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
