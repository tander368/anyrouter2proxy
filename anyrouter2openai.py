"""
AnyRouter2OpenAI - OpenAI 协议代理服务（Node.js SDK 中转模式）

将 OpenAI 协议转换为 Anthropic 协议，通过 Node.js 代理层转发请求。
支持多 API Key 负载均衡（客户端提供）。

架构:
  客户端 → Python 代理 (9999) → Node.js 代理 (4000) → anyrouter.top
                                        ↑
                                                                       官方 Node.js SDK
                                                                                                      (正确的 TLS 指纹)
                                                                                                      
                                                                                                      使用方式：
                                                                                                      1. 启动 Node.js 代理: cd node-proxy && npm install && npm start
                                                                                                      2. 启动 Python 代理: python anyrouter2openai.py
                                                                                                      3. 配置客户端 base_url 为: http://localhost:9999
                                                                                                      4. 支持多 key 负载均衡: 用逗号分隔多个 key，如 "sk-key1,sk-key2"
"""

import json
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# 加载 .env 文件
load_dotenv()

# 配置日志
logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 从环境变量读取配置
NODE_PROXY_URL = os.getenv("NODE_PROXY_URL", "http://127.0.0.1:4000")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8192"))
FORCE_NON_STREAM = os.getenv("FORCE_NON_STREAM", "false").lower() in ("true", "1", "yes")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are Claude, a helpful AI assistant.")
DEFAULT_SUPPORTED_MODELS = [
      "claude-haiku-4-5-20251001",
      "claude-3-5-haiku-20241022",
      "claude-3-5-sonnet-20241022",
      "claude-3-7-sonnet-20250219",
      "claude-sonnet-4-20250514",
      "claude-sonnet-4-5-20250929",
      "claude-sonnet-4-6",
      "claude-opus-4-5-20251101",
      "claude-opus-4-6",
]
DEFAULT_MODEL_ALIASES = {
      "claude-opus-4-6[1m]": "claude-opus-4-6",
}
SENSITIVE_HEADER_NAMES = {"authorization", "x-api-key", "cookie", "set-cookie"}


def parse_csv_env(raw: str | None, default: list[str]) -> list[str]:
      if not raw:
                return list(default)
            return [item.strip() for item in raw.split(",") if item.strip()]


def parse_model_aliases(raw: str | None) -> dict[str, str]:
      aliases: dict[str, str] = {}
    if not raw:
              return aliases
          for pair in raw.split(","):
                    alias, sep, target = pair.partition("=")
                    if sep and alias.strip() and target.strip():
                                  aliases[alias.strip()] = target.strip()
                          return aliases


SUPPORTED_MODELS = parse_csv_env(os.getenv("SUPPORTED_MODELS"), DEFAULT_SUPPORTED_MODELS)
MODEL_ALIASES = DEFAULT_MODEL_ALIASES | parse_model_aliases(os.getenv("MODEL_ALIASES"))


def normalize_model_name(model: str | None) -> str | None:
      if model is None:
                return None
    return MODEL_ALIASES.get(model, model)


def build_model_list_payload() -> dict[str, Any]:
      seen: set[str] = set()
    data: list[dict[str, Any]] = []

    for model in [*SUPPORTED_MODELS, *MODEL_ALIASES.keys()]:
              if model in seen:
                            continue
                        seen.add(model)
        data.append({
                      "id": model,
                      "object": "model",
                      "created": 0,
                      "owned_by": "anthropic",
        })

    return {"object": "list", "data": data}


def redact_headers_for_log(headers: dict[str, str]) -> dict[str, str]:
      return {
                key: ("<redacted>" if key.lower() in SENSITIVE_HEADER_NAMES else val)
                for key, val in headers.items()
      }


def parse_json_maybe(value: str | bytes | dict[str, Any] | None) -> str | dict[str, Any]:
      if value is None:
                return ""
    if isinstance(value, bytes):
              value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
              value = value.strip()
        if not value:
                      return ""
                  try:
                                return json.loads(value)
except json.JSONDecodeError:
            return value
    return value


def build_openai_error_payload(raw_error: str | bytes | dict[str, Any] | None, status_code: int) -> dict[str, Any]:
      payload = parse_json_maybe(raw_error)

    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
              nested = parse_json_maybe(payload["detail"])
        if isinstance(nested, dict):
                      payload = nested

    error: dict[str, Any] = {}
    if isinstance(payload, dict):
              if isinstance(payload.get("error"), dict):
                            error = payload["error"]
elif isinstance(payload.get("detail"), dict):
            detail = payload["detail"]
            if isinstance(detail.get("error"), dict):
                              error = detail["error"]
else:
                error = detail

    message = "Upstream request failed"
    if isinstance(payload, str) and payload:
              message = payload
elif isinstance(error.get("message"), str) and error["message"]:
        message = error["message"]
elif isinstance(payload, dict):
        if isinstance(payload.get("message"), str) and payload["message"]:
                      message = payload["message"]
elif isinstance(payload.get("detail"), str) and payload["detail"]:
            message = payload["detail"]

    error_type = "api_error"
    if isinstance(error.get("type"), str) and error["type"]:
              error_type = error["type"]
elif isinstance(payload, dict) and isinstance(payload.get("type"), str) and payload["type"]:
        error_type = payload["type"]

    error_code = error.get("code", status_code)

    return {
              "error": {
                            "message": message,
                            "type": error_type,
                            "param": None,
                            "code": error_code,
              }
    }


def openai_error_response(status_code: int, raw_error: str | bytes | dict[str, Any] | None) -> JSONResponse:
      return JSONResponse(status_code=status_code, content=build_openai_error_payload(raw_error, status_code))


@dataclass
class Account:
      """API 账号"""
    api_key: str
    name: str = ""

    def __post_init__(self):
              if not self.name:
                            self.name = f"key_{self.api_key[:8]}..."


class RequestLoadBalancer:
      """请求级负载均衡器（基于客户端提供的 keys）"""

    def __init__(self, api_keys: list[str]):
              self.accounts = [Account(api_key=k, name=f"key_{i+1}") for i, k in enumerate(api_keys)]
        self._rr_index = 0

    def select_account(self) -> Account | None:
              if not self.accounts:
                            return None
                        if len(self.accounts) == 1:
                                      return self.accounts[0]
                                  # 轮询策略
                                  account = self.accounts[self._rr_index % len(self.accounts)]
        self._rr_index += 1
        return account


# 全局变量
http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
      if http_client is None:
                raise RuntimeError("HTTP client not initialized")
    return http_client


def extract_api_keys(request: Request) -> list[str]:
      """从请求头提取 API keys（支持逗号分隔的多个 key）"""
    auth_header = request.headers.get("authorization", "")
    if not auth_header:
              return []

    # 移除 Bearer 前缀
    if auth_header.lower().startswith("bearer "):
              auth_header = auth_header[7:]

    # 支持逗号分隔的多个 key
    keys = [k.strip() for k in auth_header.split(",") if k.strip()]
    return keys


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
      global http_client
    http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    logger.info("Started: Node.js SDK proxy mode enabled")
    logger.info("Node.js proxy URL: %s", NODE_PROXY_URL)
    yield
    await http_client.aclose()


app = FastAPI(title="AnyRouter OpenAI Proxy (Node.js SDK Mode)", lifespan=lifespan)


def generate_request_id() -> str:
      return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def generate_user_id() -> str:
      user_hash = ''.join(random.choices('0123456789abcdef', k=64))
    session_uuid = uuid.uuid4()
    return f"user_{user_hash}_account__session_{session_uuid}"


def build_forwarding_headers(api_key: str, original_headers: dict[str, str] = None) -> dict[str, str]:
      """构建转发到 Node.js 代理的请求头，透传客户端所有特殊头"""
    SKIP_HEADERS = {
              "host", "content-length", "transfer-encoding", "connection",
              "keep-alive", "upgrade", "proxy-authorization", "proxy-connection",
              "accept-encoding",
    }

    headers = {}

    # 透传客户端的所有头（保留 Claude Code 发送的所有特殊头）
    if original_headers:
              for key, val in original_headers.items():
                            if key.lower() not in SKIP_HEADERS:
                                              headers[key] = val

                    # 确保关键字段
                    headers["Content-Type"] = "application/json"
    headers["x-api-key"] = api_key

    # 移除 authorization 避免重复认证
    headers.pop("authorization", None)

    return headers


def convert_message_content(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
      """转换消息内容为 Anthropic 格式"""
    if isinstance(content, str):
              return [{"type": "text", "text": content}]

    result = []
    for item in content:
              if item.get("type") == "text":
                            result.append({"type": "text", "text": item.get("text", "")})
elif item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "")
            if url.startswith("data:"):
                              media_type, _, base64_data = url.partition(";base64,")
                              media_type = media_type.replace("data:", "")
                              result.append({
                                  "type": "image",
                                  "source": {"type": "base64", "media_type": media_type, "data": base64_data}
                              })
else:
                result.append({"type": "image", "source": {"type": "url", "url": url}})
    return result if result else [{"type": "text", "text": ""}]


def convert_openai_to_anthropic(openai_request: dict[str, Any]) -> dict[str, Any]:
      """将 OpenAI 请求格式转换为 Anthropic 格式"""
    system_messages: list[dict[str, Any]] = []
    chat_messages: list[dict[str, Any]] = []

    for msg in openai_request.get("messages", []):
              content = msg.get("content", "")
        role = msg.get("role", "")

        if role == "system":
                      if isinstance(content, str):
                                        system_messages.append({"type": "text", "text": content})
        else:
                system_messages.extend(convert_message_content(content))
elif role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": convert_message_content(content)})
elif role == "tool":
            chat_messages.append({"role": "user", "content": convert_message_content(content)})

    anthropic_request: dict[str, Any] = {
              "model": normalize_model_name(openai_request.get("model")),
              "messages": chat_messages,
              "max_tokens": openai_request.get("max_tokens", DEFAULT_MAX_TOKENS),
              "stream": True,
    }

    # 只在用户显式传了 system 消息时设置，否则让 Node.js 注入 Claude Code 系统提示
    if system_messages:
              anthropic_request["system"] = system_messages

    # 不设置 metadata，让 Node.js 注入 Claude Code 格式的 user_id

    optional_params = {"temperature": "temperature", "top_p": "top_p", "stop": "stop_sequences"}
    for openai_key, anthropic_key in optional_params.items():
              if openai_key in openai_request:
                            anthropic_request[anthropic_key] = openai_request[openai_key]

    return anthropic_request


def convert_anthropic_response_to_openai(
      anthropic_response: dict[str, Any], model: str, request_id: str
) -> dict[str, Any]:
      """将 Anthropic 响应转换为 OpenAI 格式"""
    content = "".join(
              block.get("text", "") for block in anthropic_response.get("content", [])
              if block.get("type") == "text"
    )
    usage = anthropic_response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    stop_reason = anthropic_response.get("stop_reason", "")
    finish_reason = "stop" if stop_reason == "end_turn" else "length"

    return {
              "id": request_id,
              "object": "chat.completion",
              "created": int(time.time()),
              "model": model,
              "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
              "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }


def create_stream_chunk(request_id: str, model: str, content: str | None = None, finish_reason: str | None = None) -> dict[str, Any]:
      delta: dict[str, Any] = {}
    if content is not None:
              delta["content"] = content
    return {
              "id": request_id,
              "object": "chat.completion.chunk",
              "created": int(time.time()),
              "model": model,
              "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def stream_response(
      anthropic_request: dict[str, Any],
      account: Account,
      headers: dict[str, str],
      request_id: str,
      model: str
) -> AsyncGenerator[str, None]:
      """处理流式响应"""
    client = get_client()
    raw_lines: list[str] = []
    saw_sse_event = False

    try:
              async with client.stream(
                            "POST", f"{NODE_PROXY_URL}/v1/messages", headers=headers, json=anthropic_request
              ) as resp:
                            if resp.status_code != 200:
                                              error_text = await resp.aread()
                                              logger.error("[%s] Error %d: %s", account.name, resp.status_code, error_text.decode(errors="replace")[:200])
                                              yield f"data: {json.dumps(build_openai_error_payload(error_text, resp.status_code), ensure_ascii=False)}\n\n"
                                              yield "data: [DONE]\n\n"
                                              return

                            async for line in resp.aiter_lines():
                                              if not line or not line.strip():
                                                                    continue
                                                                if not line.startswith("data: "):
                                                                                      raw_lines.append(line)
                                                                                      continue

                                              saw_sse_event = True

                                try:
                                                      event = json.loads(line[6:])
                                                      event_type = event.get("type")

                    if event_type == "content_block_delta":
                                              delta = event.get("delta", {})
                                              if delta.get("type") == "text_delta":
                                                                            chunk = create_stream_chunk(request_id, model, content=delta.get("text", ""))
                                                                            yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "message_stop":
                        chunk = create_stream_chunk(request_id, model, finish_reason="stop")
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
elif event_type == "error":
                        error_msg = event.get("error", {}).get("message", "Unknown error")
                        logger.error("[%s] Stream error: %s", account.name, error_msg)
                        yield f"data: {json.dumps(build_openai_error_payload(event, 502), ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
except json.JSONDecodeError:
                    continue

            if raw_lines and not saw_sse_event:
                              raw_payload = "\n".join(raw_lines)
                              logger.error("[%s] Non-SSE upstream payload: %s", account.name, raw_payload[:200])
                              yield f"data: {json.dumps(build_openai_error_payload(raw_payload, 502), ensure_ascii=False)}\n\n"
                              yield "data: [DONE]\n\n"

except httpx.TimeoutException:
        logger.error("[%s] Timeout", account.name)
        yield f"data: {json.dumps(build_openai_error_payload({'error': {'message': 'Request timeout', 'type': 'timeout_error'}}, 504), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
except httpx.HTTPError as e:
        logger.error("[%s] HTTP error: %s", account.name, str(e))
        yield f"data: {json.dumps(build_openai_error_payload({'error': {'message': str(e), 'type': 'http_error'}}, 502), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


async def stream_from_non_stream(
      anthropic_request: dict[str, Any],
      account: Account,
      headers: dict[str, str],
      request_id: str,
      model: str
) -> AsyncGenerator[str, None]:
      """非流式后端 + 流式前端"""
    client = get_client()

    try:
              resp = await client.post(f"{NODE_PROXY_URL}/v1/messages", headers=headers, json=anthropic_request)

        if resp.status_code != 200:
                      logger.error("[%s] Error %d: %s", account.name, resp.status_code, resp.text[:200])
            yield f"data: {json.dumps(build_openai_error_payload(resp.text, resp.status_code), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        anthropic_response = resp.json()
        content = "".join(
                      block.get("text", "") for block in anthropic_response.get("content", [])
                      if block.get("type") == "text"
        )

        if content:
                      yield f"data: {json.dumps(create_stream_chunk(request_id, model, content=content))}\n\n"

        yield f"data: {json.dumps(create_stream_chunk(request_id, model, finish_reason='stop'))}\n\n"
        yield "data: [DONE]\n\n"

except httpx.TimeoutException:
        logger.error("[%s] Timeout", account.name)
        yield f"data: {json.dumps(build_openai_error_payload({'error': {'message': 'Request timeout', 'type': 'timeout_error'}}, 504), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
except httpx.HTTPError as e:
        logger.error("[%s] HTTP error: %s", account.name, str(e))
        yield f"data: {json.dumps(build_openai_error_payload({'error': {'message': str(e), 'type': 'http_error'}}, 502), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
      """OpenAI 兼容的 chat completions 接口"""
    # 提取并验证 API keys
    api_keys = extract_api_keys(request)
    if not api_keys:
              return openai_error_response(
                            401,
                            {"error": {"message": "Authorization header required. Please provide a valid API key.", "type": "authentication_error"}},
              )

    # 创建请求级负载均衡器
    lb = RequestLoadBalancer(api_keys)
    account = lb.select_account()
    if not account:
              return openai_error_response(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})

    openai_request = await request.json()
    anthropic_request = convert_openai_to_anthropic(openai_request)

    # 记录完整的客户端请求信息（用于调试）
    original_headers = dict(request.headers)
    logger.info("========== 客户端请求详情 ==========")
    logger.info("[请求头] %s", json.dumps(
              redact_headers_for_log(original_headers),
              ensure_ascii=False, indent=2
    ))
    logger.info("[OpenAI请求] model=%s stream=%s", openai_request.get("model"), openai_request.get("stream"))
    logger.info("====================================")

    # 构建转发头，透传客户端所有特殊头
    forwarding_headers = build_forwarding_headers(account.api_key, original_headers)

    request_id = generate_request_id()
    model = openai_request.get("model", "unknown")
    is_stream = openai_request.get("stream", True)

    use_non_stream_backend = FORCE_NON_STREAM or not is_stream
    if use_non_stream_backend:
              anthropic_request['stream'] = False

    logger.info("[%s] %s stream=%s backend_stream=%s", account.name, model, is_stream, not use_non_stream_backend)

    if is_stream:
              handler = stream_from_non_stream if use_non_stream_backend else stream_response
        return StreamingResponse(
                      handler(anthropic_request, account, forwarding_headers, request_id, model),
                      media_type="text/event-stream",
                      headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
else:
        client = get_client()
        try:
                      resp = await client.post(f"{NODE_PROXY_URL}/v1/messages", headers=forwarding_headers, json=anthropic_request)
            if resp.status_code != 200:
                              logger.error("[%s] Error %d: %s", account.name, resp.status_code, resp.text[:200])
                              return openai_error_response(resp.status_code, resp.text)
                          return convert_anthropic_response_to_openai(resp.json(), model, request_id)
except httpx.TimeoutException:
            return openai_error_response(504, {"error": {"message": "Request timeout", "type": "timeout_error"}})
except httpx.HTTPError as e:
            return openai_error_response(502, {"error": {"message": str(e), "type": "http_error"}})


@app.get("/v1/models")
async def list_models(request: Request):
      """列出可用模型"""
    api_keys = extract_api_keys(request)
    if not api_keys:
              return openai_error_response(
                            401,
                            {"error": {"message": "Authorization header required", "type": "authentication_error"}},
              )

    lb = RequestLoadBalancer(api_keys)
    account = lb.select_account()
    if not account:
              return openai_error_response(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})

    # Node.js 代理当前不稳定提供 /v1/models，因此这里直接返回静态模型清单，
    # 并包含常见别名（如 claude-opus-4-6[1m]），方便 Cherry Studio / LobeChat 手动选型。
    return build_model_list_payload()


@app.get("/health")
async def health():
      # 检查 Node.js 代理是否可用
      try:
                client = get_client()
                resp = await client.get(f"{NODE_PROXY_URL}/health", timeout=5)
                node_status = resp.json() if resp.status_code == 200 else {"status": "error"}
except Exception:
          node_status = {"status": "unreachable"}

      return {
                "status": "ok",
                "mode": "node-proxy",
                "node_proxy": NODE_PROXY_URL,
                "node_status": node_status,
      }


@app.get("/")
async def root():
      return {
                "service": "AnyRouter OpenAI Proxy",
                "mode": "node-proxy",
                "node_proxy_url": NODE_PROXY_URL,
                "description": "Forwarding requests to Node.js proxy (using official Anthropic SDK)",
      }


if __name__ == "__main__":
      import uvicorn

    port = int(os.getenv("OPENAI_PROXY_PORT", "9999"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║       AnyRouter OpenAI Proxy (Node.js SDK Mode)          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Python 代理: http://{host}:{port}
    ║  Node.js 代理: {NODE_PROXY_URL}
    ╠══════════════════════════════════════════════════════════╣
    ║  架构:                                                    ║
    ║    客户端 → Python (9999) → Node.js (4000) → anyrouter   ║
    ╠══════════════════════════════════════════════════════════╣
    ║  启动步骤:                                                ║
    ║    1. cd node-proxy && npm install && npm start          ║
    ║    2. python anyrouter2openai.py                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  管理接口:                                                ║
    ║    GET /health - 健康检查（含 Node.js 状态）              ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=host, port=port)
