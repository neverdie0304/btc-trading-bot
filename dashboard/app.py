"""FastAPI 웹 대시보드 앱.

봇 관리 API와 SSE 실시간 스트림을 제공한다.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from dashboard.bot_manager import BotManager

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="BTC Trading Bot Dashboard")
manager = BotManager()


# === Pydantic 모델 ===

class BotStartRequest(BaseModel):
    symbol: str
    mode: str = "paper"
    capital: float = 0


class BotStopRequest(BaseModel):
    symbol: str


# === 라이프사이클 ===

@app.on_event("startup")
async def startup():
    await manager.initialize()
    logger.info("Dashboard 서버 시작")


@app.on_event("shutdown")
async def shutdown():
    await manager.shutdown_all()
    logger.info("Dashboard 서버 종료")


# === 페이지 ===

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# === API ===

@app.get("/api/status")
async def api_status():
    return {"bots": manager.get_all_status()}


@app.get("/api/status/{symbol}")
async def api_bot_status(symbol: str):
    status = manager.get_bot_status(symbol)
    if status is None:
        return {"error": "봇을 찾을 수 없습니다"}
    return status


@app.post("/api/bot/start")
async def api_start_bot(req: BotStartRequest):
    result = await manager.start_bot(
        symbol=req.symbol,
        mode=req.mode,
        capital=req.capital,
    )
    return result


@app.post("/api/bot/stop")
async def api_stop_bot(req: BotStopRequest):
    result = await manager.stop_bot(symbol=req.symbol)
    return result


@app.get("/api/trades")
async def api_trades(symbol: str | None = None, limit: int = 30):
    trades = manager.get_recent_trades(symbol=symbol, limit=limit)
    return {"trades": trades}


# === SSE 실시간 스트림 ===

@app.get("/api/stream")
async def sse_stream():
    async def event_generator():
        while True:
            try:
                data = {
                    "bots": manager.get_all_status(),
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SSE 스트림 에러")
                await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
