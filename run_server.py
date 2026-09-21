"""
Local Server Entrypoint for Vision-Based Car Camera Navigation (High Performance)
Provides asynchronous, non-blocking WebSocket streaming and REST endpoints
for real-time video playback and instant button responsiveness.
"""

import os
import sys
import json
import base64
import asyncio
import shutil
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from modules.ugv_controller import UGVController
from modules.video_processor import VideoProcessor

app = FastAPI(title="Car Camera Vision-Based Navigation Server")

logs_dir = CURRENT_DIR / "logs"
uploads_dir = CURRENT_DIR / "uploads"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

# Controller
controller = UGVController(log_dir=str(logs_dir))

# Mount static, logs, and uploads folders
static_dir = CURRENT_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/logs", StaticFiles(directory=str(logs_dir)), name="logs")
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


@app.get("/")
async def get_index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
async def health_check():
    return {"status": "online", "system": "Car Camera Autonomous Navigation Server"}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = uploads_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ok = await asyncio.to_thread(controller.load_video, str(file_path))
    meta = controller.video_processor.get_metadata()
    return JSONResponse({
        "status": "success" if ok else "failed",
        "filename": file.filename,
        "metadata": meta,
    })


@app.post("/api/load_demo")
async def load_demo_video():
    demo_path = str(logs_dir / "demo_dashcam.mp4")
    if not os.path.exists(demo_path):
        await asyncio.to_thread(VideoProcessor.generate_demo_dashcam_video, demo_path, duration_sec=12, fps=30)

    ok = await asyncio.to_thread(controller.load_video, demo_path)
    meta = controller.video_processor.get_metadata()
    return JSONResponse({
        "status": "success" if ok else "failed",
        "filename": "demo_dashcam.mp4",
        "metadata": meta,
    })


@app.post("/api/load_demo_offroad")
async def load_demo_offroad():
    offroad_path = str(logs_dir / "demo_offroad.mp4")
    if not os.path.exists(offroad_path):
        await asyncio.to_thread(VideoProcessor.generate_demo_offroad_video, offroad_path, duration_sec=12, fps=30)

    ok = await asyncio.to_thread(controller.load_video, offroad_path)
    meta = controller.video_processor.get_metadata()
    return JSONResponse({
        "status": "success" if ok else "failed",
        "filename": "demo_offroad.mp4",
        "metadata": meta,
    })


@app.post("/api/open_webcam")
async def open_webcam(device_index: int = Form(0)):
    ok = await asyncio.to_thread(controller.load_video, device_index)
    meta = controller.video_processor.get_metadata()
    return JSONResponse({
        "status": "success" if ok else "failed",
        "metadata": meta,
    })


@app.get("/api/generate_plot")
async def get_plot():
    """Trigger plot generation in a background thread and return file details."""
    result = await asyncio.to_thread(controller.finish_run)
    return JSONResponse(result)


@app.websocket("/ws/ugv")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] Client connected to Car Camera Navigation WebSocket.")

    streaming_task: Optional[asyncio.Task] = None
    is_streaming = False

    async def stream_loop():
        nonlocal is_streaming
        try:
            while is_streaming:
                # Run OpenCV frame processing in worker thread so event loop stays free
                result = await asyncio.to_thread(controller.process_next_video_frame)
                if result is None:
                    # End of video reached
                    finish_data = await asyncio.to_thread(controller.finish_run)
                    await websocket.send_json({
                        "type": "VIDEO_ENDED",
                        "payload": finish_data,
                    })
                    is_streaming = False
                    break

                await websocket.send_json({
                    "type": "FRAME_RESULT",
                    "payload": result,
                })
                # Smooth 25-28 FPS pacing
                await asyncio.sleep(0.038)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            is_streaming = False

    try:
        while True:
            # Always actively listen for incoming client commands (PAUSE, STEP, FINISH, etc.)
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            msg_type = message.get("type", "")

            if msg_type == "PLAY":
                if not is_streaming:
                    is_streaming = True
                    if streaming_task and not streaming_task.done():
                        streaming_task.cancel()
                    streaming_task = asyncio.create_task(stream_loop())
                    print("[SERVER] Playback stream started.")
                    await websocket.send_json({"type": "STATUS", "status": "PLAYING"})

            elif msg_type == "PAUSE":
                is_streaming = False
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                print("[SERVER] Playback paused.")
                await websocket.send_json({"type": "STATUS", "status": "PAUSED"})

            elif msg_type == "STEP":
                is_streaming = False
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                result = await asyncio.to_thread(controller.process_next_video_frame)
                if result is not None:
                    await websocket.send_json({
                        "type": "FRAME_RESULT",
                        "payload": result,
                    })

            elif msg_type == "RESTART":
                is_streaming = False
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                await asyncio.to_thread(controller.video_processor.restart)
                controller.reset()
                print("[SERVER] Video rewound to beginning.")
                await websocket.send_json({"type": "STATUS", "status": "RESTARTED"})

            elif msg_type == "FINISH":
                is_streaming = False
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                finish_result = await asyncio.to_thread(controller.finish_run)
                await websocket.send_json({
                    "type": "RUN_FINISHED",
                    "payload": finish_result,
                })

            elif msg_type == "PING":
                await websocket.send_json({"type": "PONG"})

    except WebSocketDisconnect:
        is_streaming = False
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
        print("[SERVER] Client disconnected.")
    except Exception as e:
        is_streaming = False
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
        print(f"[SERVER ERROR] Exception in WebSocket: {e}")


def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 75)
    print(" VISION-BASED CAR CAMERA AUTONOMOUS NAVIGATION SERVER")
    print(f" Access Dashboard: {url}")
    print("=" * 75)

    async def open_browser_later():
        await asyncio.sleep(1.0)
        webbrowser.open(url)

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(open_browser_later())
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    main()
