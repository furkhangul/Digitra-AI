import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.inference import classifier_health, get_classifier

router = APIRouter(tags=["recognition"])


@router.websocket("/ws/recognize")
async def recognize(websocket: WebSocket) -> None:
    await websocket.accept()
    health = await asyncio.to_thread(classifier_health)
    if not health["ready"]:
        await websocket.send_json({"model_ready": False, **health})
        await websocket.close(code=1011)
        return

    classifier = await asyncio.to_thread(get_classifier)
    session = classifier.new_session()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                response = await asyncio.to_thread(session.update, payload)
            except (KeyError, TypeError, ValueError) as error:
                await websocket.send_json(
                    {"model_ready": True, "error": f"Geçersiz istek: {error}"}
                )
                continue
            await websocket.send_json(response)
    except WebSocketDisconnect:
        return
