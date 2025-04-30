import responder

api = responder.API()
sessions = {}

@api.route('/ws', websocket=True)
async def websocket(ws):
    await ws.accept()
    key = ws.headers.get('sec-websocket-key')
    sessions[key] = ws
    try:
        while True:
            msg = await ws.receive_text()
            for k in sessions.values():
                await k.send_text(msg)
    except:
        del sessions[key]
        await ws.close()

api.add_route('/', static=True)
api.run()
