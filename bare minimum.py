import websocket
import threading
import json
import time
ipAddress = "ws://localhost:9000"
wsConnected = False
wsInstance = None
sensorData = []
leftEncoder = 0
rightEncoder = 0
yawDelta = 0

def onOpen (ws):
    global wsConnected, wsInstance
    print ("Now connected to the server")
    wsConnected = True
    wsInstance = ws

def onMessage(ws, message):
    data = json.loads(message)

    print(json.dumps(data, indent=2))

def onMe (ws, message):
    global sensorData, leftEncoder, rightEncoder, yawDelta
    data = json.loads (message)
    if data.get ("type") == "DATA":
        sensorData = data.get ("sensor", [])
        leftEncoder = data.get ("leftEncoder", 0)
        rightEncoder = data.get ("rightEncoder", 0)
        yawDelta = data.get ("yawDelta", 0)
        print (leftEncoder,rightEncoder,yawDelta)

def onClose (ws, code, reason):
    global wsConnected
    print ("Disconnected from the server", reason)
    wsConnected = False

def onError (ws, error):
    print ("Error: Reason", error)

def runWS ():
    ws = websocket.WebSocketApp(
        ipAddress,
        on_open=onOpen,
        on_message=onMessage,
        on_close=onClose,
        on_error=onError
        )
    ws.run_forever ()
threading.Thread(target=runWS, daemon=True).start()
while not wsConnected:
    time.sleep (0.1)

try:
    while True:
        leftWheelVel = 0
        rightWheelVel = 0
        if wsInstance:
            try:
                wsInstance.send(f"MOVE {leftWheelVel} {rightWheelVel}")
            except Exception as e:
                print ("Sending failed", e)
        time.sleep (1/15.0)
except KeyboardInterrupt:
    print ("Exiting")
finally:
    if wsInstance:
        wsInstance.close()