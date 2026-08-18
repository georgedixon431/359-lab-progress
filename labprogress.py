import websocket
import threading
import json
import time
import numpy as np
from numba import njit
import cv2

ipAddress = "ws://localhost:9000"
wsConnected = False
wsInstance = None
sensorData = []
leftEncoder = 0
rightEncoder = 0
yawDelta = 0

currentLeftVel = 0.0
currentRightVel = 0.0
wheeldiameter = 20.0
wheelbase = 100
ticksPerRev = 100
distancePerTick = np.pi * wheeldiameter / ticksPerRev
maxVelChange = 2.0
MaxWheelVel = 20.0

mapSize = 2440
worldMap = np.zeros((mapSize, mapSize), dtype=np.uint8)

robotX = mapSize // 2
robotY = mapSize // 2
robotAngle = 0.0

prevLeftEncoder = 0
prevRightEncoder = 0

def limitVelocity(current, target):
    target = np.clip(target, -MaxWheelVel, MaxWheelVel)

    difference = target - current
    difference = np.clip(
        difference,
        -maxVelChange,
        maxVelChange
    )

    return current + difference

def onOpen (ws):
    global wsConnected, wsInstance
    print ("Now connected to the server")
    wsConnected = True
    wsInstance = ws

def onMessage (ws, message):
    global sensorData, leftEncoder, rightEncoder, yawDelta
    data = json.loads (message)
    if data.get ("type") == "DATA":
        sensorData = data.get ("sensor", [])
        leftEncoder = data.get ("leftEncoder", 0)
        rightEncoder = data.get ("rightEncoder", 0)
        yawDelta = data.get ("yawDelta", 0)
        dtime = data.get("dt", 0)
        #print (leftEncoder,rightEncoder,yawDelta, sensorData, dtime)

        updatePosition(leftEncoder, rightEncoder, yawDelta)
        updateMap(sensorData)
        frontiers = findFrontiers()
        displayMap()
        print(
            "yaw deg:", round(yawDelta, 3),
            "angle rad:", round(robotAngle, 4),
            "robot:", round(robotX, 2), round(robotY, 2),
            "mapped:", np.count_nonzero(worldMap)
        )

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

@njit
def updateMapNumba(sensorData):
    global worldMap, robotX, robotY, robotAngle

    for i, distance in enumerate(sensorData):
        angle = robotAngle + i * (2 * np.pi / len(sensorData))

        for d in range(int(distance)):
            x = int(robotX + d * np.cos(angle))
            y = int(robotY - d * np.sin(angle))
            if 0 <= x < mapSize and 0 <= y < mapSize:
                if worldMap[y, x] != 2:
                    worldMap[y, x] = 1

        # If sensor actually hit something, mark obstacle
        if distance < 500:
            endX = int(robotX + distance * np.cos(angle))
            endY = int(robotY - distance * np.sin(angle))

            if 0 <= endX < mapSize and 0 <= endY < mapSize:
                worldMap[endY, endX] = 2

def updatePosition(leftEncoder, rightEncoder, yawDelta):
    global robotX, robotY, robotAngle, prevLeftEncoder, prevRightEncoder

    deltaLeftTicks = leftEncoder - prevLeftEncoder
    deltaRightTicks = rightEncoder - prevRightEncoder

    prevRightEncoder = rightEncoder
    prevLeftEncoder = leftEncoder

    deltaLeft = deltaLeftTicks * distancePerTick
    deltaRight = deltaRightTicks * distancePerTick

    robotAngle = np.deg2rad(yawDelta)
    distanceMoved = (deltaLeft + deltaRight) /2

    robotX += distanceMoved * np.cos(robotAngle)
    robotY -= distanceMoved * np.sin(robotAngle)

def displayMap():

    display = np.zeros((mapSize, mapSize, 3), dtype=np.uint8)

    # Unknown = dark
    display[worldMap == 0] = (30, 30, 30)
    # Free = white
    display[worldMap == 1] = (255, 255, 255)
    # Obstacles = red
    display[worldMap == 2] = (0, 0, 255)

    frontiers = findFrontiers()
    for x, y in frontiers:
        display[y, x] = (255,0,0)


    cv2.circle(display, (int(robotX), int(robotY)), 8, (0,255,0), -1)

    displaySmall = cv2.resize(display, (610, 610))
    cv2.imshow("Robot Map", displaySmall)
    cv2.waitKey(1)



def findFrontiers():
    frontiers = []

    for y in range(1, mapSize - 1):
        for x in range(1, mapSize - 1):

            # Must already be known free space
            if worldMap[y, x] != 1:
                continue

            # Look at 8 neighbouring cells
            neighbours = worldMap[y-1:y+2, x-1:x+2]

            # If any neighbour is unknown, this is a frontier
            if np.any(neighbours == 0):
                frontiers.append((x, y))

    return frontiers
    
    



threading.Thread(target=runWS, daemon=True).start()
while not wsConnected:
    time.sleep (0.1)

try:
    while True:
        targetLeftVel = 15
        targetRightVel = 20

        currentLeftVel = limitVelocity(currentLeftVel, targetLeftVel)
        currentRightVel = limitVelocity(currentRightVel, targetRightVel)
        
        if wsInstance:
            try:
                wsInstance.send(f"MOVE {currentLeftVel} {currentRightVel}")
            except Exception as e:
                print ("Sending failed", e)
        time.sleep (1/15.0)

except KeyboardInterrupt:
    print ("Exiting")
finally:
    if wsInstance:
        wsInstance.close()

    cv2.destroyAllWindows()