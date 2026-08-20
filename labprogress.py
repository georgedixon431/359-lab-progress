#git add .
#git commit -m "Add frontier detection"
#git push

import websocket
import threading
import json
import time
import numpy as np
from numba import njit
import cv2
import heapq


ipAddress = "ws://localhost:9000"
wsConnected = False
wsInstance = None
sensorData = []
leftEncoder = 0
rightEncoder = 0
yawDelta = 0
startrecieved = False
startAngle = 0.0

currentLeftVel = 0.0
currentRightVel = 0.0
wheeldiameter = 20.0
wheelbase = 100.0
botRadius = 50
clearance = botRadius + 5
planningClearance = botRadius + 20
ticksPerRev = 100
distancePerTick = np.pi * wheeldiameter / ticksPerRev
maxVelChange = 2.0
maxWheelVel = 20.0
testWheelVel = 80.0
speedPrevLeft = 0
speedPrevRight = 0
speedPrevTime = 0.0
totaltime = time.time()

mapSize = 5000
worldMap = np.zeros((mapSize, mapSize), dtype=np.uint8)
frontierMask = np.zeros((mapSize, mapSize), dtype=np.uint8)
frontierTarget = None
updateCounter = 0
explorationComplete = False
noFrontier = 0

planningEvent = threading.Event()
pathLock = threading.Lock()
mapLock = threading.Lock()
currentPath = np.empty((0, 2), dtype=np.int32)
now = 0.0

robotX = mapSize//2
robotY = mapSize//2
robotAngle = 0.0
prevLeftEncoder = 0
prevRightEncoder = 0

waypointIndex = 0
waypointTolerance = 8.0
lookAhead = 30
linearGain = 0.8
angularGain = 3.0

testMode = False

photo = cv2.imread("bot.png", cv2.IMREAD_UNCHANGED)

def limitVelocity(current, target):
    global testWheelVel, maxWheelVel

    if testMode:
        maxVel = testWheelVel
    else:
        maxVel = maxWheelVel
    target = np.clip(target, -maxVel, maxVel)

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

def onMessage(ws, message):
    global sensorData, startrecieved, leftEncoder, rightEncoder, yawDelta, updateCounter, frontierMask
    global robotX, robotY, robotAngle, startAngle, speedPrevRight, speedPrevLeft, speedPrevTime, now

    data = json.loads(message)

    now = time.time()

    if data.get("type") == "DEBUG":
        if not startrecieved:
            #robotX = data["x"]
            #robotY = data["y"]
            #theta = data["theta"]
            #startAngle = np.deg2rad(theta)
            robotAngle = 0
            #print("Starting position: ", robotX, robotY, theta)
            startrecieved = True

        return

    if data.get("type") != "DATA":
        return
    if not startrecieved:
        return
        
    sensorData = data.get("sensor", [])
    leftEncoder = data.get("leftEncoder", 0)
    rightEncoder = data.get("rightEncoder", 0)
    yawDelta = data.get("yawDelta", 0)


    with mapLock:

        updatePosition(leftEncoder, rightEncoder, yawDelta)

        sensorArray = np.asarray(sensorData,dtype=np.float64)

        updateMapNumba(
            worldMap,
            sensorArray,
            robotX,
            robotY,
            robotAngle,
            mapSize
        )

        updateCounter += 1
        if updateCounter % 10 == 0:
            frontierMask = findFrontiersNumba(
                worldMap
            )

    if updateCounter % 100 == 0 and not explorationComplete:
        planningEvent.set()

def plannerLoop():
    global currentPath, waypointIndex, frontierTarget, now, explorationComplete, noFrontier

    while True:

        # Sleep here until somebody asks for a replan
        planningEvent.wait()
        planningEvent.clear()

        print("Replanning...")

        # Snapshot everything needed by A*
        with mapLock:
            mapCopy = worldMap.copy()
            frontierCopy = frontierMask.copy()
            startX = int(round(robotX))
            startY = int(round(robotY))

        planningMap = createPlanningMap(mapCopy)

        # Keep old target if still valid
        if (frontierTarget is not None and checkTarget(frontierTarget, frontierCopy, planningMap)):
            target = frontierTarget

        else:
            target = chooseFrontier(frontierCopy,planningMap, startX, startY)

        if target is None:
            noFrontier += 1
            print("No frontier available")
            if noFrontier >= 3:
                explorationComplete
                with pathLock:
                    currentPath = np.empty((0,2),dtype=np.int32)
                    waypointIndex = 0

            continue

        noFrontier = 0
        startTime = time.time()

        newPath = localAStar(planningMap, (startX, startY), target)
        newPath = simplifyPath(newPath)

        print("A* took", round(time.time() - startTime, 3), "seconds")

        if len(newPath) > 0:

            newPath = simplifyPath(newPath)

            # Very short critical section
            with pathLock:
                currentPath = newPath
                waypointIndex = 0
                frontierTarget = target

            print("New path:", len(newPath), "points")

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

@njit(nogil=True, cache=True)
def updateMapNumba(worldMap, sensorData, robotX, robotY, robotAngle, mapSize):

    n = len(sensorData)

    for i in range(n):
        distance = sensorData[i]
        angle = robotAngle + i * (2.0 * np.pi / n)
        cosA = np.cos(angle)
        sinA = np.sin(angle)

        # Mark free space along lidar ray
        for d in range(int(distance)):
            x = int(robotX + d * cosA)
            y = int(robotY - d * sinA)

            if 0 <= x < mapSize and 0 <= y < mapSize:
                if worldMap[y, x] != 2:
                    worldMap[y, x] = 1

        # Mark obstacle
        if distance < 500:
            endX = int(robotX + distance * cosA)
            endY = int(robotY - distance * sinA)

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

    robotAngle = np.deg2rad(yawDelta) + startAngle
    robotAngle = np.arctan2(np.sin(robotAngle), np.cos(robotAngle))
    distanceMoved = (deltaLeft + deltaRight) /2

    robotX += distanceMoved * np.cos(robotAngle)
    robotY -= distanceMoved * np.sin(robotAngle)

    robotX = np.clip(robotX, botRadius, mapSize - botRadius - 1)
    robotY = np.clip(robotY, botRadius, mapSize - botRadius - 1)


def createPlanningMap(mapData):

    # Actual known obstacles
    obstacleMask = (mapData == 2).astype(np.uint8)

    kernelSize = 2 * planningClearance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
    inflatedObstacles = cv2.dilate(obstacleMask, kernel)

    # Unknown space must also be blocked
    planningMap = np.zeros_like(mapData, dtype=np.uint8)

    planningMap[mapData == 0] = 1
    planningMap[inflatedObstacles == 1] = 1

    planningMap[:planningClearance, :] = 1
    planningMap[-planningClearance:, :] = 1
    planningMap[:, :planningClearance] = 1
    planningMap[:, -planningClearance:] = 1

    return planningMap


def checkTarget(target, frontierMask, planningMap):

    if target is None:
        return False

    x, y = target

    if not (0 <= x < mapSize and 0 <= y < mapSize):
        return False

    if frontierMask[y, x] != 1:
        return False

    if planningMap[y, x] == 1:
        return False

    return True


def displayMap():
    global currentRightVel, currentLeftVel, testMode, robotAngle
    
    display = np.zeros((mapSize, mapSize, 3), dtype=np.uint8)

    # Unknown = dark
    display[worldMap == 0] = (30, 30, 30)
    # Free = white
    display[worldMap == 1] = (255, 255, 255)
    # Obstacles = red
    display[worldMap == 2] = (0, 0, 255)
    display[frontierMask == 1] = (255, 0, 0)

    if currentPath is not None and len(currentPath) > 0:
        for i in range(len(currentPath) - 1):
            x1, y1 = currentPath[i]
            x2, y2 = currentPath[i + 1]
            cv2.line(
                display,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 255),
                3
            )

    if frontierTarget is not None:
        cv2.circle(display, frontierTarget, 15, (0,255,255),-1)

    if photo is not None:
        bot = cv2.resize(photo, (100, 100))
        x = int(robotX - 50)
        y = int(robotY - 50)
        circleMask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(circleMask, (50, 50), 50, 255, -1)

        if bot.shape[2] == 4:
            mask = cv2.bitwise_and(bot[:, :, 3], circleMask)
            cv2.copyTo(bot[:, :, :3], mask, display[y:y + 100, x:x + 100])
        else:
            cv2.copyTo(bot, circleMask, display[y:y + 100, x:x + 100])
    #cv2.circle(display, (int(robotX), int(robotY)), 50, (0,255,0), -1)

    arrowLength = 100
    startPoint = (int(robotX), int(robotY))
    endPoint = (
        int(robotX + arrowLength * np.cos(robotAngle)),
        int(robotY - arrowLength * np.sin(robotAngle)))
    cv2.line(display, startPoint, endPoint, (0, 0, 0), 4)

    currentTime = (time.time() - totaltime)/100
    displaySmall = cv2.resize(display, (1000, 1000))
    cv2.putText(displaySmall,f"Left wheel: {currentLeftVel:.2f} rad/s",(20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)
    cv2.putText(displaySmall,f"right wheel: {currentRightVel:.2f} rad/s",(20, 60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)
    cv2.putText(displaySmall,f"heading: {np.rad2deg(robotAngle):.2f} degrees",(20, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)
    cv2.putText(displaySmall,f"Time Taken: {np.rad2deg(currentTime):.2f} seconds",(20, 120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)
    

    cv2.imshow("Robot Map", displaySmall)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        testMode = not testMode
        if testMode:
            print("Super Duper fast")
        else:
            print("normal speed")



@njit(nogil=True, cache=True)
def findFrontiersNumba(worldMap):
    height, width = worldMap.shape

    frontierMask = np.zeros((height, width), dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if worldMap[y, x] != 1:
                continue

            foundUnknown = False

            for dy in range(-1, 2):
                for dx in range(-1, 2):

                    if dx == 0 and dy == 0:
                        continue

                    if worldMap[y + dy, x + dx] == 0:
                        foundUnknown = True
                        break

                if foundUnknown:
                    break

            if foundUnknown:
                frontierMask[y, x] = 1

    return frontierMask

def chooseFrontier(frontierMask, planningMap, robotX, robotY):

    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frontierMask,
        connectivity=8
    )

    bestTarget = None
    bestScore = float("inf")

    for label in range(1, numLabels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < 10:
            continue

        ys, xs = np.where(labels == label)

        if len(xs) == 0:
            continue

        cx, cy = centroids[label]

        distToCentre = ((xs - cx) ** 2 + (ys - cy) ** 2)

        index = np.argmin(distToCentre)

        tx = int(xs[index])
        ty = int(ys[index])

        # Skip targets inside inflated obstacle area
        if planningMap[ty, tx] == 1:
            continue

        distance = np.hypot(
            tx - robotX,
            ty - robotY
        )

        if distance < bestScore:
            bestScore = distance
            bestTarget = (tx, ty)

    return bestTarget

@njit(nogil=True, cache=True)
def astarNumba(planningMap, startX, startY, goalX, goalY):

    height, width = planningMap.shape

    # Check bounds
    if startX < 0 or startX >= width or startY < 0 or startY >= height:
        return np.empty((0, 2), dtype=np.int32)

    if goalX < 0 or goalX >= width or goalY < 0 or goalY >= height:
        return np.empty((0, 2), dtype=np.int32)

    # Start and goal must be free
    if planningMap[startY, startX] == 1:
        return np.empty((0, 2), dtype=np.int32)

    if planningMap[goalY, goalX] == 1:
        return np.empty((0, 2), dtype=np.int32)

    gcostgrid = np.full((height, width), np.inf, dtype=np.float32)
    closed = np.zeros((height, width), dtype=np.uint8)
    parentX = np.full((height, width),-1, dtype=np.int32)
    parentY = np.full((height, width), -1,dtype=np.int32)

    # 8-connected movement
    moveX = np.array([0, 1, 1, -1, 0, -1, -1, 1],dtype=np.int32)
    moveY = np.array([1, 0, 1, 0, -1, -1, 1, -1], dtype=np.int32)
    costs = np.array(
        [
            1.0,
            1.0,
            1.41421356,
            1.0,
            1.0,
            1.41421356,
            1.41421356,
            1.41421356
        ],
        dtype=np.float32)

    gcostgrid[startY, startX] = 0.0
    dx = abs(goalX - startX)
    dy = abs(goalY - startY)

    hcost = (max(dx, dy)+ (1.41421356 - 1.0) * min(dx, dy))
    openList = [(hcost, startX, startY)]
    pathFound = False

    while len(openList) > 0:

        _, currentX, currentY = heapq.heappop(openList)
        if closed[currentY, currentX] == 1:
            continue
        closed[currentY, currentX] = 1

        if currentX == goalX and currentY == goalY:
            pathFound = True
            break

        currentG = gcostgrid[currentY, currentX]

        for i in range(8):

            nextX = currentX + moveX[i]
            nextY = currentY + moveY[i]

            # Bounds
            if (
                nextX < 0 or nextX >= width or
                nextY < 0 or nextY >= height
            ):
                continue

            # Blocked
            if planningMap[nextY, nextX] == 1:
                continue

            # Already closed
            if closed[nextY, nextX] == 1:
                continue

            tentativeG = currentG + costs[i]

            if tentativeG < gcostgrid[nextY, nextX]:

                gcostgrid[nextY, nextX] = tentativeG

                parentX[nextY, nextX] = currentX
                parentY[nextY, nextX] = currentY

                dx = abs(goalX - nextX)
                dy = abs(goalY - nextY)

                hcost = (
                    max(dx, dy)
                    + (1.41421356 - 1.0) * min(dx, dy)
                )

                fcost = tentativeG + hcost

                heapq.heappush(
                    openList,
                    (fcost, nextX, nextY)
                )

    if not pathFound:
        return np.empty((0, 2), dtype=np.int32)

    pathLength = 1

    x = goalX
    y = goalY

    while x != startX or y != startY:

        px = parentX[y, x]
        py = parentY[y, x]

        if px == -1 or py == -1:
            return np.empty((0, 2), dtype=np.int32)

        x = px
        y = py

        pathLength += 1

    path = np.empty(
        (pathLength, 2),
        dtype=np.int32
    )

    x = goalX
    y = goalY

    index = pathLength - 1

    while True:

        path[index, 0] = x
        path[index, 1] = y

        if x == startX and y == startY:
            break

        px = parentX[y, x]
        py = parentY[y, x]

        x = px
        y = py

        index -= 1

    return path

def localAStar(planningMap, start, goal, margin=250):

    startX, startY = start
    goalX, goalY = goal

    minX = max(0, min(startX, goalX) - margin)
    maxX = min(mapSize, max(startX, goalX) + margin + 1)

    minY = max(0, min(startY, goalY) - margin)
    maxY = min(mapSize, max(startY, goalY) + margin + 1)

    localMap = planningMap[minY:maxY, minX:maxX]

    localStartX = startX - minX
    localStartY = startY - minY

    localGoalX = goalX - minX
    localGoalY = goalY - minY

    localPath = astarNumba(
        localMap,
        localStartX,
        localStartY,
        localGoalX,
        localGoalY
    )

    if len(localPath) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Convert back to full-map coordinates
    localPath[:, 0] += minX
    localPath[:, 1] += minY

    return localPath

def simplifyPath(path):

    if path is None or len(path) < 3:
        return path

    simplified = [path[0]]
    previousDirection = None

    for i in range(1, len(path)):

        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        direction = (np.sign(dx), np.sign(dy))

        if previousDirection is None:
            previousDirection = direction

        elif direction != previousDirection:

            # Save point immediately before direction changed
            simplified.append(path[i - 1])

            previousDirection = direction

    # Always include goal
    simplified.append(path[-1])

    return np.asarray(
        simplified,
        dtype=np.int32
    )

def getObstacleDistance(x, y, worldMap, searchRadius=100):

    x = int(round(x))
    y = int(round(y))

    minX = max(0, x - searchRadius)
    maxX = min(mapSize, x + searchRadius + 1)

    minY = max(0, y - searchRadius)
    maxY = min(mapSize, y + searchRadius + 1)

    region = worldMap[minY:maxY, minX:maxX]

    obstaclePoints = np.argwhere(region == 2)

    if len(obstaclePoints) == 0:
        return searchRadius

    robotLocalX = x - minX
    robotLocalY = y - minY

    dy = obstaclePoints[:, 0] - robotLocalY
    dx = obstaclePoints[:, 1] - robotLocalX

    distances = np.sqrt(dx * dx + dy * dy)

    return np.min(distances)
    
def followPath():
    global waypointIndex, explorationComplete

    if explorationComplete:
        return 0.0, 0.0
    
    with pathLock:
        path = currentPath
        localWaypointIndex = waypointIndex

    if path is None or len(path) == 0:
        return 0.0, 0.0

    if localWaypointIndex >= len(path) - 1:
        return 0.0, 0.0

    cornerTolerance = 20.0

    # Skip waypoints already reached
    while localWaypointIndex < len(path) - 1:

        targetIndex = localWaypointIndex + 1
        targetX, targetY = path[targetIndex]

        dx = targetX - robotX
        dy = robotY - targetY

        distanceToTarget = np.hypot(dx, dy)

        if distanceToTarget >= cornerTolerance:
            break

        localWaypointIndex = targetIndex

        with pathLock:
            waypointIndex = localWaypointIndex

    # Final waypoint reached
    if localWaypointIndex >= len(path) - 1:
        return 0.0, 0.0

    # Desired heading to next simplified waypoint
    desiredHeading = np.arctan2(dy, dx)

    headingError = np.arctan2(
        np.sin(desiredHeading - robotAngle),
        np.cos(desiredHeading - robotAngle)
    )
    headingAbs = abs(headingError)
    maxVel = testWheelVel if testMode else maxWheelVel
    obstacleDistance = getObstacleDistance(robotX, robotY, worldMap)
    closeToObstacle = obstacleDistance < 90


    if closeToObstacle and headingAbs > np.deg2rad(15) or headingAbs > np.deg2rad(100):

        # Near obstacles: turn safely on the spot if needed
        turnThreshold = np.deg2rad(15)
        if headingAbs > turnThreshold:
            turnSpeed = 20.0
            if headingError > 0:
                leftWheel = -turnSpeed
                rightWheel = turnSpeed
            else:
                leftWheel = turnSpeed
                rightWheel = -turnSpeed
        else:
            # Once aligned, move forward cautiously
            baseSpeed = 8.0
            steeringGain = 3.0
            correction = steeringGain * headingError
            leftWheel = baseSpeed - correction
            rightWheel = baseSpeed + correction

    else:

        # Open space: keep moving while turning
        forwardScale = np.cos(headingError)
        baseSpeed = maxVel * forwardScale
        steeringGain = 6.0
        correction = steeringGain * headingError
        leftWheel = baseSpeed - correction
        rightWheel = baseSpeed + correction

    largest = max(abs(leftWheel), abs(rightWheel))

    if largest > maxVel:
        scale = maxVel / largest
        leftWheel *= scale
        rightWheel *= scale
    return leftWheel, rightWheel


threading.Thread(target=runWS, daemon=True).start()
threading.Thread(target=plannerLoop, daemon=True).start()
while not wsConnected:
    time.sleep (0.1)

lastDisplay = 0.0
updateTime = time.perf_counter()
lastUpdate = time.perf_counter()
timeInterval = 1/15

try:
    while not explorationComplete:

        currentTime = time.perf_counter()

        if currentTime - updateTime >= timeInterval:

            targetLeftVel, targetRightVel = followPath()
            currentLeftVel = limitVelocity(currentLeftVel, targetLeftVel)
            currentRightVel = limitVelocity(currentRightVel, targetRightVel)

            updateTime += timeInterval

            if wsInstance:
                try:
                    wsInstance.send(f"MOVE {currentLeftVel} {currentRightVel}")
                except Exception as e:
                    print ("Sending failed", e)

        if currentTime - lastDisplay >= 0.1:
            displayMap()
            lastDisplay += 0.1

    duration = time.time() - totaltime
    print("map fully explored, it took : ", round(duration,2), " seconds" )
except KeyboardInterrupt:
    print ("Exiting")
finally:
    if wsInstance:
        wsInstance.close()

    cv2.destroyAllWindows()
