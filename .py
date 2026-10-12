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
import os
import queue

ipAddress = "ws://localhost:9000"
wsConnected = False
wsInstance = None
sensorData = []
leftEncoder = 0
rightEncoder = 0
yawDelta = 0
startrecieved = False
startAngle = 0.0
scriptFolder = os.path.dirname(os.path.abspath(__file__))

currentLeftVel = 0.0
currentRightVel = 0.0
wheeldiameter = 20.0
wheelbase = 100.0
botRadius = 50
clearance = botRadius + 5
planningClearance = botRadius + 20
ticksPerRev = 100
distancePerTick = np.pi * wheeldiameter / ticksPerRev
maxVelChange = 8.0
maxWheelVel = 30
testWheelVel = 80.0
speedPrevLeft = 0
speedPrevRight = 0
speedPrevTime = 0.0

mapSize = 5000
worldMap = np.zeros((mapSize, mapSize), dtype=np.uint8)
frontierMask = np.zeros((mapSize, mapSize), dtype=np.uint8)
frontierTarget = None
updateCounter = 0
explorationComplete = False
noFrontier = 0
mapSaved = False
failedFrontiers = []
failedFrontierRadius = 15
maxFrontierCandidates = 12
maxPointsPerFrontier = 4

planningEvent = threading.Event()
pathLock = threading.Lock()
mapLock = threading.Lock()
dataLock = threading.Lock()
dataQueue = queue.Queue(maxsize=4)
currentPath = np.empty((0, 2), dtype=np.int32)

robotX = 0.0
robotY = 0.0
robotAngle = 0.0
prevLeftEncoder = 0
prevRightEncoder = 0

waypointIndex = 0
waypointTolerance = 8.0
lookAhead = 30
linearGain = 0.8
angularGain = 3.0
now = time.time()
testMode = False

photoPath = os.path.join(scriptFolder, "bot.png")
photo = cv2.imread(photoPath, cv2.IMREAD_UNCHANGED) if os.path.exists(photoPath) else None

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
    """Keep the WebSocket callback lightweight: parse, store, queue, return."""
    global sensorData, startrecieved, leftEncoder, rightEncoder, yawDelta
    global robotX, robotY, robotAngle, startAngle

    data = json.loads(message)

    if data.get("type") == "DEBUG":
        if not startrecieved:
            robotX = mapSize / 2
            robotY = mapSize / 2
            theta = data["theta"]
            startAngle = np.deg2rad(theta)
            robotAngle = startAngle
            print("Starting position:", robotX, robotY, theta)
            startrecieved = True
        return

    if data.get("type") != "DATA" or not startrecieved:
        return

    scan = data.get("sensor", [])
    left = data.get("leftEncoder", 0)
    right = data.get("rightEncoder", 0)
    yaw = data.get("yawDelta", 0)

    # Keep a latest-value snapshot for any other code that needs it.
    with dataLock:
        sensorData = scan
        leftEncoder = left
        rightEncoder = right
        yawDelta = yaw

    # Mapping/odometry can be expensive. Put it on a worker thread so the
    # WebSocket receive thread is never blocked by Numba/OpenCV work.
    packet = (scan, left, right, yaw)
    try:
        dataQueue.put_nowait(packet)
    except queue.Full:
        # If processing falls behind, discard the oldest packet and keep the
        # newest. Encoder readings are cumulative, so odometry still catches up.
        try:
            dataQueue.get_nowait()
        except queue.Empty:
            pass
        try:
            dataQueue.put_nowait(packet)
        except queue.Full:
            pass


def mappingLoop():
    """Process odometry and occupancy-grid updates away from onMessage()."""
    global updateCounter, frontierMask

    while True:
        scan, left, right, yaw = dataQueue.get()

        if len(scan) == 0:
            continue

        sensorArray = np.asarray(scan, dtype=np.float64)

        with mapLock:
            updatePosition(left, right, yaw)
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
                frontierMask = findFrontiersNumba(worldMap)

        # Replan periodically on a separate planner thread.
        if updateCounter % 50 == 0 and not explorationComplete:
            planningEvent.set()


def saveMap():
    """Save the occupancy map once exploration is genuinely complete."""
    global mapSaved

    if mapSaved:
        return

    npyPath = os.path.join(scriptFolder, "binary_map.npy")
    pngPath = os.path.join(scriptFolder, "binary_map.png")

    np.save(npyPath, worldMap)

    # worldMap uses 0=unknown, 1=free, 2=occupied.
    mapImage = np.zeros_like(worldMap, dtype=np.uint8)
    mapImage[worldMap == 1] = 127
    mapImage[worldMap == 2] = 255
    cv2.imwrite(pngPath, mapImage)

    mapSaved = True
    print("Map saved to:", npyPath)
    print("Preview saved to:", pngPath)


def hasMeaningfulFrontiers(frontierCopy, minArea=10):
    """Return True if at least one real frontier cluster remains."""
    numLabels, _, stats, _ = cv2.connectedComponentsWithStats(
        frontierCopy,
        connectivity=8
    )

    for labelIndex in range(1, numLabels):
        if stats[labelIndex, cv2.CC_STAT_AREA] >= minArea:
            return True

    return False


def findNearestPlanningFree(planningMap, x, y, maxRadius=100):
    """Snap a planning point to nearby traversable space if inflation blocks it."""
    x = int(round(x))
    y = int(round(y))

    height, width = planningMap.shape

    if 0 <= x < width and 0 <= y < height and planningMap[y, x] == 0:
        return (x, y)

    for radius in range(1, maxRadius + 1):
        minX = max(0, x - radius)
        maxX = min(width - 1, x + radius)
        minY = max(0, y - radius)
        maxY = min(height - 1, y + radius)

        # Search only the outside ring at this radius.
        for tx in range(minX, maxX + 1):
            if planningMap[minY, tx] == 0:
                return (tx, minY)
            if planningMap[maxY, tx] == 0:
                return (tx, maxY)

        for ty in range(minY + 1, maxY):
            if planningMap[ty, minX] == 0:
                return (minX, ty)
            if planningMap[ty, maxX] == 0:
                return (maxX, ty)

    return None


def plannerLoop():
    global currentPath, waypointIndex, frontierTarget
    global explorationComplete, noFrontier, failedFrontiers

    while True:
        planningEvent.wait()
        planningEvent.clear()

        if explorationComplete:
            continue

        replanStart = time.perf_counter()
        print("Replanning...")

        # Take a snapshot quickly, then release the map lock before the expensive
        # OpenCV/A* work. Mapping and WebSocket reception can continue meanwhile.
        copyStart = time.perf_counter()
        with mapLock:
            mapCopy = worldMap.copy()
            frontierCopy = frontierMask.copy()
            startX = int(round(robotX))
            startY = int(round(robotY))
        copyEnd = time.perf_counter()

        # Only declare completion when the frontier detector itself says there
        # are no meaningful frontier clusters left. A failed/blacklisted target
        # is NOT the same as a fully explored map.
        if not hasMeaningfulFrontiers(frontierCopy):
            noFrontier += 1
            print(f"No meaningful frontier clusters ({noFrontier}/3)")

            if noFrontier >= 3:
                explorationComplete = True
                with pathLock:
                    currentPath = np.empty((0, 2), dtype=np.int32)
                    waypointIndex = 0
                    frontierTarget = None

                print("Exploration complete")
                print("Total mapping time:", round(time.time() - now, 2), "seconds")
                saveMap()
            continue

        noFrontier = 0

        planningMapStart = time.perf_counter()
        planningMap = createPlanningMap(mapCopy)
        planningMapEnd = time.perf_counter()

        # Obstacle inflation/noise can occasionally put the robot's exact map
        # cell inside a blocked planning cell. Snap to nearby known free space.
        planningStart = findNearestPlanningFree(
            planningMap,
            startX,
            startY,
            maxRadius=planningClearance + 40
        )

        if planningStart is None:
            print("Could not find traversable planning cell near robot")
            failedFrontiers.clear()
            continue

        candidateStart = time.perf_counter()
        candidates = getFrontierCandidates(
            frontierCopy,
            planningMap,
            planningStart[0],
            planningStart[1]
        )
        candidateEnd = time.perf_counter()

        if len(candidates) == 0:
            print("Frontiers exist, but all candidate cells are currently blocked/blacklisted")
            noFrontier += 1

            # The map changes as new scans arrive, so a previously failed target
            # can later become reachable. Periodically forget the blacklist.
            if noFrontier >= 2:
                print("Clearing failed-frontier list and trying again on a later replan")
                failedFrontiers.clear()
                noFrontier = 0
            continue

        chosenPath = None
        chosenTarget = None
        astarTotal = 0.0

        # Try several frontier candidates during this SAME planning cycle.
        # One unreachable frontier should not stop exploration.
        for target in candidates[:maxFrontierCandidates]:
            astarStart = time.perf_counter()
            candidatePath = localAStar(
                planningMap,
                planningStart,
                target
            )
            astarTotal += time.perf_counter() - astarStart

            if len(candidatePath) > 0:
                chosenPath = candidatePath
                chosenTarget = target
                break

            print("A* could not reach target", target)
            failedFrontiers.append(target)

        if chosenPath is None:
            print("No reachable frontier among", min(len(candidates), maxFrontierCandidates), "candidates")
            noFrontier += 1

            if noFrontier >= 2:
                print("Clearing failed-frontier list so changed map geometry can be retried")
                failedFrontiers.clear()
                noFrontier = 0
            continue

        noFrontier = 0
        newPath = simplifyPath(chosenPath)

        # Only replace the old path after a valid replacement is completely ready.
        with pathLock:
            currentPath = newPath
            waypointIndex = 0
            frontierTarget = chosenTarget

        totalEnd = time.perf_counter()
        print("New path:", len(newPath), "points ->", chosenTarget)
        print(
            "Planning times | copy:", round(copyEnd - copyStart, 3),
            "planning map:", round(planningMapEnd - planningMapStart, 3),
            "frontiers:", round(candidateEnd - candidateStart, 3),
            "A* total:", round(astarTotal, 3),
            "total:", round(totalEnd - replanStart, 3), "seconds"
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
    """Render a 800x800 preview instead of building a 5000x5000 RGB frame."""
    global currentRightVel, currentLeftVel, testMode, robotAngle

    displaySize = 800
    scale = displaySize / mapSize

    # Downsample the integer map first, then colour only the smaller image.
    mapSmall = cv2.resize(worldMap, (displaySize, displaySize), interpolation=cv2.INTER_NEAREST)
    frontierSmall = cv2.resize(frontierMask, (displaySize, displaySize), interpolation=cv2.INTER_NEAREST)

    display = np.zeros((displaySize, displaySize, 3), dtype=np.uint8)
    display[mapSmall == 0] = (30, 30, 30)       # unknown
    display[mapSmall == 1] = (255, 255, 255)    # free
    display[mapSmall == 2] = (0, 0, 255)        # obstacle
    display[frontierSmall == 1] = (255, 0, 0)   # frontier

    with pathLock:
        pathToDisplay = currentPath.copy()
        targetToDisplay = frontierTarget

    if pathToDisplay is not None and len(pathToDisplay) > 1:
        pts = np.asarray(pathToDisplay, dtype=np.float64)
        pts[:, 0] *= scale
        pts[:, 1] *= scale
        pts = np.rint(pts).astype(np.int32)
        cv2.polylines(display, [pts.reshape(-1, 1, 2)], False, (255, 0, 255), 2)

    if targetToDisplay is not None:
        tx = int(targetToDisplay[0] * scale)
        ty = int(targetToDisplay[1] * scale)
        cv2.circle(display, (tx, ty), 5, (0, 255, 255), -1)

    rx = int(robotX * scale)
    ry = int(robotY * scale)
    botRadiusDisplay = max(3, int(botRadius * scale))
    cv2.circle(display, (rx, ry), botRadiusDisplay, (0, 255, 0), -1)

    arrowLength = max(12, int(100 * scale))
    endPoint = (
        int(rx + arrowLength * np.cos(robotAngle)),
        int(ry - arrowLength * np.sin(robotAngle))
    )
    cv2.line(display, (rx, ry), endPoint, (0, 0, 0), 2)

    cv2.putText(display, f"Left wheel: {currentLeftVel:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display, f"Right wheel: {currentRightVel:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display, f"Heading: {np.rad2deg(robotAngle):.2f} degrees", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Robot Map", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        testMode = not testMode
        print("Super Duper fast" if testMode else "normal speed")



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

def getFrontierCandidates(frontierMask, planningMap, robotX, robotY):
    """Return multiple viable frontier points ordered from nearest to farthest.

    Several well-separated points are sampled from each frontier cluster. This
    avoids treating an entire frontier as unreachable just because its centroid
    (or one representative cell) cannot be reached by A*.
    """
    global failedFrontiers

    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frontierMask,
        connectivity=8
    )

    candidates = []

    for labelIndex in range(1, numLabels):
        area = stats[labelIndex, cv2.CC_STAT_AREA]
        if area < 10:
            continue

        left = stats[labelIndex, cv2.CC_STAT_LEFT]
        top = stats[labelIndex, cv2.CC_STAT_TOP]
        width = stats[labelIndex, cv2.CC_STAT_WIDTH]
        height = stats[labelIndex, cv2.CC_STAT_HEIGHT]

        # Work only inside this component's bounding box instead of calling
        # np.where() over the entire 5000x5000 label image for every component.
        labelCrop = labels[top:top + height, left:left + width]
        ysLocal, xsLocal = np.where(labelCrop == labelIndex)
        if len(xsLocal) == 0:
            continue

        xs = xsLocal + left
        ys = ysLocal + top

        cx, cy = centroids[labelIndex]
        order = np.argsort((xs - cx) ** 2 + (ys - cy) ** 2)

        pointsFromCluster = []

        for index in order:
            tx = int(xs[index])
            ty = int(ys[index])

            # Frontier target must itself be traversable on the inflated map.
            if planningMap[ty, tx] != 0:
                continue

            # Don't retry immediately next to a point already proven unreachable.
            if any(
                np.hypot(tx - fx, ty - fy) < failedFrontierRadius
                for fx, fy in failedFrontiers
            ):
                continue

            # Spread samples over the cluster rather than returning near-duplicates.
            if any(
                np.hypot(tx - px, ty - py) < 20.0
                for px, py in pointsFromCluster
            ):
                continue

            pointsFromCluster.append((tx, ty))

            distance = np.hypot(tx - robotX, ty - robotY)
            candidates.append((distance, (tx, ty)))

            if len(pointsFromCluster) >= maxPointsPerFrontier:
                break

    candidates.sort(key=lambda item: item[0])
    return [target for _, target in candidates]

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

    gcostgrid = np.full(
        (height, width),
        np.inf,
        dtype=np.float32
    )

    closed = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    parentX = np.full(
        (height, width),
        -1,
        dtype=np.int32
    )

    parentY = np.full(
        (height, width),
        -1,
        dtype=np.int32
    )

    # 8-connected movement
    moveX = np.array(
        [0, 1, 1, -1, 0, -1, -1, 1],
        dtype=np.int32
    )

    moveY = np.array(
        [1, 0, 1, 0, -1, -1, 1, -1],
        dtype=np.int32
    )

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
        dtype=np.float32
    )

    gcostgrid[startY, startX] = 0.0

    dx = abs(goalX - startX)
    dy = abs(goalY - startY)

    hcost = (
        max(dx, dy)
        + (1.41421356 - 1.0) * min(dx, dy)
    )

    # Numba supports heapq.
    # Seed it immediately so the heap type can be inferred.
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

def localAStar(planningMap, start, goal):

    margins = [250, 500, 1000]

    startX, startY = start
    goalX, goalY = goal

    for margin in margins:

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

        if len(localPath) > 0:

            localPath[:, 0] += minX
            localPath[:, 1] += minY

            return localPath

    return np.empty((0, 2), dtype=np.int32)

def simplifyPath(path):

    if path is None or len(path) < 3:
        return path

    simplified = [path[0]]

    previousDirection = None

    for i in range(1, len(path)):

        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        direction = (
            np.sign(dx),
            np.sign(dy)
        )

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
    
def closestPointOnSegment(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    lengthSquared = abx * abx + aby * aby
    if lengthSquared < 1e-9:
        return ax, ay, 0.0

    t = ((px - ax) * abx + (py - ay) * aby) / lengthSquared
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby

    # Signed cross-track error in map coordinates.
    crossTrack = (abx * (py - ay) - aby * (px - ax)) / np.sqrt(lengthSquared)
    return cx, cy, crossTrack


def findClosestPathPoint(path, px, py):
    if path is None or len(path) == 0:
        return None
    if len(path) == 1:
        return 0, float(path[0][0]), float(path[0][1]), 0.0

    best = None
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        cx, cy, crossTrack = closestPointOnSegment(px, py, ax, ay, bx, by)
        distance = np.hypot(px - cx, py - cy)
        if best is None or distance < best[0]:
            best = (distance, i, cx, cy, crossTrack)

    if best is None:
        return None

    _, segmentIndex, cx, cy, crossTrack = best
    return segmentIndex, cx, cy, crossTrack


def pickLookAheadPoint(path, distanceAhead, segmentIndex, closestX, closestY):
    remaining = float(distanceAhead)
    cx = float(closestX)
    cy = float(closestY)

    for i in range(segmentIndex, len(path) - 1):
        if i == segmentIndex:
            ax, ay = cx, cy
        else:
            ax, ay = float(path[i][0]), float(path[i][1])

        bx, by = float(path[i + 1][0]), float(path[i + 1][1])
        segmentLength = np.hypot(bx - ax, by - ay)

        if segmentLength >= remaining:
            t = remaining / segmentLength if segmentLength > 1e-9 else 0.0
            return ax + t * (bx - ax), ay + t * (by - ay)

        remaining -= segmentLength
        cx, cy = bx, by

    return float(path[-1][0]), float(path[-1][1])


def followPath():
    """Smooth look-ahead path follower with continuous steering commands."""
    global waypointIndex, explorationComplete

    if explorationComplete:
        return 0.0, 0.0

    with pathLock:
        path = currentPath.copy()

    if path is None or len(path) < 2:
        return 0.0, 0.0

    # Stop if the final target is already reached.
    finalDistance = np.hypot(
        float(path[-1][0]) - robotX,
        float(path[-1][1]) - robotY
    )

    # Start calculating the next path before this one ends
    if finalDistance < 150.0 and not explorationComplete:
        planningEvent.set()

    # Only stop when extremely close to final point
    if finalDistance < 5.0:
        return 0.0, 0.0

    closest = findClosestPathPoint(path, robotX, robotY)
    if closest is None:
        return 0.0, 0.0

    segmentIndex, closestX, closestY, crossTrack = closest
    waypointIndex = segmentIndex

    targetX, targetY = pickLookAheadPoint(
        path, lookAhead, segmentIndex, closestX, closestY
    )

    # Map y increases downward, while positive robotAngle points upward.
    dx = targetX - robotX
    dy = robotY - targetY
    desiredHeading = np.arctan2(dy, dx)

    headingError = np.arctan2(
        np.sin(desiredHeading - robotAngle),
        np.cos(desiredHeading - robotAngle)
    )

    maxVel = testWheelVel if testMode else maxWheelVel

    # Forward speed fades smoothly as heading error grows. This removes the
    # abrupt drive/turn mode switching that made the old controller jerky.
    alignment = max(0.0, np.cos(headingError))

    obstacleDistance = getObstacleDistance(robotX, robotY, worldMap)
    obstacleScale = 1.0
    if obstacleDistance < 90.0:
        obstacleScale = max(0.35, obstacleDistance / 90.0)

    baseSpeed = maxVel * alignment * obstacleScale

    # Continuous proportional steering. At ~90 degrees error the differential
    # contribution is roughly maxVel, giving a smooth transition into a turn.
    steeringGain = maxVel / (np.pi / 2.0)
    correction = steeringGain * headingError

    # Small cross-track correction nudges the bot back toward the path.
    correction += 0.03 * crossTrack
    correction = np.clip(correction, -maxVel, maxVel)

    leftWheel = baseSpeed - correction
    rightWheel = baseSpeed + correction

    largest = max(abs(leftWheel), abs(rightWheel))
    if largest > maxVel:
        scale = maxVel / largest
        leftWheel *= scale
        rightWheel *= scale

    return float(leftWheel), float(rightWheel)



threading.Thread(target=runWS, daemon=True).start()
threading.Thread(target=mappingLoop, daemon=True).start()
threading.Thread(target=plannerLoop, daemon=True).start()
while not wsConnected:
    time.sleep (0.1)

updateTime = time.perf_counter()
lastDisplay = updateTime
timeInterval = 1/15

try:
    while True:

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

        # Avoid a 100% CPU busy-loop while keeping the 15 Hz controller precise.
        time.sleep(0.001)
except KeyboardInterrupt:
    print ("Exiting")
finally:
    try:
        sendStop = wsInstance is not None
        if sendStop:
            wsInstance.send("MOVE 0 0")
    except Exception:
        pass

    if wsInstance:
        wsInstance.close()

    cv2.destroyAllWindows()