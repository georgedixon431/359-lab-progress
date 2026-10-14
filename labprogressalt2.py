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

# Normal vs narrow-passage planning/safety settings.
normalPlanningClearance = botRadius + 45
narrowPlanningClearance = botRadius
planningClearance = normalPlanningClearance

normalSafetyClearance = botRadius + 50
narrowSafetyClearance = botRadius
safetyClearance = normalSafetyClearance

# The motion controller has its own obstacle stop/slow distances. These must
# also reduce in narrow mode, otherwise A* can find a path that the controller
# refuses to physically drive through.
normalHardStopDistance = 60.0
normalSlowDistance = 120.0
narrowHardStopDistance = float(botRadius)
narrowSlowDistance = float(botRadius + 20)
hardStopDistance = normalHardStopDistance
slowDistance = normalSlowDistance

narrowClearanceMode = False
ticksPerRev = 100
distancePerTick = np.pi * wheeldiameter / ticksPerRev
maxVelChange = 8.0
maxWheelVel = 30
testWheelVel = 80.0
speedPrevLeft = 0
speedPrevRight = 0
speedPrevTime = 0.0

mapSize = 5000

# Start every run with a fresh, empty occupancy map.
worldMap = np.zeros((mapSize, mapSize), dtype=np.uint8)

frontierMask = np.zeros((mapSize, mapSize), dtype=np.uint8)
frontierTarget = None
updateCounter = 0
explorationComplete = False
noFrontier = 0
mapSaved = False
failedFrontiers = []
failedFrontierRadius = 15

# Frontiers already successfully approached.
# Keep this radius deliberately small so nearby corners/branches can still be
# selected and checked independently.
visitedFrontiers = []
visitedFrontierRadius = 25
currentFrontier = None

# Semi-manual controls
manualFrontierRequest = None   # map (x, y) selected by mouse
manualMode = False             # press M, then click desired frontier
mouseMapPosition = None        # cursor position shown in map coordinates
manualTargetLock = threading.Lock()
manualSearchRadius = 180       # map pixels around click to snap to a real frontier

# Keep the current exploration objective unless it is reached, disappears,
# or becomes unreachable. This prevents back-and-forth target switching.
minimumNewTargetDistance = 50.0

# Stuck-prevention settings are defined with the clearance-mode settings above.

maxFrontierCandidates = 12
maxPointsPerFrontier = 4

# Frontier approach settings:
# Drive to a safe known-free point near a frontier rather than onto the frontier.
frontierApproachMin = 45
frontierApproachMax = 120

planningEvent = threading.Event()
pathLock = threading.Lock()
mapLock = threading.Lock()
dataLock = threading.Lock()
dataQueue = queue.Queue(maxsize=4)
currentPath = np.empty((0, 2), dtype=np.int32)

robotX = mapSize / 2.0
robotY = mapSize / 2.0
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
            debugX = data["x"]
            robotY = mapSize / 2
            debugY = data["y"]
            theta = data["theta"]
            startAngle = np.deg2rad(theta)
            robotAngle = startAngle
            print("Starting position:", debugX, debugY, theta)
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




def saveMap(force=False):
    """Save the occupancy map once exploration is genuinely complete."""
    global mapSaved

    if mapSaved and not force:
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


def findNearestFrontierToPoint(frontierCopy, x, y, searchRadius=180):
    """Return the active frontier cell nearest a manually selected map point."""
    x = int(round(x))
    y = int(round(y))
    height, width = frontierCopy.shape

    minX = max(0, x - searchRadius)
    maxX = min(width - 1, x + searchRadius)
    minY = max(0, y - searchRadius)
    maxY = min(height - 1, y + searchRadius)

    crop = frontierCopy[minY:maxY + 1, minX:maxX + 1]
    ys, xs = np.where(crop == 1)
    if len(xs) == 0:
        return None

    xs = xs + minX
    ys = ys + minY
    distances = (xs - x) ** 2 + (ys - y) ** 2
    i = int(np.argmin(distances))
    return (int(xs[i]), int(ys[i]))


def onMapMouse(event, x, y, flags, param):
    """Track cursor coordinates and allow manual frontier selection by click."""
    global mouseMapPosition, manualFrontierRequest, manualMode

    displaySize = 800
    scale = displaySize / mapSize
    mapX = int(round(x / scale))
    mapY = int(round(y / scale))
    mapX = int(np.clip(mapX, 0, mapSize - 1))
    mapY = int(np.clip(mapY, 0, mapSize - 1))
    mouseMapPosition = (mapX, mapY)

    if event == cv2.EVENT_LBUTTONDOWN and manualMode:
        with manualTargetLock:
            manualFrontierRequest = (mapX, mapY)
        manualMode = False
        print('Manual frontier click requested near:', (mapX, mapY))
        planningEvent.set()

def plannerLoop():
    global currentPath, waypointIndex, frontierTarget, currentFrontier
    global explorationComplete, noFrontier, failedFrontiers, manualFrontierRequest

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
                    currentFrontier = None

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

        chosenPath = None
        chosenGoal = None
        chosenFrontier = None
        astarTotal = 0.0

        # --------------------------------------------------------------
        # MANUAL FRONTIER OVERRIDE
        # Press M in the map window, then left-click the frontier you want.
        # The click is snapped to the nearest active frontier and then the
        # normal safe-approach + A* pipeline is used.
        # --------------------------------------------------------------
        with manualTargetLock:
            manualRequest = manualFrontierRequest
            manualFrontierRequest = None

        if manualRequest is not None:
            requestedX, requestedY = manualRequest
            manualFrontier = findNearestFrontierToPoint(
                frontierCopy, requestedX, requestedY, manualSearchRadius
            )

            if manualFrontier is None:
                print('No active frontier found near manual click:', manualRequest)
            else:
                manualGoal = findSafeApproachGoal(
                    planningMap,
                    manualFrontier[0],
                    manualFrontier[1],
                    planningStart[0],
                    planningStart[1]
                )

                if manualGoal is None:
                    print('No safe approach goal for manually selected frontier:', manualFrontier)
                else:
                    astarStart = time.perf_counter()
                    manualPath = localAStar(
                        planningMap, planningStart, manualGoal
                    )
                    astarTotal += time.perf_counter() - astarStart

                    if len(manualPath) > 0:
                        chosenPath = manualPath
                        chosenGoal = manualGoal
                        chosenFrontier = manualFrontier

                        # Manual selection intentionally overrides previous
                        # visited/failed history for this objective.
                        currentFrontier = manualFrontier
                        frontierTarget = manualGoal
                        frontierProgressKey = None
                        frontierBestDistance = float('inf')
                        frontierNoProgressPlans = 0

                        print(
                            'MANUAL frontier accepted:', manualFrontier,
                            'approach goal:', manualGoal
                        )
                    else:
                        print(
                            'A* could not reach manually selected frontier',
                            manualFrontier, 'via', manualGoal
                        )

        # --------------------------------------------------------------
        # TARGET PERSISTENCE:
        # Keep working toward the currently selected frontier if that
        # frontier still exists and remains reachable. A periodic replan
        # should update the route, not change the exploration objective.
        # --------------------------------------------------------------
        persistentFrontier = None
        if chosenPath is None:
            persistentFrontier = findNearbyActiveFrontier(
            frontierCopy,
            currentFrontier,
            searchRadius=visitedFrontierRadius
        )

        if persistentFrontier is not None:
            persistentGoal = findSafeApproachGoal(
                planningMap,
                persistentFrontier[0],
                persistentFrontier[1],
                planningStart[0],
                planningStart[1]
            )

            if persistentGoal is not None:
                astarStart = time.perf_counter()
                persistentPath = localAStar(
                    planningMap,
                    planningStart,
                    persistentGoal
                )
                astarTotal += time.perf_counter() - astarStart

                if len(persistentPath) > 0:
                    chosenPath = persistentPath
                    chosenGoal = persistentGoal
                    chosenFrontier = persistentFrontier
                    print("Keeping current frontier:", persistentFrontier)
                else:
                    print("Current frontier became unreachable:", persistentFrontier)
                    failedFrontiers.append(persistentFrontier)

        # Only choose a new objective when the old one disappeared or became
        # unreachable.
        candidates = []
        if chosenPath is None:
            candidates = getFrontierCandidates(
                frontierCopy,
                planningMap,
                planningStart[0],
                planningStart[1]
            )

            for frontierPoint, approachGoal in candidates[:maxFrontierCandidates]:
                astarStart = time.perf_counter()
                candidatePath = localAStar(planningMap, planningStart,approachGoal)
                astarTotal += time.perf_counter() - astarStart

                if len(candidatePath) > 0:
                    chosenPath = candidatePath
                    chosenGoal = approachGoal
                    chosenFrontier = frontierPoint
                    break

                print("A* could not reach approach goal", approachGoal, "for frontier", frontierPoint)
                failedFrontiers.append(frontierPoint)

        candidateEnd = time.perf_counter()

        if chosenPath is None:
            if len(candidates) == 0:
                print("Frontiers exist, but no safe approach goals are currently available")
            else:
                print("No reachable safe approach goal among",min(len(candidates), maxFrontierCandidates),"candidates")
            noFrontier += 1
            if failedFrontiers and noFrontier >= 2:
                print("Clearing temporary failed-frontier list")
                failedFrontiers.clear()
                noFrontier = 0

            continue

        noFrontier = 0

        # Use a slightly more conservative map only when removing waypoints.
        # This prevents the simplified straight segments from cutting corners.
        safetyMap = createSafetyMap(mapCopy)
        newPath = simplifyPath(chosenPath, safetyMap)

        # Only replace the old path after a valid replacement is completely ready.
        with pathLock:
            currentPath = newPath
            waypointIndex = 0
            frontierTarget = chosenGoal
            currentFrontier = chosenFrontier

        totalEnd = time.perf_counter()
        print(
            "New path:",
            len(newPath),
            "points -> approach",
            chosenGoal,
            "for frontier",
            chosenFrontier
        )
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


def createSafetyMap(mapData):
    """Slightly more conservative map used only for path simplification.

    A* still plans with planningClearance so narrow valid routes are not thrown
    away, but shortcutting/simplification must obey the larger safety margin.
    """
    obstacleMask = (mapData == 2).astype(np.uint8)

    kernelSize = 2 * safetyClearance + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernelSize, kernelSize)
    )

    inflatedObstacles = cv2.dilate(obstacleMask, kernel)

    safetyMap = np.zeros_like(mapData, dtype=np.uint8)
    safetyMap[mapData == 0] = 1
    safetyMap[inflatedObstacles == 1] = 1

    safetyMap[:safetyClearance, :] = 1
    safetyMap[-safetyClearance:, :] = 1
    safetyMap[:, :safetyClearance] = 1
    safetyMap[:, -safetyClearance:] = 1

    return safetyMap


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
    global manualMode, mouseMapPosition
    global narrowClearanceMode, planningClearance, safetyClearance
    global hardStopDistance, slowDistance

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

    cv2.putText(
        display,
        f"Robot map XY: ({int(robotX)}, {int(robotY)})",
        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
    )

    if mouseMapPosition is not None:
        cv2.putText(
            display,
            f"Cursor map XY: {mouseMapPosition}",
            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

    modeText = "MANUAL SELECT: click frontier" if manualMode else "M: select frontier   S: save map   C: clearance"
    cv2.putText(
        display, modeText, (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 255) if manualMode else (0, 0, 255), 2
    )

    #clearanceText = (f"NARROW DEBUG: plan {planningClearance}px  controller stop OFF" if narrowClearanceMode else f"NORMAL: plan {planningClearance}px  stop {hardStopDistance:.0f}px")
    #cv2.putText(display, clearanceText, (20, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0, 255, 255) if narrowClearanceMode else (0, 0, 255), 2)

    cv2.imshow("Robot Map", display)
    cv2.setMouseCallback("Robot Map", onMapMouse)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        testMode = not testMode
        print("Super Duper fast" if testMode else "normal speed")
    elif key == ord('m'):
        manualMode = True
        print("Manual frontier mode: click the desired frontier in the Robot Map window")
    elif key == ord('s'):
        saveMap(force=True)
    elif key == ord('c'):
        narrowClearanceMode = not narrowClearanceMode

        if narrowClearanceMode:
            planningClearance = narrowPlanningClearance
            safetyClearance = narrowSafetyClearance
            hardStopDistance = narrowHardStopDistance
            slowDistance = narrowSlowDistance
            print("NARROW PASSAGE MODE")
        else:
            planningClearance = normalPlanningClearance
            safetyClearance = normalSafetyClearance
            hardStopDistance = normalHardStopDistance
            slowDistance = normalSlowDistance
            print("NORMAL CLEARANCE MODE")

        print("Planning clearance:", planningClearance)
        print("Safety clearance:", safetyClearance)
        print("Hard stop distance:", hardStopDistance)
        print("Slow distance:", slowDistance)

        # Replan immediately using the new obstacle inflation distance.
        planningEvent.set()



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

def findNearbyActiveFrontier(frontierMask, frontierPoint, searchRadius=25):
    """Return a nearby still-active frontier cell, or None if it disappeared.

    Frontier pixels can move slightly as the map grows, so exact-coordinate
    matching is too brittle for target persistence.
    """
    if frontierPoint is None:
        return None

    fx, fy = frontierPoint
    height, width = frontierMask.shape

    minX = max(0, int(fx - searchRadius))
    maxX = min(width - 1, int(fx + searchRadius))
    minY = max(0, int(fy - searchRadius))
    maxY = min(height - 1, int(fy + searchRadius))

    crop = frontierMask[minY:maxY + 1, minX:maxX + 1]
    ys, xs = np.where(crop == 1)

    if len(xs) == 0:
        return None

    xs = xs + minX
    ys = ys + minY

    distances = (xs - fx) ** 2 + (ys - fy) ** 2
    index = int(np.argmin(distances))

    return (int(xs[index]), int(ys[index]))


def findSafeApproachGoal(planningMap, frontierX, frontierY, robotX, robotY):
    """Find a safe known-free goal near a frontier.

    The bot does not need to sit on the frontier cell itself. It only needs to get
    close enough for lidar to reveal more unknown space.
    """
    height, width = planningMap.shape

    minX = max(0, int(frontierX - frontierApproachMax))
    maxX = min(width - 1, int(frontierX + frontierApproachMax))
    minY = max(0, int(frontierY - frontierApproachMax))
    maxY = min(height - 1, int(frontierY + frontierApproachMax))

    bestGoal = None
    bestScore = float("inf")

    # Sample every 4 px to keep this cheap.
    for y in range(minY, maxY + 1, 4):
        for x in range(minX, maxX + 1, 4):
            if planningMap[y, x] != 0:
                continue

            frontierDistance = np.hypot(x - frontierX, y - frontierY)

            if frontierDistance < frontierApproachMin:
                continue
            if frontierDistance > frontierApproachMax:
                continue

            robotDistance = np.hypot(x - robotX, y - robotY)

            # Mostly prefer closeness to the frontier, with a small preference for
            # points on the robot's side of the frontier.
            score = frontierDistance + 0.05 * robotDistance

            if score < bestScore:
                bestScore = score
                bestGoal = (x, y)

    return bestGoal


def getFrontierCandidates(frontierMask, planningMap, robotX, robotY):
    """Return (frontierPoint, safeApproachGoal) pairs ordered by usefulness."""
    global failedFrontiers, visitedFrontiers

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
            fx = int(xs[index])
            fy = int(ys[index])

            # A frontier cell itself may be blocked by inflation. That is okay:
            # the robot will drive to a safe approach point instead.
            if any(
                np.hypot(fx - failedX, fy - failedY) < failedFrontierRadius
                for failedX, failedY in failedFrontiers
            ):
                continue

            # Skip only a small neighbourhood around a frontier that has already
            # been successfully approached. 25 px is intentionally conservative:
            # it prevents repeatedly selecting the exact same frontier without
            # masking nearby corners or separate branches.
            if any(
                np.hypot(fx - visitedX, fy - visitedY) < visitedFrontierRadius
                for visitedX, visitedY in visitedFrontiers
            ):
                continue

            if any(
                np.hypot(fx - oldX, fy - oldY) < 20.0
                for oldX, oldY in pointsFromCluster
            ):
                continue

            approachGoal = findSafeApproachGoal(
                planningMap,
                fx,
                fy,
                robotX,
                robotY
            )

            if approachGoal is None:
                continue

            pointsFromCluster.append((fx, fy))

            gx, gy = approachGoal
            robotToGoal = np.hypot(gx - robotX, gy - robotY)

            # When choosing a NEW objective, ignore tiny nearby frontier fragments.
            # Existing active frontiers are handled separately by target persistence.
            if robotToGoal < minimumNewTargetDistance:
                continue

            # Prefer closer goals, but also favour larger frontier clusters.
            score = robotToGoal / max(1.0, np.sqrt(float(area)))
            candidates.append((score, (fx, fy), approachGoal))

            if len(pointsFromCluster) >= maxPointsPerFrontier:
                break

    candidates.sort(key=lambda item: item[0])

    return [
        (frontierPoint, approachGoal)
        for _, frontierPoint, approachGoal in candidates
    ]

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

def lineIsClear(planningMap, p1, p2):
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    x, y = x0, y0

    while True:

        if (
            x < 0 or x >= planningMap.shape[1]
            or y < 0 or y >= planningMap.shape[0]
        ):
            return False

        if planningMap[y, x] == 1:
            return False

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return True

def simplifyPath(path, planningMap):

    if path is None or len(path) < 3:
        return path

    simplified = [path[0]]

    anchor = 0

    while anchor < len(path) - 1:

        furthest = anchor + 1

        for i in range(anchor + 2, len(path)):

            if lineIsClear(
                planningMap,
                path[anchor],
                path[i]
            ):
                furthest = i
            else:
                break

        simplified.append(path[furthest])
        anchor = furthest

    return np.asarray(simplified, dtype=np.int32)

def getObstacleDistances(x, y, worldMap, robotAngle, searchRadius=120, forwardHalfAngle=np.deg2rad(50.0)):
    """Return (nearest obstacle distance, forward-cone obstacle distance).

    The old controller used the nearest obstacle in ANY direction. In a narrow
    passage that meant a wall or circular obstacle beside the robot could force
    forward speed to zero even when the requested path led safely away from it.
    The forward distance is therefore used for normal stop/slow behaviour, while
    the all-around nearest distance is kept only as an emergency collision check.
    """
    x = int(round(x))
    y = int(round(y))

    minX = max(0, x - searchRadius)
    maxX = min(mapSize, x + searchRadius + 1)
    minY = max(0, y - searchRadius)
    maxY = min(mapSize, y + searchRadius + 1)

    region = worldMap[minY:maxY, minX:maxX]
    obstaclePoints = np.argwhere(region == 2)

    if len(obstaclePoints) == 0:
        return float(searchRadius), float(searchRadius)

    robotLocalX = x - minX
    robotLocalY = y - minY

    dyMap = obstaclePoints[:, 0] - robotLocalY
    dx = obstaclePoints[:, 1] - robotLocalX
    distances = np.sqrt(dx * dx + dyMap * dyMap)

    nearestDistance = float(np.min(distances))

    # Map y increases downward, so negate dy when converting to robot/world angle.
    obstacleAngles = np.arctan2(-dyMap, dx)
    angleDifferences = np.abs(np.arctan2(
        np.sin(obstacleAngles - robotAngle),
        np.cos(obstacleAngles - robotAngle)
    ))

    forwardMask = angleDifferences <= forwardHalfAngle
    if np.any(forwardMask):
        forwardDistance = float(np.min(distances[forwardMask]))
    else:
        forwardDistance = float(searchRadius)

    return nearestDistance, forwardDistance
    
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
    global waypointIndex, explorationComplete, currentFrontier
    global visitedFrontiers

    if explorationComplete:
        return 0.0, 0.0

    with pathLock:
        path = currentPath.copy()
        frontierForPath = currentFrontier

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

    # Once the safe approach goal is reached, remember the ACTUAL frontier so
    # the next planning cycle does not keep selecting the same place forever.
    # The visited radius is only 25 px, so nearby corners remain eligible.
    if finalDistance < 5.0:
        if frontierForPath is not None:
            alreadyVisited = any(
                np.hypot(
                    frontierForPath[0] - visitedX,
                    frontierForPath[1] - visitedY
                ) < visitedFrontierRadius
                for visitedX, visitedY in visitedFrontiers
            )

            if not alreadyVisited:
                visitedFrontiers.append(frontierForPath)
                print("Visited frontier:", frontierForPath)

        # The objective has been completed, so the next planner cycle may choose
        # a new frontier.
        with pathLock:
            currentFrontier = None

        planningEvent.set()
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

    nearestObstacleDistance, forwardObstacleDistance = getObstacleDistances(
        robotX,
        robotY,
        worldMap,
        robotAngle,
        searchRadius=int(max(slowDistance, 120.0))
    )

    # DEBUG MODE FOR NARROW PASSAGES:
    # When narrowClearanceMode is ON, disable ALL controller-level obstacle
    # stopping/slowing so we can test whether the simulator/server itself allows
    # the robot through the gap. A* still uses the narrow planning clearance.
    if narrowClearanceMode:
        obstacleScale = 1.0
        baseSpeed = maxVel * alignment

        steeringGain = maxVel / (np.pi / 2.0)
        correction = steeringGain * headingError
        correction += 0.03 * crossTrack
        correction = np.clip(correction, -maxVel, maxVel)

        leftWheel = baseSpeed - correction
        rightWheel = baseSpeed + correction

        print(
            f"NARROW DEBUG | nearest={nearestObstacleDistance:.1f} "
            f"forward={forwardObstacleDistance:.1f} "
            f"headingErr={np.rad2deg(headingError):.1f} "
            f"cmd=({leftWheel:.2f},{rightWheel:.2f})"
        )
    else:
        # Normal mode keeps the safety controller active.
        emergencyDistance = max(1.0, botRadius - 5.0)

        if nearestObstacleDistance <= emergencyDistance:
            obstacleScale = 0.0
        elif forwardObstacleDistance <= hardStopDistance:
            obstacleScale = 0.0
        elif forwardObstacleDistance < slowDistance:
            obstacleScale = (
                (forwardObstacleDistance - hardStopDistance)
                / (slowDistance - hardStopDistance)
            )
        else:
            obstacleScale = 1.0

        baseSpeed = maxVel * alignment * obstacleScale

        steeringGain = maxVel / (np.pi / 2.0)

        if nearestObstacleDistance < slowDistance:
            steeringGain *= 0.6

        correction = steeringGain * headingError
        correction += 0.03 * crossTrack
        correction = np.clip(correction, -maxVel, maxVel)

        leftWheel = baseSpeed - correction
        rightWheel = baseSpeed + correction

        if (nearestObstacleDistance <= emergencyDistance
                or forwardObstacleDistance <= hardStopDistance):
            leftWheel = -correction
            rightWheel = correction

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