<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=100&lines=%F0%9F%A4%96+RoboCallee+%E2%80%93+Autonomous+Mobile+Robot;CBS+%C3%97+PID+%7C+Multi-Agent+Path+Finding+%26+Navigation" alt="Typing SVG" /></a>

<br/>

<!-- Badges -->
![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)
![C++](https://img.shields.io/badge/C++17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white)

<br/>

<!-- Project Hero GIF -->
<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/Multi%20Robots%20Driving.gif" width="700"/>

<br/>

**창고형 무인매장에서 다수의 AMR이 충돌 없이 동시 운행하며 고객에게 상품을 배달하는 자율주행 시스템**

<br/>

[경로계획 (CBS)](#-경로계획--cbs-conflict-based-search) · [내비게이션 (PID)](#-pid-내비게이션-상태머신) · [시스템 아키텍처](#-시스템-아키텍처) · [AMR 워크플로우](#-amr-워크플로우) · [기술적 성과](#-기술적-성과)

---

</div>

## 📑 Table of Contents

```
📦 RoboCallee
├── 🏗️ 시스템 아키텍처
├── 🗺️ 경로계획 - CBS (Conflict-Based Search)
│   ├── CBS High-Level 탐색
│   ├── A* Low-Level 탐색 (시간축)
│   └── 충돌 감지 (Vertex / Edge)
├── 🎮 PID 내비게이션 상태머신
│   ├── 상태 전이 다이어그램
│   ├── PID 제어기 설계
│   └── 실시간 파라미터 튜닝
├── 🔄 AMR 워크플로우
└── 📊 기술적 성과
```

---

## 🏗️ 시스템 아키텍처

```mermaid
flowchart TB
    subgraph GUI["🖥️ GUI Layer"]
        WEB["🌐 Web GUI<br/>(고객 주문)"]
        QT["📊 Qt GUI<br/>(관제 모니터링)"]
    end

    subgraph FMS["⚙️ FMS - Fleet Management System (C++)"]
        RM["📋 Request<br/>Manager"]
        CBS["🗺️ Traffic<br/>Planner<br/>(CBS)"]
        AMR_A["🤖 AMR<br/>Adapter ×3"]
        CORE["🧠 Core<br/>Controller"]

        RM -->|"주문 배정"| CORE
        CORE -->|"경로 요청"| CBS
        CBS -->|"충돌 없는 경로"| AMR_A
        CORE --- AMR_A
    end

    subgraph NAV["🎮 Navigation Layer (Python)"]
        PID["🎯 PID State<br/>Machine"]
        VEL["📡 Velocity<br/>Filter"]
    end

    subgraph HW["🔧 Hardware"]
        R1["🤖 AMR #1"]
        R2["🤖 AMR #2"]
        R3["🤖 AMR #3"]
    end

    WEB -->|"HTTP"| RM
    QT -->|"실시간 모니터링"| CORE
    AMR_A -->|"Waypoint<br/>Publish"| PID
    PID -->|"cmd_vel"| R1 & R2 & R3
    VEL -->|"filtered_vel"| PID
    R1 & R2 & R3 -->|"pose"| CORE

    style GUI fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style FMS fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    style NAV fill:#dcfce7,stroke:#22c55e,color:#14532d
    style HW fill:#f3e8ff,stroke:#a855f7,color:#3b0764
```

---

## 🗺️ 경로계획 — CBS (Conflict-Based Search)

<div align="center">

> 다수의 로봇이 **동시에** 이동할 때 서로 충돌하지 않는 **최적 경로**를 계산하는<br/>
> MAPF(Multi-Agent Path Finding) 알고리즘

<br/>

<table>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/MAPF%20Examples.gif" width="100%"/>
<br/>
<sub><b>CBS 경로계획 시뮬레이션</b></sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/MAPF%20path%20examples.gif" width="100%"/>
<br/>
<sub><b>다중 로봇 경로 생성 결과</b></sub>
</td>
</tr>
</table>

</div>

### 🔍 알고리즘 구조

```mermaid
flowchart TD
    START(["🚀 Start: planPaths(starts, goals)"])

    subgraph HL["🔷 High-Level: CBS Tree Search"]
        INIT["각 로봇별 A* 독립 경로 계산"]
        ROOT["Root Node 생성<br/>(초기 해 + 비용)"]
        PQ["Priority Queue<br/>(최소 비용 우선)"]
        DETECT{"🔴 충돌 감지<br/>detectFirstConflict()"}
        DONE(["✅ 충돌 없음!<br/>최적 해 반환"])

        BRANCH["분기: Constraint 추가"]
        LEFT["📌 좌측 자식<br/>Agent₁에 위치/시간 금지"]
        RIGHT["📌 우측 자식<br/>Agent₂에 위치/시간 금지"]
    end

    subgraph LL["🔶 Low-Level: Time-Space A*"]
        ASTAR["A* 탐색<br/>(x, y, timestep)"]
        CONSTRAINT["Constraint 적용<br/>특정 시간 × 위치 금지"]
        HEURISTIC["맨해튼 거리<br/>휴리스틱"]
    end

    START --> INIT
    INIT --> ROOT
    ROOT --> PQ
    PQ --> DETECT
    DETECT -->|"충돌 발견"| BRANCH
    DETECT -->|"충돌 없음"| DONE
    BRANCH --> LEFT & RIGHT
    LEFT --> ASTAR
    RIGHT --> ASTAR
    ASTAR --> CONSTRAINT
    CONSTRAINT --> HEURISTIC
    HEURISTIC --> PQ

    style HL fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style LL fill:#fff7ed,stroke:#f97316,color:#7c2d12
    style DONE fill:#dcfce7,stroke:#16a34a,color:#14532d
    style START fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
```

### ⚡ 충돌 감지: Vertex vs Edge

<div align="center">

```
   Vertex Conflict                    Edge Conflict
   (같은 시간, 같은 위치)               (교차 이동)

   t=3:  A → ● ← B                  t=3:  A ●───● B
              ↑                       t=4:  A ●───● B
         동시 점유!                          서로 교차!

   ┌─────────────────┐              ┌─────────────────┐
   │ if path[i][t]   │              │ if path[i][t]   │
   │ == path[j][t]   │              │ == path[j][t+1] │
   │                 │              │ && path[j][t]   │
   │   → Conflict!   │              │ == path[i][t+1] │
   └─────────────────┘              │   → Conflict!   │
                                    └─────────────────┘
```

</div>

<details>
<summary><b>📜 CBS 핵심 구현 코드 (C++) — 펼쳐보기</b></summary>

<br/>

**High-Level CBS 탐색**
```cpp
std::vector<std::vector<Position>> TrafficPlanner::planPaths(
    const std::vector<Position>& starts,
    const std::vector<Position>& goals)
{
    CBSNode root;
    root.constraints = {};
    root.id = 0;

    // 각 에이전트별 독립 A* 경로 계산
    for (size_t i = 0; i < starts.size(); ++i) {
        auto path = a_star(starts[i], goals[i], root.constraints, i);
        root.paths.push_back(path);
    }
    root.cost = computeCost(root.paths);

    std::priority_queue<CBSNode, std::vector<CBSNode>, std::greater<CBSNode>> open;
    open.push(root);

    while (!open.empty()) {
        CBSNode current = open.top();
        open.pop();

        // 충돌 감지
        Conflict conflict = detectFirstConflict(current.paths);
        if (conflict.agent1 == -1)
            return current.paths;  // ✅ 충돌 없는 최적 해!

        // 분기: 두 에이전트에 각각 Constraint 추가
        for (int agent : {conflict.agent1, conflict.agent2}) {
            CBSNode child = current;
            Constraint c = {agent, conflict.timestep, conflict.loc};
            child.constraints.push_back(c);

            // 해당 에이전트만 A* 재탐색
            child.paths[agent] = a_star(
                starts[agent], goals[agent], child.constraints, agent);
            child.cost = computeCost(child.paths);
            open.push(child);
        }
    }
    return {};
}
```

**Low-Level: 시간축 A\* 탐색**
```cpp
std::vector<Position> TrafficPlanner::a_star(
    const Position& start, const Position& goal,
    const std::vector<Constraint>& constraints, int agent)
{
    // 상태 공간: (x, y, timestep) — 일반 A*와의 핵심 차이점
    auto cmp = [](const Node* a, const Node* b) {
        return a->f_val() > b->f_val();
    };
    std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> open(cmp);

    Node* start_node = new Node{start, 0, manhattan(start, goal), 0, nullptr};
    open.push(start_node);

    while (!open.empty()) {
        Node* current = open.top();
        open.pop();

        if (current->pos == goal)
            return reconstructPath(current);

        // 4방향 이동 + 대기(wait)
        for (auto& [dx, dy] : directions) {
            Position next = {current->pos.x + dx, current->pos.y + dy};
            int next_t = current->timestep + 1;

            if (!isValid(next)) continue;
            if (isConstrained(agent, next, next_t, constraints)) continue;

            Node* neighbor = new Node{next, current->g_val + 1,
                                       manhattan(next, goal), next_t, current};
            open.push(neighbor);
        }
    }
    return {};
}
```

</details>

### 🗺️ 운영 맵 환경

<div align="center">

```
                     ← 22 cells (2.2m) →
    ┌──────────────────────────────────────────┐
  1 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ▓ = 벽/장애물
  2 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░▓▓│  ░ = 이동 가능
  3 │▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓│  ⚡ = 충전소
  4 │▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓│  📦 = 창고
  5 │▓▓▓▓▓▓▓▓░░░░░░⚡░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓│
  6 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░▓▓│
  7 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓📦░░░░▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓│
  8 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓▓░░░░░░░░▓▓│
  9 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓░░░░⚡░░▓▓│
 10 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░⚡░░▓▓│
 11 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░▓▓│
 12 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
    └──────────────────────────────────────────┘
         해상도: 0.1m/cell | AMR 3대 | 도착판정: 0.05m
```

</div>

<div align="center">
<table>
<tr>
<td>

| 파라미터 | 값 |
|:---:|:---:|
| 맵 크기 | `12 × 22` cells |
| 해상도 | `0.1m` / cell |
| AMR 수 | `3` 대 |
| 도착 허용 오차 | `0.05m` |

</td>
<td>

| 위치 | 좌표 |
|:---:|:---:|
| 📦 창고 (Storage) | `(6, 2)` |
| ⚡ 충전소 #1 | `(9, 8)` |
| ⚡ 충전소 #2 | `(9, 5)` |
| ⚡ 충전소 #3 | `(9, 4)` |

</td>
</tr>
</table>
</div>

---

## 🎮 PID 내비게이션 상태머신

<div align="center">

> Nav2를 사용하지 않고 직접 구현한 이유:<br/>
> **FMS와의 Waypoint 단위 통합이 더 직관적**이고, 경량화된 제어가 가능

</div>

### 🔄 상태 전이 다이어그램

<div align="center">

<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/navigation%20state%20machine.gif" width="700"/>

<sub><b>실제 동작하는 Navigation State Machine</b></sub>

</div>

<br/>

```mermaid
stateDiagram-v2
    [*] --> IDLE : 시작

    IDLE --> RotateToGoal : 🎯 Waypoint 수신

    RotateToGoal --> MoveToGoal : ✅ 각도 오차 < threshold
    RotateToGoal --> RotateToGoal : 🔄 Angular PID

    MoveToGoal --> RotateToFinal : ✅ 거리 < threshold
    MoveToGoal --> MoveToGoal : 🔄 Linear + Angular PID

    RotateToFinal --> GoalReached : ✅ 최종 자세 정렬
    RotateToFinal --> RotateToFinal : 🔄 Angular PID

    GoalReached --> IDLE : 📍 다음 WP 대기
    GoalReached --> RotateToGoal : 📍 다음 WP 수신

    note right of RotateToGoal : 목표 방향으로\n제자리 회전
    note right of MoveToGoal : 직진 + 방향 보정\n(듀얼 PID)
    note right of RotateToFinal : 최종 heading\n미세 조정
```

### 🎯 P 제어 vs PID 제어 비교

<div align="center">

<table>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/P.gif" width="100%"/>
<br/>
<sub><b>P 제어만 적용</b> — 오버슈트, 진동 발생</sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/PID.gif" width="100%"/>
<br/>
<sub><b>PID 제어 적용</b> — 안정적 수렴</sub>
</td>
</tr>
</table>

</div>

### 📐 PID 제어기 설계

<div align="center">

<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/PID%20Control%20logic.png" width="650"/>

</div>

<br/>

$$
u(t) = \underbrace{K_p \cdot e(t)}_{\text{Proportional}} + \underbrace{K_i \int_0^t e(\tau)\,d\tau}_{\text{Integral}} + \underbrace{K_d \frac{de(t)}{dt}}_{\text{Derivative}}
$$

<br/>

<table>
<tr>
<td width="50%">

**Angular PID (방향 제어)**
```python
# 목표 방향과의 오차 계산
error = normalize_angle(target_yaw - current_yaw)

# PID 각 항 계산
P = Kp * error
I = Ki * integral      # Anti-windup 적용
D = Kd * (error - prev_error) / dt

angular_vel = P + I + D
```

</td>
<td width="50%">

**Linear PID (속도 제어)**
```python
# 목표까지의 거리
distance = hypot(
    goal.x - current.x,
    goal.y - current.y
)

# 거리 비례 속도 제어
linear_vel = Kp_linear * distance
linear_vel = clamp(linear_vel, 0, max_vel)
```

</td>
</tr>
</table>

### 🛡️ Anti-Windup

적분항이 과도하게 누적되어 오버슈트를 유발하는 **Integral Windup** 현상을 방지합니다.

```python
# ❌ Without Anti-Windup — 적분항 무한 누적
self.integral += error * dt          # 정지 상태에서도 계속 누적 → 오버슈트

# ✅ With Anti-Windup — 적분항 클램핑
self.integral += error * dt
self.integral = max(-limit, min(limit, self.integral))  # 범위 제한
```

```
                Without Anti-Windup          With Anti-Windup
  목표 ─────    ╭──╮    ╭─╮                 ╭─────────────────
               │  │   │ │                 │
               │  ╰───╯ ╰─────           │
               │      진동 & 오버슈트       │   빠른 수렴 ✅
  ─────────────╯                ───────────╯
```

### 🎛️ 실시간 PID 튜닝

<div align="center">

<img src="https://raw.githubusercontent.com/addinedu-roscamp-5th/roscamp-repo-3/main/docs/images/PID%20tuning.gif" width="700"/>

<sub><b>Qt GUI에서 실시간으로 PID 게인을 조정하며 로봇 동작 확인</b></sub>

</div>

<br/>

<details>
<summary><b>📜 ROS2 동적 파라미터 구현 코드 — 펼쳐보기</b></summary>

```python
class MoveToGoalPID(Node):
    def __init__(self):
        super().__init__('move_to_goal_pid')

        # 📌 ROS2 Parameter Server에 PID 게인 등록
        self.declare_parameter('angular_kp', 2.0)
        self.declare_parameter('angular_ki', 0.0)
        self.declare_parameter('angular_kd', 0.1)
        self.declare_parameter('linear_kp', 0.5)
        self.declare_parameter('angle_tolerance', 0.05)
        self.declare_parameter('dist_tolerance', 0.03)
        self.declare_parameter('windup_limit', 1.0)

        # 📌 파라미터 변경 콜백 등록
        self.add_on_set_parameters_callback(self.param_callback)

    def param_callback(self, params):
        """로봇 구동 중 실시간으로 게인 변경 가능"""
        for param in params:
            if param.name == 'angular_kp':
                self.angular_kp = param.value
            elif param.name == 'angular_ki':
                self.angular_ki = param.value
            # ... 모든 파라미터 동적 반영
        return SetParametersResult(successful=True)
```

```bash
# 터미널에서 실시간 파라미터 변경
ros2 param set /move_to_goal_pid angular_kp 3.0
ros2 param set /move_to_goal_pid angular_kd 0.2
```

</details>

---

## 🔄 AMR 워크플로우

### 전체 작업 흐름

```mermaid
sequenceDiagram
    actor C as 🧑 고객
    participant W as 🌐 Web GUI
    participant F as ⚙️ FMS Core
    participant CBS as 🗺️ CBS Planner
    participant AMR as 🤖 AMR
    participant PID as 🎮 PID Controller

    C->>W: 1️⃣ 신발 주문
    W->>F: 2️⃣ GUIRequest 전송

    Note over F: BestRobotSelector<br/>배터리/상태 기반<br/>최적 로봇 선택

    F->>CBS: 3️⃣ 전체 활성 로봇 경로 요청

    Note over CBS: CBS 경로계획<br/>다중 로봇 충돌 없는<br/>최적 경로 계산

    CBS-->>F: 충돌 없는 경로 반환
    F->>AMR: 4️⃣ Waypoint 리스트 전송

    loop 각 Waypoint마다
        AMR->>PID: Waypoint 전달
        PID->>PID: RotateToGoal → MoveToGoal → RotateToFinal
        PID-->>AMR: GoalReached
        AMR->>F: 도착 보고 (dist < 0.05m)
        F->>AMR: 다음 Waypoint 전송
    end

    Note over AMR: 📦 Storage 도착
    Note over AMR: 상품 적재

    F->>CBS: 5️⃣ 목적지 경로 재계획
    CBS-->>F: 새 경로
    F->>AMR: 고객 위치로 이동

    AMR-->>C: 6️⃣ 상품 배달 완료 ✅

    Note over AMR: ⚡ 충전소 복귀
```

### AMR 상태 전이

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> BUSY : 📋 주문 배정

    state BUSY {
        [*] --> CheckPath
        CheckPath --> MoveToStorage : 🗺️ CBS 경로 수신
        MoveToStorage --> MoveToDestination : 📦 상품 적재 완료
        MoveToDestination --> [*] : 🎯 고객 도착
    }

    BUSY --> RETURN : ✅ 배달 완료
    RETURN --> IDLE : ⚡ 충전소 도착

    note right of IDLE : 대기 상태\n배터리 충전 중
    note right of BUSY : 주문 처리 중\nWaypoint 추종
    note right of RETURN : 충전소 복귀 중
```

<details>
<summary><b>📜 AMR Adapter 핵심 코드 (C++) — 펼쳐보기</b></summary>

```cpp
// 🎯 Waypoint 도착 판정
bool AmrAdapter::handleWaypointArrival(const pose2f& pos) {
    Position wp = getCurrentWayPoint();
    float dist = std::hypot(pos.x - wp.x, pos.y - wp.y);

    if (dist <= 0.05f) {           // ARRIVAL_TOLERANCE
        sendNextpoint();            // → 다음 Waypoint로 진행
    }
    return true;
}

// 📍 다음 Waypoint 전송
void AmrAdapter::sendNextpoint() {
    if (isGoal()) {                 // 최종 목적지 도달?
        MoveToDone();               // → 상태 전이
        return;
    }
    incrementWaypointIndex();
    Position wp = getCurrentWayPoint();
    core->publishNavGoal(robot_id, wp);  // ROS2 토픽 퍼블리시
}

// 🔄 작업 완료 후 상태 전이
void AmrAdapter::MoveToDone() {
    switch (step_) {
        case MoveTo_Storage:         // 창고 도착
            SendPickupRequest();     // → 로봇팔에 상품 요청
            SetAmrStep(MoveTo_dst);  // → 다음: 고객에게 이동
            break;
        case MoveTo_charging_station:// 충전소 도착
            SetAmrState(IDLE);       // → 대기 상태로 전환
            break;
    }
}
```

</details>

---

## 📊 기술적 성과

<div align="center">

<table>
<tr>
<td align="center" width="25%">

### 🗺️
### CBS MAPF
**3대 AMR**<br/>동시 충돌 없는<br/>경로계획

</td>
<td align="center" width="25%">

### 🎮
### PID 제어
**Anti-Windup**<br/>실시간 튜닝<br/>안정적 수렴

</td>
<td align="center" width="25%">

### 🔄
### 실시간 재계획
**로봇 추가/복귀 시**<br/>전체 경로<br/>자동 재계산

</td>
<td align="center" width="25%">

### ⚡
### Nav2 대체
**경량 제어기**<br/>FMS 연동 최적화<br/>Waypoint 추종

</td>
</tr>
</table>

</div>

### 💡 배운 점

<table>
<tr>
<td>🗺️</td>
<td><b>MAPF 알고리즘</b></td>
<td>단일 로봇 경로계획과 달리 다중 로봇 환경에서는 <b>시간축</b>까지 고려해야 하며, CBS가 최적성을 보장하면서도 실용적인 해를 제공한다는 것을 체감</td>
</tr>
<tr>
<td>🎛️</td>
<td><b>PID 튜닝</b></td>
<td>이론적 게인과 실제 로봇에서의 최적 게인은 큰 차이가 있으며, <b>실시간 튜닝 인프라</b>의 필요성을 경험</td>
</tr>
<tr>
<td>🔧</td>
<td><b>시스템 통합</b></td>
<td>경로계획 → 제어기 → 하드웨어 간의 <b>인터페이스 설계</b>가 전체 시스템 안정성에 결정적 영향을 미침</td>
</tr>
<tr>
<td>🤖</td>
<td><b>Nav2 vs 직접 구현</b></td>
<td>기존 프레임워크의 장단점을 비교하고, 프로젝트 요구사항에 맞는 <b>기술 선택의 중요성</b>을 학습</td>
</tr>
</table>

---

<div align="center">

### 🔗 Links

[![GitHub](https://img.shields.io/badge/Project_Repo-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/addinedu-roscamp-5th/roscamp-repo-3)

<br/>

<sub>Built with ROS2 Humble · C++17 · Python 3 · CBS Algorithm · PID Control</sub>

</div>
