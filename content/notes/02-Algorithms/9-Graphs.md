---
title: "Graphs"
date: 2026-08-18
summary: "Graph types, representations, connected components, BFS/DFS, cycle detection, and problems (Provinces, Islands, Flood Fill, Rotting Oranges, 01 Matrix, Surrounded Regions)."
tags: [Graphs, DSA, Algorithms]
---

Graph basics: types, degree rules, and the two representations every graph problem starts from.

---

# Graph and types

A graph is `G = (V, E)`: a set of nodes and a set of edges connecting them.

### ***Undirected***

- Edge `u-v` can be walked both ways.
- `degree(node)` = number of edges touching it: `1:2, 2:3, 3:2, 4:1`.
- **Handshaking lemma:** `sum of degrees = 2 * E` (here `8 = 2 * 4`), since each edge adds 1 to both endpoints.

### ***Directed***

- Edge `u→v` is one way only.
- Each node has an **in-degree** and an **out-degree**: node `2` has `in=1, out=2`.
- `sum of in-degrees = sum of out-degrees = E`.

---

## Representation of a graph

Input in most questions: `n` nodes, `m` edges, directed or undirected, plus the edge list `[[u, v], ...]` (or `[[u, v, wt], ...]` if weighted).

### 1) Adjacency matrix, `O(n^2)` space

```
        1  2  3  4
     ┌──────────────┐
   1 │ 0  1  1  0   │
   2 │ 1  0  1  1   │   adj[i][j] = 1 if edge i-j exists
   3 │ 1  1  0  0   │   undirected → symmetric about the diagonal
   4 │ 0  1  0  0   │
     └──────────────┘
```

- Size `(n+1) x (n+1)` for 1-indexed nodes, `n x n` for 0-indexed.
- Fill per edge: undirected sets `adj[u][v] = adj[v][u] = 1`, directed sets only `adj[u][v] = 1`.
- Weighted: store `wt` instead of `1`.
- Edge lookup is `O(1)`, but space is `O(n^2)` even for a graph with 3 edges.

```python
def build_adj_matrix(n, edges, directed=False):
    adj = [[0] * (n + 1) for _ in range(n + 1)]   # 1-indexed
    for u, v in edges:
        adj[u][v] = 1
        if not directed:
            adj[v][u] = 1
    return adj
```

### 2) Adjacency list, `O(n + 2E)` space, prefer this

```
   adj[1] → [2, 3]
   adj[2] → [1, 3, 4]        every edge stored twice (undirected)
   adj[3] → [1, 2]           total stored = 2 * E = 8
   adj[4] → [2]
```

- `n + 1` empty lists, then for each node store only its neighbors.
- Undirected appends both ways, directed appends `v` to `adj[u]` only.
- Weighted: store tuples, `adj[u].append((v, wt))`.
- Space scales with edges, not `n^2`, and neighbor iteration is `O(degree(u))`, which is what BFS and DFS need.

```python
def build_adj_list(n, edges, directed=False):
    adj = [[] for _ in range(n + 1)]   # 1-indexed
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return adj
```

Use a `defaultdict(list)` instead when node ids are sparse or non-numeric.

---

## Which one to pick

|                          | Adjacency matrix                | Adjacency list            |
| ------------------------ | ------------------------------- | ------------------------- |
| Space                    | `O(n^2)`                        | `O(n + 2E)`               |
| Is edge `u-v` present    | `O(1)`                          | `O(degree(u))`            |
| Iterate neighbors of `u` | `O(n)`                          | `O(degree(u))`            |
| Best for                 | dense graphs, many edge queries | sparse graphs, traversals |

**Default:** adjacency list. Switch to a matrix only for dense graphs or when repeated `O(1)` edge checks dominate.

---

# Connected components and tricks

A "graph" in a question is often **several disconnected pieces**, not one connected blob. One traversal from node `1` only ever reaches node `1`'s piece.

```
n = 7,  edges = {1-2, 2-3, 4-5, 6-7}

   component 1        component 2     component 3
   1 ─── 2 ─── 3        4 ─── 5         6 ─── 7

   loop node = 1..7, traverse only when node is unvisited
   traversals start at 1, 4, 6  →  3 components
```

- Wrap every traversal in a loop over all nodes, and start one only from an unvisited node.
- Number of starts = number of components.
- `visited` lives **outside** the loop, otherwise nodes get re-visited and the count blows up.
- The traversal itself can be DFS or BFS, the component logic is identical.

Time `O(n + 2E)`, space `O(n)`. Every node is pushed once, every adjacency entry is read once.

---

# Traversal techniques

Both visit every reachable node once. Pick based on **what you need from the order**, not complexity (both are `O(n + 2E)`).

```
same graph, start at 1

        1
       / \
      2   3
     /
    4

BFS (queue, level by level)     DFS (stack, go deep first)
  visit: 1 → 2 → 3 → 4            visit: 1 → 2 → 4 → 3
  dist[1]=0, dist[2]=1, ...       explores one branch fully before the next
```

|                            | BFS                                            | DFS                                                 |
| -------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| Structure                  | queue (`deque`), pop **front**                 | stack or recursion, pop **back**                    |
| Order                      | nearest nodes first                            | deepest branch first                                |
| Shortest path (unweighted) | **yes**, first time you reach `v` is min steps | no                                                  |
| Typical use                | min steps, level order, multi-source spread    | connectivity, cycle detect, topo sort, backtracking |
| Space                      | `O(n)` queue, can hold a full level            | `O(n)` stack, can hit `O(n)` depth on a path        |

Always keep a `visited` set/array. Without it, cycles cause infinite loops.

### BFS

- Push `src`, mark visited, then repeatedly pop front and push unvisited neighbors to the back.
- Track `dist[node]` when you first visit: that is the shortest step count from `src` on an unweighted graph.
- **Multi-source BFS:** seed the queue with all sources at once (e.g. all rotten oranges, all gates). Same template, no extra loop.

```python
from collections import deque

def bfs(adj, src):
    visited = [False] * len(adj)
    dist = [-1] * len(adj)
    q = deque([src])
    visited[src] = True
    dist[src] = 0
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def multi_source_bfs(adj, sources):
    visited = [False] * len(adj)
    dist = [-1] * len(adj)
    q = deque()
    for s in sources:
        visited[s] = True
        dist[s] = 0
        q.append(s)
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist
```

### DFS

- Push `src`, mark visited, pop and push unvisited neighbors. Same `visited` logic as BFS, different container.
- Prefer an **explicit stack** over recursion on large graphs (path graphs can exceed Python's recursion limit around `10^4` depth).
- **Post-order DFS** (process node after all children) gives topological order on a DAG.

```python
def dfs(adj, src):
    visited = [False] * len(adj)
    stack = [src]
    visited[src] = True
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)
    return order


def dfs_recursive(adj, src, visited=None):
    if visited is None:
        visited = [False] * len(adj)
    visited[src] = True
    for nb in adj[src]:
        if not visited[nb]:
            dfs_recursive(adj, nb, visited)
    return visited
```

### When to use what

| Problem signal                                         | Reach for                                             |
| ------------------------------------------------------ | ----------------------------------------------------- |
| "minimum steps", "shortest path", all edges cost 1     | **BFS**                                               |
| "level by level", "distance from multiple sources"     | **BFS** (multi-source)                                |
| "count components", "is connected", flood fill on grid | **DFS or BFS** (same logic, see connected components) |
| "detect cycle", "topological order", "dependencies"    | **DFS** (3-color cycle detect, post-order topo)       |
| "explore all paths", backtracking                      | **DFS**                                               |
| edge weights > 1 or varying                            | **Dijkstra** (not plain BFS)                          |
| edge weights only 0 or 1                               | **0-1 BFS** with deque (push front for 0, back for 1) |

**Grid as graph:** cell `(r, c)` is a node, 4/8 neighbors are edges. BFS/DFS templates are identical; the neighbor loop walks directions instead of `adj[node]`.

---

# BFS and DFS Patterns:

### Problem 1: Number of Provinces ([LC 547](https://leetcode.com/problems/number-of-provinces/))

`isConnected` is an adjacency matrix: `isConnected[i][j] == 1` means cities `i` and `j` are directly linked. Count connected components (provinces).

```
isConnected =
  0 1 2
0 1 1 0
1 1 1 0
2 0 0 1

  0       1          2
  province A    province B   → answer = 2
```

- Pattern: connected components on a matrix graph (not an edge list).
- Outer loop over all cities; start a DFS/BFS only when unvisited; each start = one province.
- Neighbors of `i` = every `j` where `isConnected[i][j] == 1` (skip `j == i`).
- Time `O(n^2)`, space `O(n)` for `visited`. Matrix is dense, so scanning all columns per city is fine.

```python
from typing import List

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = set()
        result = 0

        def dfs(i):
            visited.add(i)
            # matrix graph: scan all columns to find neighbors of city i
            for neigh in range(n):
                if isConnected[i][neigh] == 1 and neigh not in visited:
                    dfs(neigh)

        # each unvisited start = one new province
        for i in range(n):
            if i not in visited:
                result += 1
                dfs(i)
        return result
```

### Problem 2: Number of Islands ([LC 200](https://leetcode.com/problems/number-of-islands/))

Grid of `"1"` (land) and `"0"` (water). An island is a max connected group of land cells (4-directional). Count islands.

```
  1 1 0 0 0
  1 1 0 0 0
  0 0 1 0 0
  0 0 0 1 1

  island A (top-left)   island B (middle)   island C (bottom-right)
  answer = 3
```

- Pattern: connected components on a **grid** (implicit graph).
- Outer loop over every cell; when you see land, flood-fill the whole island, then `count += 1`.
- Neighbors = 4 directions `(±1, 0)`, `(0, ±1)`. Mark visited by flipping `"1"` → `"0"` (or use a `visited` set).
- BFS or DFS both work. Time `O(m * n)`, space `O(m * n)` worst case (queue/stack).

**BFS**

```python
from collections import deque
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def bfs(row, col):
            q = deque([(row, col)])
            grid[row][col] = "0"
            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == "1":
                        grid[nr][nc] = "0"
                        q.append((nr, nc))

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    bfs(i, j)
                    res += 1
        return res
```

**DFS**

```python
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0

        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] == "0":
                return
            grid[i][j] = "0"
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    count += 1
                    dfs(i, j)
        return count
```

**Same pattern, different graph:** Provinces = components on an adjacency matrix. Islands = components on a grid. Both are "loop all nodes → traverse if unvisited → count starts."

### Problem 3: Flood Fill ([LC 733](https://leetcode.com/problems/flood-fill/))

Start at `(sr, sc)`. Recolor every 4-connected cell that shares the **original** color with the start cell to `color`. Return the modified image.

```
image, start (1,1), color = 2

  before          after
  1 1 1           2 2 2
  1 1 0    →      2 2 0
  1 0 1           2 0 1

  only cells matching start color 1 get recolored
```

- Pattern: DFS/BFS flood from **one** seed cell (not a count-all-components loop).
- Capture `ic = image[sr][sc]` first. Recolor only cells equal to `ic`.
- Early return if `ic == color` (already filled). Without this, DFS never terminates: every neighbor still "matches" and you recurse forever.
- Time `O(m * n)`, space `O(m * n)` recursion/queue worst case.

```python
from typing import List

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        m, n = len(image), len(image[0])
        ic = image[sr][sc]
        if ic == color:  # already filled → avoid infinite recursion
            return image

        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or image[i][j] != ic:
                return
            image[i][j] = color
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                dfs(i + dr, j + dc)

        dfs(sr, sc)
        return image
```

### Problem 4: Rotting Oranges ([LC 994](https://leetcode.com/problems/rotting-oranges/))

Grid cells: `0` empty, `1` fresh, `2` rotten. Every minute, any fresh orange 4-adjacent to a rotten one becomes rotten. Return minutes until all are rotten, or `-1` if impossible.

```
minute 0          minute 1          minute 2
  2 1 1             2 2 1             2 2 2
  1 1 0      →      2 1 0      →      2 2 0
  0 1 1             0 1 1             0 2 1  ... → answer = 4
```

- Pattern: **multi-source BFS**. Seed the queue with **all** initially rotten cells at time `0`.
- Track `fresh` count. Each time you rot a fresh orange, decrement it.
- Answer is the max time on the queue. If `fresh > 0` when BFS ends, return `-1`.
- Mark rotten **before** enqueue (same as Islands) so a cell is not queued twice.
- Time `O(m * n)`, space `O(m * n)`.

```python
from collections import deque
from typing import List

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        q = deque()  # (r, c, time)

        # seed all rotten oranges at time 0; count fresh
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    q.append((i, j, 0))

        res = 0
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while q:
            r, c, time = q.popleft()
            res = time
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2  # rot + mark visited
                    fresh -= 1
                    q.append((nr, nc, time + 1))

        return -1 if fresh > 0 else res
```

---

# Cycle detection

A cycle means you can walk edges and return to a node you already visited. The check differs for **undirected** vs **directed** graphs.

```
undirected (cycle)              undirected (no cycle)         directed (cycle)
  1 ─── 2                         1 ─── 2                     1 ──→ 2
  │     │                         │                           ↑     │
  └── 3 ┘                         3                           └─────┘
  back edge to node on path       tree, no back edge          back edge to GRAY node
```

|              | Undirected                                                             | Directed                                                         |
| ------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Cycle signal | visited neighbor that is **not** your parent                           | neighbor already **GRAY** (on current DFS path)                  |
| Algorithm    | BFS or DFS + `parent`                                                  | DFS + 3 colors (WHITE / GRAY / BLACK)                            |
| Why parent?  | edge `u-v` appears twice in adj list; parent is the node you came from | parent trick does not work; need to track active recursion stack |

Loop over all nodes before each traversal (graph may be disconnected).

### Undirected graph

**Idea:** during traversal, if you reach a visited neighbor that is not your parent, that neighbor is a back edge → cycle.

**DFS**

```python
def has_cycle_undirected_dfs(n, adj):
    visited = [False] * n

    def dfs(node, parent):
        visited[node] = True
        for nb in adj[node]:
            if not visited[nb]:
                if dfs(nb, node):
                    return True
            elif nb != parent:  # back edge, not the edge we arrived on
                return True
        return False

    for i in range(n):
        if not visited[i] and dfs(i, -1):
            return True
    return False
```

**BFS**

```python
from collections import deque

def has_cycle_undirected_bfs(n, adj):
    visited = [False] * n

    def bfs(src):
        visited[src] = True
        q = deque([(src, -1)])  # (node, parent)
        while q:
            node, parent = q.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append((nb, node))
                elif nb != parent:  # back edge
                    return True
        return False

    for i in range(n):
        if not visited[i] and bfs(i):
            return True
    return False
```

### Directed graph

**Idea:** 3-color DFS. GRAY = on the current recursion path. A GRAY neighbor means a back edge → cycle.

```
WHITE = unvisited     GRAY = in current DFS path     BLACK = fully explored

  1(W) → 2(W) → 3(W)
         ↑____________↓
  when 3 points back to 2 (GRAY) → cycle
```

```python
def has_cycle_directed(n, adj):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY
        for nb in adj[node]:
            if color[nb] == GRAY:  # back edge to node on current path
                return True
            if color[nb] == WHITE and dfs(nb):
                return True
        color[node] = BLACK
        return False

    return any(color[i] == WHITE and dfs(i) for i in range(n))
```

Time `O(n + 2E)` for all versions. Space `O(n)`.

| Problem signal                                      | Use                             |
| --------------------------------------------------- | ------------------------------- |
| undirected graph, "has a cycle?"                    | parent-based BFS/DFS            |
| directed graph, "has a cycle?" / invalid topo order | 3-color DFS                     |
| directed graph, "can finish all courses?"           | topo sort; empty result = cycle |

---

### Problem 5: 01 Matrix ([LC 542](https://leetcode.com/problems/01-matrix/))

Binary matrix: `0` = empty, `1` = obstacle. For each cell, return Manhattan distance to the **nearest** `0`.

```
input                    output (dist to nearest 0)
  0 0 0                    0 0 0
  0 1 1          →         0 1 2
  1 1 1                    1 2 3
```

- Pattern: **multi-source BFS**. Seed the queue with **all** `0` cells at distance `0` (inverse of Rotting Oranges: spread from targets, not sources).
- First time a `1` is reached = shortest distance to any `0` (BFS guarantees min steps on unweighted grid).
- No need to store `steps` in the queue; set `dist[r][c]` when you dequeue (or when you enqueue).
- Time `O(m * n)`, space `O(m * n)`.

```python
from collections import deque
from typing import List

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        dist = [[0] * n for _ in range(m)]
        visited = [[False] * n for _ in range(m)]
        q = deque()

        # all 0s are sources at distance 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    visited[i][j] = True
                    q.append((i, j))

        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                    visited[nr][nc] = True
                    dist[nr][nc] = dist[r][c] + 1  # first visit = nearest 0
                    q.append((nr, nc))
        return dist
```

**vs Rotting Oranges:** both multi-source BFS on a grid. Oranges spread from all `2`s and track time; 01 Matrix spreads from all `0`s and records distance to the nearest zero.

### Problem 6: Surrounded Regions ([LC 130](https://leetcode.com/problems/surrounded-regions/))

Board of `"X"` and `"O"`. Capture every `"O"` that is **fully surrounded** by `"X"` (flip to `"X"`). Any `"O"` on the border, or connected to a border `"O"`, stays `"O"`.

```
before                    after
  X X X X                   X X X X
  X O O X         →         X X X X
  X X O X                   X X X X
  X O X X                   X O X X   ← bottom O touches border, safe
```

- Pattern: **invert the problem**. Do not search for surrounded regions. Mark all `"O"`s reachable from the **border**, then flip everything else.
- Seed DFS/BFS from every border cell that is `"O"`. Those regions are **safe**.
- After that pass, any unvisited `"O"` is landlocked → flip to `"X"`.
- Time `O(m * n)`, space `O(m * n)` for `visited` (or mark safe cells as `"T"` in-place and convert back).

```python
from typing import List

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        def dfs(i, j):
            visited[i][j] = True  # mark safe (connected to border)
            for dr, dc in dirs:
                nr, nc = i + dr, j + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc] and board[nr][nc] == "O":
                    dfs(nr, nc)

        # flood from all border O's
        for j in range(n):
            if board[0][j] == "O" and not visited[0][j]:
                dfs(0, j)
            if board[m - 1][j] == "O" and not visited[m - 1][j]:
                dfs(m - 1, j)
        for i in range(m):
            if board[i][0] == "O" and not visited[i][0]:
                dfs(i, 0)
            if board[i][n - 1] == "O" and not visited[i][n - 1]:
                dfs(i, n - 1)

        # unvisited O = surrounded → capture
        for i in range(m):
            for j in range(n):
                if board[i][j] == "O" and not visited[i][j]:
                    board[i][j] = "X"
```
