---
title: "Graphs"
date: 2026-08-18
summary: "Graph types, representations, connected components, BFS/DFS, cycle detection, bipartite check, topological sort (DFS + Kahn), and related problems."
tags: [Graphs, DSA, Algorithms]
---

Graph basics: types, degree rules, representations, traversals, cycle/bipartite checks, and topological sort.

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
| "detect cycle", "topological order", "dependencies"    | **DFS** (3-color) or **Kahn's** (indegree BFS)        |
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
- Matrix is dense, so scanning all columns per city is fine.
- **Time** `O(n^2)` | **Space** `O(n)` (`visited` + recursion stack)

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
- BFS or DFS both work.
- **Time** `O(m * n)` | **Space** `O(m * n)` worst case (queue / recursion stack)

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
- **Time** `O(m * n)` | **Space** `O(m * n)` worst case (recursion / queue)

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
- **Time** `O(m * n)` | **Space** `O(m * n)` (queue)

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

Time `O(n + E)` for all versions. Space `O(n)` (`visited` / `color` + stack / queue).

| Problem signal                                      | Use                                                         |
| --------------------------------------------------- | ----------------------------------------------------------- |
| undirected graph, "has a cycle?"                    | parent-based BFS/DFS                                        |
| directed graph, "has a cycle?" / invalid topo order | 3-color DFS or Kahn's (`len < V`)                           |
| directed graph, "can finish all courses?"           | [topo sort](#topological-sorting); incomplete order = cycle |

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
- **Time** `O(m * n)` | **Space** `O(m * n)` (`dist` + `visited` + queue)

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
- **Time** `O(m * n)` | **Space** `O(m * n)` (`visited`; or mark `"T"` in-place for `O(1)` extra)

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

### Problem 7: Number of Enclaves ([LC 1020](https://leetcode.com/problems/number-of-enclaves/))

Grid: `0` = sea, `1` = land. An **enclave** is a land cell that cannot walk off the grid (4-directional moves on land). Return the count of such cells.

```
  0 0 0 0
  1 0 1 0
  0 1 1 0
  0 0 0 0

  border-connected land is reachable (walks off grid)
  the 1s in the middle are enclaves → answer = 3
```

- Pattern: same invert as Surrounded Regions. Flood from all **border land** cells, mark them visited (can walk off).
- Count remaining unvisited `1`s. That count is the answer (cells, not components).
- Surrounded Regions flips `"O"` → `"X"`. Here you **count** leftover land instead of flipping.
- BFS or DFS both work.
- **Time** `O(m * n)` | **Space** `O(m * n)` (`visited` + queue / recursion)

**DFS**

```python
from typing import List

class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        def dfs(i, j):
            visited[i][j] = True  # can walk off the grid
            for dr, dc in dirs:
                nr, nc = i + dr, j + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc] and grid[nr][nc] == 1:
                    dfs(nr, nc)

        # flood from all border land
        for i in range(m):
            if grid[i][0] == 1 and not visited[i][0]:
                dfs(i, 0)
            if grid[i][n - 1] == 1 and not visited[i][n - 1]:
                dfs(i, n - 1)
        for j in range(n):
            if grid[0][j] == 1 and not visited[0][j]:
                dfs(0, j)
            if grid[m - 1][j] == 1 and not visited[m - 1][j]:
                dfs(m - 1, j)

        # leftover unvisited land = enclaves
        return sum(
            1 for i in range(m) for j in range(n)
            if grid[i][j] == 1 and not visited[i][j]
        )
```

**BFS**

```python
from collections import deque
from typing import List

class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        q = deque()

        # seed queue with all border land
        for i in range(m):
            for j in (0, n - 1):
                if grid[i][j] == 1 and not visited[i][j]:
                    visited[i][j] = True
                    q.append((i, j))
        for j in range(n):
            for i in (0, m - 1):
                if grid[i][j] == 1 and not visited[i][j]:
                    visited[i][j] = True
                    q.append((i, j))

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc] and grid[nr][nc] == 1:
                    visited[nr][nc] = True  # mark before enqueue
                    q.append((nr, nc))

        return sum(
            1 for i in range(m) for j in range(n)
            if grid[i][j] == 1 and not visited[i][j]
        )
```

**vs Surrounded Regions:** identical border flood. Regions flips safe-unreachable `"O"`s; Enclaves counts safe-unreachable `1`s.

### Problem 8: Number of Distinct Islands ([GFG](https://www.geeksforgeeks.org/problems/number-of-distinct-islands/1))

Count islands that are **unique by shape**. Two islands are the same if one is a translate of the other (same relative layout). Rotations / reflections count as different.

```
grid (1 = land)                 shapes (relative to start)

  1 1 0 0 0                     island A start (0,0):
  1 0 0 1 1                       (0,0),(0,1),(1,0)
  0 0 0 1 0
  0 1 1 0 0                     island B start (1,3):
  1 1 0 0 0                       (0,0),(0,1),(1,0)  ← same shape as A

  island C start (3,1): (0,0),(0,1),(1,0),(1,1)  ← different
  answer = 2  (A and B share a shape; C is new)
```

- Pattern: Number of Islands + **shape signature**. While flooding, store each cell as `(r - r0, c - c0)` relative to the island's start.
- Put each signature in a `set` (as a tuple of coords, or a path string of moves). `len(set)` = answer.
- Order of DFS/BFS must be **deterministic** (same dir order always) so identical shapes hash the same.
- **Time** `O(m * n)` | **Space** `O(m * n)` (`visited` + set of shapes)

```python
class Solution:
    def countDistinctIslands(self, grid):
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        shapes = set()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def dfs(i, j, r0, c0, shape):
            visited[i][j] = True
            for dr, dc in dirs:
                nr, nc = i + dr, j + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1 and not visited[nr][nc]:
                    shape.append((nr - r0, nc - c0))  # relative to island origin
                    dfs(nr, nc, r0, c0, shape)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    shape = [(0, 0)]
                    dfs(i, j, i, j, shape)
                    shapes.add(tuple(shape))  # hashable signature
        return len(shapes)
```

**vs Number of Islands:** Islands counts components. Distinct Islands counts **unique shapes** (translate-invariant signatures).

### Problem 9: Detect Cycle in a Directed Graph ([GFG](https://www.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1))

Given `V` nodes and a directed edge list, return whether the graph has a cycle.

```
edges = [[0,1],[1,2],[2,0]]          edges = [[0,1],[1,2]]

  0 → 1 → 2                           0 → 1 → 2
  ↑_______↓                           no back edge → False
  cycle (2→0) → True
```

- Pattern: directed cycle DFS with **`visited` + `pathVisited**` (same idea as 3-color: `pathVisited` = GRAY = on current recursion path).
- Build adj list from edges first (`u → v` only).
- On enter: mark both `visited` and `pathVisited`. On leave: clear `pathVisited` only (node is done, like BLACK).
- Neighbor already on current path (`pathVisited[nb]`) → back edge → cycle.
- Neighbor visited but not on path → safe (another finished branch).
- See [Cycle detection → Directed](#directed-graph) for the WHITE/GRAY/BLACK form.
- **Time** `O(V + E)` | **Space** `O(V + E)` (adj + two bool arrays + recursion)

```python
class Solution:
    def isCyclic(self, V: int, edges: list[list[int]]) -> bool:
        adj = [[] for _ in range(V)]
        for u, v in edges:
            adj[u].append(v)

        visited = [False] * V
        path_visited = [False] * V  # nodes on current DFS path (GRAY)

        def dfs(node):
            visited[node] = True
            path_visited[node] = True
            for nb in adj[node]:
                if not visited[nb]:
                    if dfs(nb):
                        return True
                elif path_visited[nb]:  # back edge to current path
                    return True
            path_visited[node] = False  # leave path (become BLACK)
            return False

        for i in range(V):
            if not visited[i] and dfs(i):
                return True
        return False
```

---

# Bipartite graph

A graph is bipartite if its nodes can be split into two sets so every edge goes between the sets (equivalently: **2-colorable** with no adjacent nodes sharing a color).

```
bipartite (even cycle)          not bipartite (odd cycle)
  A ─── B                         1 ─── 2
  │     │                         │   ╱
  D ─── C                         └── 3
  color: A,C = 0; B,D = 1         1-2-3-1 has length 3 → impossible
```

**Properties**

- Linear path / tree → always bipartite.
- Even-length cycle → bipartite.
- Odd-length cycle → **not** bipartite.
- Equivalent check: no odd cycle exists.

**Algorithm:** BFS or DFS coloring. Assign color `0` to start, flip (`1 - color`) for each neighbor. Conflict (neighbor already same color) → not bipartite. Loop all nodes (graph may be disconnected).

- **Time** `O(n + E)` | **Space** `O(n)` (`color` + queue / stack)

### Problem 10: Is Graph Bipartite ([LC 785](https://leetcode.com/problems/is-graph-bipartite/))

`graph[i]` = neighbors of node `i` (adjacency list, undirected). Return whether the graph is bipartite.

```
graph = [[1,3],[0,2],[1,3],[0,2]]

  0 ─── 1
  │     │
  3 ─── 2     even cycle → True (color 0/1 alternate)
```

- Pattern: **2-coloring** via BFS or DFS.
- `color[i] = -1` means uncolored. Start each component with color `0`.
- Neighbor uncolored → assign opposite color and continue. Neighbor same color → `False`.
- Use `1 - color[node]` (not `not color[node]`) so colors stay `0`/`1` ints.
- **Time** `O(n + E)` | **Space** `O(n)`

**BFS**

```python
from collections import deque
from typing import List

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [-1] * n  # -1 = uncolored, 0/1 = two sides

        for start in range(n):
            if color[start] != -1:
                continue
            q = deque([start])
            color[start] = 0
            while q:
                node = q.popleft()
                for nb in graph[node]:
                    if color[nb] == -1:
                        color[nb] = 1 - color[node]  # opposite side
                        q.append(nb)
                    elif color[nb] == color[node]:  # same side conflict
                        return False
        return True
```

**DFS**

```python
from typing import List

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [-1] * n

        def dfs(node, c):
            color[node] = c
            for nb in graph[node]:
                if color[nb] == -1:
                    if not dfs(nb, 1 - c):  # opposite side
                        return False
                elif color[nb] == c:  # same side conflict
                    return False
            return True

        for start in range(n):
            if color[start] == -1 and not dfs(start, 0):
                return False
        return True
```

---

# Topological sorting

A **topological order** of a **DAG** (directed acyclic graph) is a linear order of nodes such that for every edge `u → v`, `u` appears **before** `v`.

Think: prerequisites. If course `u` must be taken before course `v`, then `u` comes first in the order.

```
DAG (has a topo order)              Not a DAG (cycle → no topo order)

  0 → 1 → 3                           0 → 1 → 2
  ↓   ↓                               ↑_______↓
  2 →─┘                               cycle → impossible

  valid orders: 0,1,2,3  or  0,2,1,3
  (any order that respects all arrows)
```

**When it applies**

- Graph must be **directed**.
- Graph must be **acyclic**. A cycle means mutual dependency → no valid order.
- Multiple valid orders can exist; any one is fine unless the problem asks for lexicographically smallest, etc.

**Two standard algorithms**

|                | DFS (post-order)                        | Kahn's (BFS + indegree)                                    |
| -------------- | --------------------------------------- | ---------------------------------------------------------- |
| Idea           | finish dependents first, then push node | repeatedly take nodes with indegree `0`                    |
| Detects cycle? | need 3-color / pathVisited separately   | **yes** , if `len(result) < V`, there is a cycle           |
| Structure      | recursion / stack + result stack        | queue + indegree array                                     |
| Typical use    | plain topo order                        | Course Schedule, alien dictionary, "detect cycle via topo" |

---

## Method 1: DFS post-order

**Idea:** run DFS. When a node finishes (all its outgoing neighbors are done), push it onto a stack. At the end, reverse the stack → topological order.

Why reverse? The first nodes that finish are the sinks (no outgoing edges left). Sources finish last, so reversing puts sources first.

```
DFS post-order flow

  start DFS(u)
       │
       ▼
  mark visited
       │
       ▼
  for each nb of u ──► DFS(nb) if unvisited
       │
       ▼
  append u to stack   ← post-order: after all children
       │
       ▼
  after all starts: reverse stack → topo order
```

```
example: 0 → 1 → 3
         ↓   ↓
         2 →─┘

DFS from 0:
  visit 0 → 1 → 3 (leaf) → push 3
              → 2 (leaf) → push 2
         push 1
  push 0

  stack (bottom→top): 3, 2, 1, 0
  reverse / pop:      0, 1, 2, 3   ← topo order
```

```python
class Solution:
    def topoSort(self, V, adj):
        visited = [False] * V
        stack = []  # post-order finish times

        def dfs(node):
            visited[node] = True
            for nb in adj[node]:
                if not visited[nb]:
                    dfs(nb)
            stack.append(node)  # push after all dependents

        for i in range(V):
            if not visited[i]:
                dfs(i)

        # reverse post-order = topological order
        return stack[::-1]
```

- Loop all nodes (graph may be disconnected).
- This version assumes the graph **is** a DAG. To also detect cycles, use 3-color / `pathVisited` (see [Directed cycle](#directed-graph)) and abort if a back edge appears.
- **Time** `O(V + E)` | **Space** `O(V)` (`visited` + stack + recursion)

---

## Method 2: Kahn's algorithm (BFS + indegree)

**Idea:** nodes with indegree `0` have no prerequisites → safe to place next. Process them in a queue. For each processed node, "remove" its outgoing edges (decrement neighbor indegrees). Newly zero-indegree nodes enter the queue.

```
Kahn's flow

  compute indegree[v] for all v
       │
       ▼
  enqueue all nodes with indegree == 0
       │
       ▼
  ┌── while queue not empty ──────────────────┐
  │  pop node → append to result               │
  │  for each nb:                              │
  │      indegree[nb] -= 1                     │
  │      if indegree[nb] == 0 → enqueue nb     │
  └────────────────────────────────────────────┘
       │
       ▼
  len(result) == V  →  valid topo order
  len(result) <  V  →  cycle (some nodes never reached indegree 0)
```

```
example: 0 → 1 → 3
         ↓   ↓
         2 →─┘

indegree: 0:0  1:1  2:1  3:2

step 0: queue = [0]           result = []
step 1: take 0 → unlock 1,2   queue = [1,2]   result = [0]
        indegree: 1:0  2:0  3:2
step 2: take 1 → unlock 3     queue = [2,3]   result = [0,1]
        indegree: 3:1
step 3: take 2 → unlock 3     queue = [3]     result = [0,1,2]
        indegree: 3:0
step 4: take 3                queue = []      result = [0,1,2,3]
```

```python
from collections import deque

class Solution:
    def topoSort(self, V, adj):
        indegree = [0] * V
        for u in range(V):
            for v in adj[u]:
                indegree[v] += 1

        q = deque(i for i in range(V) if indegree[i] == 0)
        result = []

        while q:
            node = q.popleft()
            result.append(node)
            for nb in adj[node]:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    q.append(nb)

        # if len(result) < V → cycle exists; no full topo order
        return result
```

- Built-in **cycle check**: incomplete result means a cycle.
- Prefer Kahn's when the problem is "can you finish all courses?" / "return order or empty if impossible."
- **Time** `O(V + E)` | **Space** `O(V)` (`indegree` + queue + result)

---

### DFS vs Kahn

| Problem signal                                     | Reach for                 |
| -------------------------------------------------- | ------------------------- |
| "return any topological order" (DAG guaranteed)    | DFS post-order or Kahn's  |
| "order if possible, else detect cycle / return []" | **Kahn's** (length check) |
| "Course Schedule", prerequisites                   | Kahn's or 3-color DFS     |
| already doing DFS cycle detect                     | extend to post-order push |

---

### Problem 11: Course Schedule ([LC 207](https://leetcode.com/problems/course-schedule/))

`numCourses` courses labeled `0..n-1`. `prerequisites[i] = [a, b]` means take `b` before `a` (edge `b → a`). Return whether you can finish all courses.

```
n = 2, prerequisites = [[1,0]]     n = 2, prerequisites = [[1,0],[0,1]]

  0 → 1   (take 0 before 1)          0 ⇄ 1   cycle
  True                               False
```

- Pattern: **topo sort on a DAG**. Build directed adj from `b → a`. If a full order exists → `True`.
- Kahn's: if `len(result) == numCourses` → can finish. Else cycle → `False`.
- Equivalent: directed cycle DFS; cycle → cannot finish.
- **Time** `O(V + E)` | **Space** `O(V + E)`

```python
from collections import deque
from typing import List

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for a, b in prerequisites:  # b before a → edge b → a
            adj[b].append(a)
            indegree[a] += 1

        q = deque(i for i in range(numCourses) if indegree[i] == 0)
        taken = 0
        while q:
            node = q.popleft()
            taken += 1
            for nb in adj[node]:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    q.append(nb)

        return taken == numCourses  # incomplete → cycle
```

### Problem 12: Course Schedule II ([LC 210](https://leetcode.com/problems/course-schedule-ii/))

Same setup as Course Schedule, but return **any** valid order of courses. If impossible, return `[]`.

```
n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]

  0 → 1 → 3
  ↓   ↑
  2 ──┘

  one valid order: [0,1,2,3]  (or [0,2,1,3])
```

- Pattern: Kahn's, but **return the order** instead of a boolean.
- Empty list when `len(result) < numCourses` (cycle).
- **Time** `O(V + E)` | **Space** `O(V + E)`

```python
from collections import deque
from typing import List

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for a, b in prerequisites:
            adj[b].append(a)
            indegree[a] += 1

        q = deque(i for i in range(numCourses) if indegree[i] == 0)
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for nb in adj[node]:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    q.append(nb)

        return order if len(order) == numCourses else []
```

**vs Course Schedule:** same Kahn's template. 207 checks `taken == n`; 210 returns the order (or `[]`).
