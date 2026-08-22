---
title: "System Design Fundamentals"
date: 2026-08-22
summary: "The standard building blocks: SQL vs NoSQL, vertical vs horizontal scaling, load balancers, replication, cache, CDN, stateless web tier, multiple data centers, message queues, and sharding."
tags: [System Design, Scalability, Databases, Caching, Sharding]
---

How a single-server app grows into one that serves millions, one component at a time. Each component below fixes the bottleneck the previous step created.

---

# The whole picture

```
   client
    │
    ▼
   DNS ─────────────▶ CDN            static assets (js, css, img, video)
    │
    ▼
  ┌────────────────────────────────┐
  │        LOAD BALANCER           │  public VIP · health checks · L4/L7
  └───────┬──────────┬──────────┬──┘
          ▼          ▼          ▼
      ┌───────┐  ┌───────┐  ┌───────┐
      │ WEB 1 │  │ WEB 2 │  │ WEB N │   stateless, autoscaled
      └───┬───┘  └───┬───┘  └───┬───┘
          └──────────┼──────────┘
        ┌────────────┼────────────┬──────────────┐
        ▼            ▼            ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────────┐
   │  CACHE  │  │ SESSION │  │  QUEUE   │  │  SEARCH /  │
   │ (Redis) │  │  STORE  │  │ ─▶ worker│  │  ANALYTICS │
   └────┬────┘  └─────────┘  └──────────┘  └────────────┘
        │ cache miss
        ▼
   ┌──────────────┐   replication   ┌───────────┐  ┌───────────┐
   │  MASTER DB   │ ──────────────▶ │  SLAVE 1  │  │  SLAVE 2  │
   │   (writes)   │                 │  (reads)  │  │  (reads)  │
   └──────────────┘                 └───────────┘  └───────────┘
```

| Bottleneck                             | What you add                     |
| -------------------------------------- | -------------------------------- |
| One server is maxed out                | Split web tier and data tier     |
| Server dies, site dies                 | Load balancer + more web servers |
| Reads are drowning the DB              | Read replicas                    |
| Same queries hit the DB repeatedly     | Cache                            |
| Users far away see lag                 | CDN                              |
| Web tier cannot autoscale              | Move session out, go stateless   |
| Region-wide outage                     | Multiple data centers            |
| Slow work blocks the response          | Message queue + workers          |
| Writes or data size exceed one machine | Sharding                         |

---

# Databases

|            | Relational (SQL / RDBMS)                | Non-relational (NoSQL)                     |
| ---------- | --------------------------------------- | ------------------------------------------ |
| Model      | Tables, rows, columns, joins            | Key-value, document, wide-column, graph    |
| Schema     | Fixed, enforced upfront                 | Flexible, per record                       |
| Guarantees | ACID transactions                       | Eventual consistency, often tunable        |
| Scaling    | Vertical first, sharding is manual work | Horizontal by design                       |
| Examples   | MySQL, Postgres, Oracle                 | Redis, DynamoDB, MongoDB, Cassandra, Neo4j |

Default to SQL. Move to NoSQL when you need single-digit-ms reads, the data is unstructured, you never join (only read and write whole objects), or the volume is too large for one machine.

NoSQL types:

- **Key-value** (Redis, DynamoDB): lookup by exact key. Sessions, cache, feature flags.
- **Document** (MongoDB, CouchDB): query fields inside a JSON doc. Profiles, catalogs.
- **Wide-column** (Cassandra, HBase): partition key + sort key. Time series, event logs.
- **Graph** (Neo4j): query relationships. Social graph, fraud rings.

> Real systems are polyglot: transactional data in Postgres, sessions in Redis, event logs in Cassandra. Pick per access pattern.

---

# Scaling: vertical vs horizontal

```
 VERTICAL (scale up)              HORIZONTAL (scale out)

   ┌──────────┐                   ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │  bigger  │  8 → 64 GB        │ 8G │ │ 8G │ │ 8G │ │ 8G │
   │  server  │  4 → 32 cores     └────┘ └────┘ └────┘ └────┘
   └──────────┘                   many boxes behind a LB
```

|              | Vertical (scale up)   | Horizontal (scale out)         |
| ------------ | --------------------- | ------------------------------ |
| What changes | Nothing in the app    | Needs a LB and a stateless app |
| Ceiling      | One machine's max     | Add nodes as needed            |
| A node dies  | Site goes down        | Pool absorbs it                |
| Cost         | Super-linear          | Roughly linear                 |
| Use when     | Small scale, early on | Any real scale                 |

Vertical is the correct first move: one config change, no code. It runs out on two counts, a hardware ceiling and no failover. Everything below assumes horizontal.

---

# Load balancers

Sits behind a public IP and spreads traffic across a pool of servers. Web servers keep private IPs only, so clients never reach them directly.

```
              public IP
   clients ──────────────▶ ┌───────────────┐
                           │ LOAD BALANCER │
                           └───┬───────┬───┘
                    private IP │       │ private IP
                        ┌──────▼──┐ ┌──▼──────┐
                        │  WEB 1  │ │  WEB 2  │
                        └─────────┘ └─────────┘
                     health check every few seconds:
                     failing server is pulled from the pool
```

What you get:

- Spread load across the pool.
- Failover: a server failing health checks is pulled out.
- Zero-downtime deploy: drain one server, deploy, put it back.
- Security: servers are unreachable except through the LB.

| Layer            | Sees                              | Use for                                                     |
| ---------------- | --------------------------------- | ----------------------------------------------------------- |
| L4 (transport)   | IP, port, TCP/UDP                 | Raw throughput, non-HTTP protocols                          |
| L7 (application) | HTTP path, host, headers, cookies | Route `/api` vs `/static`, TLS termination, sticky sessions |

Routing:

- **Round robin**: default when servers are identical.
- **Weighted round robin**: servers have different capacity.
- **Least connections**: long-lived or uneven requests.
- **Hash on IP or key**: same user to same server, helps cache locality.

---

# Database replication

One master takes all writes. Slaves (replicas) copy the master's write log and serve reads. Real traffic is read-heavy, often around 100 reads per write, so this is usually the first fix for a struggling database.

```
                     writes
   client ──────────────────────▶ ┌──────────┐
                                  │  MASTER  │
                                  └────┬─────┘
                    replication log    │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                   ┌─────────┐   ┌─────────┐   ┌─────────┐
                   │ SLAVE 1 │   │ SLAVE 2 │   │ SLAVE 3 │
                   └─────────┘   └─────────┘   └─────────┘
                        ▲              ▲              ▲
   client ──────────────┴──────────────┴──────────────┘
                              reads
```

What you get:

- Performance: reads scale out.
- Reliability: data survives one disk.
- Availability: reads still work while a node is down.

Failure handling:

- One slave dies: reads go to the remaining slaves.
- All slaves die: reads temporarily hit the master.
- Master dies: promote a slave, repoint writes, replay missing writes from the log.

|               | Sync                                     | Async (default)                      |
| ------------- | ---------------------------------------- | ------------------------------------ |
| Write returns | After replicas acknowledge               | After the master writes              |
| Speed         | Slower                                   | Fast                                 |
| Replica lag   | None                                     | Milliseconds to seconds              |
| Risk          | Write latency, blocked on a slow replica | Stale reads, lost writes on failover |

Async lag causes the **read-your-own-writes** bug: a user posts a comment, the read lands on a stale replica, and the comment looks like it vanished. Fix by routing recently written keys to the master, or pinning that user to the master for a short window.

---

# Cache

A small in-memory store (Redis, Memcached) in front of the database that absorbs repeated reads.

```
   1. read      ┌───────┐  hit → return immediately
   web ────────▶│ CACHE │
                └───┬───┘  miss
                    │ 2. query
                    ▼
                ┌───────┐
                │  DB   │
                └───┬───┘
                    │ 3. write into cache with a TTL, then return
                    └──────────────▶ CACHE
```

| Pattern                    | Flow                           | Trade-off                            |
| -------------------------- | ------------------------------ | ------------------------------------ |
| Cache-aside (read-through) | Miss, read DB, fill cache      | Default. First request pays the miss |
| Write-through              | Write cache and DB together    | Consistent, slower writes            |
| Write-back                 | Write cache, flush to DB later | Fast writes, can lose data on crash  |

Considerations:

- Cache data that is read often and written rarely. Caching volatile data just adds a hop.
- Always set a TTL. Too short loses hit rate, too long serves stale data.
- Cache and DB are written separately, so invalidate on write or accept a known stale window.
- Eviction: LRU by default, LFU for stable hot sets, FIFO.
- A dead cache sends 100% of traffic to the DB. Cluster it across zones.
- Stampede: a hot key expires and every request misses at once. Lock the refill, or jitter TTLs.
- Hot key: one key saturates a node. Replicate it, or add a small local cache.

---

# CDN (Content Delivery Network)

Edge servers spread across regions that cache static content close to users. The first request pulls from your origin and caches at the edge; later requests in that region are served locally.

Considerations:

- Cost: billed per GB, so caching rarely requested assets is waste.
- Expiry: too short kills hit rate, too long serves stale files.
- Invalidation: purge API, or versioned URLs (`app.a1b2c3.js`), which can never go stale.
- Fallback: client should detect CDN failure and fall back to the origin.
- Do not cache per-user or dynamic responses.

---

# Stateless web tier

|                  | Stateful                                     | Stateless                        |
| ---------------- | -------------------------------------------- | -------------------------------- |
| Session lives in | The server's memory                          | A shared store (Redis, DynamoDB) |
| Routing          | Sticky, same user must reach the same server | Any request to any server        |
| Server dies      | That user's session is lost                  | Nothing is lost                  |
| Autoscaling      | Awkward, sticky sessions fight the LB        | Add and remove servers freely    |

```
 STATEFUL (avoid)                 STATELESS (want)

  user A ──▶ WEB 1 [A's session]   user A ──▶ any WEB ──┐
  user B ──▶ WEB 2 [B's session]   user B ──▶ any WEB ──┤
             ▲                                          ▼
       must always come back            ┌─────────────────────┐
       to the same server               │ SHARED STATE STORE  │
                                        └─────────────────────┘
```

Moving state out of the web servers is what makes horizontal scaling work. Keep every web server interchangeable and all state in the data tier.

---

# Multiple data centers

GeoDNS routes users to their nearest region. This cuts latency and lets you survive losing a whole data center by failing traffic over.

```
   US users ──▶ GeoDNS ──▶ ┌──────────────┐
                           │  DC 1 (US)   │
                           └──────┬───────┘
                                  │ cross-region data sync
                           ┌──────▼───────┐
   EU users ──▶ GeoDNS ──▶ │  DC 2 (EU)   │
                           └──────────────┘
```

Challenges:

- Traffic redirection: GeoDNS plus health checks. DNS TTLs mean failover is not instant.
- Data synchronization: cross-region replication is slow, so consistency becomes a design decision.
- Test and deployment: every region must be tested and rolled out gradually, not all at once.

---

# Message queue

A durable buffer between producers and consumers. Producers (publishers) push messages onto the queue; consumers (subscribers) pull them off and do the work on their own schedule.

```
                ┌────────────────────────────┐ ──▶ worker 1
   web ────────▶│  QUEUE  [m4][m3][m2][m1]   │ ──▶ worker 2
  (producer)    └────────────────────────────┘ ──▶ worker 3
                                                     (consumers)
                                                          │ keeps failing
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ DEAD LETTER QUEUE│
                                                 └──────────────────┘
```

Why it helps:

- Decoupling: producer and consumer scale, deploy, and fail independently.
- Buffering: a traffic spike grows the queue instead of dropping requests.
- Async work: slow jobs (thumbnails, emails, invoices) leave the request path.
- Retries: failed messages are redelivered. Poison messages go to a dead-letter queue.

Delivery is usually at-least-once, so consumers must be **idempotent**: handling the same message twice must be harmless. Strict global ordering costs throughput, so most systems only order per key or partition.

---

# Database scaling

Replicas fix read load. When writes or raw data size are the problem, there are two moves.

|        | Vertical                         | Sharding (horizontal)                   |
| ------ | -------------------------------- | --------------------------------------- |
| Move   | Bigger machine (disk, CPU, RAM)  | Split rows across N databases           |
| Cost   | Rises steeply                    | Linear in commodity nodes               |
| Limits | Hardware ceiling, still one SPOF | Needs a shard key and app-level routing |

Each shard holds a disjoint slice of the rows, picked by hashing a **shard key**.

```
   user_id ──▶ hash(user_id) % 4 ──▶
             ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
             │ shard0 │ │ shard1 │ │ shard2 │ │ shard3 │
             └────────┘ └────────┘ └────────┘ └────────┘
              users     users       users      users
              0,4,8..   1,5,9..     2,6,10..   3,7,11..
```

The shard key is the whole design decision. It has to spread data evenly and match how you query, since any query without it fans out to every shard.

Challenges:

- **Resharding**: changing `n` in `% n` remaps nearly every key. Use consistent hashing, or virtual shards mapped to physical nodes.
- **Celebrity / hotspot**: one popular key sends all traffic to one shard. Give it a dedicated shard, or suffix the key to spread it.
- **Joins and denormalization**: cross-shard joins are impractical. Denormalize and duplicate so each query hits one shard.

Try the cheap options first: cache harder, push reads to replicas, archive cold rows, and move logs, sessions, and analytics to a NoSQL store.
