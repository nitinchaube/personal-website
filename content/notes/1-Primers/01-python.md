---
title: "Python"
date: 2026-05-13
summary: "Revision sheet for Python: built-in types, collections, itertools, functions and closures, generators, decorators, context managers, OOP in depth, exceptions, logging, JSON, concurrency, and the gotchas that show up in interviews."
tags: [Python Foundation and intermediate level]
---

Revision notes, not a tutorial. Each section is a definition, the syntax I forget, and the gotchas. Read top to bottom in about 40 minutes, or jump to a section from the table of contents.

Deeper dives that live in their own notes: [Concurrency in Python](/notes/Primers/concurrency).

---

# Part 1: Core built-in types

## 1. The object model, first

Everything is an object with an identity, a type, and a value. Variables are names bound to objects, never boxes holding values.

### 1.1 Mutable vs immutable

| Immutable                                                            | Mutable                                                 |
| -------------------------------------------------------------------- | ------------------------------------------------------- |
| `int`, `float`, `bool`, `str`, `bytes`, `tuple`, `frozenset`, `None` | `list`, `dict`, `set`, `bytearray`, most custom classes |

- Immutable objects can be dict keys and set members. Mutable ones cannot (no stable hash).
- "Changing" an immutable creates a new object. `s.upper()` returns a new string, it does not edit `s`.

### 1.2 `is` vs `==`

```python
a = [1, 2]; b = [1, 2]
a == b        # True    same value
a is b        # False   different objects
id(a), id(b)  # (4382948160, 4382949120)   two different addresses

x = 256; y = 256; x is y   # True    small ints (-5..256) are cached
x = 257; y = 257; x is y   # False   (may be True inside a single code block)
```

Rule: `==` for values, `is` only for `None`, `True`, `False`, and sentinels.

### 1.3 Numbers

```python
7 / 2        # 3.5    true division, always float
7 // 2       # 3      floor division
-7 // 2      # -4     floors toward negative infinity
7 % 3        # 1
-7 % 3       # 2      sign follows the divisor
divmod(7, 2) # (3, 1)
2 ** 10      # 1024
round(2.675, 2)   # 2.67, binary float, not a bug in round
0.1 + 0.2 == 0.3  # False. Use math.isclose or Decimal
```

`int` has unlimited precision. For money use `decimal.Decimal("0.1")`, for exact ratios `fractions.Fraction`.

### 1.4 Truthiness

Falsy: `False`, `None`, `0`, `0.0`, `""`, `[]`, `()`, `{}`, `set()`, and objects whose `__bool__` or `__len__` says so. Everything else is truthy.

```python
bool(0), bool(""), bool([]), bool({}), bool(None)   # all False
bool(-1), bool("0"), bool([0]), bool(" ")           # all True

items = [1, 2]
if items: ...        # prefer this over len(items) > 0

0 or "fallback"      # 'fallback'   returns the first truthy operand, not a bool
"a" or "b"           # 'a'
1 and 0 and 2        # 0            returns the first falsy operand
1 and 2              # 2            all truthy, so the last one

x = 0
if not x: ...        # fires for 0, "", [] and None alike
if x is None: ...    # what you usually mean
```

### 1.5 Type conversion

```python
int("42"), int("ff", 16), int(3.9)   # (42, 255, 3)   int() truncates, never rounds
float("3.14"), str(42)               # (3.14, '42')
bool("False")                        # True    any non-empty string is truthy
int("3.9")                           # ValueError: invalid literal for int()
list("abc")                          # ['a', 'b', 'c']
tuple([1, 2])                        # (1, 2)
set("aab")                           # {'a', 'b'}
dict([("a", 1)])                     # {'a': 1}
type(3.0)                            # <class 'float'>
isinstance(3, (int, float))          # True
isinstance(True, int)                # True    bool is a subclass of int
```

---

## 2. Lists

Ordered, mutable, allows duplicates, mixed types allowed.

### 2.1 Create and index

```python
nums = [3, 1, 4, 1, 5]
empty = []                          # or list()
grid = [[0] * 3 for _ in range(3)]  # [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
nums[0], nums[-1]                   # (3, 5)   first, last
nums[1:4]                           # [1, 4, 1]   stop is exclusive
nums[:3]                            # [3, 1, 4]
nums[3:]                            # [1, 5]
nums[::2]                           # [3, 4, 5]
nums[::-1]                          # [5, 1, 4, 1, 3]   reversed copy
nums[2:99]                          # [4, 1, 5]   slices clamp
nums[99]                            # IndexError: list index out of range
```

Slices return a new list and never raise for out-of-range bounds. Plain indexing raises `IndexError`.

### 2.2 Methods

| Method              | Does                                              | Cost        |
| ------------------- | ------------------------------------------------- | ----------- |
| `append(x)`         | Add one at the end                                | O(1)        |
| `extend(it)`        | Add all items of an iterable                      | O(k)        |
| `insert(i, x)`      | Insert at index                                   | O(n)        |
| `remove(x)`         | Delete first item equal to `x`, else `ValueError` | O(n)        |
| `pop()` / `pop(i)`  | Remove and return last / at index                 | O(1) / O(n) |
| `index(x)`          | First position of `x`, else `ValueError`          | O(n)        |
| `count(x)`          | Occurrences                                       | O(n)        |
| `sort()`            | Sort in place, returns `None`                     | O(n log n)  |
| `reverse()`         | Reverse in place                                  | O(n)        |
| `clear()`, `copy()` | Empty it, shallow copy                            | O(n)        |

Mutating methods return `None`. `nums = nums.sort()` is the classic bug, use `sorted(nums)` for a new list.

### 2.3 Sorting

```python
nums = [3, 1, 4, 1, 5]
nums.sort()                                 # returns None, nums is [1, 1, 3, 4, 5]
sorted(nums, reverse=True)                  # [5, 4, 3, 1, 1]   nums untouched

words = ["banana", "kiwi", "Apple"]
sorted(words, key=len)                      # ['kiwi', 'Apple', 'banana']
sorted(words)                               # ['Apple', 'banana', 'kiwi']   capitals first
sorted(words, key=str.lower)                # ['Apple', 'banana', 'kiwi']

people = [("ann", 30), ("bob", 25), ("cal", 30)]
sorted(people, key=lambda p: p[1])          # [('bob', 25), ('ann', 30), ('cal', 30)]
sorted(people, key=lambda p: (-p[1], p[0])) # [('ann', 30), ('cal', 30), ('bob', 25)]
```

`key` is called once per element. Sort is stable, so equal keys keep their original order.

### 2.4 Gotchas

```python
a = [1, 2, 3]
b = a                 # alias, not a copy
b.append(4)
a                     # [1, 2, 3, 4]   a changed too

bad = [[0] * 3] * 3   # three references to ONE row
bad[0][0] = 9
bad                   # [[9, 0, 0], [9, 0, 0], [9, 0, 0]]

def f(x, acc=[]):     # mutable default, created once at definition
    acc.append(x); return acc
f(1), f(2)            # ([1, 2], [1, 2])   the same list, and both names see it

nums = [1, 2, 2, 3]
for n in nums:        # never add or remove while iterating
    if n == 2: nums.remove(n)
nums                  # [1, 2, 3]   one 2 survived, the loop skipped past it
[n for n in nums if n != 2]   # [1, 3]   do this instead

x in big_list         # O(n). Convert to a set before a loop of lookups
```

### 2.5 Sequence built-ins

```python
nums = [3, 1, 4, 1, 5]
len(nums), sum(nums), min(nums), max(nums)   # (5, 14, 1, 5)
sorted(nums)                                 # [1, 1, 3, 4, 5]
list(reversed(nums))                         # [5, 1, 4, 1, 3]
any(n > 4 for n in nums)                     # True
all(n > 0 for n in nums)                     # True   (also True for an empty iterable)

list(enumerate(["a", "b"], start=1))         # [(1, 'a'), (2, 'b')]
list(zip([1, 2, 3], "ab"))                   # [(1, 'a'), (2, 'b')]   stops at the shortest
list(range(2, 10, 3))                        # [2, 5, 8]

first, *rest = nums                          # first=3, rest=[1, 4, 1, 5]
a, b = 1, 2
a, b = b, a                                  # a=2, b=1
```

---

## 3. Tuples

Ordered, immutable, allows duplicates. Use it for a fixed record, use a list for a growable collection.

```python
point = (3, 4)
rgb = 255, 128, 0     # (255, 128, 0)   the comma makes the tuple, parens optional
single = (5,)         # (5,)   a one-item tuple NEEDS the trailing comma
not_tuple = (5)       # 5      just the int
type(single), type(not_tuple)    # (<class 'tuple'>, <class 'int'>)
empty = ()            # ()
point.count(3), point.index(4)   # (1, 1)
```

Only two methods survive immutability: `count()` and `index()`. Indexing and slicing work as in lists.

### 3.1 Unpacking

```python
name, age, city = ("Ann", 30, "Pune")   # name='Ann', age=30, city='Pune'
first, *middle, last = (1, 2, 3, 4, 5)  # first=1, middle=[2, 3, 4], last=5

def min_max(vals): return min(vals), max(vals)
lo, hi = min_max([3, 1, 4])             # lo=1, hi=4   returning "two things" is a tuple

for i, (k, v) in enumerate({"a": 1, "b": 2}.items()):
    print(i, k, v)                      # 0 a 1
                                        # 1 b 2

a, b = (1, 2, 3)                        # ValueError: too many values to unpack
```

Counts must match, or `ValueError`.

### 3.2 Immutable is shallow

```python
t = (1, [2, 3])
t[1].append(4)     # allowed, the inner list is still mutable
t                  # (1, [2, 3, 4])
t[1] = [9]         # TypeError: 'tuple' object does not support item assignment
hash((1, 2))       # works, an int (the exact value is an implementation detail)
hash((1, [2]))     # TypeError: unhashable type: 'list'
```

A tuple is hashable only if everything inside it is. That is why `(row, col)` is a perfect dict key.

### 3.3 Why pick a tuple

Fixed-shape records, dict keys and set members, multiple return values, constants, slightly smaller and faster. When positions get hard to remember, move up to `namedtuple` or `dataclass`.

---

## 4. Dictionaries

Key to value, hash table, average O(1) lookup. Insertion-ordered since 3.7. Keys must be hashable and unique, values can be anything.

### 4.1 Create

```python
person = {"name": "Ann", "age": 30}
dict(name="Ann", age=30)             # {'name': 'Ann', 'age': 30}
{n: n * n for n in range(5)}         # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
dict(zip(["a", "b"], [1, 2]))        # {'a': 1, 'b': 2}
dict.fromkeys(["a", "b"], 0)         # {'a': 0, 'b': 0}
empty = {}                           # {}   a dict. An empty set is set()
len(person), "name" in person        # (2, True)
```

### 4.2 Access and methods

| Call                                  | Does                                        |
| ------------------------------------- | ------------------------------------------- |
| `d[k]`                                | Value, `KeyError` if missing                |
| `d.get(k, default)`                   | Value or default, never raises              |
| `d.setdefault(k, default)`            | Get, inserting the default first if missing |
| `d.keys()`, `d.values()`, `d.items()` | Live views                                  |
| `d.pop(k, default)`                   | Remove and return                           |
| `d.popitem()`                         | Remove and return the last inserted pair    |
| `d.update(other)`                     | Merge, right side wins                      |
| `del d[k]`, `d.clear()`, `d.copy()`   | Delete, empty, shallow copy                 |

`k in d` tests keys and is O(1). Views are live, and `d1.keys() & d2.keys()` works like a set.

### 4.3 Iterate, merge, sort

```python
d = {"a": 1, "b": 2}
list(d)                          # ['a', 'b']   iterating a dict yields keys
list(d.items())                  # [('a', 1), ('b', 2)]
list(d.values())                 # [1, 2]

x, y = {"p": 1, "q": 2}, {"q": 9}
{**x, **y}                       # {'p': 1, 'q': 9}   any version, right side wins
x | y                            # {'p': 1, 'q': 9}   3.9+
x |= y                           # x is now {'p': 1, 'q': 9}

{v: k for k, v in d.items()}     # {1: 'a', 2: 'b'}

scores = {"ann": 90, "bob": 72, "cal": 85}
dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
                                 # {'ann': 90, 'cal': 85, 'bob': 72}
max(scores, key=scores.get)      # 'ann'
scores.keys() & {"ann", "zoe"}   # {'ann'}   key views behave like sets
```

Never add or delete keys while iterating (`RuntimeError`). Iterate over `list(d)` if you must.

### 4.4 Patterns

```python
words = ["ant", "bee", "ant", "koala"]

counts = {}
for w in words: counts[w] = counts.get(w, 0) + 1
counts            # {'ant': 2, 'bee': 1, 'koala': 1}   Counter does this better

groups = {}
for w in words: groups.setdefault(len(w), []).append(w)
groups            # {3: ['ant', 'bee', 'ant'], 5: ['koala']}   defaultdict is cleaner

actions = {"add": do_add, "del": do_del}
actions["add"](1)         # calls do_add(1). A dispatch table beats a long if/elif
actions.get("nope", noop) # fall back instead of KeyError
```

Gotcha: `dict.fromkeys(keys, [])` gives every key the same list object.

---

## 5. Sets

Unordered, unique, mutable, hashable items only. Same hash table as dict, minus the values.

```python
s = {1, 2, 3}
set([1, 2, 2, 3])              # {1, 2, 3}   duplicates collapse
set("aab")                     # {'a', 'b'}
empty = set()                  # set()   note that {} is a dict
frozenset({1, 2})              # frozenset({1, 2})   immutable, hashable, valid dict key
{len(w) for w in ["ab", "cde", "fg"]}   # {2, 3}   set comprehension

s[0]                           # TypeError: 'set' object is not subscriptable
sorted({3, 1, 2})              # [1, 2, 3]   sort on the way out
list(dict.fromkeys([2, 1, 2]))  # [2, 1]   dedupe while keeping first-seen order
```

No indexing (`s[0]` is a `TypeError`), and iteration order is arbitrary. Use `sorted(s)` when order matters. To dedupe while keeping order: `list(dict.fromkeys(items))`.

### 5.1 Add and remove

| Method                 | Does                                |
| ---------------------- | ----------------------------------- |
| `add(x)`, `update(it)` | Add one, add many                   |
| `remove(x)`            | Remove, `KeyError` if absent        |
| `discard(x)`           | Remove, silent if absent            |
| `pop()`                | Remove and return an arbitrary item |
| `clear()`, `copy()`    | Empty, shallow copy                 |

### 5.2 Set algebra

| Idea                 | Method                      | Result                    |
| -------------------- | --------------------------- | ------------------------- |
| Union                | `a.union(b)`                | Everything in either      |
| Intersection         | `a.intersection(b)`         | Only the shared items     |
| Difference           | `a.difference(b)`           | In `a` but not in `b`     |
| Symmetric difference | `a.symmetric_difference(b)` | In exactly one of the two |

Each one also has an operator form, in order: `|`, `&`, `-`, `^`, plus an in-place variant (`|=`, `&=`, `-=`, `^=`).

```python
a, b = {1, 2, 3, 4}, {3, 4, 5}
a | b            # {1, 2, 3, 4, 5}
a & b            # {3, 4}
a - b            # {1, 2}     not symmetric, b - a is {5}
a ^ b            # {1, 2, 5}
a.union([9])     # {1, 2, 3, 4, 9}   the method takes any iterable
a | [9]          # TypeError: unsupported operand type(s)
a |= b           # a is now {1, 2, 3, 4, 5}. Also &=, -=, ^=
```

Operators need both sides to be sets, the methods accept any iterable.

### 5.3 Comparisons and uses

```python
{1, 2} <= {1, 2, 3}         # True    subset, same as issubset
{1, 2} < {1, 2}             # False   a proper subset must be strictly smaller
{1, 2, 3} >= {1, 2}         # True    superset
{1, 2}.isdisjoint({5})      # True    nothing in common

old, new = {"a", "b"}, {"b", "c"}
new - old                   # {'c'}   added
old - new                   # {'a'}   removed
old & new                   # {'b'}   unchanged
```

Two real uses: O(1) membership tests, and diffing collections (`new - old` is added, `old - new` is removed, `old & new` is kept).

---

## 6. Strings

Ordered, immutable sequence of Unicode characters. Every method returns a new string.

```python
s = "hello world"
s[0]             # 'h'
s[-5:]           # 'world'
s[::-1]          # 'dlrow olleh'   the reverse idiom
len(s)           # 11
s.upper()        # 'HELLO WORLD'
s                # 'hello world'   unchanged, upper() returned a new string
s[0] = "H"       # TypeError: 'str' object does not support item assignment

'a' == "a"       # True    the two quote styles are the same thing
"""multi
line"""          # 'multi\nline'
r"C:\new"        # 'C:\\new'   raw, the backslash stays literal (use for regex)
"café".encode()  # b'caf\xc3\xa9'   4 characters, 5 bytes
```

### 6.1 Methods

| Method                                             | Does                                          |
| -------------------------------------------------- | --------------------------------------------- |
| `upper()`, `lower()`, `casefold()`                 | Case, `casefold` for comparisons              |
| `title()`, `capitalize()`                          | Title Case, Sentence case                     |
| `strip()`, `lstrip()`, `rstrip()`                  | Trim whitespace, or a set of characters       |
| `removeprefix()`, `removesuffix()`                 | Strip an exact affix (3.9+)                   |
| `startswith(p)`, `endswith(p)`                     | Prefix, suffix. Accept a tuple of options     |
| `find(sub)` / `index(sub)`                         | First position, `-1` / `ValueError` if absent |
| `count(sub)`                                       | Non-overlapping occurrences                   |
| `replace(old, new, count)`                         | Replace all, or the first `count`             |
| `split(sep)`, `rsplit`, `splitlines()`             | To a list of pieces                           |
| `join(iterable)`                                   | Called on the separator                       |
| `isdigit()`, `isalpha()`, `isalnum()`, `isspace()` | Content checks, `False` for `""`              |
| `zfill(n)`, `ljust`, `rjust`, `center`             | Padding                                       |
| `encode()`                                         | To bytes                                      |

```python
"a,b,,c".split(",")        # ['a', 'b', '', 'c']   empty field kept
"a b  c".split()           # ['a', 'b', 'c']       whitespace runs collapse
"a-b-c".split("-", 1)      # ['a', 'b-c']          split at most once
", ".join(["a", "b"])      # 'a, b'
", ".join([1, 2])          # TypeError: sequence item 0: expected str, int found
", ".join(str(n) for n in [1, 2])    # '1, 2'
"file.txt".endswith((".txt", ".md")) # True    a tuple tests several at once
"  hi  ".strip()           # 'hi'
"xoxo".strip("ox")         # ''      the argument is a character set, not a prefix
"xoxo".removeprefix("x")   # 'oxo'   what people expect strip to do
"banana".find("na")        # 2
"banana".find("z")         # -1      index() would raise ValueError
"banana".count("na")       # 2       non-overlapping
"banana".replace("a", "A", 2)   # 'bAnAna'
"Hello World".title()      # 'Hello World'
"don't".title()            # "Don'T"   title() mangles apostrophes
"7".zfill(3)               # '007'
"42".isdigit(), "".isdigit()    # (True, False)
```

### 6.2 Build with join, not `+=`

Strings are immutable, so `+=` in a loop is O(n^2). Collect pieces and join once.

```python
words = ["hi", "there"]

out = ""
for w in words: out += w.upper() + " "     # allocates a new string every iteration
out                                        # 'HI THERE '

" ".join(w.upper() for w in words)         # 'HI THERE'   one allocation
```

### 6.3 Formatting

```python
name, score = "Ann", 91.5678
"Hi %s, %.2f" % (name, score)         # 'Hi Ann, 91.57'   old style
"Hi {}, {:.2f}".format(name, score)   # 'Hi Ann, 91.57'   str.format
f"Hi {name}, {score:.2f}"             # 'Hi Ann, 91.57'   f-string, 3.6+, use this
```

| Spec                                              | Result                                            |
| ------------------------------------------------- | ------------------------------------------------- |
| `f"{score:.2f}"`                                  | `'91.57'`                                         |
| `f"{score:10.2f}"`                                | `'     91.57'`, width 10, right-aligned           |
| `f"{name:>10}"`, `f"{name:<10}"`, `f"{name:^10}"` | `'       Ann'`, `'Ann       '`, `'   Ann    '`     |
| `f"{name:*^10}"`                                  | `'***Ann****'`, custom fill                       |
| `f"{1234567:,}"`                                  | `'1,234,567'`                                     |
| `f"{0.8734:.1%}"`                                 | `'87.3%'`                                         |
| `f"{255:04d}"`, `f"{255:#x}"`, `f"{255:b}"`       | `'0255'`, `'0xff'`, `'11111111'`                  |
| `f"{score=}"`                                     | `'score=91.5678'`, best print-debug tool (3.8+)   |
| `f"{obj!r}"`                                      | `repr()` instead of `str()`                       |
| `f"{{}}"`                                         | `'{}'`, literal braces                            |

Do not use f-strings for logging (`logger.info("hi %s", name)` defers formatting) or SQL (injection risk, use parameters).

---

## 7. Collections

### 7.1 `Counter`

```python
from collections import Counter
c = Counter("mississippi")        # Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
c["s"]                            # 4
c["z"]                            # 0    no KeyError, and z is not inserted
c.most_common(2)                  # [('i', 4), ('s', 4)]
c.most_common()[-1]               # ('m', 1)   least common
c.total()                         # 11   (3.10+)
Counter("abc") == Counter("cab")  # True   the anagram test

list(Counter("aab").elements())   # ['a', 'a', 'b']
Counter(["x", "y"]).update(["x"]) # counts become {'x': 2, 'y': 1}

a, b = Counter("aab"), Counter("abb")   # {'a': 2, 'b': 1}, {'a': 1, 'b': 2}
a + b                             # Counter({'a': 3, 'b': 3})
a - b                             # Counter({'a': 1})   drops zero and negative
a & b                             # Counter({'a': 1, 'b': 1})   per-key minimum
a | b                             # Counter({'a': 2, 'b': 2})   per-key maximum
```

### 7.2 `defaultdict`

```python
from collections import defaultdict

groups = defaultdict(list)                  # pass a factory, not a value
for w in ["ant", "bee", "ape"]: groups[w[0]].append(w)
groups          # defaultdict(<class 'list'>, {'a': ['ant', 'ape'], 'b': ['bee']})
dict(groups)    # {'a': ['ant', 'ape'], 'b': ['bee']}

counts = defaultdict(int)
for w in ["x", "y", "x"]: counts[w] += 1
counts["x"]     # 2
counts["zzz"]   # 0, but this READ also inserts 'zzz'
len(counts)     # 3

graph = defaultdict(list)                   # adjacency lists for graphs
graph[1].append(2)                          # {1: [2]}, no key check needed
nested = defaultdict(lambda: defaultdict(int))
nested["a"]["b"] += 1                       # {'a': {'b': 1}}   two-level counting
```

Reading a missing key inserts it. Use `.get()` when you do not want that.

### 7.3 `namedtuple`

```python
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
p                   # Point(x=3, y=4)   a readable repr for free
p.x, p[0]           # (3, 3)   by name and by index
x, y = p            # x=3, y=4   still a tuple
p._replace(x=10)    # Point(x=10, y=4)   a NEW object, p is unchanged
p._asdict()         # {'x': 3, 'y': 4}
Point._fields       # ('x', 'y')
p.x = 9             # AttributeError: can't set attribute
```

Same memory as a tuple, readable field names. Need mutability or methods? Use `@dataclass`.

### 7.4 `deque`

```python
from collections import deque
dq = deque([1, 2, 3])
dq.append(4); dq.appendleft(0)        # both O(1)
dq.pop(); dq.popleft()                # both O(1), list.pop(0) is O(n)
dq.extendleft([1, 2])                 # inserts in reverse
dq.rotate(1)
window = deque(maxlen=3)              # full deque drops from the far end
```

Right structure for queues, BFS, and sliding windows. Middle indexing is O(n).

### 7.5 `OrderedDict` and `ChainMap`

```python
from collections import OrderedDict, ChainMap

od = OrderedDict(a=1, b=2, c=3)
od.move_to_end("a")            # OrderedDict([('b', 2), ('c', 3), ('a', 1)])
od.popitem(last=False)         # ('b', 2)   pops from the FRONT
OrderedDict(a=1, b=2) == OrderedDict(b=2, a=1)   # False   order is part of equality
dict(a=1, b=2) == dict(b=2, a=1)                 # True    order ignored

cfg = ChainMap({"level": 5}, {"level": 1, "debug": False})
cfg["level"]                   # 5      first mapping wins
cfg["debug"]                   # False  falls through to the next one
dict(cfg)                      # {'level': 5, 'debug': False}
```

Plain dicts keep order now, so reach for `OrderedDict` only when order affects equality or you need `move_to_end`.

---

## 8. Itertools

Lazy iterators. Nothing is computed until you loop, and each one is single-use.

### 8.1 Infinite

```python
from itertools import count, cycle, repeat, islice

count(10, 2)                        # count(10, 2)   a lazy object, prints nothing
list(islice(count(10, 2), 4))       # [10, 12, 14, 16]
list(islice(cycle("AB"), 5))        # ['A', 'B', 'A', 'B', 'A']
list(repeat(1, times=3))            # [1, 1, 1]
list(zip("abc", count()))           # [('a', 0), ('b', 1), ('c', 2)]
```

Always bound them with `zip`, `islice`, or a `break`.

### 8.2 Combinatorics

```python
from itertools import product, permutations, combinations, combinations_with_replacement

["".join(p) for p in product("AB", repeat=2)]
                        # ['AA', 'AB', 'BA', 'BB']   replaces nested loops
["".join(p) for p in permutations("ABC", 2)]
                        # ['AB', 'AC', 'BA', 'BC', 'CA', 'CB']   order matters
["".join(c) for c in combinations("ABC", 2)]
                        # ['AB', 'AC', 'BC']   order does not matter
["".join(c) for c in combinations_with_replacement("AB", 2)]
                        # ['AA', 'AB', 'BB']

list(product([1, 2], "ab"))    # [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
len(list(permutations(range(5), 2)))   # 20   which is 5!/3!
```

Counts: permutations n!/(n-k)!, combinations n!/(k!(n-k)!). All of these blow up fast.

### 8.3 Everyday ones

```python
from itertools import chain, islice, accumulate, groupby, tee, zip_longest, starmap, pairwise, compress, takewhile, dropwhile, filterfalse

nums = [1, 2, 3, 1, 5]

list(chain([1, 2], [3]))                    # [1, 2, 3]   flatten one level
list(chain.from_iterable([[1, 2], [3]]))    # [1, 2, 3]   same, from one iterable
list(islice(nums, 1, 4))                    # [2, 3, 1]   slicing for iterators
list(islice(nums, 0, None, 2))              # [1, 3, 5]   every second item
list(accumulate([1, 2, 3, 4]))              # [1, 3, 6, 10]   running total
list(accumulate(nums, max))                 # [1, 2, 3, 3, 5]   running maximum
list(zip_longest("ab", [1], fillvalue=0))   # [('a', 1), ('b', 0)]
list(starmap(pow, [(2, 3), (3, 2)]))        # [8, 9]   each tuple is unpacked as args
list(pairwise([1, 2, 3]))                   # [(1, 2), (2, 3)]   (3.10+)
list(takewhile(lambda n: n < 3, nums))      # [1, 2]      stops at the first failure
list(dropwhile(lambda n: n < 3, nums))      # [3, 1, 5]   skips until the first failure
list(filterfalse(lambda n: n < 3, nums))    # [3, 5]      the opposite of filter
list(compress("abcd", [1, 0, 1, 0]))        # ['a', 'c']

it = iter([1, 2])
a, b = tee(it, 2)
list(a), list(b)                            # ([1, 2], [1, 2])   two independent copies
```

`groupby` groups **consecutive** equal keys, so sort by the same key first:

```python
from itertools import groupby

words = ["ant", "ape", "bee", "ax"]
for k, grp in groupby(words, key=lambda w: w[0]):
    print(k, list(grp))
# a ['ant', 'ape']
# b ['bee']
# a ['ax']            two separate 'a' groups, because they were not adjacent

for k, grp in groupby(sorted(words), key=lambda w: w[0]):
    print(k, list(grp))
# a ['ant', 'ape', 'ax']
# b ['bee']           sort by the same key first and you get one group per key
```

---

# Part 2: Functions and language mechanics

## 9. Control flow

```python
x = 5
if x > 0: ...
elif x == 0: ...
else: ...

"pos" if x > 0 else "neg"       # 'pos'    ternary
0 < x < 10                      # True     chained comparison

list(range(5))                  # [0, 1, 2, 3, 4]
list(range(2, 10, 3))           # [2, 5, 8]
list(range(5, 0, -1))           # [5, 4, 3, 2, 1]

for n in [1, 2, 3]:
    if n == 9: break
else:
    print("not found")          # not found    for/else runs only when no break fired

items = [1, 2, 3, 4]
if (n := len(items)) > 3:       # walrus assigns inside the expression
    print(n)                    # 4
```

`match` (3.10+), which is structural pattern matching, not a C switch:

```python
command = "go north"

match command.split():
    case ["go", direction]: print("moving", direction)   # moving north
    case ["quit" | "exit"]: sys.exit()                   # alternatives with |
    case ["take", *items]: print(items)                  # rest captured as a list
    case {"type": "user", "id": uid}: load(uid)          # dict patterns
    case Point(x=0, y=0): print("origin")                # class patterns
    case [x] if x > 0: print("guarded")                  # pattern plus condition
    case _: print("unknown")                             # default
```

---

## 10. Comprehensions

```python
nums = [1, 2, 3, 4]
words = ["ab", "cde"]
grid = [[1, 2], [3, 4]]

[n * n for n in range(5)]          # [0, 1, 4, 9, 16]
{n: n * n for n in range(3)}       # {0: 0, 1: 1, 2: 4}
{len(w) for w in words}            # {2, 3}
(n * n for n in range(5))          # <generator object ...>   lazy, no list built
sum(n * n for n in range(5))       # 30    parens dropped inside a call

[n for n in nums if n % 2 == 0]    # [2, 4]   trailing if filters
["even" if n % 2 == 0 else "odd" for n in nums]
                                   # ['odd', 'even', 'odd', 'even']   if/else transforms
[n for n in nums if n > 1 if n < 4]   # [2, 3]   two filters
[i for row in grid for i in row]      # [1, 2, 3, 4]   flatten, loops read left to right
[[c * 2 for c in row] for row in grid]  # [[2, 4], [6, 8]]   nested comprehension
[(x, y) for x in "ab" for y in (1, 2)]
                                   # [('a', 1), ('a', 2), ('b', 1), ('b', 2)]
```

Use a generator expression when you only iterate once, especially inside `sum()`, `any()`, `min()`, or `join()`. Drop the extra parentheses there: `sum(x * x for x in nums)`.

Comprehensions have their own scope, so the loop variable does not leak.

---

## 11. Function arguments

```python
def f(a, b=2, *args, c, d=4, **kwargs):
    print(a, b, args, c, d, kwargs)

f(1, c=3)                       # 1 2 () 3 4 {}
f(1, 9, 8, 7, c=3, e=5)         # 1 9 (8, 7) 3 4 {'e': 5}
f(1)                            # TypeError: missing keyword-only argument: 'c'
```

Order in a definition: positional, defaults, `*args`, keyword-only, `**kwargs`.

| Kind                | Definition                    | Call                            |
| ------------------- | ----------------------------- | ------------------------------- |
| Positional          | `def f(a, b)`                 | `f(1, 2)`                       |
| Keyword             | any parameter                 | `f(b=2, a=1)`, order free       |
| Default             | `def f(a, b=2)`               | `f(1)`                          |
| Variadic positional | `def f(*args)`                | `f(1, 2, 3)`, `args` is a tuple |
| Variadic keyword    | `def f(**kwargs)`             | `f(x=1)`, `kwargs` is a dict    |
| Keyword-only        | after `*`: `def f(a, *, key)` | must call `f(1, key=2)`         |
| Positional-only     | before `/`: `def f(a, /, b)`  | `a` cannot be passed by name    |

### 11.1 Gotchas

```python
def bad(item, bucket=[]):        # default evaluated ONCE, at definition time
    bucket.append(item)
    return bucket

bad(1)                           # [1]
bad(2)                           # [1, 2]      the same list is still there
bad.__defaults__                 # ([1, 2],)   you can see it on the function

def good(item, bucket=None):
    bucket = [] if bucket is None else bucket
    bucket.append(item)
    return bucket

good(1), good(2)                 # ([1], [2])  fresh list each call
```

Argument passing is by object reference. Rebinding a parameter inside the function does not affect the caller, but mutating a mutable argument does.

```python
def rebind(lst): lst = [9]       # rebinds the local name only
def mutate(lst): lst.append(9)   # mutates the caller's object

a = [1]
rebind(a); a                     # [1]      unaffected
mutate(a); a                     # [1, 9]   changed
```

---

## 12. The asterisk operator

```python
# 1. Multiply, power, repeat
2 * 3, 2 ** 3, [0] * 3, "ab" * 2      # (6, 8, [0, 0, 0], 'abab')

# 2. Pack in a definition
def show(*args, **kwargs): print(args, kwargs)
show(1, 2, x=3)                       # (1, 2) {'x': 3}

# 3. Unpack at a call site
def add(a, b, c): return a + b + c
add(*[1, 2, 3])                       # 6
add(*[1, 2], **{"c": 3})              # 6

# 4. Unpack into new literals
a, b = [1, 2], [3]
[*a, *b]                              # [1, 2, 3]
{*a, *b}                              # {1, 2, 3}
{**{"x": 1}, **{"y": 2}}              # {'x': 1, 'y': 2}
[*"abc"]                              # ['a', 'b', 'c']

# 5. Extended unpacking
first, *rest = [1, 2, 3]              # first=1, rest=[2, 3]
*init, last = [1, 2, 3]               # init=[1, 2], last=3
a, *_, z = [1, 2, 3, 4]               # a=1, z=4, middle ignored

# 6. Signature markers
def kw_only(a, *, key): ...
kw_only(1, 2)                         # TypeError: takes 1 positional argument
def pos_only(a, /, b): ...
pos_only(a=1, b=2)                    # TypeError: 'a' is positional-only

# 7. Everyday spreads
print(*[1, 2, 3], sep=", ")           # 1, 2, 3
list(zip(*[[1, 2], [3, 4]]))          # [(1, 3), (2, 4)]   transpose
```

---

## 13. Lambda functions

Single-expression anonymous function. No statements, no annotations, no docstring.

```python
square = lambda x: x * x           # if you name it, just use def
square(4)                          # 16
(lambda a, b=1: a + b)(2)          # 3
```

Real uses are as a throwaway `key` or callback:

```python
nums = [1, 2, 3]
pairs = [("a", 2), ("b", 5), ("c", 2)]
d = {"x": 1, "y": 9}

sorted(pairs, key=lambda kv: kv[1])            # [('a', 2), ('c', 2), ('b', 5)]
sorted(pairs, key=lambda kv: (-kv[1], kv[0]))  # [('b', 5), ('a', 2), ('c', 2)]
max(d, key=lambda k: d[k])                     # 'y'
list(map(lambda x: x * 2, nums))               # [2, 4, 6]
list(filter(lambda x: x % 2, nums))            # [1, 3]

from functools import reduce
reduce(lambda a, b: a + b, nums, 0)            # 6
reduce(lambda a, b: a * b, nums)               # 6
```

Prefer a comprehension over `map`/`filter` with a lambda: `[x * 2 for x in nums]` reads better. `operator.itemgetter(1)` and `attrgetter("age")` are faster and clearer than an index lambda.

Late-binding trap:

```python
fs = [lambda: i for i in range(3)]
[f() for f in fs]                         # [2, 2, 2]   i is looked up when called

fs = [lambda i=i: i for i in range(3)]
[f() for f in fs]                         # [0, 1, 2]   the default binds now
```

---

## 14. Scope and closures

LEGB lookup order: Local, Enclosing, Global, Built-in.

```python
count = 0
def outer():
    total = 0
    def inner():
        global count       # rebind the module-level name
        nonlocal total     # rebind the enclosing function's name
        count += 1
        total += 1
    inner(); inner()
    return total

outer()                    # 2
count                      # 2

def broken():
    x = x + 1              # UnboundLocalError: x referenced before assignment
x = 10                     # the global x is shadowed by the local one
```

Without `global` or `nonlocal`, assigning to a name inside a function creates a new local, and reading it before assignment raises `UnboundLocalError`.

A closure is a nested function that remembers the enclosing variables it uses, even after the outer call returns. It is the mechanism behind decorators.

```python
def multiplier(n):
    def mul(x): return x * n
    return mul

double = multiplier(2)
triple = multiplier(3)
double(5), triple(5)                  # (10, 15)   each keeps its own n
double.__closure__[0].cell_contents   # 2          the captured value
double.__name__                       # 'mul'
```

---

## 15. Iterators and generators

### 15.1 Iterable vs iterator

- **Iterable**: has `__iter__`, can be looped many times (list, dict, str).
- **Iterator**: has `__iter__` **and** `__next__`, is consumed once, raises `StopIteration` at the end.

```python
nums = [1, 2]
it = iter(nums)
next(it)                 # 1
next(it)                 # 2
next(it)                 # StopIteration
next(it, "done")         # 'done'   a default instead of the exception

iter(nums) is nums       # False   a list is iterable but is not an iterator
iter(it) is it           # True    an iterator returns itself
```

A `for` loop is just `iter()` plus repeated `next()` inside a `try`.

### 15.2 Generator functions

Any function with `yield` returns a generator. It pauses at each `yield` and resumes on the next `next()`, keeping local state.

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

g = countdown(3)
g                                   # <generator object countdown at 0x...>
next(g)                             # 3     runs up to the first yield, then pauses
list(g)                             # [2, 1]   picks up where it left off
list(g)                             # []       exhausted, and it stays that way

def read_big(path):                 # constant memory over a huge file
    with open(path) as f:
        for line in f:
            yield line.strip()

def fib():                          # infinite, safe because it is lazy
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

list(islice(fib(), 8))              # [0, 1, 1, 2, 3, 5, 8, 13]

def flatten(items):
    for i in items:
        if isinstance(i, list):
            yield from flatten(i)   # delegate to a sub-generator
        else:
            yield i

list(flatten([1, [2, [3, 4]], 5]))  # [1, 2, 3, 4, 5]
```

### 15.3 Why and watch out

Why: lazy (nothing computed until needed), memory-flat (no list held in RAM), composable into pipelines, can be infinite.

```python
lines = (l for l in open("f.txt"))           # generator expression
values = (int(l) for l in lines if l.strip())
sum(values)                                  # one pass, one item in memory at a time

import sys
sys.getsizeof([n for n in range(10_000)])    # 85176   the whole list
sys.getsizeof((n for n in range(10_000)))    # 192     just the generator

g = (n for n in range(3))
list(g), list(g)                             # ([0, 1, 2], [])   single use
len(g)                                       # TypeError: object of type has no len()
```

Watch out: single use (loop twice and the second is empty), no `len()` and no indexing, and a `return` inside a generator just ends it. `gen.send(value)`, `throw()`, and `close()` exist for coroutine-style use, which is rare in day-to-day code.

---

## 16. Decorators

A decorator is a callable that takes a function and returns a replacement. `@dec` above `def f` means `f = dec(f)`.

```python
import functools, time

def timer(func):
    @functools.wraps(func)            # keep __name__, __doc__, signature
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timer
def work(n):
    time.sleep(n)
    return n * 2

work(0.5)                # prints: work took 0.501s
                         # returns 1.0
work.__name__            # 'work'      without @wraps this would be 'wrapper'
work.__wrapped__         # <function work at 0x...>   the original
```

Always use `*args, **kwargs` so any signature works, always return the result, always add `functools.wraps`.

### 16.1 Decorator with arguments

Three levels: arguments, then the function, then the call.

```python
def retry(times=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"attempt {attempt + 1} failed: {e}")
                    if attempt == times - 1: raise
        return wrapper
    return decorator

@retry(times=3)
def flaky():
    raise ValueError("boom")

flaky()
# attempt 1 failed: boom
# attempt 2 failed: boom
# attempt 3 failed: boom
# ValueError: boom          re-raised on the last attempt
```

### 16.2 Stacking and built-ins

```python
def a(f): print("a applied"); return f
def b(f): print("b applied"); return f

@a
@b
def g(): ...
# b applied
# a applied        f = a(b(f)), so the bottom one is applied first
```

| Decorator                                            | Use                                            |
| ---------------------------------------------------- | ---------------------------------------------- |
| `@property`, `@x.setter`                             | Managed attribute                              |
| `@staticmethod`, `@classmethod`                      | Method kinds                                   |
| `@functools.lru_cache(maxsize=None)` / `@cache`      | Memoize on the arguments (must be hashable)    |
| `@functools.cached_property`                         | Compute once per instance, store on it         |
| `@functools.wraps`                                   | Preserve metadata in your own decorators       |
| `@contextlib.contextmanager`                         | Generator to context manager                   |
| `@dataclasses.dataclass`                             | Generated `__init__`, `__repr__`, `__eq__`     |
| `@functools.singledispatch`, `@singledispatchmethod` | Overload on the first argument's type          |
| `@functools.total_ordering`                          | Fill in comparisons from `__eq__` and `__lt__` |
| `@abc.abstractmethod`                                | Force subclasses to implement                  |
| `@typing.final`, `@typing.override`                  | Signal intent to type checkers (3.12)          |

### 16.3 The patterns worth knowing

```python
# 1. Registry: record the function and return it UNCHANGED, no wrapper needed
HANDLERS = {}
def handler(name):
    def deco(func):
        HANDLERS[name] = func
        return func
    return deco

@handler("csv")
def load_csv(path): ...

HANDLERS                   # {'csv': <function load_csv at 0x...>}
load_csv                   # still the plain function, nothing was wrapped

# 2. Stateful: hang the state on the wrapper instead of using a global
def counted(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@counted
def ping(): return "pong"

ping(); ping()
ping.calls                 # 2

# 3. Class-based: __call__ makes the instance the decorator. Good when state is real
class Retry:
    def __init__(self, times=3): self.times = times
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(self.times):
                try: return func(*args, **kwargs)
                except Exception:
                    if i == self.times - 1: raise
        return wrapper

# 4. Usable both bare and with arguments
def log(func=None, *, level="INFO"):
    if func is None:
        return functools.partial(log, level=level)     # called as @log(level=...)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.log(level, "calling %s", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@log                       # works bare
def a(): ...
@log(level="DEBUG")        # and with arguments
def b(): ...

# 5. Async, and the version that handles either kind
import inspect
def timed(func):
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)         # must await inside
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    return wrapper

# 6. Class decorator: takes a class, returns a class. This is how @dataclass works
def auto_repr(cls):
    cls.__repr__ = lambda self: f"{cls.__name__}({self.__dict__})"
    return cls

@auto_repr
class P:
    def __init__(self): self.x = 1

P()                        # P({'x': 1})

# 7. Typed decorator that preserves the signature for mypy (3.10+)
from typing import Callable, ParamSpec, TypeVar
P = ParamSpec("P"); R = TypeVar("R")
def deco(func: Callable[P, R]) -> Callable[P, R]: ...

# 8. A context manager that is also a decorator
class Timer(contextlib.ContextDecorator):
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *exc): print(time.perf_counter() - self.t); return False

@Timer()
def work(): ...            # and "with Timer():" still works
```

`@contextmanager` functions get this for free, so `@my_cm()` works as a decorator without extra effort.

### 16.4 Pitfalls

- **Decorators run at import time**, the wrapper runs per call. Heavy work in the decorator body slows startup and is easy to miss.
- **Order matters with method decorators.** `@property`, `@classmethod`, and `@staticmethod` go outermost, your decorator goes closest to `def`.
- **A cached method leaks.** `@lru_cache` on a method puts `self` in the key and keeps the instance alive forever. Use `@cached_property`, or cache a module-level function instead.
- **Caches need hashable arguments.** A `dict` or `list` argument raises `TypeError: unhashable type`. The cache is also per-process, not shared.
- **Forgetting to return.** If the wrapper does not `return func(...)`, every decorated call returns `None`. Silent and common.
- **Skipping `functools.wraps`** loses `__name__`, `__doc__`, and annotations, which breaks docs tooling and test discovery. It also sets `__wrapped__`, so `inspect.signature` sees through the wrapper and `inspect.unwrap(f)` recovers the original.
- **Recursion goes through the wrapper every time.** That is what makes `@cache` turn exponential recursion into linear, and what makes a logging decorator extremely noisy.
- Debugging: check `f.__wrapped__`, `f.__name__`, and `inspect.signature(f)` to confirm what you actually got.

---

## 17. Context managers

`with` guarantees setup and teardown, even when an exception is raised.

```python
with open("f.txt") as f:              # closed automatically
    data = f.read()

with open("in") as src, open("out", "w") as dst:      # multiple
    dst.write(src.read())
```

### 17.1 Two ways to write one

```python
class Timer:
    def __enter__(self):
        print("enter")
        self.start = time.perf_counter()
        return self                       # this is what "as t" gets
    def __exit__(self, exc_type, exc_val, tb):
        self.elapsed = time.perf_counter() - self.start
        print("exit", exc_type)
        return False                      # False propagates, True swallows

with Timer() as t:
    time.sleep(0.1)
# enter
# exit None
t.elapsed                                 # 0.1003...

with Timer() as t:
    raise ValueError("boom")
# enter
# exit <class 'ValueError'>
# ValueError: boom                        teardown ran, then the error propagated

from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield "handle"                    # the with body runs here
    finally:
        print(f"took {time.perf_counter() - start:.1f}s")   # runs even on exception

with timer() as h:
    h                                     # 'handle'
    time.sleep(0.2)
# took 0.2s
```

Put teardown in `finally` in the generator form, or it is skipped when the body raises.

### 17.2 `contextlib` helpers

```python
from contextlib import suppress, closing, redirect_stdout, ExitStack, nullcontext
import io

with suppress(FileNotFoundError):
    os.remove("nope.txt")                 # no error, no traceback, one line

buf = io.StringIO()
with redirect_stdout(buf):
    print("captured")
buf.getvalue()                            # 'captured\n'

with ExitStack() as stack:                # a variable number of managers
    files = [stack.enter_context(open(p)) for p in paths]
    len(files)                            # all open here, all closed after

cm = nullcontext() if not debug else Timer()
with cm:                                  # a no-op stand-in, keeps one code path
    ...
```

Async version: `async with`, backed by `__aenter__` and `__aexit__`.

---

## 18. Shallow vs deep copying

```python
import copy
a = [1, 2]
b = a                      # 1. alias. Same object, no copy at all
b is a                     # True
b = a[:]                   # 2. shallow: list(a), a.copy(), dict(d), copy.copy(a)
b is a, b == a             # (False, True)
b = copy.deepcopy(a)       # 3. deep: recursively copies nested objects
```

|              | New outer object | New inner objects |
| ------------ | ---------------- | ----------------- |
| Assignment   | No               | No                |
| Shallow copy | Yes              | No, shared        |
| Deep copy    | Yes              | Yes               |

```python
a = [[1, 2], [3]]

s = copy.copy(a)
s[0].append(9)
a                        # [[1, 2, 9], [3]]   the inner list was shared
s.append([4])
a                        # [[1, 2, 9], [3]]   but the outer list is separate

a = [[1, 2], [3]]
d = copy.deepcopy(a)
d[0].append(9)
a                        # [[1, 2], [3]]      untouched all the way down

t = (1, 2)
tuple(t) is t            # True   copying an immutable is a no-op
```

Notes: shallow is enough when items are immutable, `deepcopy` is slow and handles cycles, and it can fail on file handles, sockets, and locks. Customize with `__copy__` and `__deepcopy__`. For immutable types copying is pointless, `tuple(t) is t`.

---

# Part 3: OOP in Python

## 19. Classes and instances

A class is a blueprint, an instance is one object built from it. `self` is the instance, passed automatically.

```python
class Dog:
    species = "Canis familiaris"        # class attribute, shared

    def __init__(self, name, age):      # initializer, not a constructor
        self.name = name                # instance attributes, per object
        self.age = age

    def bark(self):                     # instance method
        return f"{self.name} says woof"

    def __repr__(self):
        return f"Dog({self.name!r}, {self.age})"

d = Dog("Rex", 3)
d                                       # Dog('Rex', 3)      from __repr__
d.bark()                                # 'Rex says woof'
Dog.bark(d)                             # 'Rex says woof'    the same call
isinstance(d, Dog), type(d) is Dog      # (True, True)
d.name, d.species                       # ('Rex', 'Canis familiaris')
```

### 19.1 Instance vs class attributes

```python
d, e = Dog("Rex", 3), Dog("Sam", 5)

Dog.species          # 'Canis familiaris'
d.species            # 'Canis familiaris'   found on the class
d.__dict__           # {'name': 'Rex', 'age': 3}   instance attributes only

d.species = "x"      # creates an INSTANCE attribute that shadows the class one
d.species, e.species # ('x', 'Canis familiaris')
d.__dict__           # {'name': 'Rex', 'age': 3, 'species': 'x'}

Dog.species = "y"
d.species, e.species # ('x', 'y')   d still shadows, e sees the new class value
del d.species
d.species            # 'y'   the shadow is gone, lookup falls back again
```

Lookup order: instance `__dict__`, then the class, then base classes along the MRO.

Gotcha, the class-level mutable:

```python
class Bad:
    items = []           # SHARED by every instance
class Good:
    def __init__(self):
        self.items = []  # one list per instance
```

### 19.2 Three kinds of method

| Kind            | First arg | Sees                       | Use for                              |
| --------------- | --------- | -------------------------- | ------------------------------------ |
| Instance method | `self`    | The instance and the class | Normal behaviour                     |
| `@classmethod`  | `cls`     | The class only             | Alternative constructors, factories  |
| `@staticmethod` | none      | Nothing                    | A helper that belongs with the class |

```python
KNOWN = {"tomato", "mozzarella"}

class Pizza:
    def __init__(self, toppings): self.toppings = toppings
    def describe(self): return f"{type(self).__name__}: {self.toppings}"

    @classmethod
    def margherita(cls):                       # cls respects subclasses
        return cls(["tomato", "mozzarella"])

    @staticmethod
    def is_valid(topping):
        return topping in KNOWN

class DeepDish(Pizza): pass

Pizza.margherita().describe()      # "Pizza: ['tomato', 'mozzarella']"
type(DeepDish.margherita())        # <class '__main__.DeepDish'>
                                   # a DeepDish, not a Pizza, because of cls
Pizza.is_valid("pineapple")        # False   callable on the class or an instance
Pizza(["tomato"]).is_valid("tomato")   # True
```

### 19.3 Encapsulation

Python relies on convention, not enforcement.

```python
class Account:
    def __init__(self, balance):
        self.owner = "Ann"        # public
        self._balance = balance   # internal by convention, "please do not touch"
        self.__secret = 1         # name-mangled to _Account__secret

a = Account(100)
a.owner                # 'Ann'
a._balance             # 100    accessible, the underscore is only a signal
a.__secret             # AttributeError: 'Account' object has no attribute '__secret'
a._Account__secret     # 1      mangling, not security
a.__dict__             # {'owner': 'Ann', '_balance': 100, '_Account__secret': 1}
```

Single underscore is a hint and is what real code uses. Double underscore mangles the name to avoid collisions in subclasses, it is not security. Dunder names like `__init__` are reserved by the language, never invent your own.

### 19.4 Properties

Turn a method into an attribute so you can add validation later without changing callers.

```python
class Circle:
    def __init__(self, radius): self.radius = radius     # goes through the setter

    @property
    def radius(self): return self._radius

    @radius.setter
    def radius(self, value):
        if value <= 0: raise ValueError("radius must be positive")
        self._radius = value

    @property
    def area(self):                    # computed, read-only
        return 3.14159 * self._radius ** 2

c = Circle(2)
c.radius        # 2         calls the getter, no parentheses
c.area          # 12.56636  computed on access
c.radius = 5
c.area          # 78.53975  stays consistent automatically
c.radius = -1   # ValueError: radius must be positive
Circle(-1)      # ValueError too, because __init__ goes through the setter
c.area = 10     # AttributeError: property 'area' has no setter
```

Start with a plain attribute. Promote it to a property only when you need validation, computation, or a deprecation shim. `@functools.cached_property` is the version that computes once and caches on the instance.

---

## 20. Inheritance and polymorphism

### 20.1 Inheritance and `super()`

```python
class Animal:
    def __init__(self, name): self.name = name
    def speak(self): raise NotImplementedError
    def info(self): return f"{self.name} is a {type(self).__name__}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)          # always call it, do not repeat the parent
        self.breed = breed
    def speak(self): return "Woof"      # override

class Puppy(Dog):
    def speak(self):
        return super().speak() + "!"    # extend rather than replace

p = Puppy("Rex", "corgi")
p.speak()                     # 'Woof!'
p.info()                      # 'Rex is a Puppy'   inherited, type(self) is Puppy
p.name, p.breed               # ('Rex', 'corgi')
Animal("x").speak()           # NotImplementedError
issubclass(Puppy, Animal)     # True
isinstance(p, Animal)         # True
```

`super()` follows the MRO, not simply "the parent class", which is what makes cooperative multiple inheritance work. `issubclass(Dog, Animal)` is `True`.

### 20.2 Multiple inheritance and the MRO

```python
class A:
    def who(self): return "A"
class B(A):
    def who(self): return "B"
class C(A):
    def who(self): return "C"
class D(B, C): pass

D().who()          # 'B'   first match along the MRO
D.__mro__          # (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
[c.__name__ for c in D.mro()]   # ['D', 'B', 'C', 'A', 'object']

class Bad(A, B): pass
# TypeError: Cannot create a consistent method resolution order (MRO) for bases A, B
```

The MRO is computed by C3 linearization: depth-first, left to right, but a class always appears after all of its subclasses. Ambiguous hierarchies raise `TypeError` at class creation time.

Mixins are the sane use of multiple inheritance: small classes providing one behaviour, no state, named `SomethingMixin`.

```python
class JsonMixin:
    def to_json(self): return json.dumps(self.__dict__)

class User(JsonMixin):
    def __init__(self, name): self.name = name

User("Ann").to_json()      # '{"name": "Ann"}'   leftmost base wins on conflicts
```

### 20.3 Polymorphism and duck typing

Same call, different behaviour by type. Python cares that the method exists, not what the class is.

```python
class Duck:
    def speak(self): return "Quack"
class Robot:                      # unrelated class, no shared base
    def speak(self): return "Beep"

for thing in [Duck(), Robot()]:
    print(thing.speak())
# Quack
# Beep

len("abc"), len([1]), len({"a": 1})    # (3, 1, 1)   anything with __len__
hasattr(Duck(), "speak")               # True   the duck-typing check
```

Python has no method overloading by signature. Use default arguments, `*args`, or `@functools.singledispatch`. Operator overloading is done with dunder methods (see 21).

### 20.4 Abstract base classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): ...

    def describe(self):                 # concrete methods are allowed
        return f"area={self.area()}"

class Square(Shape):
    def __init__(self, s): self.s = s
    def area(self): return self.s ** 2

Square(3).area()        # 9
Square(3).describe()    # 'area=9'
Shape()                 # TypeError: Can't instantiate abstract class Shape
                        #            with abstract method area

class Blob(Shape): pass
Blob()                  # TypeError too, area was never implemented
```

Use an ABC to declare a required interface. `typing.Protocol` is the structural alternative: it checks shape at type-check time with no inheritance needed.

### 20.5 Composition over inheritance

Inheritance is "is a", composition is "has a". Prefer composition, it couples less.

```python
class Engine:
    def start(self): return "vroom"

class Car:                       # a Car HAS an Engine
    def __init__(self): self.engine = Engine()
    def start(self): return self.engine.start()

Car().start()                    # 'vroom'   delegated, not inherited
isinstance(Car(), Engine)        # False     which is the honest answer
```

Inherit only when the subclass is genuinely substitutable for the base. Deep hierarchies and multiple inheritance for code reuse are the usual mistakes.

---

## 21. Dunder methods

Special methods that hook into Python's syntax. Implement only what you need.

| Group          | Methods                                                                | Triggered by                            |
| -------------- | ---------------------------------------------------------------------- | --------------------------------------- |
| Construction   | `__new__`, `__init__`, `__del__`                                       | Allocation, init, finalization          |
| Representation | `__repr__`, `__str__`, `__format__`                                    | `repr()`, `print()`, f-strings          |
| Comparison     | `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__`, `__ne__`             | `==`, `<`, sorting                      |
| Hashing        | `__hash__`                                                             | Dict keys, set members                  |
| Truthiness     | `__bool__`, `__len__`                                                  | `if obj`                                |
| Container      | `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__` | `len`, `obj[k]`, `in`                   |
| Iteration      | `__iter__`, `__next__`                                                 | `for`, `next()`                         |
| Callable       | `__call__`                                                             | `obj()`                                 |
| Arithmetic     | `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__radd__`, `__iadd__` | `+`, `-`, `*`, `/`, reflected, in place |
| Attributes     | `__getattr__`, `__getattribute__`, `__setattr__`, `__dir__`            | Attribute access                        |
| Context        | `__enter__`, `__exit__`                                                | `with`                                  |
| Async          | `__aiter__`, `__anext__`, `__aenter__`, `__aexit__`                    | `async for`, `async with`               |

```python
class Vector:
    def __init__(self, x, y): self.x, self.y = x, y
    def __repr__(self): return f"Vector({self.x}, {self.y})"      # for developers
    def __str__(self): return f"({self.x}, {self.y})"             # for users
    def __add__(self, o): return Vector(self.x + o.x, self.y + o.y)
    def __mul__(self, k): return Vector(self.x * k, self.y * k)
    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)
    def __hash__(self): return hash((self.x, self.y))
    def __len__(self): return 2
    def __getitem__(self, i): return (self.x, self.y)[i]
    def __iter__(self): return iter((self.x, self.y))

v, w = Vector(1, 2), Vector(3, 4)
v                       # Vector(1, 2)      repr, in the REPL and in containers
print(v)                # (1, 2)            str
f"{v} {v!r}"            # '(1, 2) Vector(1, 2)'
v + w                   # Vector(4, 6)
v * 3                   # Vector(3, 6)
3 * v                   # TypeError, until you add __rmul__
v == Vector(1, 2)       # True
len(v), v[0]            # (2, 1)
list(v)                 # [1, 2]            __iter__
px, py = v              # px=1, py=2        unpacking follows from __iter__
{v, w}                  # {Vector(1, 2), Vector(3, 4)}   works via __hash__
[v, w] == [Vector(1, 2), Vector(3, 4)]    # True
```

Rules to remember:

- `__repr__` should be unambiguous and ideally reconstruct the object. If you write only one, write this one, since `__str__` falls back to it.
- Defining `__eq__` sets `__hash__` to `None`, making the class unhashable. Define `__hash__` too if instances go in sets or dicts, and hash only immutable fields.
- `functools.total_ordering` fills in the rest of the comparisons from `__eq__` plus `__lt__`.
- `__getattr__` runs only when normal lookup fails, `__getattribute__` runs on every access (easy to break).
- Do not rely on `__del__`, its timing is not guaranteed. Use a context manager.

---

## 22. Object internals

### 22.1 `__new__` vs `__init__`

`__new__` creates and returns the object, `__init__` configures it. You only need `__new__` when subclassing an immutable type or controlling creation.

```python
class Singleton:
    _instance = None
    def __new__(cls, *a, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

Singleton() is Singleton()      # True   the same object every time

class Noisy:
    def __new__(cls): print("new"); return super().__new__(cls)
    def __init__(self): print("init")

Noisy()
# new
# init                          __new__ builds it, then __init__ configures it
```

### 22.2 `__slots__`

Declares a fixed set of attributes, drops the per-instance `__dict__`, saves memory and speeds up access. The cost is no new attributes at runtime and no `__dict__`.

```python
class Slotted:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x, self.y = x, y

class Plain:
    def __init__(self, x, y): self.x, self.y = x, y

s = Slotted(1, 2)
s.x                     # 1
s.z = 3                 # AttributeError: 'Slotted' object has no attribute 'z'
s.__dict__              # AttributeError, there is no instance dict

sys.getsizeof(Plain(1, 2).__dict__)   # 296   a whole dict per instance, on top of it
Slotted.__slots__                     # ('x', 'y')
```

Worth it for millions of small objects, not worth the constraint otherwise.

### 22.3 Class bodies, descriptors, metaclasses

```python
class Dog:
    species = "canis"
    def __init__(self, name): self.name = name

d = Dog("Rex")
d.__dict__                       # {'name': 'Rex'}
vars(d)                          # {'name': 'Rex'}   the same thing
Dog.__bases__                    # (<class 'object'>,)
d.__class__.__name__             # 'Dog'
getattr(d, "name")               # 'Rex'
getattr(d, "nope", "default")    # 'default'
hasattr(d, "species")            # True
setattr(d, "age", 3); d.age      # 3
[a for a in dir(d) if not a.startswith("_")]    # ['age', 'name', 'species']

Cat = type("Cat", (), {"speak": lambda self: "Meow"})   # a class built at runtime
Cat().speak()                    # 'Meow'
```

A class body executes once at import time, and the resulting names become class attributes. Classes are objects, so `type("Dog", (Animal,), {"speak": f})` builds one at runtime.

A **descriptor** is any object defining `__get__`, `__set__`, or `__delete__`, and it is the machinery behind `property`, methods, `classmethod`, and `staticmethod`. A **metaclass** is the class of a class (`type` by default), used by frameworks such as ORMs to validate or register subclasses. Awareness is enough for most work, and `__init_subclass__` handles most cases far more simply.

---

## 23. Dataclasses, NamedTuple, Enum

```python
from dataclasses import dataclass, field, asdict, replace

@dataclass(order=True)
class Point:
    x: int
    y: int = 0
    tags: list[str] = field(default_factory=list)   # never tags: list = []

    def dist(self): return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
p                       # Point(x=3, y=4, tags=[])   generated __repr__
p == Point(3, 4)        # True                        generated __eq__
p.dist()                # 5.0
Point(1) < Point(2)     # True                        because order=True
p.tags.append("a")      # mutable, since frozen was not set
asdict(p)               # {'x': 3, 'y': 4, 'tags': ['a']}
replace(p, x=9)         # Point(x=9, y=4, tags=['a'])   returns a new object

@dataclass(frozen=True)
class Frozen:
    x: int

f = Frozen(1)
f.x = 2                 # FrozenInstanceError: cannot assign to field 'x'
{Frozen(1)}             # works, frozen dataclasses are hashable
```

Generates `__init__`, `__repr__`, and `__eq__`. `frozen=True` makes it immutable and hashable, `order=True` adds comparisons, `slots=True` adds `__slots__` (3.10+). `field(default_factory=...)` is mandatory for mutable defaults. Helpers: `asdict()`, `astuple()`, `replace()`, and `__post_init__` for validation.

Which container for a record:

| Need                                     | Use                       |
| ---------------------------------------- | ------------------------- |
| Immutable, tuple-like, unpackable        | `NamedTuple`              |
| Mutable record with methods and defaults | `@dataclass`              |
| Immutable, hashable, with methods        | `@dataclass(frozen=True)` |
| Validation and parsing at the boundary   | `pydantic.BaseModel`      |
| Arbitrary dynamic keys                   | plain `dict`              |

```python
from enum import Enum, IntEnum, StrEnum, auto

class Status(Enum):
    PENDING = "pending"
    DONE = "done"

Status.PENDING            # <Status.PENDING: 'pending'>
Status.PENDING.name       # 'PENDING'
Status.PENDING.value      # 'pending'
Status("pending")         # <Status.PENDING: 'pending'>   lookup by value
Status["PENDING"]         # <Status.PENDING: 'pending'>   lookup by name
list(Status)              # [<Status.PENDING: 'pending'>, <Status.DONE: 'done'>]
Status.PENDING == "pending"      # False   an Enum is not its value
Status.PENDING is Status("pending")   # True

class Level(IntEnum):
    LOW = 1
    HIGH = 2

Level.HIGH == 2           # True    IntEnum compares equal to plain ints
Level.HIGH + 1            # 3
```

`auto()` assigns values for you when they do not matter. `IntEnum` and `StrEnum` compare equal to plain ints and strings, which is handy at API and database boundaries. Enums replace magic strings, and members compare by identity.

---

## 24. Design principles, quick list

- **SOLID**: single responsibility, open for extension and closed for modification, subtypes are substitutable, small interfaces, depend on abstractions.
- **Prefer composition** to inheritance, **prefer duck typing** to type checks.
- **Ask for forgiveness, not permission** (try/except) over checking first, which is the Python idiom.
- Patterns in Python shorthand: singleton is usually just a module, factory is a `@classmethod`, strategy is passing a function, decorator is `@decorator`, iterator is a generator, observer is a list of callbacks.
- Keep `__init__` cheap and side-effect free, do I/O in explicit methods or classmethods.

---

# Part 4: Runtime, standard library, tooling

## 25. Exceptions and errors

Syntax errors break at parse time, exceptions happen at runtime.

```python
def parse(text):
    try:
        value = int(text)
    except (ValueError, TypeError) as e:   # several types in one tuple
        print("bad input:", e)
        return None
    except ZeroDivisionError:
        value = 0
    else:
        print("clean parse")               # runs only when try succeeded
        return value
    finally:
        print("always runs")               # even on return, raise or break

parse("42")
# clean parse
# always runs
# returns 42

parse("abc")
# bad input: invalid literal for int() with base 10: 'abc'
# always runs
# returns None
```

### 25.1 Hierarchy

`BaseException` sits at the top, with `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit` beside `Exception`. Catch `Exception`, never `BaseException`, and never a bare `except:` (it swallows Ctrl-C).

Common ones: `ValueError` (right type, wrong value), `TypeError` (wrong type), `KeyError`, `IndexError`, `AttributeError`, `FileNotFoundError` (an `OSError`), `ZeroDivisionError`, `StopIteration`, `ImportError`, `RuntimeError`, `NotImplementedError`.

### 25.2 Raising

```python
status = 500
raise ValueError(f"bad status {status}")   # ValueError: bad status 500

try:
    int("abc")
except ValueError as err:
    raise RuntimeError("parse failed") from err
# RuntimeError: parse failed
# The above exception was the direct cause of the following exception: ...
# both errors appear in the traceback, which is the point of "from"

class AppError(Exception):
    """Base for this app."""
class RetryableError(AppError): pass       # one base per app, then subclass

try:
    raise RetryableError("later")
except AppError as e:                      # catching the base catches subclasses
    print(type(e).__name__, e.args)        # RetryableError ('later',)

x = -1
assert x > 0, "x must be positive"         # AssertionError: x must be positive
                                           # removed entirely under python -O
```

### 25.3 Habits

- Catch the narrowest exception you can actually handle, let the rest bubble up.
- Keep the `try` block small so it is obvious what raised.
- `logger.exception("failed")` inside an `except` logs the message plus the traceback.
- `except ... : pass` hides bugs. Use `contextlib.suppress(SpecificError)` when it really is fine.
- `return` in `finally` swallows the pending exception, so avoid it.
- 3.11+: `ExceptionGroup` plus `except*` for concurrent failures, and `e.add_note()` for extra context.

---

## 26. Logging

Five levels, from lowest: `DEBUG`, `INFO`, `WARNING` (the default threshold), `ERROR`, `CRITICAL`. Use logging instead of `print` because you get levels, timestamps, module names, and destinations you can change without editing code.

```python
import logging

logger = logging.getLogger(__name__)      # per-module logger, the standard pattern
logger.name                               # 'mypkg.core'   or '__main__' in a script

logger.debug("value=%s", 42)              # nothing, DEBUG is below the default level
logger.warning("retrying")                # retrying
                                          # bare, because no handler is configured yet

logging.basicConfig(level=logging.DEBUG)  # now the default format applies
logger.debug("value=%s", 42)              # DEBUG:mypkg.core:value=42   lazy formatting
logger.error("failed")                    # ERROR:mypkg.core:failed

try:
    1 / 0
except ZeroDivisionError:
    logger.exception("math went wrong")
# ERROR:mypkg.core:math went wrong
# Traceback (most recent call last):
#   File "app.py", line 2, in <module>
#     1 / 0
#     ~~^~~
# ZeroDivisionError: division by zero
```

Configure once, at program entry only:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logging.getLogger("app").info("started")
# 2026-08-13 21:40:02,113 INFO app:7 started      to stderr and to app.log
```

Pieces: a **logger** is what you call, a **handler** is where records go (stream, file, `RotatingFileHandler`, SMTP, HTTP), a **formatter** shapes the text, a **filter** drops records. Records travel up the logger hierarchy by dotted name to the root, so configuring the root covers everything, and `logger.propagate = False` stops it.

Notes: libraries should log and never configure, add per-call context with `extra={...}`, `dictConfig` is the way to configure from a file, and JSON formatters make logs queryable in production.

---

## 27. JSON

`json` converts between Python objects and JSON text. Text, not objects, is the boundary.

| Python                    | JSON                              |
| ------------------------- | --------------------------------- |
| `dict`                    | object                            |
| `list`, `tuple`           | array (tuples come back as lists) |
| `str`                     | string                            |
| `int`, `float`            | number                            |
| `True` / `False` / `None` | `true` / `false` / `null`         |

```python
import json

obj = {"b": 1, "a": [1, 2], "ok": True, "none": None}

json.dumps(obj)                     # '{"b": 1, "a": [1, 2], "ok": true, "none": null}'
json.dumps(obj, sort_keys=True)     # '{"a": [1, 2], "b": 1, "none": null, "ok": true}'
print(json.dumps({"a": 1}, indent=2))
# {
#   "a": 1
# }

json.loads('{"a": 1, "b": null}')   # {'a': 1, 'b': None}
json.loads("[1, 2]")                # [1, 2]
type(json.dumps(obj))               # <class 'str'>   dumps gives text, not a file

json.dumps((1, 2))                  # '[1, 2]'    tuples become arrays
json.loads("[1, 2]") == (1, 2)      # False       and come back as lists
json.dumps({1: "a"})                # '{"1": "a"}'   int keys become strings

with open("d.json", "w") as f: json.dump(obj, f)   # to a file, note: dump not dumps
with open("d.json") as f: loaded = json.load(f)    # from a file
```

Custom types need an encoder or a `default` hook:

```python
from datetime import datetime

json.dumps({"at": datetime(2026, 1, 1)})
# TypeError: Object of type datetime is not JSON serializable

def encode(o):
    if isinstance(o, datetime): return o.isoformat()
    raise TypeError(f"{type(o)} is not serializable")

json.dumps({"at": datetime(2026, 1, 1)}, default=encode)
# '{"at": "2026-01-01T00:00:00"}'

class Enc(json.JSONEncoder):
    def default(self, o): return o.__dict__

json.dumps(Point(1, 2), cls=Enc)          # '{"x": 1, "y": 2}'

json.loads('{"x": 1, "y": 2}', object_hook=lambda d: Point(**d))
# Point(x=1, y=2)                         dict to object on the way in

json.loads("{bad}")                       # json.JSONDecodeError: Expecting property name
print(json.dumps({"k": "café"}))          # {"k": "caf\u00e9"}   escaped by default
print(json.dumps({"k": "café"}, ensure_ascii=False))    # {"k": "café"}
```

Gotchas: dict keys become strings (`{1: "a"}` comes back as `{"1": "a"}`), sets and datetimes are not serializable, `NaN` and `Infinity` are written but are not valid strict JSON, bad input raises `json.JSONDecodeError` (a `ValueError`), and `json.dumps` on untrusted deep structures can be slow. `ensure_ascii=False` keeps Unicode readable.

---

## 28. Random numbers

`random` is a Mersenne Twister, fast and reproducible but **not** secure.

```python
import random
random.seed(42)                        # fix the sequence, so these outputs are exact

random.random()                        # 0.6394267984578837   float in [0.0, 1.0)
random.uniform(1, 10)                  # 1.2250967970040025   float in a range
random.randint(1, 6)                   # 3    int, BOTH ends included
random.randrange(0, 10, 2)             # 2    like range, stop excluded
random.choice("abcde")                 # 'b'  one item
random.choices([0, 1], weights=[1, 5], k=5)   # [0, 0, 1, 1, 1]   with replacement
random.sample(range(10), k=3)          # [0, 9, 1]   without replacement
random.gauss(mu=0, sigma=1)            # 0.23229773690672087

lst = [1, 2, 3, 4]
random.shuffle(lst)                    # returns None
lst                                    # [2, 4, 3, 1]   shuffled in place

random.seed(42); random.random()       # 0.6394267984578837   same seed, same value
rng = random.Random(42)                # an independent generator
rng.random()                           # 0.6394267984578837   unaffected by global seed
```

For tokens, passwords, and keys use `secrets`:

```python
import secrets
secrets.token_hex(16)        # '9f8c1d...' 32 hex characters, unpredictable
secrets.token_urlsafe(32)    # 'Yk3n-...'  safe in a URL
secrets.choice("abcde")      # 'c'
secrets.randbelow(100)       # 57    and there is no seeding, by design
```

For arrays use `numpy`: `rng = np.random.default_rng(42)`, then `rng.integers`, `rng.normal`, `rng.choice`. Seeding in ML means seeding every library you use (`random`, `numpy`, and the framework), and `random.seed` does not affect `numpy`.

---

## 29. Files and paths

```python
with open("f.txt", "w") as f:
    f.write("a\nb\n")               # returns 4, the number of characters written

with open("f.txt", "r", encoding="utf-8") as f:
    f.read()                        # 'a\nb\n'      the whole file as one string
with open("f.txt") as f:
    f.readlines()                   # ['a\n', 'b\n']   newlines are kept
with open("f.txt") as f:
    [line.rstrip() for line in f]   # ['a', 'b']    lazy, the memory-safe way

with open("f.txt", "a") as f: f.write("c\n")     # appends
open("f.txt").read()                             # 'a\nb\nc\n'
open("f.txt", "rb").read()                       # b'a\nb\nc\n'   binary gives bytes
open("f.txt", "x")                               # FileExistsError
```

Modes: `r`, `w`, `a`, `x` (fail if it exists), plus `b` or `t` and `+` for read and write. Always pass `encoding="utf-8"` for text so behaviour does not change across platforms.

```python
from pathlib import Path

p = Path("data") / "raw" / "f.csv"   # / builds paths, no os.path.join
p                                    # PosixPath('data/raw/f.csv')
str(p)                               # 'data/raw/f.csv'
p.name, p.stem, p.suffix             # ('f.csv', 'f', '.csv')
p.parent                             # PosixPath('data/raw')
p.parts                              # ('data', 'raw', 'f.csv')
p.with_suffix(".parquet")            # PosixPath('data/raw/f.parquet')
p.exists(), p.is_file()              # (False, False)
Path("out").mkdir(parents=True, exist_ok=True)    # no error if it already exists
Path("out/a.txt").write_text("hi")   # 2   characters written
Path("out/a.txt").read_text()        # 'hi'
list(Path("out").glob("*.txt"))      # [PosixPath('out/a.txt')]
list(Path(".").rglob("*.py"))        # recursive, same as glob("**/*.py")
```

Related: `csv` for tables, `pickle` for Python-only objects (never load untrusted pickles), `shutil` for copy and move, `tempfile` for scratch files, `os.environ` for configuration.

---

## 30. Concurrency, in brief

Full treatment in the [Concurrency in Python](/notes/Primers/concurrency) note. The revision version:

The one question first: is the work **waiting** (network, disk, database) or **computing** (loops and math in Python)?

|                  | Multiprocessing          | Threading                        | Asyncio                     |
| ---------------- | ------------------------ | -------------------------------- | --------------------------- |
| Unit             | Processes                | OS threads                       | Coroutines on one thread    |
| Memory           | Separate, needs pickling | Shared                           | Shared                      |
| Switching        | OS, preemptive           | OS, preemptive                   | Event loop, only at `await` |
| True parallelism | Yes                      | No, the GIL serializes bytecode  | No                          |
| Best for         | CPU-bound Python         | Blocking I/O with sync libraries | High-volume async I/O       |

The GIL lets only one thread execute Python bytecode at a time, which is why threads help with waiting but not with crunching. It is released during I/O and by many C extensions such as NumPy.

```python
import threading, time

def work(n):
    time.sleep(n)
    print("done", n, threading.current_thread().name)

t = threading.Thread(target=work, args=(1,), daemon=True)
t.start()                      # returns immediately
t.is_alive()                   # True
t.join()                       # blocks until it finishes
# done 1 Thread-1 (work)

counter = 0
lock = threading.Lock()
def bump():
    global counter
    for _ in range(100_000):
        with lock:             # without the lock the total comes out short
            counter += 1

ts = [threading.Thread(target=bump) for _ in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
counter                        # 200000   exact, because of the lock

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
with ThreadPoolExecutor(max_workers=8) as ex:      # I/O bound
    list(ex.map(str.upper, ["a", "b"]))            # ['A', 'B']   order preserved

with ProcessPoolExecutor() as ex:                  # CPU bound
    list(ex.map(abs, [-1, -2]))                    # [1, 2]

import asyncio
async def fetch(u):
    await asyncio.sleep(0.1)
    return u.upper()

async def main():
    return await asyncio.gather(fetch("a"), fetch("b"))

asyncio.run(main())            # ['A', 'B']   both waits overlap, so ~0.1s total
```

Primitives: `Lock`, `RLock`, `Semaphore`, `Event`, `Condition`, `Barrier`, plus `queue.Queue` for handing work between threads. Multiprocessing equivalents live in `multiprocessing`, and sharing data means `Queue`, `Pipe`, `Value`, `Array`, or `Manager`. Guard process-spawning code with `if __name__ == "__main__":`.

Common bugs: race conditions on shared state, deadlocks from taking locks in different orders, and blocking calls inside an event loop (offload with `asyncio.to_thread`).

---

## 31. Modules, packages, environments

```python
import math                      # math.pi   ->  3.141592653589793
import numpy as np               # alias
from math import pi, sqrt        # names directly: sqrt(9) -> 3.0
from math import *               # avoid, pollutes the namespace

math.__name__                    # 'math'
__name__                         # '__main__' when run directly,
                                 # 'mypkg.core' when imported
if __name__ == "__main__":       # so this block runs only on direct execution
    main()

import sys
"math" in sys.modules            # True   the import cache
sys.modules["math"] is math      # True
```

A module is one `.py` file, a package is a directory (usually with `__init__.py`). Imports are resolved along `sys.path` and cached in `sys.modules`, so a module's top-level code runs once per process. Prefer absolute imports (`from pkg.sub import x`); relative imports (`from .sub import x`) work inside packages only.

Circular imports usually mean the split is wrong. Short-term fixes are importing inside a function or importing the module rather than the name.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt
python -m module_name        # run a module as a script
```

### 31.1 Package layout and exports

```text
project/
  pyproject.toml         # metadata, dependencies, build backend, tool config
  src/mypkg/
    __init__.py          # marks the package, defines the public API
    core.py
  tests/test_core.py
```

```python
# mypkg/__init__.py
from .core import Client, connect
__all__ = ["Client", "connect"]      # what "from mypkg import *" exports
__version__ = "0.1.0"
```

`__all__` documents the public surface and is what linters use to tell an intentional re-export from an unused import. Keep `__init__.py` thin, since everything in it runs on import.

`pyproject.toml` is the modern replacement for `setup.py`, and `pip install -e .` gives an editable install so imports resolve without path hacks. For anything beyond a script, `uv` or `poetry` gives you a lock file, which `pip freeze` does not.

Import internals worth knowing: `sys.modules` is the cache (so re-importing is free and a module's top level runs once), `importlib.import_module("name")` imports by string at runtime, and `importlib.reload(mod)` re-executes a module (useful in notebooks, fragile elsewhere).

---

## 32. Type hints

Hints are documentation for humans and tools. Python does not enforce them at runtime.

```python
def greet(name: str, times: int = 1) -> str:
    return f"hi {name}" * times

greet(42)                                # 'hi 42'   runs fine, hints are not checked
greet.__annotations__                    # {'name': <class 'str'>, 'times': <class 'int'>,
                                         #  'return': <class 'str'>}

x: int = 5
names: list[str] = []
scores: dict[str, float] = {}
pair: tuple[int, str] = (1, "a")
maybe: str | None = None                 # Optional[str], 3.10+ syntax
value: int | str                         # Union
fn: Callable[[int, str], bool]
items: Iterable[int]                     # accept broadly, return concretely
```

```python
from typing import Any, TypeVar, Generic, Protocol, Literal, TypedDict, Final, cast

T = TypeVar("T")
def first(items: list[T]) -> T | None: ...

class Sized(Protocol):                    # structural typing, no inheritance
    def __len__(self) -> int: ...

Mode = Literal["r", "w"]

class Row(TypedDict):
    id: int
    name: str
```

### 32.1 The advanced pieces

```python
from typing import ParamSpec, Self, overload, NewType, Annotated, TypeAlias, TYPE_CHECKING

P = ParamSpec("P")                        # capture a full signature (decorators)
def deco(f: Callable[P, R]) -> Callable[P, R]: ...

class Builder:
    def add(self, x: int) -> Self: ...    # fluent API, correct in subclasses (3.11+)

UserId = NewType("UserId", int)           # a distinct type at check time, int at runtime
Port = Annotated[int, "1-65535"]          # attach metadata (pydantic and FastAPI use this)
Vector: TypeAlias = list[float]

@overload
def get(k: str) -> str: ...
@overload
def get(k: int) -> bytes: ...
def get(k): ...                           # one real implementation

if TYPE_CHECKING:
    from heavy import Thing               # import only for hints, no runtime cost

class Box[T]:                             # 3.12 generic syntax, no TypeVar needed
    def __init__(self, item: T) -> None: self.item = item
```

Rules that matter in practice: annotate the boundaries (public functions, dataclasses, config) and skip the obvious locals; accept broad types (`Iterable`, `Mapping`) and return concrete ones (`list`, `dict`); never annotate a mutable default as `list` with `[]`; and remember hints are just metadata, readable at runtime with `typing.get_type_hints(func)`, which is how pydantic and dataclasses work.

Run `mypy` or `pyright` in CI, otherwise the hints drift. `from __future__ import annotations` makes annotations lazy and lets you reference names defined later.

---

## 33. functools and other stdlib workhorses

```python
from functools import lru_cache, cache, wraps, partial, reduce, cached_property, total_ordering

import operator

@cache                              # unbounded memoization (3.9+)
def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)

fib(100)                            # 354224848179261915075   instant, not exponential
fib.cache_info()                    # CacheInfo(hits=98, misses=101, maxsize=None,
                                    #           currsize=101)
fib.cache_clear()                   # back to empty

@lru_cache(maxsize=2)               # bounded, least-recently-used is evicted
def load(key): print("miss", key); return key * 2

load("a"); load("a")                # miss a       the second call is served from cache
load.cache_info()                   # CacheInfo(hits=1, misses=1, maxsize=2, currsize=1)
load(["a"])                         # TypeError: unhashable type: 'list'

add5 = partial(operator.add, 5)     # freeze the first argument
add5(3)                             # 8
reduce(operator.mul, [1, 2, 3, 4])  # 24   fold, but prefer sum/math.prod when they fit
```

```python
import re
s = "order 66 on 2026-01-02"

re.search(r"\d+", s)                  # <re.Match object; span=(6, 8), match='66'>
re.search(r"\d+", s).group()          # '66'    first match anywhere
re.search(r"zzz", s)                  # None    falsy, so "if m:" is the usual guard
re.match(r"\d+", s)                   # None    match is anchored at the start
re.findall(r"\d+", s)                 # ['66', '2026', '01', '02']
re.sub(r"\d+", "N", s)                # 'order N on N-N-N'
re.split(r"\s+", s)                   # ['order', '66', 'on', '2026-01-02']

m = re.search(r"(?P<y>\d{4})-(?P<m>\d{2})", s)
m.group(0)                            # '2026-01'
m.group(1), m.group("y")              # ('2026', '2026')
m.groups()                            # ('2026', '01')
m.groupdict()                         # {'y': '2026', 'm': '01'}
m.span()                              # (12, 19)

pat = re.compile(r"\d+")              # compile once when reused in a loop
pat.findall(s)                        # ['66', '2026', '01', '02']
```

```python
from datetime import datetime, date, timedelta, timezone

datetime.now(timezone.utc)         # datetime(2026, 8, 13, 21, 40, 2, 113, tzinfo=utc)
                                   # always store UTC, aware rather than naive
dt = datetime.fromisoformat("2026-01-01T10:00:00")
dt.strftime("%Y-%m-%d %H:%M")      # '2026-01-01 10:00'
datetime.strptime("2026-01-01", "%Y-%m-%d")     # datetime(2026, 1, 1, 0, 0)
dt.isoformat()                     # '2026-01-01T10:00:00'
dt.date(), dt.year, dt.weekday()   # (date(2026, 1, 1), 2026, 3)   Monday is 0

dt + timedelta(days=7)             # datetime(2026, 1, 8, 10, 0)
(datetime(2026, 1, 2) - datetime(2026, 1, 1))   # timedelta(days=1)
(datetime(2026, 1, 2) - datetime(2026, 1, 1)).total_seconds()   # 86400.0
dt - datetime.now(timezone.utc)    # TypeError: can't subtract naive from aware
```

### 33.1 heapq and bisect

Both come up constantly in coding rounds, and both operate on a plain list.

```python
import heapq

h = []
heapq.heappush(h, (2, "write"))         # min-heap, O(log n)
heapq.heappush(h, (1, "read"))
heapq.heappush(h, (3, "sync"))
h                                       # [(1, 'read'), (2, 'write'), (3, 'sync')]
h[0]                                    # (1, 'read')   peek, O(1)
heapq.heappop(h)                        # (1, 'read')   smallest, O(log n)

lst = [5, 1, 3]
heapq.heapify(lst)                      # in place, O(n)
lst                                     # [1, 5, 3]   heap order, NOT sorted order

nums = [5, 1, 9, 3]
heapq.nlargest(2, nums)                 # [9, 5]
heapq.nsmallest(2, nums)                # [1, 3]
scores = {"ann": 90, "bob": 72}
heapq.nlargest(1, scores.items(), key=lambda kv: kv[1])     # [('ann', 90)]
list(heapq.merge([1, 4], [2, 3]))       # [1, 2, 3, 4]   lazily merges sorted inputs

maxheap = [-n for n in nums]            # negate for a max-heap
heapq.heapify(maxheap)
-heapq.heappop(maxheap)                 # 9
```

Python only has a min-heap, so push `-value` for a max-heap. Ties compare the next tuple element, so add a counter (`(priority, count, task)`) when the payload is not comparable.

```python
import bisect
a = [10, 20, 20, 30]

bisect.bisect_left(a, 20)    # 1    first position where 20 could go, O(log n)
bisect.bisect_right(a, 20)   # 3    after the existing 20s
bisect.bisect_left(a, 25)    # 3    insertion point for a value that is not present

bisect.insort(a, 25)
a                            # [10, 20, 20, 25, 30]   the insert itself is O(n)

bisect.bisect_right(a, 25) - bisect.bisect_left(a, 20)    # 3   items in [20, 25]
```

### 33.2 Introspection and the remaining odds

```python
import inspect

def f(a, b=2): return a
inspect.signature(f)              # <Signature (a, b=2)>
str(inspect.signature(f))         # '(a, b=2)'
inspect.iscoroutinefunction(f)    # False
inspect.isclass(int)              # True

import operator
people = [("bob", 25), ("ann", 30)]
sorted(people, key=operator.itemgetter(1))         # [('bob', 25), ('ann', 30)]
operator.itemgetter(0)(("x", "y"))                 # 'x'
operator.attrgetter("real")(3 + 4j)                # 3.0

import weakref
class Big: pass
obj = Big()
ref = weakref.ref(obj)
ref()                             # <__main__.Big object at 0x...>
del obj
ref()                             # None   the weak reference did not keep it alive

import contextvars
request_id = contextvars.ContextVar("request_id", default=None)
request_id.get()                  # None
request_id.set("abc")             # <Token ...>
request_id.get()                  # 'abc'   per-task, the async-safe threading.local

text = "café"
len(text)                         # 4
data = text.encode("utf-8")       # b'caf\xc3\xa9'
len(data)                         # 5   characters and bytes are not the same count
data.decode("utf-8")              # 'café'
data.decode("ascii")              # UnicodeDecodeError
```

Also worth knowing: `os` and `sys` (environment, argv, exit), `argparse` for CLIs, `math` and `statistics`, `uuid`, `hashlib`, `subprocess`, `shlex`, `timeit`, `cProfile`, `dis` (see the bytecode), and `pickle` with `__reduce__` for custom serialization. `sys.setrecursionlimit` exists because Python has no tail-call optimization, so deep recursion needs an explicit stack instead.

---

## 34. Testing

```python
# test_math.py, run with: pytest -q
import pytest
from app.calc import add, divide

def test_add():
    assert add(2, 3) == 5

@pytest.mark.parametrize("a,b,expected", [(1, 1, 2), (0, 0, 0), (-1, 1, 0)])
def test_add_cases(a, b, expected):
    assert add(a, b) == expected

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

@pytest.fixture
def client():
    c = make_client()
    yield c              # setup before, teardown after
    c.close()

def test_with_fixture(client):     # request it by parameter name
    assert client.ping() == "pong"
```

```text
pytest -q

....F                                                        [100%]
=================================== FAILURES ===================================
_______________________________ test_add_cases[-1-1-0] _________________________
    assert add(a, b) == expected
E   assert 1 == 0
E    +  where 1 = add(-1, 1)
1 failed, 4 passed in 0.03s
```

The failure line shows the actual values, which is why a plain `assert` is enough in pytest and you rarely need `assertEqual`.

`unittest` is the stdlib alternative (`class T(unittest.TestCase)`, `self.assertEqual`, `setUp`). Use `unittest.mock.patch` to replace I/O, `monkeypatch` for environment and attributes, `pytest-cov` for coverage, and keep tests fast, isolated, and independent of execution order.

---

## 35. Memory and performance

CPython frees objects by **reference counting**, and a cyclic **garbage collector** handles reference cycles. Both are automatic.

```python
import sys, gc

sys.getsizeof(0), sys.getsizeof(1.0)      # (28, 24)   bytes, even for a small int
sys.getsizeof([])                         # 56
sys.getsizeof([1, 2, 3])                  # 88         plus the ints themselves
sys.getsizeof("")                         # 41

a = []
sys.getrefcount(a)                        # 2   one for a, one for the argument itself
b = a
sys.getrefcount(a)                        # 3
del b                                     # unbinds the name, drops one reference
sys.getrefcount(a)                        # 2

gc.collect()                              # 0   number of unreachable cycles collected
```

Interning and caching: small ints (-5 to 256) and short identifier-like strings are reused, which is why `is` occasionally looks like it works on values.

```python
a, b = 256, 256
a is b                # True    cached
a, b = 257, 257
a is b                # False   two separate objects
a, b = "hi", "hi"
a is b                # True    short literals are interned
a, b = "h i", "h i"
a is b                # False   not identifier-like, so no interning
```

Quick wins, in the order I try them:

1. Pick the right data structure (set or dict for membership, `deque` for ends, `heapq` for top-K).
2. Use built-ins and comprehensions, they run in C.
3. `join` instead of `+=` for strings.
4. Generators instead of building large intermediate lists.
5. Cache repeated pure calls with `functools.cache`.
6. Hoist attribute and global lookups out of hot loops.
7. Measure with `timeit` and `cProfile` before optimizing, then reach for `numpy`, Cython, or another process only if it is still needed.

---

# Part 5: Revision checklist

## 36. Gotchas I get asked about

1. Mutable default argument is created once at definition. Use `None`.
2. `[[0] * 3] * 3` and `dict.fromkeys(k, [])` share one inner object.
3. Mutating methods return `None`, so `nums = nums.sort()` loses the list.
4. Never add or remove items while iterating a list or dict.
5. Assignment aliases, `[:]` is shallow, `copy.deepcopy` is deep.
6. A one-item tuple needs the trailing comma, `(5)` is an int.
7. `{}` is an empty dict, an empty set is `set()`.
8. `is` compares identity, `==` compares value. Use `is` only with `None` and sentinels.
9. `0.1 + 0.2 != 0.3`. Use `math.isclose` or `Decimal`.
10. Late binding in closures and lambdas inside loops. Bind with a default argument.
11. Defining `__eq__` without `__hash__` makes the class unhashable.
12. `x in list` is O(n), `x in set` is O(1).
13. A generator is exhausted after one pass.
14. `except:` bare or `except Exception: pass` hides real bugs, including Ctrl-C.
15. Class-level mutable attributes are shared across instances.
16. Threads do not speed up CPU-bound Python because of the GIL.
17. `str.strip("chars")` strips a character set, not a prefix. Use `removeprefix`.
18. Integer division `//` floors toward negative infinity, so `-7 // 2 == -4`.

## 37. Complexity cheat sheet

| Operation                     | list | dict / set   | deque            |
| ----------------------------- | ---- | ------------ | ---------------- |
| Index or key access           | O(1) | O(1) average | O(1) at the ends |
| Insert or delete at the end   | O(1) | O(1)         | O(1)             |
| Insert or delete at the front | O(n) | n/a          | O(1)             |
| Search by value               | O(n) | O(1)         | O(n)             |
| Iterate everything            | O(n) | O(n)         | O(n)             |

Sorting is O(n log n) with Timsort, and it is stable.

## 38. One-liners worth memorizing

```python
items = [3, 1, 3, 2]
d = {"a": 2, "b": 0, "c": 5}
matrix = [[1, 2], [3, 4]]

list(dict.fromkeys(items))                    # [3, 1, 2]        dedupe, keeping order
Counter("aabbbc").most_common(2)              # [('b', 3), ('a', 2)]   top frequencies
sorted(d.items(), key=lambda kv: -kv[1])      # [('c', 5), ('a', 2), ('b', 0)]
{k: v for k, v in d.items() if v}             # {'a': 2, 'c': 5}   drop falsy values
list(zip(*matrix))                            # [(1, 3), (2, 4)]  transpose
[i for row in matrix for i in row]            # [1, 2, 3, 4]      flatten one level
" ".join(w.capitalize() for w in "ab cd".split())    # 'Ab Cd'
"racecar" == "racecar"[::-1]                  # True              palindrome
Counter("listen") == Counter("silent")        # True              anagram
max(d, key=d.get)                             # 'c'               largest value's key
len(items) != len(set(items))                 # True              has duplicates
dict(zip("ab", [1, 2]))                       # {'a': 1, 'b': 2}
[items[i:i + 2] for i in range(0, len(items), 2)]    # [[3, 1], [3, 2]]   chunk
next((x for x in items if x > 2), None)       # 3                 first match or None
sum(1 for _ in iter([1, 2, 3]))               # 3                 count an iterator
sorted(d, key=d.get, reverse=True)            # ['c', 'a', 'b']   keys by value
```
