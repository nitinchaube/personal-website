---
title: "Concepts and Problems"
date: 2026-06-07
summary: "This consist of all the problems I have solved with thier concepts in python."
tags: [DP, DSA, Algorithms]
---

## Two-Pointer Pattern

**When to use it:**

- You are dealing with a sorted array or linked list.
- You need to find a set of elements (pairs, triplets) that satisfy a specific condition (e.g., sum to a target).
- You need to compare elements from opposite ends (e.g., checking for palindromes).
- Keywords: "Sorted array", "Pair that sums to...", "Reverse the array".

```python
# Template
def two_pointer_template(arr, target):
    left = 0
    right = len(arr)-1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return -1
```

### Problem LC 125: Valid Palindrome ([link](https://leetcode.com/problems/valid-palindrome/description/))

```python
def isPalindrome(s: str) -> bool:
    s = "".join(char for char in s if char.isalnum()).lower()
    left, right = 0, len(s)-1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### (GOOGLE) Problem LC 15: 3Sum ([link](https://leetcode.com/problems/3sum/description/))

```python
from typing import List
def threeSum(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left = i+1
        right = len(nums)-1
        while left < right:
            curr_sum = nums[i] + nums[left] + nums[right]
            if curr_sum == 0:
                res.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left-1]:
                    left += 1
                while left < right and nums[right] == nums[right+1]:
                    right -= 1
            elif curr_sum > 0:
                right -= 1
            else:
                left += 1
    return res
```

### Problem LC 11: Container with Most Water ([link](https://leetcode.com/problems/container-with-most-water/))

```python
def maxArea(height: List[int]) -> int:
    max_result = float('-inf')
    left = 0
    right = len(height)-1
    while left < right:
        area = min(height[left], height[right]) * (right-left)
        if area > max_result:
            max_result = area
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return max_result
```

### Problem LC 121: Best Time to Buy and Sell Stock ([link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/))

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxprofit = 0
        min_element = prices[0]
        for i in range(1, len(prices)):
            if prices[i]-min_element > maxprofit:
                maxprofit = prices[i] - min_element
            if prices[i] < min_element:
                min_element = prices[i]
        return maxprofit
```

### Problem LC 167: Two Sum II ([link](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/))

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers)-1
        while left < right:
            num = numbers[left]+numbers[right]
            if num == target:
                return [left+1, right+1]
            elif num < target:
                left += 1
            else:
                right -= 1
```

### Problem LC 42: Trapping Rain Water ([link](https://leetcode.com/problems/trapping-rain-water/description/))

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # Optimized O(n) time, O(1) space using two pointers
        res = 0
        left, right = 0, len(height)-1
        leftMax = height[left]
        rightMax = height[right]
        while left < right:
            if leftMax < rightMax:
                left += 1
                leftMax = max(height[left], leftMax)
                res += leftMax - height[left]
            else:
                right -= 1
                rightMax = max(height[right], rightMax)
                res += rightMax - height[right]
        return res
```

---

## MATRIX Problems

### Problem LC 54: Spiral Matrix [https://leetcode.com/problems/spiral-matrix/description/](https://leetcode.com/problems/spiral-matrix/description/)

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        row = len(matrix)
        col = len(matrix[0])
        top = 0
        left = 0
        right = col-1
        bottom = row - 1
        result = []
        while top<=bottom and left<=right:
            #left to right
            for i in range(left, right+1):
                result.append(matrix[top][i])
            top +=1
            # top to bottm
            for i in range(top, bottom+1):
                result.append(matrix[i][right])
            right -= 1
            # right to left
            if top<=bottom:
                for i in range(right, left-1, -1):
                    result.append(matrix[bottom][i])
                bottom -=1
            # bottom to top
            if left<= right:
                for i in range(bottom, top-1, -1):
                    result.append(matrix[i][left])
                left+=1
        return result
```

### Problem LC 73: Set Matrix Zero ([https://leetcode.com/problems/set-matrix-zeroes/description/](https://leetcode.com/problems/set-matrix-zeroes/description/))

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        is_first_row_zero = False
        is_first_col_zero = False
        for c in range(n):
            if matrix[0][c] == 0:
                    is_first_row_zero = True
        for r in range(m):
            if matrix[r][0]==0:
                is_first_col_zero = True

        for r in range(1,m):
            for c in range(1,n):
                if matrix[r][c]==0:
                    matrix[r][0] = 0
                    matrix[0][c] = 0
            
        for r in range(1,m):
            for c in range(1,n):
                if matrix[r][0]==0 or matrix[0][c]==0:
                    matrix[r][c] = 0
        
        if is_first_row_zero:
            for c in range(n):
                matrix[0][c] = 0
        
        if is_first_col_zero:
            for r in range(m):
                matrix[r][0] = 0
        return matrix
```

### Problem LC 48: Rotate Image ([https://leetcode.com/problems/rotate-image/description/](https://leetcode.com/problems/rotate-image/description/))

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        col = len(matrix[0])
        row = len(matrix)
        for i in range(row):
            for j in range(i, col):
                if i==j:
                    continue
                matrix[i][j] , matrix[j][i] = matrix[j][i], matrix[i][j]
        
        for i in range(row):
            left = 0 
            right = col-1
            while left<right:
                matrix[i][left], matrix[i][right] = matrix[i][right], matrix[i][left]
                left+=1
                right-=1
        return matrix
```

---

## Sliding Window Pattern

**When to use it:**

- You need to find a contiguous subarray or substring.
- You are asked to calculate the longest, shortest, maximum, or minimum of a contiguous sequence.
- Keywords: "Longest substring", "Shortest subarray", "Maximum sum of a contiguous subarray".

```python
def sliding_window_template(s: str):
    left = 0
    best_result = 0  # Use float('inf') if looking for a minimum

    for right in range(len(s)):
        # A. ADD TO WINDOW STATE
        # e.g., window_counts[s[right]] = window_counts.get(s[right], 0) + 1

        # Shrink: while INVALID (for max window) or while VALID (for min window)
        while condition_to_shrink_window:
            # B. RECORD MINIMUM RESULT (if finding shortest valid window)
            # e.g., best_result = min(best_result, right - left + 1)

            # C. REMOVE FROM WINDOW STATE
            # e.g., window_counts[s[left]] -= 1
            left += 1

        # D. RECORD MAXIMUM RESULT (if finding longest valid window)
        # e.g., best_result = max(best_result, right - left + 1)

    return best_result
```

### (GOOGLE) Problem LC 3: Longest Substring without Repeating Characters ([link](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/))

```python
def lengthOfLongestSubstring(s):
    max_count = 0
    char_index = {}
    l = 0
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= l:
            l = char_index[char] + 1
        char_index[char] = i
        count = i - l + 1
        if count > max_count:
            max_count = count
    return max_count
```

### Problem LC 424: Longest Repeating Character Replacement ([link](https://leetcode.com/problems/longest-repeating-character-replacement/description/))

```python
def characterReplacement(s, k):
    counts = {}
    result = 0
    left = 0
    max_freq_elem = 0
    for right, char in enumerate(s):
        counts[char] = counts.get(char, 0) + 1
        max_freq_elem = max(max_freq_elem, counts[char])
        while (right-left+1)-max_freq_elem > k:
            counts[s[left]] -= 1
            left += 1
        result = max(result, right-left+1)
    return result
```

### Problem LC 209: Minimum Size Subarray Sum ([link](https://leetcode.com/problems/minimum-size-subarray-sum/description/))

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        min_len = float("inf")
        n = len(nums)
        total_sum = 0
        for right in range(n):
            total_sum += nums[right]
            while total_sum >= target and left <= right:
                min_len = min(min_len, right-left+1)
                total_sum -= nums[left]
                left += 1
        return min_len if min_len != float("inf") else 0
```

### Problem LC 239: Sliding Window Maximum ([link](https://leetcode.com/problems/sliding-window-maximum/))

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        result = []
        for i in range(len(nums)):
            if q and q[0] <= i-k:
                q.popleft()
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if i >= k-1:
                result.append(nums[q[0]])
        return result
```

---

## String and Advanced Hashing

### (GOOGLE) Problem LC 5: Longest Palindromic Substring ([link](https://leetcode.com/problems/longest-palindromic-substring/description/))

```python
def longestPalindromicSubstring(s):
    start, end = 0, 0
    def expand_from_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right-left-1

    for i in range(len(s)):
        len1 = expand_from_center(i, i)    # odd length
        len2 = expand_from_center(i, i+1)  # even length
        max_len = max(len1, len2)

        if max_len > end-start:
            start = i - (max_len-1)//2
            end = i + max_len//2

    return s[start:end+1]
```

### Problem LC 647: Count all Palindromic Substrings ([link](https://leetcode.com/problems/palindromic-substrings/description/))

```python
def countSubstring(s):
    n = len(s)
    palindromic_count = 0
    def expand_center(left, right):
        count = 0
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
            count += 1
        return count

    for i in range(n):
        palindromic_count += expand_center(i, i)
        palindromic_count += expand_center(i, i+1)
    return palindromic_count
```

### (GOOGLE) Problem LC 271: Encode and Decode String ([link](https://neetcode.io/problems/string-encode-and-decode/question))

```python
class Solution:
    def encode(self, strs: List[str]) -> str:
        encoded = ""
        for s in strs:
            encoded += str(len(s)) + "#" + s
        return encoded

    def decode(self, s: str) -> List[str]:
        decoded = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            parsed = s[j:j+length+1]
            decoded.append(parsed)
            i = j + length + 1
        return decoded
```

### (GOOGLE) Problem LC 76: Minimum Window Substring ([link](https://leetcode.com/problems/minimum-window-substring/description/))

```python
def minwindow(s, t):
    t_dict = Counter(t)
    need = len(t_dict)
    have = 0
    window = {}
    left = 0
    result = [-1, -1]
    res_len = float("inf")
    for right, char in enumerate(s):
        window[char] = window.get(char, 0)+1
        if char in t_dict and window[char] == t_dict[char]:
            have += 1
        while need == have:
            if right-left+1 < res_len:
                res_len = right-left+1
                result = [left, right]
            window[s[left]] -= 1
            if s[left] in t_dict and window[s[left]] < t_dict[s[left]]:
                have -= 1
            left += 1

    l, r = result
    return s[l:r+1] if res_len != float("inf") else ""
```

### Problem LC 567: Permutation in String ([link](https://leetcode.com/problems/permutation-in-string/description/))

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        s1_count = [0]*26
        s2_count = [0]*26
        k = len(s1)
        for i in range(k):
            s1_count[ord(s1[i]) - ord("a")] += 1
            s2_count[ord(s2[i]) - ord("a")] += 1

        if s1_count == s2_count:
            return True

        for i in range(k, len(s2)):
            char = s2[i]
            s2_count[ord(char)-ord('a')] += 1
            char_out = s2[i-k]
            s2_count[ord(char_out)-ord('a')] -= 1

            if s1_count == s2_count:
                return True
        return False
```

---

## Stacks, Queues and Monotonic Patterns

> **Tip:** In most problems, store the **index** `i` in the stack rather than the value `arr[i]`. The index gives you the value (`arr[stack[-1]]`) *and* the distance (`current_index - stack[-1]`).

```python
# Monotonic Increasing Stack
def monotonicIncreasing(arr):
    stack = []
    for i in range(len(arr)):
        while stack and arr[i] < stack[-1]:
            stack.pop()
        stack.append(arr[i])
    return stack

# Monotonic Decreasing Stack
def monotonicDecreasing(arr):
    stack = []
    for i in range(len(arr)):
        while stack and arr[i] > stack[-1]:
            stack.pop()
        stack.append(arr[i])
    return stack
```

### Problem LC 20: Valid Parentheses ([link](https://leetcode.com/problems/valid-parentheses/description/))

```python
def isValid(s):
    partList = []
    for w in s:
        if w in ["(", "[", "{"]:
            partList.append(w)
        else:
            if not partList:
                return False
            top = partList.pop()
            if w == "]" and top != "[":
                return False
            elif w == "}" and top != "{":
                return False
            elif w == ")" and top != "(":
                return False
    return len(partList) == 0
```

### Problem LC 155: MinStack ([link](https://leetcode.com/problems/min-stack/description/))

```python
class MinStack:
    def __init__(self):
        self.minstack = []
        self.minElement = []

    def push(self, val: int) -> None:
        self.minstack.append(val)
        if not self.minElement or val <= self.minElement[-1]:
            self.minElement.append(val)

    def pop(self) -> None:
        popped_elem = self.minstack.pop()
        if popped_elem == self.minElement[-1]:
            self.minElement.pop()

    def top(self) -> int:
        return self.minstack[-1]

    def getMin(self) -> int:
        return self.minElement[-1]
```

### (GOOGLE) Problem LC 739: Daily Temperatures ([link](https://leetcode.com/problems/daily-temperatures/description/))

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stk = []
        res = [0]*len(temperatures)
        for i in range(len(temperatures)):
            while stk and temperatures[i] > temperatures[stk[-1]]:
                res[stk[-1]] = i - stk[-1]
                stk.pop()
            stk.append(i)
        return res
```

### Problem LC 496: Next Greater Element I ([link](https://leetcode.com/problems/next-greater-element-i/description/))

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stk = []
        greater = {}
        for num in reversed(nums2):
            while stk and stk[-1] <= num:
                stk.pop()
            greater[num] = -1 if not stk else stk[-1]
            stk.append(num)
        return [greater[n] for n in nums1]
```

### Problem LC 84: Largest Rectangle in Histogram ([link](https://leetcode.com/problems/largest-rectangle-in-histogram/description/))

```python
def largestRectangleArea(heights):
    mono_stk = []
    heights.append(0)
    max_area = 0
    for i, h in enumerate(heights):
        while mono_stk and heights[mono_stk[-1]] > h:
            height = heights[mono_stk.pop()]
            width = i if not mono_stk else i - mono_stk[-1] - 1
            max_area = max(height*width, max_area)
        mono_stk.append(i)
    return max_area
```

### Problem LC 853: Car Fleet ([link](https://leetcode.com/problems/car-fleet/))

```python
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        cars = sorted(zip(position, speed), reverse=True)
        stack = []
        for p, s in cars:
            time = (target-p)/s
            if not stack or time > stack[-1]:
                stack.append(time)
        return len(stack)
```

---

## Binary Search Mastery

#### 1. Exact Match

Use when you need **some** index where `nums[i] == target`.

```python
def binary_search_exact(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1
```

#### 2. Lower Bound (first `i` with `nums[i] >= target`)

```python
def binary_search_lower_bound(nums, target):
    l, r = 0, len(nums)  # note: len(nums), not len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] >= target:
            r = mid
        else:
            l = mid + 1
    return l
```

#### 3. Upper Bound (first `i` with `nums[i] > target`)

```python
def binary_search_upper_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > target:
            r = mid
        else:
            l = mid + 1
    return l
```

### Problem LC 704: Binary Search ([link](https://leetcode.com/problems/binary-search/description/))

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums)-1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid
            elif target > nums[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return -1
```

### (GOOGLE) Problem LC 74: Search a 2D Matrix ([link](https://leetcode.com/problems/search-a-2d-matrix/description/))

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        left = 0
        right = (m*n) - 1
        while left <= right:
            mid = (left+right)//2
            mid_val = matrix[mid//n][mid%n]
            if mid_val == target:
                return True
            elif mid_val > target:
                right = mid - 1
            else:
                left = mid + 1
        return False
```

### (GOOGLE) Problem LC 875: Koko Eating Bananas ([link](https://leetcode.com/problems/koko-eating-bananas/description/))

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        right = max(piles)
        left = 1
        res = right
        while left <= right:
            mid = (left+right)//2
            hours = 0
            for i in range(len(piles)):
                hours += math.ceil(piles[i]/mid)
            if hours <= h:
                res = mid
                right = mid-1
            else:
                left = mid+1
        return res
```

### Problem LC 153: Find Minimum in Rotated Sorted Array ([link](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/))

```python
def findMin(nums):
    left = 0
    right = len(nums)-1
    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] > nums[right]:
            left = mid+1
        else:
            right = mid-1
    return nums[left]
```

---

## Linked List and Fast/Slow Pointers

### Problem LC 206: Reverse Linked List ([link](https://leetcode.com/problems/reverse-linked-list/description/))

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr is not None:
            nextN = curr.next
            curr.next = prev
            prev = curr
            curr = nextN
        return prev
```

### Problem LC 21: Merge Two Sorted Lists ([link](https://leetcode.com/problems/merge-two-sorted-lists/description/))

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list2 is None:
            return list1
        if list1 is None:
            return list2

        if list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
```

### Problem LC 141: Linked List Cycle — Floyd's Algorithm ([link](https://leetcode.com/problems/linked-list-cycle/))

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### Problem LC 146: LRU Cache ([link](https://leetcode.com/problems/lru-cache/description/))

```python
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def remove(self, node):
        prv, nxt = node.prev, node.next
        prv.next = nxt
        nxt.prev = prv

    def add_to_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.remove(node)
        self.add_to_front(node)
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self.remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self.add_to_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self.remove(lru)
            del self.cache[lru.key]
```

### (GOOGLE) Problem LC 23: Merge k Sorted Lists ([link](https://leetcode.com/problems/merge-k-sorted-lists/description/))

```python
def mergeKLists(lists):
    min_heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))

    dum = ListNode(0)
    curr = dum

    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))
    return dum.next
```

---

## Binary Tree

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1. PreOrder (Root -> Left -> Right)
def preorder(root):
    result = []
    def traverse(node):
        if not node: return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)
    traverse(root)
    return result

# 2. Inorder (Left -> Root -> Right) — sorts BSTs
def inorder(root):
    result = []
    def traverse(node):
        if not node: return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)
    traverse(root)
    return result

# 3. PostOrder (Left -> Right -> Root)
def postorder(root):
    result = []
    def traverse(node):
        if not node: return
        traverse(node.left)
        traverse(node.right)
        result.append(node.val)
    traverse(root)
    return result

# 4. BFS / Level Order
def levelorder(root):
    if not root: return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(current_level)
    return result
```

### Problem LC 226: Invert Binary Tree ([link](https://leetcode.com/problems/invert-binary-tree/description/))

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

### Problem LC 104: Maximum Depth of Binary Tree ([link](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/))

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return 1 + max(left_depth, right_depth)
```

### Problem LC 100: Same Tree ([link](https://leetcode.com/problems/same-tree/description/))

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        left = self.isSameTree(p.left, q.left)
        right = self.isSameTree(p.right, q.right)
        return left and right
```

### (GOOGLE) Problem LC 102: Binary Tree Level Order Traversal ([link](https://leetcode.com/problems/binary-tree-level-order-traversal/))

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        level_order = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            level_order.append(level)
        return level_order
```

### Problem LC 297: Serialize and Deserialize Binary Tree ([link](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/))

```python
class Codec:
    def serialize(self, root):
        result = []
        def traverse(node):
            if not node:
                result.append("null")
                return
            result.append(str(node.val))
            traverse(node.left)
            traverse(node.right)
        traverse(root)
        return ",".join(result)

    def deserialize(self, data):
        values = data.split(',')
        iterator = iter(values)
        def build_tree():
            val = next(iterator)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build_tree()
            node.right = build_tree()
            return node
        return build_tree()
```

---

## BST and Validations

### (GOOGLE) Problem LC 98: Validate Binary Search Tree ([link](https://leetcode.com/problems/validate-binary-search-tree/description/))

```python
def isValidBST(root: Optional[TreeNode]) -> bool:
    # DFS with bounds
    def dfs(node, min_value, max_value):
        if not node:
            return True
        if not min_value < node.val < max_value:
            return False
        return dfs(node.left, min_value, node.val) and dfs(node.right, node.val, max_value)
    return dfs(root, float("-inf"), float("inf"))
```

### Problem LC 230: Kth Smallest Element in BST ([link](https://leetcode.com/problems/kth-smallest-element-in-a-bst/))

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # Iterative inorder traversal
        stack = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right
```

### Problem LC 235: LCA of BST ([link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/))

```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        curr = root
        while curr:
            if curr.val < p.val and curr.val < q.val:
                curr = curr.right
            elif curr.val > p.val and curr.val > q.val:
                curr = curr.left
            else:
                return curr
```

### Problem LC 236: LCA of Binary Tree ([link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/))

```python
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root or root == p or root == q:
            return root
        left_res = self.lowestCommonAncestor(root.left, p, q)
        right_res = self.lowestCommonAncestor(root.right, p, q)
        if left_res and right_res:
            return root
        return left_res if left_res else right_res
```

### Problem LC 124: Binary Tree Maximum Path Sum ([link](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/))

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        max_sum = float("-inf")

        def get_max(node):
            if not node:
                return 0
            nonlocal max_sum
            left = max(get_max(node.left), 0)
            right = max(get_max(node.right), 0)
            current_path_sum = node.val + left + right
            max_sum = max(max_sum, current_path_sum)
            return node.val + max(left, right)

        get_max(root)
        return max_sum
```

---

## Graph (BFS / DFS)

```python
# DFS on a Graph
def dfs_graph(graph, start_node, visited=None):
    if visited is None:
        visited = set()
    visited.add(start_node)
    for neighbor in graph[start_node]:
        if neighbor not in visited:
            dfs_graph(graph, neighbor, visited)
    return visited

# BFS on a Graph
def bfs_graph(graph, start_node):
    queue = deque([start_node])
    visited = set([start_node])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### (GOOGLE) Problem LC 200: Number of Islands ([link](https://leetcode.com/problems/number-of-islands/))

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        m = len(grid)
        n = len(grid[0])

        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n:
                return
            if grid[i][j] == "0":
                return
            grid[i][j] = "0"
            dfs(i+1, j)
            dfs(i, j+1)
            dfs(i-1, j)
            dfs(i, j-1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    count += 1
                    dfs(i, j)
        return count
```

### Problem LC 133: Clone Graph ([link](https://leetcode.com/problems/clone-graph/))

```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return
        clone = {}
        def dfs(node):
            if not node:
                return
            if node in clone:
                return clone[node]
            copy = Node(node.val)
            clone[node] = copy
            for n in node.neighbors:
                copy.neighbors.append(dfs(n))
            return copy
        return dfs(node)
```

### Problem LC 417: Pacific Atlantic Water Flow ([link](https://leetcode.com/problems/pacific-atlantic-water-flow/description/))

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []
        m, n = len(heights), len(heights[0])
        pacific, atlantic = set(), set()

        def dfs(i, j, visited, previous):
            if (i < 0 or j < 0 or i >= m or j >= n
                    or (i, j) in visited or heights[i][j] < previous):
                return
            visited.add((i, j))
            dfs(i+1, j, visited, heights[i][j])
            dfs(i-1, j, visited, heights[i][j])
            dfs(i, j+1, visited, heights[i][j])
            dfs(i, j-1, visited, heights[i][j])

        for i in range(m):
            dfs(i, 0, pacific, heights[i][0])
            dfs(i, n-1, atlantic, heights[i][n-1])
        for j in range(n):
            dfs(0, j, pacific, heights[0][j])
            dfs(m-1, j, atlantic, heights[m-1][j])

        return list(pacific & atlantic)
```

### Problem LC 207: Course Schedule ([link](https://leetcode.com/problems/course-schedule/description/))

```python
class Solution:
    def canFinish(self, numCourses, prerequisites):
        graph = defaultdict(list)
        indegree = [0] * numCourses
        for i, j in prerequisites:
            graph[j].append(i)
            indegree[i] += 1

        q = deque([i for i in range(numCourses) if indegree[i] == 0])
        visited = 0
        while q:
            course = q.popleft()
            visited += 1
            for next_course in graph[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    q.append(next_course)
        return visited == numCourses
```

### Problem LC 210: Course Schedule II ([link](https://leetcode.com/problems/course-schedule-ii/description/))

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        indegree = [0]*numCourses
        for i, j in prerequisites:
            indegree[i] += 1
            graph[j].append(i)

        result = [i for i in range(numCourses) if indegree[i] == 0]
        q = deque(result)
        visited = 0

        while q:
            course = q.popleft()
            visited += 1
            for n in graph[course]:
                indegree[n] -= 1
                if indegree[n] == 0:
                    q.append(n)
                    result.append(n)

        return [] if numCourses != visited else result
```

---

## Dijkstra's and Union Find

```python
# Union Find Template
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, u, v):
        p1, p2 = self.find(u), self.find(v)
        if p1 == p2:
            return False
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.parent[p1] = p2
            self.rank[p2] += self.rank[p1]
        return True
```

### (GOOGLE) Problem LC 743: Network Delay Time ([link](https://leetcode.com/problems/network-delay-time/))

```python
def networkDelayTime(times, n, k):
    adj = defaultdict(list)
    for i, j, t in times:
        adj[i].append((j, t))

    min_heap = [(0, k)]
    distances = [float("inf")] * (n+1)
    distances[k] = 0
    visited_count = 0
    while min_heap:
        time_taken, node = heapq.heappop(min_heap)
        if time_taken > distances[node]:
            continue
        visited_count += 1
        if visited_count == n:
            return time_taken
        for neighbor, time in adj[node]:
            new_time = time_taken + time
            if distances[neighbor] > new_time:
                distances[neighbor] = new_time
                heapq.heappush(min_heap, (new_time, neighbor))
    return -1
```

### (GOOGLE) Problem LC 684: Redundant Connection ([link](https://leetcode.com/problems/redundant-connection/description/))

```python
def findRedundantConnection(edges):
    n = len(edges)
    rank = [1]*(n+1)
    parent = list(range(n+1))

    def find(n):
        if parent[n] != n:
            parent[n] = find(parent[n])
        return parent[n]

    def union(n1, n2):
        p1, p2 = find(n1), find(n2)
        if p1 == p2:
            return False
        if rank[p1] > rank[p2]:
            parent[p2] = p1
            rank[p1] += rank[p2]
        else:
            parent[p1] = p2
            rank[p2] += rank[p1]
        return True

    for u, v in edges:
        if not union(u, v):
            return [u, v]
```

### Problem LC 323: Number of Connected Components ([link](https://neetcode.io/problems/count-connected-components/question))

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        rank = [1] * n
        parent = list(range(n))

        def find(n):
            if parent[n] != n:
                parent[n] = find(parent[n])
            return parent[n]

        def union(u, v):
            p1, p2 = find(u), find(v)
            if p1 == p2:
                return 0
            if rank[p1] > rank[p2]:
                parent[p2] = p1
                rank[p1] += rank[p2]
            else:
                parent[p1] = p2
                rank[p2] += rank[p1]
            return 1

        components = n
        for u, v in edges:
            components -= union(u, v)
        return components
```

### Problem LC 721: Accounts Merge ([link](https://leetcode.com/problems/accounts-merge/description/))

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        parent = {}
        email_to_name = {}

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(u, v):
            p1, p2 = find(u), find(v)
            if p1 != p2:
                parent[p1] = p2

        for acc in accounts:
            name = acc[0]
            first_email = acc[1]
            for email in acc[1:]:
                if email not in parent:
                    parent[email] = email
                email_to_name[email] = name
                union(email, first_email)

        groups = defaultdict(list)
        for email in parent:
            root = find(email)
            groups[root].append(email)

        res = []
        for root, emails in groups.items():
            res.append([email_to_name[root]] + sorted(emails))
        return res
```

---

## Heaps and Priority Queues

### MinHeap from Scratch

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val: int):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def extract_min(self) -> int:
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return min_val

    def _sift_up(self, index: int):
        parent_index = (index - 1) // 2
        while index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            index = parent_index
            parent_index = (index - 1) // 2

    def _sift_down(self, index: int):
        length = len(self.heap)
        while True:
            left_child_idx = 2 * index + 1
            right_child_idx = 2 * index + 2
            smallest = index
            if left_child_idx < length and self.heap[left_child_idx] < self.heap[smallest]:
                smallest = left_child_idx
            if right_child_idx < length and self.heap[right_child_idx] < self.heap[smallest]:
                smallest = right_child_idx
            if smallest == index:
                break
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest
```

### Problem LC 703: Kth Largest Element in a Stream ([link](https://leetcode.com/problems/kth-largest-element-in-a-stream/description/))

```python
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.min_heap = nums
        heapq.heapify(self.min_heap)
        while len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)

    def add(self, val):
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

### (GOOGLE) Problem LC 973: K Closest Points to Origin ([link](https://leetcode.com/problems/k-closest-points-to-origin/description/))

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        max_heap = []
        for point1, point2 in points:
            dist = point1**2 + point2**2
            heapq.heappush(max_heap, (-dist, point1, point2))
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        return [[x, y] for _, x, y in max_heap]
```

### Problem LC 295: Find Median From Data Stream ([link](https://leetcode.com/problems/find-median-from-data-stream/))

```python
class MedianFinder:
    def __init__(self):
        self.upper_half = []  # min_heap
        self.lower_half = []  # max_heap (negate values)

    def addNum(self, num):
        heapq.heappush(self.lower_half, -num)
        if (self.lower_half and self.upper_half and
                (-self.lower_half[0] > self.upper_half[0])):
            val = -heapq.heappop(self.lower_half)
            heapq.heappush(self.upper_half, val)
        if len(self.lower_half) > len(self.upper_half) + 1:
            val = -heapq.heappop(self.lower_half)
            heapq.heappush(self.upper_half, val)
        elif len(self.upper_half) > len(self.lower_half) + 1:
            val = heapq.heappop(self.upper_half)
            heapq.heappush(self.lower_half, -val)

    def findMedian(self):
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        elif len(self.upper_half) > len(self.lower_half):
            return self.upper_half[0]
        else:
            return (-self.lower_half[0] + self.upper_half[0]) / 2
```

### (GOOGLE) Problem LC 621: Task Scheduler ([link](https://leetcode.com/problems/task-scheduler/))

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # O(N) math approach
        frequencies = Counter(tasks)
        max_freq = max(frequencies.values())
        num_of_max_freq = sum(1 for v in frequencies.values() if v == max_freq)
        num_of_chunks = max_freq - 1
        size_of_chunks = n + 1
        total_size = num_of_chunks * size_of_chunks + num_of_max_freq
        return max(total_size, len(tasks))
```

---

## Tries and String Algorithms

### (GOOGLE) Problem LC 208: Implement Trie ([link](https://leetcode.com/problems/implement-trie-prefix-tree/description/))

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.iswordend = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.iswordend = True

    def search(self, word: str) -> bool:
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.iswordend

    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True
```

### Problem LC 211: Design Add and Search Words ([link](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/))

```python
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.iswordend = True

    def search(self, word: str) -> bool:
        def dfs(index, node):
            if index == len(word):
                return node.iswordend
            char = word[index]
            if char == ".":
                for ch in node.children.values():
                    if dfs(index+1, ch):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(index+1, node.children[char])
        return dfs(0, self.root)
```

### (GOOGLE) Problem LC 212: Word Search II ([link](https://leetcode.com/problems/word-search-ii/))

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None
 Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for word in words:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = word

        m, n = len(board), len(board[0])
        res = []

        def dfs(r, c, node):
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] not in node.children:
                return
            char = board[r][c]
            child = node.children[char]
            if child.word is not None:
                res.append(child.word)
                child.word = None
            board[r][c] = "#"
            dfs(r+1, c, child)
            dfs(r-1, c, child)
            dfs(r, c+1, child)
            dfs(r, c-1, child)
            board[r][c] = char
            if not child.children:
                del node.children[char]

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        return res
```

### Problem LC 139: Word Break ([link](https://leetcode.com/problems/word-break/description/))

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Trie + DP
        root = TrieNode()
        for word in wordDict:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True

        n = len(s)
        dp = [False] * (n+1)
        dp[0] = True

        for i in range(n):
            if not dp[i]:
                continue
            node = root
            for j in range(i, n):
                char = s[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_word:
                    dp[j+1] = True
        return dp[n]
```

### Problem LC 14: Longest Common Prefix

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        prefix = strs[0]
        for i in range(1, len(strs)):
            while strs[i].find(prefix) != 0:
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
```

---

## 1D Dynamic Programming

### Problem LC 70: Climbing Stairs ([link](https://leetcode.com/problems/climbing-stairs/description/))

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 0 or n == 1:
            return 1
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```

### Problem LC 198: House Robber ([link](https://leetcode.com/problems/house-robber/description/))

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        rob1 = 0
        rob2 = 0
        for n in nums:
            current_max = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = current_max
        return rob2
```

### (GOOGLE) Problem LC 322: Coin Change ([link](https://leetcode.com/problems/coin-change/description/))

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount+1)
        dp[0] = 0
        for a in range(1, amount+1):
            for coin in coins:
                if a-coin >= 0:
                    dp[a] = min(dp[a], 1+dp[a-coin])
        return dp[amount] if dp[amount] != float("inf") else -1
```

### Problem LC 300: Longest Increasing Subsequence ([link](https://leetcode.com/problems/longest-increasing-subsequence/description/))

```python
# DP — O(n²)
class Solution:
    def lengthOfLIS(self, nums):
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], 1+dp[j])
        return max(dp)

# Binary search — O(n log n)
from bisect import bisect_left

class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        sub = []
        for num in nums:
            if len(sub) == 0 or num > sub[-1]:
                sub.append(num)
            else:
                replace_index = bisect_left(sub, num)
                sub[replace_index] = num
        return len(sub)
```

### Problem LC 72: Edit Distance ([link](https://leetcode.com/problems/edit-distance/description/))

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],   # delete
                        dp[i][j-1],   # insert
                        dp[i-1][j-1]  # replace
                    )
        return dp[m][n]
```

### Problem LC 97: Interleaving String ([link](https://leetcode.com/problems/interleaving-string/description/))

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1)+len(s2) != len(s3):
            return False
        m, n = len(s1), len(s2)
        dp = [[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        for j in range(1, n+1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        for i in range(1, m+1):
            for j in range(1, n+1):
                current_index = i+j-1
                from_above = dp[i-1][j] and s1[i-1] == s3[current_index]
                from_left = dp[i][j-1] and s2[j-1] == s3[current_index]
                dp[i][j] = from_above or from_left
        return dp[m][n]
```

---

## Backtracking and Recursion

### Problem LC 78: Subsets ([link](https://leetcode.com/problems/subsets/description/))

```python
class Solution:
    def subsets(self, nums):
        result = []
        def bt(start, path):
            result.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                bt(i+1, path)
                path.pop()
        bt(0, [])
        return result
```

### Problem LC 46: Permutations ([link](https://leetcode.com/problems/permutations/description/))

```python
class Solution:
    def permutation(self, nums):
        res = []
        def bt(start_index):
            if start_index == len(nums):
                res.append(nums[:])
                return
            for i in range(start_index, len(nums)):
                nums[start_index], nums[i] = nums[i], nums[start_index]
                bt(start_index+1)
                nums[start_index], nums[i] = nums[i], nums[start_index]
        bt(0)
        return res
```

### (GOOGLE) Problem LC 39: Combination Sum ([link](https://leetcode.com/problems/combination-sum/description/))

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        candidates.sort()

        def bt(start, path, current_sum):
            if current_sum == target:
                result.append(path[:])
                return
            for i in range(start, len(candidates)):
                if current_sum + candidates[i] > target:
                    return
                path.append(candidates[i])
                bt(i, path, current_sum + candidates[i])
                path.pop()

        bt(0, [], 0)
        return result
```

### (GOOGLE) Problem LC 51: N-Queens ([link](https://leetcode.com/problems/n-queens/description/))

```python
class Solution:
    def solveNQueens(self, n):
        res = []
        board = [["."]*n for _ in range(n)]
        cols, left_diag, right_diag = set(), set(), set()

        def bt(r):
            if r == n:
                res.append(["".join(row) for row in board])
                return
            for col in range(n):
                if col in cols or (col+r) in right_diag or (r-col) in left_diag:
                    continue
                board[r][col] = "Q"
                cols.add(col)
                left_diag.add(r-col)
                right_diag.add(col+r)
                bt(r+1)
                board[r][col] = "."
                cols.remove(col)
                left_diag.remove(r-col)
                right_diag.remove(col+r)
        bt(0)
        return res
```

### (GOOGLE) Problem LC 17: Letter Combinations of a Phone Number ([link](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/))

```python
class Solution:
    def letterCombination(self, digits):
        if not digits:
            return []
        phone = {
            '2': 'abc', '3': 'def', '4': 'ghi',
            '5': 'jkl', '6': 'mno', '7': 'pqrs',
            '8': 'tuv', '9': 'wxyz'
        }
        res = []
        def bt(start, path):
            if len(path) == len(digits):
                res.append("".join(path))
                return
            for char in phone[digits[start]]:
                path.append(char)
                bt(start+1, path)
                path.pop()
        bt(0, [])
        return res
```

---

## Interval and Greedy

### Interval Problem-Solving Framework

**Step 1 — Sort:** Choose what to sort by.

- **Sort by start time** → merge/cover (Merge Intervals, Insert Interval)
- **Sort by end time** → maximize non-overlapping (Activity Selection, Non-overlapping Intervals)

**Step 2 — Process Linearly:** Two intervals `[a, b]` and `[c, d]` overlap if `c <= b` (sorted by start).

**Step 3 — Track the "Active" Set:**

| What to track              | When to use it                              |
| -------------------------- | ------------------------------------------- |
| Last merged interval's end | Merge Intervals                             |
| Min-heap of end times      | Meeting Rooms II / max simultaneous overlap |
| Count of intervals removed | Non-overlapping Intervals                   |

### (GOOGLE) Problem LC 56: Merge Intervals ([link](https://leetcode.com/problems/merge-intervals/))

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        result = [intervals[0]]
        for left, right in intervals[1:]:
            if left <= result[-1][1]:
                result[-1][1] = max(result[-1][1], right)
            else:
                result.append([left, right])
        return result
```

### (GOOGLE) Problem LC 57: Insert Interval ([link](https://leetcode.com/problems/insert-interval/description/))

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        new_left, new_right = newInterval
        result = []
        for interval in intervals:
            left, right = interval
            if right < new_left:
                result.append(interval)
            elif left > new_right:
                result.append([new_left, new_right])
                new_left, new_right = left, right
            else:
                new_left = min(new_left, left)
                new_right = max(new_right, right)
        result.append([new_left, new_right])
        return result
```

### (GOOGLE) Problem LC 435: Non-overlapping Intervals ([link](https://leetcode.com/problems/non-overlapping-intervals/description/))

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        removed_count = 0
        last_kept_end = intervals[0][1]
        for left, right in intervals[1:]:
            if left < last_kept_end:
                removed_count += 1
            else:
                last_kept_end = right
        return removed_count
```

### Problem LC 253: Meeting Rooms II ([link](https://neetcode.io/problems/meeting-schedule-ii/question))

```python
class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        if not intervals:
            return 0
        starts = sorted(x.start for x in intervals)
        ends = sorted(x.end for x in intervals)
        s = e = 0
        used_rooms = 0
        while s < len(intervals):
            if starts[s] < ends[e]:
                used_rooms += 1
            else:
                e += 1
            s += 1
        return used_rooms
```

### Problem LC 452: Minimum Arrows to Burst Balloons ([link](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/))

```python
def findMinArrowShots(points: list[list[int]]) -> int:
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    current_arrow_pos = points[0][1]
    for i in range(1, len(points)):
        if points[i][0] > current_arrow_pos:
            arrows += 1
            current_arrow_pos = points[i][1]
    return arrows
```

---

## Advanced DP (State Machine and Bitmask)

### Problem LC 309: Best Time to Buy and Sell Stock with Cooldown ([link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/))

```python
# State machine DP — O(n) time, O(1) space
def maxProfit(prices):
    rest = 0
    hold = -float('inf')
    sold = 0
    for price in prices:
        prev_hold = hold
        hold = max(hold, rest - price)
        rest = max(rest, sold)
        sold = prev_hold + price
    return max(rest, sold)
```

### Problem LC 188: Best Time to Buy and Sell Stock IV ([link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/))

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if n <= 1 or k == 0:
            return 0
        if k > n//2:
            return sum(max(prices[i]-prices[i-1], 0) for i in range(1, n))
        dp = [[0]*n for _ in range(k+1)]
        for i in range(1, k+1):
            max_diff = -prices[0]
            for j in range(1, n):
                dp[i][j] = max(dp[i][j-1], prices[j]+max_diff)
                max_diff = max(max_diff, dp[i-1][j]-prices[j])
        return dp[k][n-1]
```

### Problem LC 416: Partition Equal Subset Sum ([link](https://leetcode.com/problems/partition-equal-subset-sum/))

```python
class Solution:
    def canPartition(self, nums):
        if sum(nums) % 2 != 0:
            return False
        target = sum(nums)//2
        dp = [False]*(target+1)
        dp[0] = True
        for num in nums:
            for i in range(target, num-1, -1):
                dp[i] = dp[i] or dp[i-num]
        return dp[target]
```

### Problem LC 131: Palindrome Partitioning ([link](https://leetcode.com/problems/palindrome-partitioning/description/))

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = length <= 2 or dp[i+1][j-1]

        res = []
        current_partition = []

        def dfs(start_index):
            if start_index >= n:
                res.append(current_partition.copy())
                return
            for end_index in range(start_index, n):
                if dp[start_index][end_index]:
                    current_partition.append(s[start_index:end_index+1])
                    dfs(end_index + 1)
                    current_partition.pop()

        dfs(0)
        return res
```

### Problem LC 312: Burst Balloons ([link](https://leetcode.com/problems/burst-balloons/))

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        padded_nums = [1] + nums + [1]
        n = len(padded_nums)
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n):
            for left in range(0, n - length):
                right = left + length
                for i in range(left + 1, right):
                    coins_for_i = padded_nums[left] * padded_nums[i] * padded_nums[right]
                    total_coins = coins_for_i + dp[left][i] + dp[i][right]
                    dp[left][right] = max(dp[left][right], total_coins)
        return dp[0][n - 1]
```

---

## Bit Manipulation and Math

### Problem LC 136: Single Number ([link](https://leetcode.com/problems/single-number/description/))

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num
        return ans
```

### Problem LC 191: Number of 1 Bits ([link](https://leetcode.com/problems/number-of-1-bits/description/))

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        # Brian Kernighan's Algorithm — clears lowest set bit each iteration
        count = 0
        while n:
            n &= n-1
            count += 1
        return count
```

### Problem LC 338: Counting Bits ([link](https://leetcode.com/problems/counting-bits/description/))

```python
def countBits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp
```

### (GOOGLE) Problem LC 50: Pow(x, n) — Fast Exponentiation

```python
def fast_power(base, exp):
    is_negative = exp < 0
    if is_negative:
        exp = -exp
    result = 1
    while exp > 0:
        if exp & 1:
            result = result * base
        base = base * base
        exp >>= 1
    return 1 / result if is_negative else result
```

### (GOOGLE) Problem LC 43: Multiply Strings ([link](https://leetcode.com/problems/multiply-strings/description/))

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        res = [0] * (len(num1) + len(num2))
        for i in range(len(num1)-1, -1, -1):
            for j in range(len(num2)-1, -1, -1):
                product = int(num1[i]) * int(num2[j])
                p1, p2 = i+j, i+j+1
                total_sum = product + res[p2]
                res[p2] = total_sum % 10
                res[p1] += total_sum // 10
        res_str = ""
        for num in res:
            if not (res_str == "" and num == 0):
                res_str += str(num)
        return res_str
```

---

## Segment Tree and Advanced Data Structures

### Segment Tree — Range Sum Query ([link](https://leetcode.com/problems/range-sum-query-mutable/description/))

```python
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def update(self, index, val):
        pos = index + self.n
        self.tree[pos] = val
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.tree[2*pos] + self.tree[2*pos+1]

    def query(self, L, R):
        left = L + self.n
        right = R + self.n + 1
        res = 0
        while left < right:
            if left % 2 == 1:
                res += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                res += self.tree[right]
            left //= 2
            right //= 2
        return res
```

### (GOOGLE) Problem LC 218: The Skyline Problem ([link](https://leetcode.com/problems/the-skyline-problem/description/))

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        events = []
        for left, right, height in buildings:
            events.append((left, -height, right))
            events.append((right, height, 0))
        events.sort()

        result = []
        heap = [(0, float('inf'))]

        for x, height, right in events:
            if height < 0:
                heapq.heappush(heap, (height, right))
            while heap[0][1] <= x:
                heapq.heappop(heap)
            current_max_height = -heap[0][0]
            if not result or result[-1][1] != current_max_height:
                result.append([x, current_max_height])
        return result
```

### Problem LC 223: Rectangle Area ([link](https://leetcode.com/problems/rectangle-area/description/))

```python
class Solution:
    def computeArea(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> int:
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        overlapping_width = min(ax2, bx2) - max(ax1, bx1)
        overlapping_height = min(ay2, by2) - max(ay1, by1)
        overlap_area = 0
        if overlapping_width > 0 and overlapping_height > 0:
            overlap_area = overlapping_width * overlapping_height
        return area_a + area_b - overlap_area
```

### (GOOGLE) Problem LC 149: Max Points on a Line ([link](https://leetcode.com/problems/max-points-on-a-line/description/))

```python
import math
from collections import defaultdict

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) <= 1:
            return len(points)
        global_max = 1
        for i in range(len(points)):
            x1, y1 = points[i]
            maps = defaultdict(int)
            for j in range(len(points)):
                if i == j:
                    continue
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1
                g = math.gcd(dx, dy)
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                maps[(dx, dy)] += 1
            if maps:
                global_max = max(global_max, max(maps.values()) + 1)
        return global_max
```
