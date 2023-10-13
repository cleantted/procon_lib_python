mod = 998244353
INF = 1 << 61
DIFF = 10 ** -9


class Val:
    def __init__(self, f, v=None):
        self.f = f
        self.v = v
 
    def __str__(self):
        return str(self.v)
 
    def ud(self, n):
        if n is None:
            return

        if self.v is None:
            self.v = n
            return
        self.v = self.f(self.v, n)


class Comb:
    __slots__ = ["d_comb", "F", "FI"]
    def __init__(self, N):
        F = [1] * (N + 1)
        for i in range(N):
            F[i + 1] = (i + 1) * F[i] % mod

        FI = [1] * (N + 1)
        FI[N] = pow(F[N], mod - 2, mod)
        for i in reversed(range(N)):
            FI[i] = (i + 1) * FI[i + 1] % mod

        self.d_comb = {}
        self.F = F
        self.FI = FI

    def comb(self, a, b):
        if (a, b) not in self.d_comb:
            self.d_comb[(a, b)] = ((self.F[a] * self.FI[a - b]) % mod) * self.FI[b] % mod
            self.d_comb[(a, a - b)] = self.d_comb[(a, b)]
        return self.d_comb[(a, b)]


def LCS(a, b):
    dp = [[0 for _ in range(len(a) + 1)] for __ in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[j][i] = dp[j - 1][i - 1] + 1
            else:
                dp[j][i] = max(dp[j - 1][i], dp[j][i - 1])
    return dp[len(b)][len(a)]


class UnionFind:
    def __init__(self, N):
        self.state = [-1] * N
        self.rank = [0] * N
        self.num_group = N
    
    def get_parent(self, a):
        p = self.state[a]
        if p < 0:
            return a
        
        q = self.get_parent(p)
        self.state[a] = q
        return q

    def make_pair(self, a, b):
        pa = self.get_parent(a)
        pb = self.get_parent(b)
        if pa == pb:
            return

        if self.rank[pa] > self.rank[pb]:
            pa, pb = pb, pa
            a, b = b, a
        elif self.rank[pa] == self.rank[pb]:
            self.rank[pb] += 1

        self.state[pb] += self.state[pa]
        self.state[pa] = pb
        self.state[a] = pb
        self.num_group -= 1
    
    def is_pair(self, a, b):
        return self.get_parent(a) == self.get_parent(b)

    def get_size(self, a):
        return -self.state[self.get_parent(a)]


def mat_mul(X, Y):
    m = len(X)
    return [[sum(X[i][k] * Y[k][j] % mod for k in range(m)) % mod for j in range(m)] for i in range(m)]


def mat_pow(M, k):
    m = len(M)
    res = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    while k > 0:
        if k & 1:
            res = mat_mul(M, res)
        M = mat_mul(M, M)
        k >>= 1

    return res


def dijk(A, h0, w0):
    import heapq
    H = len(A)
    W = len(A[0])
 
    Q = [(0, (h0, w0))]
    P = [[False for _ in range(W)] for _ in range(H)]
    D = [[10 ** 10 for _ in range(W)] for _ in range(H)]
    D[h0][w0] = 0
    while Q:
        c, (h, w) = heapq.heappop(Q)
        if P[h][w]:
            continue
        P[h][w] = True
        D[h][w] = c
 
        if h != 0 and not P[h - 1][w]:
            if D[h - 1][w] > c + A[h - 1][w]:
                heapq.heappush(Q, (c + A[h - 1][w], (h - 1, w)))
 
        if h != H - 1 and not P[h + 1][w]:
            if D[h + 1][w] > c + A[h + 1][w]:
                heapq.heappush(Q, (c + A[h + 1][w], (h + 1, w)))
 
        if w != 0 and not P[h][w - 1]:
            if D[h][w - 1] > c + A[h][w - 1]:
                heapq.heappush(Q, (c + A[h][w - 1], (h, w - 1)))
 
        if w != W - 1 and not P[h][w + 1]:
            if D[h][w + 1] > c + A[h][w + 1]:
                heapq.heappush(Q, (c + A[h][w + 1], (h, w + 1)))
 
    return D


def LIS(A, strict=True):
    # TODO: strict=False は未確認
    # 最長増加部分列(strict=Falseで広義単調増加になる)
    from bisect import bisect_left
    T = []
    position = []

    for a in A:
        if len(T) == 0 or (strict and T[-1] < a) or (not strict and T[-1] <= a):
            position.append(len(T))
            T.append(a)
        else:
            if strict:
                k = bisect_left(T, a)
            else:
                k = bisect_left(T, a + 1)
            position.append(k)
            T[k] = a

    res = []
    t = len(T) - 1
    for i, p in enumerate(reversed(position)):
        if t == p:
            res.append(len(A) - 1 - i)
            t -= 1
    res.reverse()
    return res


class LCA:
    def __init__(self, N, L, root=0):
        max_depth = (N - 1).bit_length()
        parent = [[-1] * N for _ in range(max_depth + 1)]
        self.depth, parent[0] = self._dfs(L, root)

        for i in range(max_depth):
            for j in range(N):
                if parent[i][j] != -1:
                    parent[i + 1][j] = parent[i][parent[i][j]]
        
        self.parent = parent
        self.max_depth = max_depth

    def _dfs(self, L, root):
        N = len(L)
        depth = [-1] * N
        depth[0] = 0
        parent = [-1] * N
        S = [(root, -1)]
        for i, p in S:
            for j in L[i]:
                if j == p:
                    continue
                if depth[j] != -1:
                    continue
                depth[j] = depth[i] + 1
                parent[j] = i
                S.append((j, i))
        return depth, parent
 
    def q(self, u, v):
        d = self.depth[v] - self.depth[u]
        if d < 0:
            u, v = v, u
            d *= -1
 
        for k in range(self.max_depth + 1):
            if d & 1:
                v = self.parent[k][v]
            d >>= 1
 
        if u == v:
            return u
 
        for k in reversed(range(self.max_depth + 1)):
            pu = self.parent[k][u]
            pv = self.parent[k][v]
            if pu != pv:
                u = pu
                v = pv
        if u != v:
            u = self.parent[k][u]

        return u

    def calc_dist(self, a, b):
        return self.depth[a] + self.depth[b] - 2 * self.depth[self.q(a, b)]


class CS1:
    def __init__(self, A):
        N = len(A)
        S = [0] * (N + 1)
        for i, a in enumerate(A):
            S[i + 1] = S[i] + a
        self.S = S

    # [l, r)
    def get(self, l, r):
        S = self.S
        return S[r] - S[l]


class CS2:
    def __init__(self, A):
        H, W = len(A), len(A[0])
        S = [[0 for _ in range(W + 1)] for _ in range(H + 1)]
        for i in range(H):
            for j in range(W):
                S[i + 1][j + 1] = S[i + 1][j] + S[i][j + 1] - S[i][j] + A[i][j]
        self.S = S

    # [h1, h2) * [w1, w2)
    def get(self, h1, w1, h2, w2):
        S = self.S
        return S[h2][w2] - S[h2][w1] - S[h1][w2] + S[h1][w1]
 

class SegmentTree:
    def __init__(self, S, unit, funct):
        N = len(S)
        M = 1 << (N - 1).bit_length()
        T = [unit for _ in range(2 * M)]
        for i, s in enumerate(S):
            T[M + i] = s
        for i in range(M - 1, 0, -1):
            T[i] = funct(T[2 * i], T[2 * i + 1])
        self.M = M
        self.T = T
        self.unit = unit
        self.funct = funct

    def update(self, i, a):
        k = i + self.M
        self.T[k] = a
        while k > 1:
            k >>= 1
            self.T[k] = self.funct(self.T[k << 1], self.T[(k << 1) + 1])

    def apply_right(self, i, a):
        k = i + self.M
        v = self.funct(self.T[k], a)
        if v == self.T[k]:
            return

        self.T[k] = v
        while k > 1:
            k >>= 1
            self.T[k] = self.funct(self.T[k << 1], self.T[(k << 1) + 1])

    def apply_left(self, i, a):
        k = i + self.M
        v = self.funct(a, self.T[k])
        if v == self.T[k]:
            return 

        self.T[k] = v
        while k > 1:
            k >>= 1
            self.T[k] = self.funct(self.T[k << 1], self.T[(k << 1) + 1])

    def query(self, l, r):
        l += self.M
        r += self.M
        left_value = self.unit
        right_value = self.unit
        while l < r:
            if l & 1:
                left_value = self.funct(left_value, self.T[l])
                l += 1
            if r & 1:
                right_value = self.funct(self.T[r - 1], right_value)
            l >>= 1
            r >>= 1
        
        res = self.funct(left_value, right_value)

        return res

    def at(self, i):
        return self.T[i + self.M]

    def __str__(self):
        return str(self.T[self.M:])


class LasySegmentTree:
    __slots__ = ["N", "num", "T", "lazy", "unit", "op", "mapping", "composition", "id_f"]
    def __init__(self, S, unit, op, mapping, composition, id_f):
        N = len(S)
        num = 1 << ((N - 1).bit_length())
        T = [unit for _ in range(2 * num)]
        for i, s in enumerate(S):
            T[i + num] = s
        for i in range(num - 1, 0, -1):
            T[i] = op(T[2 * i], T[2 * i + 1])
        
        self.N = N
        self.num = num

        self.T = T
        self.lazy = [id_f for _ in range(2 * num)]
        self.unit = unit
        self.op = op
        self.mapping = mapping
        self.composition = composition
        self.id_f = id_f

    def update(self, a, b, f):
        a += self.num
        b += self.num

        am = a // (a & -a)
        bm = b // (b & -b) - 1

        self.propagate_above(am)
        self.propagate_above(bm)
        while a < b:
            if a & 1:
                self.lazy[a] = self.composition(self.lazy[a], f)
                a += 1
            if b & 1:
                b -= 1
                self.lazy[b] = self.composition(self.lazy[b], f)
            a >>= 1
            b >>= 1

        self.calc_above(am)
        self.calc_above(bm)

    def calc(self, i):
        return self.mapping(self.lazy[i], self.T[i])

    def calc_above(self, i):
        i >>= 1
        while i:
            self.T[i] = self.op(self.calc(2 * i), self.calc(2 * i + 1))
            i >>= 1

    def propagate(self, i):
        self.T[i] = self.mapping(self.lazy[i], self.T[i])
        self.lazy[2 * i] = self.composition(self.lazy[2 * i], self.lazy[i])
        self.lazy[2 * i + 1] = self.composition(self.lazy[2 * i + 1], self.lazy[i])
        self.lazy[i] = self.id_f

    def propagate_above(self, i):
        for h in range(i.bit_length(), 0, -1):
            self.propagate(i >> h)

    def query(self, a, b):
        a += self.num
        b += self.num

        am = a // (a & -a)
        bm = b // (b & -b) - 1
        self.propagate_above(am)
        self.propagate_above(bm)

        vl = self.unit
        vr = self.unit
        while a < b:
            if a & 1:
                vl = self.op(vl, self.calc(a))
                a += 1
            if b & 1:
                b -= 1
                vr = self.op(self.calc(b), vr)
            a >>= 1
            b >>= 1
        return self.op(vl, vr)

    def at(self, i):
        return self.T[i + self.num]

    def __str__(self):
        S = []
        for i in range(self.N):
            t = self.query(i, i + 1)
            S.append(t)
        return str(S)


class BIT:
    def __init__(self, N):
        self.N = N
        self.T = [0] * (N + 1)
 
    def add(self, i, x):
        i += 1
        while i <= self.N:
            self.T[i] += x
            i += i & -i
 
    def _sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.T[i]
            i -= i & -i
        return s
 
    def sum(self, i, j):
        si = self._sum(i - 1)
        sj = self._sum(j)
        return sj - si

    def lower_left(self, v):
        if v < 0:
            return -1
        x = 0
        k = 1 << (self.N.bit_length() - 1)
        while k > 0:
            if x + k < self.N and self.T[x + k] < v:
                v -= self.T[x + k]
                x += k
            k //= 2
        return x


# https://juppy.hatenablog.com/entry/2020/09/03/%E9%A0%86%E5%BA%8F%E4%BB%98%E3%81%8D%E9%9B%86%E5%90%88%E3%82%82%E3%81%A9%E3%81%8D_Python_1
class OrderedBIT:
    import bisect

    def __init__(self, A):
        self.A = A
        self.bit = BIT(len(A) + 1)
        self.num = 0

    def insert_value(self, v, c=1):
        k = bisect.bisect_left(self.A, v)
        self.bit.add(k, c)
        self.num += c
    
    def delete_value(self, v, c=1):
        k = bisect.bisect_left(self.A, v)
        self.bit.add(k, -c)
        self.num -= c

    # 0-index 
    def find_kth_val(self, k):
        if self.num <= k or k < 0:
            return None
        
        return self.A[self.bit.lower_left(k + 1)]

    # len(c <= x for c in C) 
    def count_lower(self, x):
        if x < self.A[0]:
            return 0
        return self.bit.sum(0, bisect.bisect_right(self.A, x) - 1)

    # len(c >= x for c in C)
    def count_upper(self, x):
        if x > self.A[-1]:
            return 0
        return self.bit.sum(bisect.bisect_left(self.A, x), len(self.A))

    def find_nearest_higher_value(self, x):
        return self.find_kth_val(self.count_lower(x))

    # return max(c < x for c in C)
    def find_nearest_lower_value(self, x):
        return self.find_kth_val(self.num - self.count_upper(x) - 1)


class DPTravel:
    def __init__(self, N, C):
        self.C = C
        self.N = N
        assert (2 ** N) * (N ** 2) < 10 ** 9


    def query(self, start):
        N = self.N
        C = self.C
        INF = 10 ** 20
        dp = [[INF] * (1 << N) for _ in range(N)]
        dp[start][0] = 0
        for i in range(1 << N):
            for k in range(N):
                if dp[k][i] == INF:
                    continue

                for j in range(N):
                    if (i >> j) & 1:
                        continue
                    dp[j][i | (1 << j)] = min(dp[j][i | (1 << j)], dp[k][i] + C[k][j])
        
        return dp[start][(1 << N) - 1]


def ext_gcd(a, b, p, q):
    if b == 0:
        return a, (1, 0)
    
    d, (q, p) = ext_gcd(b, a % b, q, p)
    q -= a // b * p
    return d, (p, q)


def chinese_rem(b1, m1, b2, m2):
    d, (p, q) = ext_gcd(m1, m2, 0, 0)
    if (b2 - b1) % d != 0:
        return (0, -1)
    
    m = m1 * (m2 // d)
    tmp = (b2 - b1) // d * p % (m2 // d)
    r = (b1 + m1 * tmp) % m
    return (r, m)


def binary_search(init_ok, init_ng, is_ok):
    ok, ng = init_ok, init_ng
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok


def next_comb(n):
    x = n & -n
    y = n + x
    return (((n & ~y) // x) >> 1) | y


class SortedSet:
    # https://github.com/tatyam-prime/SortedSet/blob/main/SortedSet.py
    from math import sqrt, ceil
    from bisect import bisect_left, bisect_right
    from typing import Iterable, TypeVar, Union, Tuple
    T = TypeVar('T')
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    @classmethod   
    def _new_bucket_size(cls, size: int) -> int:
        return int(ceil(sqrt(size / cls.BUCKET_RATIO)))

    def _build(self, a: list):
        size = self.size = len(a)
        bucket_size = self._new_bucket_size(self.size)
        self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
    
    def __init__(self, a: Iterable = []):
        "Make a new SortedSet from iterable. / O(N) if sorted and unique / O(N log N)"
        a = list(a)
        if not all(a[i] < a[i + 1] for i in range(len(a) - 1)):
            a = sorted(set(a))
        self._build(a)

    def __iter__(self):
        for i in self.a:
            for j in i: yield j

    def __reversed__(self):
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _bucket_index(self, x: T) -> int:
        "Find the index of the bucket which should contain x. / O(log N)"
        ok = -1
        ng = len(self.a)
        a = self.a
        while ng - ok > 1:
            mid = (ng + ok) >> 1
            if a[mid][0] <= x: ok = mid
            else: ng = mid
        if ok == -1: return 0
        if ng == len(self.a): return ok
        if a[ok][-1] < x:
            return ok + (len(a[ok]) > len(a[ok + 1]))
        return ok

    def __contains__(self, x: T) -> bool:
        "O(log N)"
        if self.size == 0: return False
        a = self.a[self._bucket_index(x)]
        i = bisect_left(a, x)
        return i != len(a) and a[i] == x

    def add(self, x: T) -> bool:
        "Add an element and return True if added. / O(N ** 0.5)"
        if self.size == 0:
            self._build([x])
            return True
        a = self.a[self._bucket_index(x)]
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x: return False
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build(list(self))
        return True

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(N ** 0.5)"
        if self.size == 0: return False
        a = self.a[self._bucket_index(x)]
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: return False
        a.pop(i)
        self.size -= 1
        if len(a) == 0:
            self._build(list(self))
        return True
    
    def lt(self, x: T) -> Union[T, None]:
        "Return the largest element < x, or None if it doesn't exist. / O(log N)"
        if self.size == 0: return None
        i = self._bucket_index(x)
        a = self.a
        if a[i][0] >= x:
            return a[i - 1][-1] if i else None
        return a[i][bisect_left(a[i], x) - 1]

    def le(self, x: T) -> Union[T, None]:
        "Return the largest element <= x, or None if it doesn't exist. / O(log N)"
        if self.size == 0: return None
        i = self._bucket_index(x)
        a = self.a
        if a[i][0] > x:
            return a[i - 1][-1] if i else None
        return a[i][bisect_right(a[i], x) - 1]

    def gt(self, x: T) -> Union[T, None]:
        "Return the smallest element > x, or None if it doesn't exist. / O(log N)"
        if self.size == 0: return None
        i = self._bucket_index(x)
        a = self.a
        if a[i][-1] <= x:
            return a[i + 1][0] if i + 1 < len(self.a) else None
        return a[i][bisect_right(a[i], x)]

    def ge(self, x: T) -> Union[T, None]:
        "Return the smallest element >= x, or None if it doesn't exist. / O(log N)"
        if self.size == 0: return None
        i = self._bucket_index(x)
        a = self.a
        if a[i][-1] < x:
            return a[i + 1][0] if i + 1 < len(self.a) else None
        return a[i][bisect_left(a[i], x)]
    
    def __getitem__(self, x: int) -> T:
        "Take (i, j) and return the j-th element in the i-th bucket, or IndexError if it doesn't exist. / O(1)"
        "Take x and return the x-th element, or IndexError if it doesn't exist. / O(N ** 0.5) (fast)"
        if isinstance(x, tuple):
            return self.a[x[0]][x[1]]
        if x < 0: x += self.size
        if x < 0 or x >= self.size: raise IndexError
        for a in self.a:
            if x < len(a): return a[x]
            x -= len(a)
        assert False
    
    def index(self, x: T) -> int:
        "Return the index of x, or ValueError if it doesn't exist. / O(N ** 0.5) (fast)"
        if self.size == 0: raise ValueError
        idx = self._bucket_index(x)
        a = self.a[idx]
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: raise ValueError
        for j in range(idx): i += len(self.a[j])
        return i

    def lower_bound(self, x: T) -> Tuple[int, int]:
        "Find the smallest element self.a[i][j] >= x and return (i, j), or (len(a), 0) if it doesn't exist. / O(log N)"
        if self.size == 0:
            return (0, 0)
        i = self._bucket_index(x)
        a = self.a
        if a[i][-1] < x:
            return (i + 1, 0)
        return (i, bisect_left(a[i], x))

    def upper_bound(self, x: T) -> Tuple[int, int]:
        "Find the smallest element self.a[i][j] > x and return (i, j), or (len(a), 0) if it doesn't exist. / O(log N)"
        if self.size == 0:
            return (0, 0)
        i = self._bucket_index(x)
        a = self.a
        if a[i][-1] <= x:
            return (i + 1, 0)
        return (i, bisect_right(a[i], x))


class SortedMultiset():
    # https://github.com/tatyam-prime/SortedSet/blob/main/SortedMultiset.py
    import math
    from bisect import bisect_left, bisect_right, insort
    from typing import Generic, Iterable, Iterator, TypeVar, Union, List
    T = TypeVar('T')
    BUCKET_RATIO = 50
    REBUILD_RATIO = 170

    def _build(self, a=None) -> None:
        "Evenly divide `a` into buckets."
        if a is None: a = list(self)
        size = self.size = len(a)
        bucket_size = int(math.ceil(math.sqrt(size / self.BUCKET_RATIO)))
        self.a = [a[size * i // bucket_size : size * (i + 1) // bucket_size] for i in range(bucket_size)]
    
    def __init__(self, a: Iterable[T] = []) -> None:
        "Make a new SortedMultiset from iterable. / O(N) if sorted / O(N log N)"
        a = list(a)
        if not all(a[i] <= a[i + 1] for i in range(len(a) - 1)):
            a = sorted(a)
        self._build(a)

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i: yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i): yield j
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "SortedMultiset" + str(self.a)
    
    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _find_bucket(self, x: T) -> List[T]:
        "Find the bucket which should contain x. self must not be empty."
        for a in self.a:
            if x <= a[-1]: return a
        return a

    def __contains__(self, x: T) -> bool:
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        return i != len(a) and a[i] == x

    def count(self, x: T) -> int:
        "Count the number of x."
        return self.index_right(x) - self.index(x)

    def add(self, x: T) -> None:
        "Add an element. / O(√N)"
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a = self._find_bucket(x)
        insort(a, x)
        self.size += 1
        if len(a) > len(self.a) * self.REBUILD_RATIO:
            self._build()

    def discard(self, x: T) -> bool:
        "Remove an element and return True if removed. / O(√N)"
        if self.size == 0: return False
        a = self._find_bucket(x)
        i = bisect_left(a, x)
        if i == len(a) or a[i] != x: return False
        a.pop(i)
        self.size -= 1
        if len(a) == 0: self._build()
        return True

    def lt(self, x: T) -> Union[T, None]:
        "Find the largest element < x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Union[T, None]:
        "Find the largest element <= x, or None if it doesn't exist."
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Union[T, None]:
        "Find the smallest element > x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Union[T, None]:
        "Find the smallest element >= x, or None if it doesn't exist."
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]
    
    def __getitem__(self, x: int) -> T:
        "Return the x-th element, or IndexError if it doesn't exist."
        if x < 0: x += self.size
        if x < 0: raise IndexError
        for a in self.a:
            if x < len(a): return a[x]
            x -= len(a)
        raise IndexError

    def index(self, x: T) -> int:
        "Count the number of elements < x."
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        "Count the number of elements <= x."
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans


# V: int: 頂点数
# G: List[Dict[int, List[int, int]]]: 各頂点の出リンク(index->(cost, capacity)
# f: 流すフロー
def min_flow(V, G, f):
    INF = 1 << 61
    res = 0
    while f > 0:
        P = [-1] * V
        dist = [INF] * V
        dist[0] = 0
        update = True
        while update:
            update = False
            for i in range(V):
                if dist[i] == INF:
                    continue

                for j, (ff, c) in G[i].items():
                    if ff > 0 and dist[j] > dist[i] + c:
                        dist[j] = dist[i] + c
                        P[j] = i
                        update = True

        if dist[V - 1] == INF:
            return INF

        min_f = INF
        u = V - 1
        while u != 0:
            p = P[u]
            min_f = min(min_f, G[p][u][0])
            u = p
        
        u = V - 1
        while u != 0:
            p = P[u]
            G[p][u][0] -= min_f
            G[u][p][0] += min_f
            u = p
        f -= min_f
        res += dist[V - 1]

    return res


def min_cost_flow(V, G, f):
    INF = 1 << 61
    res = 0
    while f > 0:
        P = [-1] * V
        dist = [INF] * V
        dist[0] = 0
        update = True
        while update:
            update = False
            for i in range(V):
                if dist[i] == INF:
                    continue

                for j, (ff, c) in G[i].items():
                    if ff > 0 and dist[j] > dist[i] + c:
                        dist[j] = dist[i] + c
                        P[j] = i
                        update = True

        if dist[V - 1] == INF:
            return INF

        min_f = INF
        u = V - 1
        while u != 0:
            p = P[u]
            min_f = min(min_f, G[p][u][0])
            u = p
        
        u = V - 1
        while u != 0:
            p = P[u]
            G[p][u][0] -= min_f
            G[u][p][0] += min_f
            u = p
        f -= min_f
        res += dist[V - 1]

    return res


class Flow:
    def __init__(self, V):
        self.V = V
        self.G = [dict() for _ in range(V)]

    def add_edge(self, u, v, cap):
        self.G[u][v] = cap
        self.G[v][u] = 0

    def add_multi_edge(self, u, v, cap1, cap2):
        self.G[u][v] = cap1
        self.G[v][u] = cap2

    def wfs(self, s, g):
        S = [(s, -1, INF)]
        P = [-1] * self.V
        while S:
            u, p, f = S.pop()
            if u == g:
                break
            
            for v, c in self.G[u].items():
                if c <= 0:
                    continue
    
                if P[v] != -1:
                    continue
    
                t = min(f, c)
                P[v] = u
                S.append((v, u, t))
        else:
            return 0

        while u != s:
            p = P[u]
            self.G[p][u] -= f
            self.G[u][p] += f
            u = p
    
        return f

    def flow(self, s, g):
        res = 0
        r = self.wfs(s, g)
        while r > 0:
            res += r
            r = self.wfs(s, g)
        return res



class MinCostFlow:
    import heapq

    def __init__(self, N):
        self.N = N
        self.graph = [[] for _ in range(N)]

    def add_edge(self, u, v, capacity, cost):
        # to_node_id, cost, capacity, inverse_link_id
        self.graph[u].append([v, cost, capacity, len(self.graph[v])])
        self.graph[v].append([u, -cost, 0, len(self.graph[u]) - 1])

    def flow(self, s, t, f):
        INF = 1 << 61
        N, graph = self.N, self.graph

        res = 0
        H = [0] * N
        parent = [(-1, -1)] * N

        while f > 0:
            dist = [INF] * N
            dist[s] = 0
            que = [(0, s)]
            while que:
                c, v = heapq.heappop(que)
                if dist[v] < c:
                    continue

                r0 = dist[v] + H[v]
                for link_id, (w, cost, cap, inv_link_id) in enumerate(graph[v]):
                    if cap > 0 and r0 + cost - H[w] < dist[w]:
                        r = r0 + cost - H[w]
                        dist[w] = r
                        parent[w] = (v, link_id, inv_link_id)
                        heapq.heappush(que, (r, w))

            if dist[t] == INF:
                return -1

            for i in range(N):
                H[i] += dist[i]

            d = f
            p = t
            log = []
            while p != s:
                u, link_id, _ = parent[p]
                d = min(d, graph[u][link_id][2])
                p = u
                log.append((u, p, cost))

            f -= d
            res += d * H[t]

            p = t
            while p != s:
                u, link_id, inv_link_id = parent[p]
                graph[u][link_id][2] -= d
                graph[p][inv_link_id][2] += d
                p = u

        return res


# https://atcoder.jp/contests/practice2/submissions/16771746
class Convolution():
    def __init__(self, mod):
        self.mod = mod
        self.root = self.primitive_root(mod)
        self.first1 = True
        self.first2 = True
        self.sum_e = [0] * 30
        self.sum_ie = [0] * 30
 
    def primitive_root(self, m):
        if m == 2: return 1
        if m == 167772161: return 3
        if m == 469762049: return 3
        if m == 754974721: return 11
        if m == 998244353: return 3
        divs = [0] * 20
        cnt = 1
        x = (m - 1) // 2
        while x % 2 == 0: x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0: x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while True:
            for i in range(cnt):
                if pow(g, (m - 1) // divs[i], m) == 1: break
            else:
                return g
            g += 1
 
    def inv_gcd(self, a, b):
        a %= b
        if a == 0: return b, 0
        s = b
        t = a
        m0 = 0
        m1 = 1
        while t:
            u = s // t
            s -= t * u
            m0 -= m1 * u
            s, t = t, s
            m0, m1 = m1, m0
        if m0 < 0: m0 += b // s
        return s, m0
 
    def butterfly(self, arr):
        mod = self.mod
        g = self.root
        n = len(arr)
        h = (n - 1).bit_length()
        if self.first1:
            self.first1 = False
            es = [0] * 30
            ies = [0] * 30
            m = mod - 1
            cnt2 = (m & -m).bit_length() - 1
            e = pow(g, m >> cnt2, mod)
            ie = pow(e, mod - 2, mod)
            for i in range(cnt2 - 1)[::-1]:
                es[i] = e
                ies[i] = ie
                e *= e
                e %= mod
                ie *= ie
                ie %= mod
            now = 1
            for i in range(cnt2 - 2):
                self.sum_e[i] = es[i] * now % mod
                now *= ies[i]
                now %= mod
        for ph in range(1, h + 1):
            w = 1 << (ph - 1)
            p = 1 << (h - ph)
            now = 1
            for s in range(w):
                offset = s << (h - ph + 1)
                for i in range(p):
                    l = arr[i + offset]
                    r = arr[i + offset + p] * now
                    arr[i + offset] = (l + r) % mod
                    arr[i + offset + p] = (l - r) % mod
                now *= self.sum_e[(~s & -~s).bit_length() - 1]
                now %= mod
 
    def butterfly_inv(self, arr):
        mod = self.mod
        g = self.root
        n = len(arr)
        h = (n - 1).bit_length()
        if self.first2:
            self.first2 = False
            es = [0] * 30
            ies = [0] * 30
            m = mod - 1
            cnt2 = (m & -m).bit_length() - 1
            e = pow(g, m >> cnt2, mod)
            ie = pow(e, mod - 2, mod)
            for i in range(cnt2 - 1)[::-1]:
                es[i] = e
                ies[i] = ie
                e *= e
                e %= mod
                ie *= ie
                ie %= mod
            now = 1
            for i in range(cnt2 - 2):
                self.sum_ie[i] = ies[i] * now % mod
                now *= es[i]
                now %= mod
        for ph in range(1, h + 1)[::-1]:
            w = 1 << (ph - 1)
            p = 1 << (h - ph)
            inow = 1
            for s in range(w):
                offset = s << (h - ph + 1)
                for i in range(p):
                    l = arr[i + offset]
                    r = arr[i + offset + p]
                    arr[i + offset] = (l + r) % mod
                    arr[i + offset + p] = (mod + l - r) * inow % mod
                inow *= self.sum_ie[(~s & -~s).bit_length() - 1]
                inow %= mod
 
    def convolution(self, a, b):
        mod = self.mod
        n = len(a)
        m = len(b)
        if not n or not m: return []
        if min(n, m) <= 50:
            if n < m:
                n, m = m, n
                a, b = b, a
            res = [0] * (n + m - 1)
            for i in range(n):
                for j in range(m):
                    res[i + j] += a[i] * b[j]
                    res[i + j] %= mod
            return res
        z = 1 << (n + m - 2).bit_length()
        a += [0] * (z - n)
        b += [0] * (z - m)
        self.butterfly(a)
        self.butterfly(b)
        for i in range(z):
            a[i] *= b[i]
            a[i] %= mod
        self.butterfly_inv(a)
        a = a[:n + m - 1]
        iz = pow(z, mod - 2, mod)
        for i in range(n + m - 1):
            a[i] *= iz
            a[i] %= mod
        return a
 
    def convolution_ll(self, a, b):
        n = len(a)
        m = len(b)
        if not n or not m: return []
        mod1 = 754974721
        mod2 = 167772161
        mod3 = 469762049
        m2m3 = mod2 * mod3
        m1m3 = mod1 * mod3
        m1m2 = mod1 * mod2
        m1m2m3 = mod1 * mod2 * mod3
        i1 = self.inv_gcd(m2m3, mod1)[1]
        i2 = self.inv_gcd(m1m3, mod2)[1]
        i3 = self.inv_gcd(m1m2, mod3)[1]
        c1 = self.convolution(a.copy(), b.copy(), mod1)
        c2 = self.convolution(a.copy(), b.copy(), mod2)
        c3 = self.convolution(a.copy(), b.copy(), mod3)
        c = [0] * (n + m - 1)
        for i in range(n + m - 1):
            x = 0
            x += (c1[i] * i1) % mod1 * m2m3
            x += (c2[i] * i2) % mod2 * m1m3
            x += (c3[i] * i3) % mod3 * m1m2
            x %= m1m2m3
            c[i] = x
        return c


class Flow:
    def __init__(self, V):
        self.V = V
        self.G = [dict() for _ in range(V)]

    def add_edge(self, u, v, cap):
        self.G[u][v] = cap
        self.G[v][u] = 0

    def add_multi_edge(self, u, v, cap1, cap2):
        self.G[u][v] = cap1
        self.G[v][u] = cap2

    def wfs(self, s, g):
        S = [(s, -1, INF)]
        P = [-1] * self.V
        while S:
            u, p, f = S.pop()
            if u == g:
                break
            
            for v, c in self.G[u].items():
                if c <= 0:
                    continue
    
                if P[v] != -1:
                    continue
    
                t = min(f, c)
                P[v] = u
                S.append((v, u, t))
        else:
            return 0

        while u != s:
            p = P[u]
            self.G[p][u] -= f
            self.G[u][p] += f
            u = p
    
        return f

    def flow(self, s, g):
        res = 0
        r = self.wfs(s, g)
        while r > 0:
            res += r
            r = self.wfs(s, g)
        return res


class Fraction(object):
    import math

    def __init__(self, a=0, b=1):
        assert b != 0
        if b < 0:
            a *= -1
            b *= -1
        g = math.gcd(a, b)
        self.a = a // g
        self.b = b // g

    def __add__(self, f):
        b = self.b * f.b // math.gcd(self.b, f.b)
        m1 = b // self.b
        m2 = b // f.b

        a = self.a * m1 + self.b * m2
        return Fraction(a, b)


# Mo's Algorithm
from typing import List, Tuple, Callable
import math
def mo_algorithm(
        N: int,
        Q: int,
        querys: List[Tuple[int, int]],
        func_add: Callable[[int], int],
        func_pop: Callable[[int], int]) -> List[int]:
    M = math.ceil(N / (Q ** 0.5))
    
    QL = [(l, r, i) for i, (l, r) in enumerate(querys)]
    QL.sort(key=lambda x: (x[0] // M, x[1]))

    res = [-1] * Q
    p_l = 0
    p_r = -1
    v = 0
    for l, r, q in QL:
        while l < p_l:
            p_l -= 1
            v += func_add(p_l)

        while p_r < r:
            p_r += 1 
            v += func_add(p_r)
        
        while p_l < l:
            v += func_pop(p_l)
            p_l += 1
        
        while r < p_r:
            v += func_pop(p_r)
            p_r -= 1

        res[q] = v
    return res


# for MHC

def solve():
    return ""


def main():
    T = int(input())

    for t in range(T):
        print("Case #{}: {}".format(t + 1, solve()))
