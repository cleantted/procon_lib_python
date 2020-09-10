import bisect
import copy
import heapq
import sys
import itertools
import math
import queue
input = sys.stdin.readline
sys.setrecursionlimit(1000000)
mod = 10 ** 9 + 7

def read_values(): return map(int, input().split())
def read_index(): return map(lambda x: int(x) - 1, input().split())
def read_list(): return list(read_values())
def read_lists(N): return [read_list() for n in range(N)]
def init_dp1(init, N): return [init for _ in range(N)]
def init_dp2(init, N, M): return [[init for _ in range(M)] for _ in range(N)]


class V:
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



def functional(N):
    F = [1] * (N + 1)
    for i in range(N):
        F[i + 1] = (i + 1) * F[i] % mod
    return F


def inv(a):
    return pow(a, mod - 2, mod)


def C(F, a, b):
    return F[a] * inv(F[a - b]) * inv(F[b]) % mod 


def get_bit(n, i):
    return n >> i & 1


def read_graph(H, W, wall="#"):
    return [wall * (W + 2)] + [wall + input() + wall for h in range(H)] + [wall * (W + 2)]
 

def LCS(a, b):
    dp = [[0 for _ in range(len(a) + 1)] for __ in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[j][i] = dp[j - 1][i - 1] + 1
            else:
                dp[j][i] = max(dp[j - 1][i], dp[j][i - 1])
    return dp[len(b)][len(a)]


class UF:
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


class LCA:
    def __init__(self, N, L, root=0):
        self.L = L
        self.parent = [None] * N
        self.depth = [None] * N
        self._dfs(root, 0, None)
        self.k_parent = [self.parent]
        self.max_depth = (N - 1).bit_length()
        S = self.parent
        for _ in range(self.max_depth):
            T = [0] * N
            for i in range(N):
                if S[i] is None:
                    continue
                T[i] = S[S[i]]
            self.k_parent.append(T)
            S = T
 
    def _dfs(self, v, d, p):
        self.parent[v] = p
        self.depth[v] = d
        for w in self.L[v]:
            if w == p:
                continue
            self._dfs(w, d + 1, v)
 
    def q(self, u, v):
        d = self.depth[v] - self.depth[u]
        if d < 0:
            u, v = v, u
            d *= -1
 
        for k in range(self.max_depth + 1):
            if d & 1:
                v = self.k_parent[k][v]
            d >>= 1
 
        if u == v:
            return u
 
        for k in range(self.max_depth - 1, -1, -1):
            pu = self.k_parent[k][u]
            pv = self.k_parent[k][v]
            if pu != pv:
                u = pu
                v = pv
 
        return self.k_parent[0][u]


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
 

# for CodeJam

def solve():
    return ""


def main():
    T = int(input())

    for t in range(T):
        print("Case #{}: {}".format(t + 1, solve()))
