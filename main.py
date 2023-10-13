import copy
import heapq
import itertools
import math
import operator
import sys
from bisect import bisect, bisect_left, bisect_right, insort
from collections import Counter, deque
from fractions import Fraction
from functools import cmp_to_key, lru_cache, partial
from inspect import currentframe
from math import ceil, gcd, log10, pi, sqrt

# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
input = sys.stdin.readline
sys.setrecursionlimit(10000000)
# mod = 10 ** 9 + 7
mod = 998244353
# mod = 1 << 128
# mod = 10 ** 30 + 1
INF = 1 << 61
DIFF = 10 ** -9
DX = [1, 0, -1, 0, 1, 1, -1, -1]
DY = [0, 1, 0, -1, 1, -1, 1, -1]

def read_values(): return tuple(map(int, input().split()))
def read_index(): return tuple(map(lambda x: int(x) - 1, input().split()))
def read_list(): return list(read_values())
def read_lists(N): return [read_list() for _ in range(N)]
def dprint(*values): print(*values, file=sys.stderr)
def dprint2(*values):
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    dprint(", ".join(f"{names.get(id(value), '???')}={repr(value)}" for value in values))


def main():
    N = int(input())
    S = input().strip()
    N, M = read_values()
    A = read_list()
    L = read_lists(M)


if __name__ == "__main__":
    main()
