import numpy as np

def fill_auxilary_table(source, target, lambd, k):
    n, m = len(source), len(target)
    table = np.zeros((n, m, k+1))
    table[:, :, 0] = 1
    for l in range(1, k+1):
        for p in range(n+m):
            for i in range(l-1, n):
                j = p - i
                if i < 0 or j < 0 or i >= n or j >= m:
                    continue
                if i < l-1 or j < l-1:
                    table[i, j, l] = 0
                    continue
                value = lambd * (table[i, j-1, l] + table[i-1, j, l]) - lambd**2 * table[i-1, j-1, l]
                if source[i] == target[j]:
                    value += lambd**2 * table[i-1, j-1, l-1]
                table[i, j, l] = value
    return table

def fill_kernel_table(source, target, lambd, k):
    n, m = len(source), len(target)
    auxilary_table = fill_auxilary_table(source, target, lambd, k-1)
    table = np.zeros((n, m))
    for p in range(n+m):
        for i in range(n):
            j = p - i
            if i < 0 or j < 0 or i >= n or j >= m:
                continue
            if i < k-1 or j < k-1:
                continue
            value = table[i-1, j] 
            a = source[i]
            for q in range(j+1):
                if target[q] == a:
                    value += lambd**2 * auxilary_table[i-1, q-1, k-1]
            table[i, j] = value
    return table

def substring_kernel(source, target, lambd, k):
    table = fill_kernel_table(source, target, lambd, k)
    return table [-1, -1]


# test the kernel

# source = 'cat'
# target = 'cat'
# lambd = 2
# k = 2
# K, B = fill_kernel_table(source, target, lambd, k)
