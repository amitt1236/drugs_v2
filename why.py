import numpy as np

num = 1000000
same= 0
for _ in range(num):
    lst = np.arange(6000)
    np.random.shuffle(lst)
    lst2 = lst[:500]
    same = same + np.count_nonzero(lst2[lst2<500])

print(same/num)