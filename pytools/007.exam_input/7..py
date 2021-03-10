# order = ['1', '5 8', 'L', 'R', 'G 10', 'R', 'R', 'G 10', 'P', 'R', 'R', 'G 10', 'P']
"""
input:
3
1 2
E
E
4 2
L
L
1 1
R
output:
3      测试案例个数
[1, 4, 1] [2, 2, 1]
[[['E'], ['E']], [['L'], ['L']], [['R']]]
"""


num = int(input())
n_, line_, order_ = [], [], []
for _ in range(num):
    n, line = input().split()
    n_.append(int(n))
    line_.append(int(line))
    order = []
    for i in range(int(line)):
        order.append(input().split())
    order_.append(order)
print(num)
print(n_, line_)
print(order_)

lst = [1, 2, 3, 4]


def list_move(lst, k):
    return lst[k:] + lst[:k]

init_xy = [0, 0]
for t in range(1):
    print("Case #{}:".format(t+1))
    n = n_[t]
    for order_ in order[t]:
        if len(order_) == 1:
            if order_[0] == "L":
                lst = list_move(lst, 3)
                # print(lst)
            elif order_[0] == "R":
                lst = list_move(lst, 1)
                # print(lst)
            elif order_[0] == "P":
                print(init_xy)
        else:
            bias = int(order_[1])
            if lst[0] == 1:
                for _ in range(bias):
                    if init_xy[1] > 1:
                        init_xy[1] -= 1
            elif lst[0] == 2:
                for _ in range(bias):
                    if init_xy[0] < n-1:
                        init_xy[0] += 1
            elif lst[0] == 3:
                for _ in range(bias):
                    if init_xy[1] < n-1:
                        init_xy[1] += 1
            elif lst[0] == 4:
                for _ in range(bias):
                    if init_xy[0] > 1:
                        init_xy[0] -= bias




