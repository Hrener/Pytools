import torch

# ---------------------1.张量裁剪
a = torch.rand((3, 4))
print(a)
print(a.narrow(0, 0, 2))    # torch.narrow(维度, 起始索引, 终止索引)
print(a.narrow(1, 0, 2))

# ---------------------2.张量条件赋值
a = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).to(torch.float32)
b = torch.zeros(1, 2, 3)
print(torch.where(a % 2 == 0, b, a))    # 如果张量中的值大于或小于某一数值，它们可以很容易地被替换

# ---------------------3.fliter
# fiter(function. Iterable)
# function: 用来筛选的函数. 在ﬁlter中会自动的把iterable中的元素传递给function. 然后根据function返回的True或者False来判断是否保留留此项数据 , Iterable: 可迭代对象


def func(i):    # 判断奇数
    return i % 2 == 1


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
l1 = filter(func, lst)  # l1是迭代器
print(l1)   # <filter object at 0x000001CE3CA98AC8>
print(list(l1))
