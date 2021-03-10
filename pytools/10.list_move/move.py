

def move_right(lst, k):
    return lst[-k:] + lst[:-k]


def move_left(lst, k):
    return lst[k:] + lst[:k]


lst_ = [1, 2, 3, 4]
for i in range(len(lst_)):
    print("move_left", move_left(lst_, i))
    # print("move_right", move_right(lst_, i))
