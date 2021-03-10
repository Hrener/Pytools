# def partition(arr, low, high):
#     i = (low - 1)  # 最小元素索引
#     pivot = arr[high]
#
#     for j in range(low, high):
#
#         # 当前元素小于或等于 pivot
#         if arr[j] <= pivot:
#             i = i + 1
#             arr[i], arr[j] = arr[j], arr[i]
#
#     arr[i + 1], arr[high] = arr[high], arr[i + 1]
#     return i + 1
#
#
# # 快速排序函数
# def quickSort(arr, low, high):
#     # arr[] --> 排序数组
#     # low  --> 起始索引
#     # high  --> 结束索引
#     if low < high:
#         pi = partition(arr, low, high)
#
#         quickSort(arr, low, pi - 1)
#         quickSort(arr, pi + 1, high)


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90, 78]
    print("排序前:", arr)
    arr = quicksort(arr)
    print("排序后:", arr)
