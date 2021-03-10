import json

with open('train.json')as json_f:
    data = json.load(json_f)
print(data.keys())
images = data["images"]
annotations = data["annotations"]
categories = data["categories"]
print(len(annotations))
print(categories)
num2category = {8:'boat', 84:'hat', 64:'mouse', 62:'tv', 36:'skateboard', 63:'laptop', 40:'wineglass', 1:'bicycle', 72:'refrigerator', 61:'toilet', 5:'bus', 81:'watch', 16:'dog', 25:'umbrella'}
category_id_num = {'boat': 0, 'hat': 0, 'mouse': 0, 'tv': 0, 'skateboard': 0, 'laptop': 0, 'wineglass': 0, 'bicycle': 0, 'refrigerator': 0, 'toilet': 0, 'bus': 0, 'watch': 0, 'dog': 0, 'umbrella': 0}
for anno in annotations:
    print(anno)
    category_id_num[num2category[anno["category_id"]]] += 1
    # break
print(category_id_num)# {'boat': 743, 'hat': 4578, 'mouse': 198, 'tv': 1312, 'skateboard': 48, 'laptop': 436, 'wineglass': 1650, 'bicycle': 248, 'refrigerator': 261, 'toilet': 9, 'bus': 143, 'watch': 1798, 'dog': 355, 'umbrella': 125}