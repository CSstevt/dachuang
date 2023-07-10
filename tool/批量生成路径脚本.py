import os
def shift(input, out):
    list = []
    for root, _, files in os.walk(input):
        for file in files:
            file_path = os.path.join(root, file)
            list.append(file_path)
    with open(out, 'w') as f:
        f.write('\n'.join(list))

# 指定文件夹路径和输出文件路径
input = 'D:\\Users\\ma\\dachuang\\coco128\\images\\test2017'
out = 'D:\\Users\\ma\\dachuang\\coco128\\test.txt'  # 输出文件路径


shift(input, out)
