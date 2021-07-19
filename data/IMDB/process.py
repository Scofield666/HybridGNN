# 做图序号的reid, 保证节点序号从0开始
if __name__ == '__main__':
    reid = dict()
    reading_files = ['train.txt', 'valid.txt', 'test.txt']
    for reading_file in reading_files:
        with open(reading_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        for datum in data:
            datum = datum.replace('\n', '').split(" ")
            if reading_file == 'train.txt':
                gid, node1, node2 = int(datum[0]), int(datum[1]), int(datum[2])
            else:
                gid, node1, node2, conn = int(datum[0]), int(datum[1]), int(datum[2]), int(datum[3])
            if node1 not in reid.keys():
                reid[node1] = len(reid)
            if node2 not in reid.keys():
                reid[node2] = len(reid)

    for reading_file in reading_files:
        with open(reading_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        rewrite_list = []
        for datum in data:
            datum = datum.replace('\n', '').split(" ")
            if reading_file == 'train.txt':
                gid, node1, node2 = int(datum[0]), int(datum[1]), int(datum[2])
                rewrite_list.append(' '.join([str(gid), str(reid[node1]), str(reid[node2])]))
            else:
                gid, node1, node2, conn = int(datum[0]), int(datum[1]), int(datum[2]), int(datum[3])
                rewrite_list.append(' '.join([str(gid), str(reid[node1]), str(reid[node2]), str(conn)]))

        with open(reading_file, 'w+', encoding='utf-8') as f:
            for data in rewrite_list:
                f.write(data + '\n')

    with open('nodetype.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    rewrite_list = []
    for datum in data:
        datum = datum.replace('\n', '').split(" ")
        ids, types = int(datum[0]), datum[1]
        rewrite_list.append(' '.join([str(reid[ids]), types]))
    with open('nodetype.txt', 'w+', encoding='utf-8') as f:
        for data in rewrite_list:
            f.write(data + '\n')

    print(reid)