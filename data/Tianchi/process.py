import random

random.seed(0)

# 进行节点切分而不是边切分！
if __name__ == '__main__':
    split = {
        "train": 0.85,
        "valid": 0.05,
        "test": 0.1,
    }
    all_user_ids = set()
    all_item_ids = set()
    behav_dict = dict()
    neighbors = dict()
    nodetype = dict()
    subset_ratio = 0.01

    with open('UserBehavior.csv', 'r', encoding='utf-8') as f:
        for datum in f:
            datum = datum.strip().replace('\n', '').split(",")
            uid, iid, bev_type = int(datum[0]), int(datum[1]), datum[3]
            # reid behavior type, make sure starts from 0
            if bev_type not in behav_dict.keys():
                behav_dict[bev_type] = len(behav_dict)
            reid_type = behav_dict[bev_type]

            all_user_ids.add(uid)
            all_item_ids.add(iid)

    print("preserve ids finished")

    all_user_ids = list(all_user_ids)
    all_item_ids = list(all_item_ids)
    user_id_key = {uid: uid for uid in all_user_ids}
    random.shuffle(all_user_ids)

    print("shuffle finished")

    sub_user_ids = all_user_ids[:int(len(all_user_ids) * subset_ratio)]
    sub_user_id_key = {uid: uid for uid in sub_user_ids}
    neighbors = dict()
    for sub_id in sub_user_ids:
        for _, reid_type in behav_dict.items():
            neighbors[(reid_type, sub_id)] = []

    with open('UserBehavior.csv', 'r', encoding='utf-8') as f:
        for datum in f:
            datum = datum.strip().replace('\n', '').split(",")
            uid, iid, bev_type = int(datum[0]), int(datum[1]), datum[3]
            # 选取uid中的子集
            if uid not in sub_user_id_key.keys():
                continue
            # 截取id
            nodetype[uid] = 'U'
            nodetype[iid] = 'I'
            reid_type = behav_dict[bev_type]
            if (reid_type, uid) not in neighbors.keys():
                neighbors[(reid_type, uid)] = []
            neighbors[(reid_type, uid)].append(iid)

    print("subgraph neighbors finished")

    # reshuffle
    for cxt, neibs in neighbors.items():
        random.shuffle(neibs)
        neighbors[cxt] = neibs

    print("reshuffle finished")
    train_neighbors = dict()
    valid_neighbors = dict()
    test_neighbors = dict()

    for cxt, neibs in neighbors.items():
        til_valid = int(round(len(neibs) * split["train"]))
        til_test = int(round(len(neibs) * (split["train"] + split["valid"])))
        train_neighbors[cxt] = neibs[:til_valid]
        valid_neighbors[cxt] = neibs[til_valid: til_test]
        test_neighbors[cxt] = neibs[til_test:]


    node_type_list = []
    for k, v in nodetype.items():
        node_type_list.append(str(k) + " " + str(v) + "\n")

    train_list = []
    valid_list = []
    test_list = []

    for cxt, neibs in train_neighbors.items():
        for iid in neibs:
            train_list.append(str(cxt[0]) + " " + str(cxt[1]) + " " + str(iid) + "\n")
    print("train list finished")

    for cxt, neibs in valid_neighbors.items():
        for iid in neibs:
            valid_list.append(str(cxt[0]) + " " + str(cxt[1]) + " " + str(iid) + " " + str(1) + "\n")
            neg_id = random.choice(all_item_ids)
            while neg_id in neibs:
                neg_id = random.choice(all_item_ids)
            valid_list.append(str(cxt[0]) + " " + str(cxt[1]) + " " + str(neg_id) + " " + str(0) + "\n")
    print("valid list finished")

    for cxt, neibs in test_neighbors.items():
        for iid in neibs:
            test_list.append(str(cxt[0]) + " " + str(cxt[1]) + " " + str(iid) + " " + str(1) + "\n")
            neg_id = random.choice(all_item_ids)
            while neg_id in neibs:
                neg_id = random.choice(all_item_ids)
            test_list.append(str(cxt[0]) + " " + str(cxt[1]) + " " + str(neg_id) + " " + str(0) + "\n")
    print("test list finished")

    with open('train.txt', 'w+', encoding='utf-8') as f:
        f.writelines(''.join(train_list))
    with open('valid.txt', 'w+', encoding='utf-8') as f:
        f.writelines(''.join(valid_list))
    with open('test.txt', 'w+', encoding='utf-8') as f:
        f.writelines(''.join(test_list))
    with open('nodetype.txt', 'w+', encoding='utf-8') as f:
        f.writelines(''.join(node_type_list))
