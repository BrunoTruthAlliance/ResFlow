import re
import numpy as np

# use_columns = [
#     '101',
#     '121',
#     '122',
#     '124',
#     '125',
#     '126',
#     '127',
#     '128',
#     '129',
#     '205',
#     '206',
#     '207',
#     '216',
#     '508',
#     '509',
#     '702',
#     '853',
#     '301']

print("training common fea...")
fin = open("../../../AITM-main/data/common_features_train.csv", 'r')
fout = open("../../../AITM-main/data/common_features_train_out.csv", "w")
header_list = ['common_feature_index', '101', '121',
               '122', '124', '125', '126', '127', '128', '129']
fout.write(','.join(header_list) + '\n')
for i, line in enumerate(fin):
    if i % 100000 == 0:
        print(i)
    line_list = line.strip().split(',')
    kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
    key = kv[range(0, len(kv), 3)]
    id = kv[range(1, len(kv), 3)]

    row = []
    row.append(line_list[0])

    for j, k in enumerate(key):
        if k == '101':
            row.append(id[j])
    if len(row) < 2:
        row.append('0')

    for j, k in enumerate(key):
        if k == '121':
            row.append(id[j])
    if len(row) < 3:
        row.append('0')

    for j, k in enumerate(key):
        if k == '122':
            row.append(id[j])
    if len(row) < 4:
        row.append('0')

    for j, k in enumerate(key):
        if k == '124':
            row.append(id[j])
    if len(row) < 5:
        row.append('0')

    for j, k in enumerate(key):
        if k == '125':
            row.append(id[j])
    if len(row) < 6:
        row.append('0')

    for j, k in enumerate(key):
        if k == '126':
            row.append(id[j])
    if len(row) < 7:
        row.append('0')

    for j, k in enumerate(key):
        if k == '127':
            row.append(id[j])
    if len(row) < 8:
        row.append('0')

    for j, k in enumerate(key):
        if k == '128':
            row.append(id[j])
    if len(row) < 9:
        row.append('0')

    for j, k in enumerate(key):
        if k == '129':
            row.append(id[j])
    if len(row) < 10:
        row.append('0')

    fout.write(','.join(row) + '\n')

print("testing common fea...")
fin = open("../../../AITM-main/data/common_features_test.csv", 'r')
fout = open("../../../AITM-main/data/common_features_test_out.csv", "w")
header_list = ['common_feature_index', '101', '121',
               '122', '124', '125', '126', '127', '128', '129']
fout.write(','.join(header_list) + '\n')
for i, line in enumerate(fin):
    if i % 100000 == 0:
        print(i)
    line_list = line.strip().split(',')
    kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
    key = kv[range(0, len(kv), 3)]
    id = kv[range(1, len(kv), 3)]

    row = []
    row.append(line_list[0])

    for j, k in enumerate(key):
        if k == '101':
            row.append(id[j])
    if len(row) < 2:
        row.append('0')

    for j, k in enumerate(key):
        if k == '121':
            row.append(id[j])
    if len(row) < 3:
        row.append('0')

    for j, k in enumerate(key):
        if k == '122':
            row.append(id[j])
    if len(row) < 4:
        row.append('0')

    for j, k in enumerate(key):
        if k == '124':
            row.append(id[j])
    if len(row) < 5:
        row.append('0')

    for j, k in enumerate(key):
        if k == '125':
            row.append(id[j])
    if len(row) < 6:
        row.append('0')

    for j, k in enumerate(key):
        if k == '126':
            row.append(id[j])
    if len(row) < 7:
        row.append('0')

    for j, k in enumerate(key):
        if k == '127':
            row.append(id[j])
    if len(row) < 8:
        row.append('0')

    for j, k in enumerate(key):
        if k == '128':
            row.append(id[j])
    if len(row) < 9:
        row.append('0')

    for j, k in enumerate(key):
        if k == '129':
            row.append(id[j])
    if len(row) < 10:
        row.append('0')

    fout.write(','.join(row) + '\n')

print("sample_skeleton_train ...")
fin = open("../../../AITM-main/data/sample_skeleton_train.csv", 'r')
fout = open("../../../AITM-main/data/sample_skeleton_train_out.csv", "w")
header_list = ['sample_id', 'ctr_label', 'cvr_label', 'common_feature_index', '205', '206', '207',
               '216', '301', '508_id', '508_val', '509_id', '509_val', '702_id', '702_val', '853_id', '853_val']
fout.write(','.join(header_list) + '\n')
for i, line in enumerate(fin):
    if i % 100000 == 0:
        print(i)
    line_list = line.strip().split(',')
    if line_list[1] == '0' and line_list[2] == '1':
        continue
    row = []
    row.append(line_list[0])
    row.append(line_list[1])
    row.append(line_list[2])
    row.append(line_list[3])

    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
    key = kv[range(0, len(kv), 3)]
    id = kv[range(1, len(kv), 3)]
    value = kv[range(2, len(kv), 3)]

    for j, k in enumerate(key):
        if k == '205':
            row.append(id[j])
    if len(row) < 5:
        row.append('0')

    for j, k in enumerate(key):
        if k == '206':
            row.append(id[j])
    if len(row) < 6:
        row.append('0')

    for j, k in enumerate(key):
        if k == '207':
            row.append(id[j])
    if len(row) < 7:
        row.append('0')

    for j, k in enumerate(key):
        if k == '216':
            row.append(id[j])
    if len(row) < 8:
        row.append('0')

    for j, k in enumerate(key):
        if k == '301':
            row.append(id[j])
    if len(row) < 9:
        row.append('0')

    for j, k in enumerate(key):
        if k == '508':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 11:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '509':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 13:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '702':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 15:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '853':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 17:
        row.append('0')
        row.append('0')

    fout.write(','.join(row) + '\n')


print("sample_skeleton_test ...")
fin = open("../../../AITM-main/data/sample_skeleton_test.csv", 'r')
fout = open("../../../AITM-main/data/sample_skeleton_test_out.csv", "w")
header_list = ['sample_id', 'ctr_label', 'cvr_label', 'common_feature_index', '205', '206', '207',
               '216', '301', '508_id', '508_val', '509_id', '509_val', '702_id', '702_val', '853_id', '853_val']
fout.write(','.join(header_list) + '\n')
for i, line in enumerate(fin):
    if i % 100000 == 0:
        print(i)
    line_list = line.strip().split(',')
    if line_list[1] == '0' and line_list[2] == '1':
        continue
    row = []
    row.append(line_list[0])
    row.append(line_list[1])
    row.append(line_list[2])
    row.append(line_list[3])

    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
    key = kv[range(0, len(kv), 3)]
    id = kv[range(1, len(kv), 3)]
    value = kv[range(2, len(kv), 3)]

    for j, k in enumerate(key):
        if k == '205':
            row.append(id[j])
    if len(row) < 5:
        row.append('0')

    for j, k in enumerate(key):
        if k == '206':
            row.append(id[j])
    if len(row) < 6:
        row.append('0')

    for j, k in enumerate(key):
        if k == '207':
            row.append(id[j])
    if len(row) < 7:
        row.append('0')

    for j, k in enumerate(key):
        if k == '216':
            row.append(id[j])
    if len(row) < 8:
        row.append('0')

    for j, k in enumerate(key):
        if k == '301':
            row.append(id[j])
    if len(row) < 9:
        row.append('0')

    for j, k in enumerate(key):
        if k == '508':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 11:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '509':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 13:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '702':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 15:
        row.append('0')
        row.append('0')

    for j, k in enumerate(key):
        if k == '853':
            row.append(id[j])
            row.append(value[j])
    if len(row) < 17:
        row.append('0')
        row.append('0')

    fout.write(','.join(row) + '\n')
