import re
import numpy as np

fin = open("../../../AITM-main/data/sample_skeleton_train.csv", 'r')

for i, line in enumerate(fin):

    line_list = line.strip().split(',')
    if line_list[0] == '0078413e72587a5e':
        print(line_list)
    else:
        continue

    kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
    key = kv[range(0, len(kv), 3)]
    id = kv[range(1, len(kv), 3)]
    value = kv[range(2, len(kv), 3)]

    print("now", i, line_list[0])
    print(key)
    print(id)
    print(value)

    if line_list[0] == '0078413e72587a5e':
        break
