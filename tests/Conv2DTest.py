import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
res = []
with open("tests/test_file.txt") as f:
    lines = f.readlines()
    h, w, c = [int(x) for x in lines[0].split()]
    fh, fw, fc = [int(x) for x in lines[1].split()]
    dy, dx, sy, sx = [int(x) for x in lines[2].split()]
    pt, pl, pb, pr = [int(x) for x in lines[3].split()]
    tensor = [int(x) for x in lines[4].split()]
    filter = [int(x) for x in lines[5].split()]
    tensor = tf.reshape(tensor, [1, h, w, c])
    filter = tf.reshape(filter, [fh, fw, c, fc])
    res = tf.nn.conv2d(tensor, filter, [1, sy, sx, 1], [[0, 0], [pt, pb], [pl, pr], [0, 0]])
    res = [x % 256 for x in res.numpy()]

with open("tests/test_file.txt", "w") as f:
    for batch in res:
        for tensor in batch:
            for line in tensor:
                for x in line:
                    f.write(str(x) + ' ')
