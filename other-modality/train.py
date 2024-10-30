from pyexpat import model
import sys
import argparse
import os
import time
from importlib import import_module
from pathlib import Path
import tensorflow as tf
import numpy as np

buffer_size = 5*1024*1024
cycle_length = 16
block_length = 32
compression_type = ''
num_parallel_calls = tf.data.experimental.AUTOTUNE
shuffle_buffer_size = 1024

model_config = {
    "embedding_dim": 8,
    "dense_dim1": 128,
    "dense_dim2": 64,
    "dense_dim3": 1,
    "learning_rate": 1e-3,
    "target_num": 9,
    "weight_increment": 0.05
}


def check_module_path(module_path: str):
    if not Path(module_path).exists():
        raise ValueError(f'Module file path {module_path} does not exist')
    if not os.path.isabs(module_path):
        raise ValueError(
            f'Module file path {module_path} is relative, please use absolute path')
    return True


def is_module_file(path: str):
    return os.path.isfile(path) and path.endswith('.py') and not path.endswith('__init__.py')


def try_import_custom_module(module_config: str):
    module_paths = []
    if ',' in module_config:
        module_paths = module_config.split(',')
    elif isinstance(module_config, str):
        module_paths = [module_config]
    module_paths = [path for path in module_paths if check_module_path(path)]

    modules_to_load = []
    module_dirs = []
    for module_path in module_paths:
        if os.path.isdir(module_path):
            paths = [str(file)
                     for file in Path(module_path).iterdir()
                     if is_module_file(str(file))]
            if paths:
                modules_to_load.extend(paths)
                module_dirs.append(module_path)
        elif is_module_file(module_path):
            modules_to_load.append(module_path)
            module_dirs.append(os.path.dirname(module_path))

    module_dirs = set(module_dirs)
    for p in module_dirs:
        if p not in sys.path:
            sys.path.append(p)

    modules = dict()
    module_names = set()
    for module_path in modules_to_load:
        try:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            if module_name in module_names:
                raise RuntimeError(f'Module {module_name} was imported already, indicating that'
                                   ' there are duplicated file names. Please rename the one of them to'
                                   ' make the module name unique.')
            module = import_module(module_name)
            print('Imported custom module ', module_name)
            modules[module_name] = module
            module_names.add(module_name)
        except ImportError:
            print('Failed to import customized module from ', module_path)
            raise
    return modules


def train(module, num_epochs, save_ckpt_per_step, train_batch_size, test_batch_size, train_dir, test_dir, model_dir, random_seed, restore_ckpt, max_train_steps, max_test_steps, load_ckpt_path, auc_save_path, test_start_step):
    print(module)
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()

    if module is not None:
        modules = try_import_custom_module(module)

    # decoding
    decode_fn = modules['decoder'].preprocess_row
    record_defaults = modules['decoder'].kuairand_defaults
    feature_config = modules['decoder'].feature_config

    train_set = tf.data.Dataset.list_files(
        train_dir, shuffle=True, seed=random_seed)
    train_set = train_set.interleave(lambda f: tf.data.experimental.CsvDataset(f,
                                                                               record_defaults=record_defaults,
                                                                               buffer_size=buffer_size,
                                                                               header=True,
                                                                               compression_type=None
                                                                               # select_cols=[i for i in range(21)]
                                                                               ),
                                     cycle_length=cycle_length,
                                     block_length=block_length,
                                     num_parallel_calls=num_parallel_calls)
    # train_set = train_set.shuffle(shuffle_buffer_size, seed=random_seed)
    train_set = train_set.repeat(num_epochs).batch(train_batch_size)
    train_set = train_set.map(
        map_func=decode_fn, num_parallel_calls=num_parallel_calls)
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)

    test_set = tf.data.Dataset.list_files(
        test_dir, shuffle=True, seed=random_seed)
    test_set = test_set.interleave(lambda f: tf.data.experimental.CsvDataset(f,
                                                                             record_defaults=record_defaults,
                                                                             buffer_size=buffer_size,
                                                                             header=True,
                                                                             compression_type=None
                                                                             # select_cols=[i for i in range(21)]
                                                                             ),
                                   cycle_length=cycle_length,
                                   block_length=block_length,
                                   num_parallel_calls=num_parallel_calls)
    # test_set = test_set.shuffle(shuffle_buffer_size, seed=random_seed)
    test_set = test_set.batch(test_batch_size)
    test_set = test_set.map(
        map_func=decode_fn, num_parallel_calls=num_parallel_calls)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    # session config
    gpus = tf.config.experimental.list_physical_devices('GPU')
    session_config = tf.compat.v1.ConfigProto()

    if gpus:
        print("In train, using GPU", gpus)
        # tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        gpu_list = ""
        delimiter = ""
        for gpu_name in gpus:
            gpu_list = gpu_list + delimiter
            gpu_list = gpu_list + gpu_name.name[-1]
            delimiter = ","
        print("Configuring gpu_list: {gl}".format(gl=gpu_list))
        session_config.gpu_options.visible_device_list = gpu_list

    session_config.log_device_placement = False
    session_config.intra_op_parallelism_threads = 128
    session_config.inter_op_parallelism_threads = 128
    session_config.gpu_options.allow_growth = True

    # build model and metrics
    model = modules['model'].ResMultiTask(feature_config, model_config)

    train_batch = tf.compat.v1.data.make_one_shot_iterator(
        train_set).get_next()
    test_iterator = tf.compat.v1.data.make_initializable_iterator(test_set)
    test_batch = test_iterator.get_next()

    print("Start training ......")
    # saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)

    with tf.compat.v1.Session(config=session_config) as sess:
        if restore_ckpt:
            ckpt_file_name = tf.train.latest_checkpoint(model_dir)
            if ckpt_file_name is None:
                print("Error msg, no checkpoint file found.")
                return
            else:
                # print("Restoring checkpoint from {n}".format(n=ckpt_file_name))
                # saver.restore(sess, ckpt_file_name)
                print("Initializing local variables......")
                print(tf.compat.v1.local_variables())
                sess.run(tf.compat.v1.local_variables_initializer())

        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

        loss_acc = 0
        mse_acc = 0
        avg_score_agg = 0
        stat_steps = 100
        # ckpt_file_name = os.path.join(model_dir, "Model")

        starttime = time.time()
        train_finish = 0
        train_step = 0

        # print("Write loss and auc to file {vf}".format(vf=auc_save_path))
        # out_f = tf.io.gfile.GFile(auc_save_path, mode='w')
        # auc_line = "type,steps,loss,ctr_auc,ctcvr_auc\n"

        while True:
            # bacth_input = sess.run(one_batch)
            for i in range(save_ckpt_per_step):
                try:
                    avg_score, loss, mse, train_step, a1, a2, a3, a4, a5, a6, a7, a8, a9 = model.forward(
                        sess.run(train_batch), "train", sess)  # a11, a12, a13, a14, a15, a16, a17, a18, a19, a20
                    loss_acc = loss_acc + loss
                    mse_acc = mse_acc + mse
                    avg_score_agg += avg_score

                except tf.errors.OutOfRangeError:
                    print("End Training due to run out of data at step {s}......".format(
                        s=train_step))
                    # print("Saving ckpt to {path}......".format(path=ckpt_file_name))
                    # saver.save(sess, ckpt_file_name, global_step=train_step)
                    # auc_line = "train," + str(train_step) + "," + str(loss_acc/stat_steps) + "," + str(auc_ctr_value) + "," + str(auc_cvr_value) + "\n"
                    train_finish = 1
                    break

            if train_finish <= 0:
                # print("Saving ckpt to {path}......".format(path=ckpt_file_name))
                # saver.save(sess, ckpt_file_name, global_step=train_step)
                endtime = time.time()

                speed = save_ckpt_per_step * 1.0 / \
                    (endtime - starttime + 0.001)
                print("trained step:", train_step, f"avg score: {avg_score_agg/save_ckpt_per_step:.2f}", "loss: ", "%.7f" % (loss_acc/save_ckpt_per_step), "mse: ", "%.7f" % (mse_acc/save_ckpt_per_step), "a1: ", "%.7f" % (a1), "a2: ", "%.7f" % (a2), "a3: ", "%.7f" % (a3), "a4: ", "%.7f" % (a4), "a5: ", "%.7f" % (a5), "a6: ", "%.7f" % (
                    a6), "a7: ", "%.7f" % (a7), "a8: ", "%.7f" % (a8), "a9: ", "%.7f" % (a9), "speed: ", "%.7f" % speed)  # "a11: ", "%.7f"%(a11),"a12: ", "%.7f"%(a12), "a13: ", "%.7f"%(a13), "a14: ", "%.7f"%(a14), "a15: ", "%.7f"%(a15), "a16: ", "%.7f"%(a16), "a17: ", "%.7f"%(a17), "a18: ", "%.7f"%(a18),"a19: ", "%.7f"%(a19), "a20: ", "%.7f"%(a20),
                # auc_line = "train," + str(train_step) + "," + str(loss_acc/save_ckpt_per_step) + "," + str(auc_ctr_value) + "," + str(auc_cvr_value) + "\n"
                loss_acc = 0
                mse_acc = 0
                avg_score_agg = 0
                starttime = endtime

            # out_f.write(auc_line)

            if train_finish <= 0 and (train_step + 2) < test_start_step:
                continue

            test_step = 0
            mse_acc = 0
            avg_score_agg = 0
            stat_steps = 1000
            starttime = time.time()
            sess.run(test_iterator.initializer)
            sess.run(tf.compat.v1.local_variables_initializer())
            while True:
                try:
                    test_step = test_step + 1
                    avg_score, mse, a1, a2, a3, a4, a5, a6, a7, a8, a9 = model.forward(sess.run(
                        test_batch), "test", sess)  # , a11, a12, a13, a14, a15, a16, a17, a18, a19, a20
                    mse_acc = mse_acc + mse
                    avg_score_agg += avg_score

                    if test_step >= stat_steps and test_step % stat_steps == 0:
                        endtime = time.time()
                        speed = stat_steps * 1.0 / (endtime - starttime)
                        print("tested step:", test_step, f"avg score: {avg_score_agg/test_step:.2f}", "mse: ", "%.7f" % (mse_acc/test_step), "a1: ", "%.7f" % (a1), "a2: ", "%.7f" % (a2), "a3: ", "%.7f" % (a3), "a4: ", "%.7f" % (a4), "a5: ", "%.7f" % (a5), "a6: ", "%.7f" % (a6), "a7: ", "%.7f" % (a7), "a8: ", "%.7f" % (
                            a8), "a9: ", "%.7f" % (a9), "speed: ", "%.7f" % speed)  # "a11: ", "%.7f"%(a11),"a12: ", "%.7f"%(a12), "a13: ", "%.7f"%(a13), "a14: ", "%.7f"%(a14), "a15: ", "%.7f"%(a15), "a16: ", "%.7f"%(a16), "a17: ", "%.7f"%(a17), "a18: ", "%.7f"%(a18),"a19: ", "%.7f"%(a19), "a20: ", "%.7f"%(a20),
                        # auc_line = "test," + str(test_step) + ",0," + str(ctr_auc_acc/stat_steps) + "," + str(cvr_auc_acc/stat_steps) + "\n"
                        starttime = endtime

                except tf.errors.OutOfRangeError:
                    print("tested step:", test_step, f"avg score: {avg_score_agg/test_step:.2f}", "mse: ", "%.7f" % (mse_acc/test_step), "a1: ", "%.7f" % (a1), "a2: ", "%.7f" % (a2), "a3: ", "%.7f" % (a3), "a4: ", "%.7f" % (a4), "a5: ", "%.7f" % (a5), "a6: ", "%.7f" % (a6), "a7: ", "%.7f" % (a7), "a8: ", "%.7f" % (
                        a8), "a9: ", "%.7f" % (a9), "speed: ", "%.7f" % speed)  # "a11: ", "%.7f"%(a11),"a12: ", "%.7f"%(a12), "a13: ", "%.7f"%(a13), "a14: ", "%.7f"%(a14), "a15: ", "%.7f"%(a15), "a16: ", "%.7f"%(a16), "a17: ", "%.7f"%(a17), "a18: ", "%.7f"%(a18),"a19: ", "%.7f"%(a19), "a20: ", "%.7f"%(a20),
                    print("End Testing due to run out of data......")
                    break

                if test_step >= max_test_steps:
                    break
            # out_f.write(auc_line)
            sess.run(tf.compat.v1.local_variables_initializer())
            if train_finish > 0 or (train_step + 1) >= max_train_steps:
                # out_f.close()
                break


def run(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--module', type=str)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--save_ckpt_per_step", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--restore_ckpt", default=False, action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_test_steps", type=int, default=0)
    parser.add_argument("--load_ckpt_path", type=str)
    parser.add_argument("--auc_save_path", type=str)
    parser.add_argument("--test_start_step", type=int, default=0)

    args, unknown_args = parser.parse_known_args(argv)
    print("parsed args", args)
    print("unknown args", unknown_args)
    train(**vars(args))


if __name__ == '__main__':
    run(sys.argv)
