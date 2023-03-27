# Please edit the file names according to the code first.

import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing for fault dataset")
    parser.add_argument('--data_dir', type=str, default=None, required=True)
    parser.add_argument('--out_dir', type=str, default=None, required=True)
    parser.add_argument('--data_name', type=str, default=None, required=True)
    parser.add_argument('--num_cdt', type=int, default=4, required=False)
    parser.add_argument('--num_cls', type=int, default=10, required=False)
    parser.add_argument('--num_para_train', type=int, default=100, required=False)
    parser.add_argument('--num_para_val', type=int, default=100, required=False)
    parser.add_argument('--para_length', type=int, default=1024, required=False)
    parser.add_argument('--split_ratio', type=float, default=0.8, required=False)
    args = parser.parse_args()

    for n in range(args.num_cdt):
        dataset_train = np.empty([args.num_cls, args.num_para_train, args.para_length])
        dataset_val = np.empty([args.num_cls, args.num_para_val, args.para_length])

        for i in range(args.num_cls):
            data_path = os.path.join(args.data_dir, str(n) + 'HP\\fault_data_' + str(i) + '.txt')
            data = np.loadtxt(data_path)

            for j in range(args.num_para_train):
                num = np.random.randint(low=0, high=(len(data) * args.split_ratio) - args.para_length)
                dataset_train[i][j] = data[num:num + args.para_length]
            for k in range(args.num_para_val):
                num = np.random.randint(low=(len(data) * args.split_ratio) - args.para_length,
                                        high=len(data) - args.para_length)
                dataset_val[i][k] = data[num:num + args.para_length]

        dataset_train = dataset_train.reshape((args.num_cls, args.num_para_train, args.para_length, 1))
        dataset_val = dataset_val.reshape((args.num_cls, args.num_para_val, args.para_length, 1))

        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

        train_save_path = os.path.join(args.out_dir, args.data_name + '_condition' + str(n) + '_train.npy')
        val_save_path = os.path.join(args.out_dir, args.data_name + '_condition' + str(n) + '_val.npy')

        print(train_save_path, ': ', dataset_train.shape)
        print(val_save_path, ': ', dataset_val.shape)

        np.save(train_save_path, dataset_train)
        np.save(val_save_path, dataset_val)
