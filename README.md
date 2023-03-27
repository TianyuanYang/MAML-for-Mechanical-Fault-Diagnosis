# MAML_for_Mechanical_Fault_Diagnosis

A pytorch implementation of the mechanical fault diagnosis method using [Model Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).
This code is a modification of [this repository](https://github.com/fmu2/PyTorch-MAML).
The target of this method is one-dimensional mechanical fault signal data. 



## How to use
### Environment
* Python 3
* PyTorch > 1.0
* Other packages: numpy, pyyaml, tqdm, tensorboardX

### Datasets
Preprocess the data into the shape of `(category, number, length, 1)`, e.g. `(10, 100, 1024, 1)`, and save them as npy files. 
The naming format is `dataset_condition_split.npy`, e.g. `CWRU_2HP_train.npy`.

Add the register module at the end of `datasets\bearingdataset.py` in the same way as in the file.

An example of pre-processing for [CWRU data](https://drive.google.com/file/d/1T5G6yEe8Fnv07jdgpHrWHHfu0xZwz_5k/view?usp=share_link) is given here.
```
python preprocessing.py --data_dir=$path_to_data_folder$ --out_dir=$path_to_save_folder$ --data_name=CWRU
```

### Configurations
Add or modify configuration files in `configs/`.


### Modifying the model
The backbone network consists of the encoder and the classifier, taking care that the shape of the output of the encoder matches the shape of the input required of the classifier.

If you add new model files, remember to update `models/classifiers/__init__.py` and `models/encoders/__init__.py`.

### Training
```
python train.py --config=configs/cwru_ticnn_train.yaml
```

### Testing
```
python test.py --config=configs/cwru_ticnn_test.yaml
```

## Notes

You can carry out different experiments by：
* using different datasets
* changing different models
* freezing the head or body of the backbone network
* etc

If the input data is two-dimensional, you can refer to the reference to modify the code.

## Reference
```
@misc{pytorch_maml,
  title={maml in pytorch - re-implementation and beyond},
  author={Mu, Fangzhou},
  howpublished={\url{https://github.com/fmu2/PyTorch-MAML}},
  year={2020}
}
```
