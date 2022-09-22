# MAML_for_Mechanical_Fault_Diagnosis

A pytorch implementation of the mechanical fault diagnosis method using [Model Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).
This code is a modification of [this repository](https://github.com/fmu2/PyTorch-MAML).
The target of this method is one-dimensional mechanical fault signal data. 



## How to use
### Environment
* Python 3
* PyTorch > 1.0

### Datasets
Preprocess the data into the shape of `(category, number, length)`, e.g. `(10, 100, 1024)`, and save them as npy files. 
The naming format is `dataset_split_condition.npy`, e.g. `CWRU_2HP_train.npy`.

Add the register module at the end of `datasets\bearingdataset.py` in the same way as in the file.

### Configurations
Add or modify configuration files in `configs/`.

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
