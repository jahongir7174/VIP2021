This is the source code of 1<sup>st</sup> place solution for Instance Segmentation track in [ICCV 2021 | VIPriors](https://competitions.codalab.org/competitions/33340) challenge

#### Steps

* Place `train`, `val`, `test`, `annotation` folders into `../Dataset/Ins2021` folder
    * `train`, `val` and `test` folders contain provided train, val, test images, respectively
    * `annotation` folder contains provided `train.json`, `val.json` and `test.json` files
* Run `python tools/parse.py` for generating train dataset
* Run `bash ./tools/dist_train.sh ./configs/exp07.py 3` for reproducing the training result
* Run `bash ./tools/dist_test.sh ./configs/exp07.py ./weights/exp07/epoch_73.pth 3` for generating submission file

#### Dataset structure
    ├── Ins2021 
        ├── train
        ├── val
        ├── test
        ├── annotation
            ├── train.json
            ├── val.json
            ├── test.json