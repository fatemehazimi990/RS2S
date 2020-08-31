# S2S-
_Pytorch code for Revisiting Sequence-to-Sequence Video Object Segmentation with Multi-Task Loss and Skip-Memory_ <br /> 
Dependencie:
```
numpy==1.18.1,
sacred==0.8.1,
torch==1.5,
torchvision==0.6,
tqdm
```
For training, download the YoutubeVOS from [here](https://competitions.codalab.org/competitions/19544#participate-get_data).
Modify the configurations and the paths in train.py and run the following command.
```
python train.py
```
For inference with pretrained model, download the weights from [here](https://drive.google.com/open?id=16EeELoziIlucqExwtxn4eGVXD4mJpn-p) and put them under Model directory.

Modify the data paths in submit_ytvos.py and run the following command.

```
python submit_ytvos.py with model_name='weights.pth'
```
