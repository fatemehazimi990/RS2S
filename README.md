# S2S-
Pytorch code for Revisiting Sequence-to-Sequence Video Object Segmentation with Multi-Task Loss and Skip-Memory
Dependencie:

Training

Download the YoutubeVOS from here.
Modify the data paths in train.py and run the file.
```
python train.py
```
For inference with pretrained model, Download the weights from [here](https://drive.google.com/open?id=147mxYW4kYUHccETQrPNv6qDW8bUGl7urxRLB7Di6CIs) and put them under model directory.

```
mkdir model
cd model
wget 
```
Modify the data paths in submit_ytvos.py and run the file.

```
python submit_ytvos.py with model_name='weights.pth'
```
