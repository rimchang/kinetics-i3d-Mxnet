# I3D models trained on Kinetics Mxnet

this repo implements the network of I3D with Mxnet, pre-trained model weights are converted from tensorflow. 

### Sample code

you can convert tensorflow model to mxnet

```
# ./convert

./convert.sh

```

you can evaluate sample 

```
./multi-evaluate.py

```


There is a slight difference from the original model. you can compare original model output with pytorch model output in out directory

### Original Model (imagenet_joint.txt)

```
Norm of logits: 138.468658

Top classes and probabilities
1.0 41.8137 playing cricket
1.49716e-09 21.494 hurling (sport)
3.84312e-10 20.1341 catching or throwing baseball
1.54923e-10 19.2256 catching or throwing softball
1.13602e-10 18.9154 hitting baseball
8.80112e-11 18.6601 playing tennis
2.44157e-11 17.3779 playing kickball
1.15319e-11 16.6278 playing squash or racquetball
6.13194e-12 15.9962 shooting goal (soccer)
4.39177e-12 15.6624 hammer throw
2.21341e-12 14.9772 golf putting
1.63072e-12 14.6717 throwing discus
1.54564e-12 14.6181 javelin throw
7.66915e-13 13.9173 pumping fist
5.19298e-13 13.5274 shot put
4.26817e-13 13.3313 celebrating
2.72057e-13 12.8809 applauding
1.8357e-13 12.4875 throwing ball
1.61348e-13 12.3585 dodgeball
1.13884e-13 12.0101 tap dancing
```

### Mxnet Converted Model (I3D_MX_TF_full/imagenet_joint.txt)
```
RGB checkpoint restored
RGB data loaded, shape= (1, 3, 79, 224, 224)
flow checkpoint restored
FLOW data loaded, shape= (1, 2, 79, 224, 224)
Norm of logits: 142.029556

Top classes and probabilities
1.0 44.2655 playing cricket
1.66801e-09 24.0539 hurling (sport)
4.99325e-10 22.8478 catching or throwing baseball
1.86202e-10 21.8613 catching or throwing softball
1.28303e-10 21.4889 hitting baseball
1.07527e-11 19.0097 playing tennis
5.45804e-12 18.3316 playing kickball
2.96073e-12 17.7199 playing squash or racquetball
7.68403e-13 16.3711 shooting goal (soccer)
1.37728e-13 14.652 hammer throw
1.04936e-13 14.3801 throwing discus
7.07885e-14 13.9864 dodgeball
6.75171e-14 13.9391 golf putting
5.38292e-14 13.7126 shot put
4.89977e-14 13.6185 javelin throw
3.99016e-14 13.4132 pumping fist
3.34444e-14 13.2366 celebrating
3.21526e-14 13.1972 passing American football (not in game)
2.45824e-14 12.9288 sword fighting
2.37701e-14 12.8952 throwing ball

```


Reference:

[kinetics-i3d](https://github.com/deepmind/kinetics-i3d)  
[kinetics-i3d-Pytorch](https://github.com/rimchang/kinetics-i3d-Pytorch)
[tensorflow-model-zoo.torch](https://github.com/Cadene/tensorflow-model-zoo.torch)
