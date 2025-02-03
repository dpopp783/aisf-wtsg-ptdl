NOTES ON RESULTS

CIFAR10 (3 epochs)
Alexnet finetuned on 25000 examples CIFAR: 88.61% accuracy
Resnet50 finetuned on ground truth: 90.52
Resnet50 WTSG: 86.45%
Resnet50 PGR: (.8645 - .8861) / (.9052 - .8861) = -1.1309
DINO GT:   96.31%
DINO WTSG: 94.74%
DINO PGR:  (.9474 - .8861) / (.9631 - .8861) = 0.7961

CIFAR100
Alexnet: 58.40%
Resnet50 (GT): 71.24%
Resnet50 (WTSG): 59.53%
Resnet50 PGR: 0.0880
DINO (GT): 85.40%
DINO (WTSG): 72.18%
DINO PGR: 0.5103703703703706

IMAGENET-1K
AlexNet:         52.14%
Resnet50 (WTSG): 74.09%
Resnet50 (GT):   74.87%
Resnet50 PGR:    (.7409 - .5214) / (.7487 - .5214) = 0.9656841179058512
DINO (WTSG):     72.67%
DINO (GT):       78.70%
DINO PGR:        (.7267 - .5214) / (.7870 - .5214) = 0.7729668674698795
