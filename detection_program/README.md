Lane Detection
==

## Table of Contents

- [Introduction](#introduction)
- [Technologies](#technologies)
- [Setup](#setup)
- [Results](#results)
- [License](#license)

## Introduction

Final project codes for Autonomous Driving car course in 2020 spring semester. This is a basic lane detection using TuSimple dataset

## Technologies

### Prerequisite:

```
conda environment
1080Ti or 2080Ti GPU device
pytorch
torchvision
visdom
cudatoolkit=10.1
opencv
sklearn
ujson
matplotlib
```

### Method Description

2 Hourglass Network + aggregation of SGPN loss

## Results

We train for 500 epochs the best evaluation result we can get right now is in the 100th epoch
| Loss | Accuracy | FP   | FN  |
| ---- | -------- | ---- | --- |
| 1.3  | 0.96     | 0.03 | 0.3 |

<img src='./image/lane.png'>

## Contact

Created by [@joycenerd](mailto:joycenerd@csie.io)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
