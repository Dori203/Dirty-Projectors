# Isometric 3d creator
Create 3d artifacts using 3 or more 2d side projections, for artistic or modeling purposes.
Using search algorithms and CSP, the program creates 3d shapes that correspond to given 2d images from different perspectives.
The resulting 3d object's projecting will match the given images.

![alt text](https://dori203.github.io/images/AI_3d/ai_ying.gif)

This project includes:
* Image processing module.
* Search algorithms configurations.
* Genetic algorithms configurations.

## Overview
**Dirty_Projectors.py**:
Main class for running the program.

**image_processing.py**:
Includes all image processing methods required for preprocessing of images, and calculating various loss values.

**successor_functions.py**:
Various successor functions for the localsearch algorithms.

**genetic3rd.py**
Implementation and configurations for solving the problem using genetic algorithms.


## Setup
Run "chmod +x setup.sh"
Run "source ./setup.sh"

## Configuration
to run the program run:

```
python3 Dirty_projectors.py --flags
```

where the flags are:
```
flags = {
  -p path to folder of images (required)
  -i number of iterations (default = 5000)
  -d height\width of pictures (images should be square, default is 40 pixels)
  -s type of solver to use = ['mrv', 'mc', 'genetic', 'greedy', 'hill', 'local', 'annealing']
  -is initial state = ['blank', 'black', 'random']
  -f fitness function = ['l1', 'l2', 'punish_empty', 'punish_black']
  -sf successor function = ['naive', 'neighbouring', 'column']
}
```
