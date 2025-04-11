# CMPT419-Group18-FinalProject
This project was aimed at trying to create a game with adaptive difficulty, depending on the detected visual emotional response from the player. The game periodically queries a web cam classification loop, determining the most present emotional response detected and using it to adjust the games difficulty in an attempt to improve the enoyability of the game.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

1.
Clone the repository:
```bash
git clone https://github.com/M1tch311/CMPT419-Group18-FinalProject
```

2. 
Install libraries:
```bash
pip install -r requirements.txt
```

## Usage
To run the project, use the following command:
```bash
python main.py
```

To run the project with camera visualization, use the following command:
```bash
python main.py --debug True
```

To run the game without the camera/emotional adaptiveness, use the following command:
```bash
python game.py
```

NOTES:
- To run the game, ensure the game.py, main.py, assets, config.py, and utils.py are in the same directory.
- CV2 takes time to start up the camera and loop. This is why the game is initially paused. Unpause using `ESC`.
- With the camera open, closing the game will automatically close the camera.
- Clicking into the camera window and pressing `q` will close the window and stop the detection, allowing the game to continue.


## Project File Structure
.
├── assets                  # Game assets
├── config.py               # Global variables shared between game and classifier
├── data/images             # Dataset images
│   ├── train               # Train dataset images
│   │   ├── angry
│   │   ├── happy           
│   │   ├── neutral
│   ├── validation          # Validation (test) dataset images
│   │   ├── angry
│   │   ├── happy           
│   │   ├── neutral
├── game.py                 # Defined game
├── main.py                 # Runs the camera classification
├── model.ipynb             # Defining and training the model
├── saved_models            # Saved trained models
├── utils.py                # Shared items
└── README.md


## Self Evaluation
We believe that we have completed a project within the confines of our original proposal. While in the proposal, we originally planned on using video sequence classification using RNNs, we decided against it for performance reasons. Classification of a sequence of images would consume significantly more resources than the frame-based approach we decided on. In addition, while we speculated that we would think about other ways to include the emotional response into the game, we were unable to do so within the scope of this project.