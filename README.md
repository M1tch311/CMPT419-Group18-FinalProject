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

To run the game without the camera/emotional adaptiveness, use the following command:
```bash
python game.py
```

NOTES:
- To run the game, ensure the game.py, main.py, assets, config.py, and utils.py are in the same directory.
- CV2 takes time to start up the camera and loop. This is why the game is initially paused. Unpause using `ESC`.


## Run


Directory Structure
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