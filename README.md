# ML101 Workshop Code
Code written during an ML101 workshop, introducing attendees to both ML concepts and PyTorch syntax. 

## Data
The notebooks assume you have some image data available.  We used data from [a Kaggle polyhedral dice dataset](https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images), which we downloaded and extracted into a `data/` folder in the root of this repository.  Some data cleaning, or at least image resizing, may be necessary.

## Setup
With Docker installed on your machine, it's a breeze!  After cloning the repository, type `docker-compose up` in the command line and the container will build, then start as a Jupyter notebook server and provide instructions for connecting to your console window.

