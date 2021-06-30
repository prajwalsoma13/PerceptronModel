import json

with open("perceptronModel/config.json") as f:
    Configuration = json.load(f)


learningRate = Configuration["learningRate"]
epochs = Configuration["epochs"]
