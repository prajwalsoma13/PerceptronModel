import json

with open("config.json") as f:
    Configuration = json.load(f)


learningRate = Configuration["learningRate"]
epochs = Configuration["epochs"]
