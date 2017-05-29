import yaml
import math

functions = {
    'parabola' : lambda x : math.pow(x,2)
}

def loadConfig():
    with open("config.yaml", 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
