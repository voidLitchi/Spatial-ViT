from .voc import VOCWSSS

def get_dataset(name):
    return {
        "vocwsss": VOCWSSS
    }[name]