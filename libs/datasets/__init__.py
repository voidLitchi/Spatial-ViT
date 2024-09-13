from .voc import VOCWSSS, VOCCls


def get_dataset(name):
    return {
        "vocwsss": VOCWSSS,
        "voccls": VOCCls,
    }[name]
