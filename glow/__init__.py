from .glow import WaveGlow, WaveGlowLoss,WN,Invertible1x1Conv
from .inference import infer_glow


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self