import torch
import hifigan
import univnet

import os
import json
import glow
from logging import error

from omegaconf import OmegaConf
from univnet import TacotronSTFT

import numpy as np
from scipy.io.wavfile import read

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import librosa

def get_vocoder(name, speaker, device):
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load("descriptinc/melgan-neurips",
                                     "load_melgan", "linda_johnson")
        elif speaker == "universal":
            vocoder = torch.hub.load("descriptinc/melgan-neurips",
                                     "load_melgan", "multi_speaker")
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        #if speaker == "LJSpeech":
        #    ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        #elif speaker == "universal":
        #    ckpt = torch.load("hifigan/generator_universal.pth.tar")
        state_dict_g = torch.load(config['ckpt'])
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "waveglow":
        with open("glow/config.json", "r") as f:
            config = json.load(f)
        print("load waveglow")
        vocoder = torch.load("glow/waveglow_256channels.pt")['model']
        print("load done")

        vocoder = vocoder.remove_weightnorm(vocoder)
        vocoder.cuda().eval()
        is_fp16 = False
        if is_fp16:
            from apex import amp
            print("apex")
            vocoder, _ = amp.initialize(vocoder, [], opt_level="O3")

        #if denoiser_strength > 0:
        #    denoiser = Denoiser(waveglow).cuda()
        vocoder.to(device)
    elif name == "univnet":
        if speaker == "LJSpeech":
            checkpoint = torch.load(
                "/media/dell/disk/LJSpeech/chkpt/LJS/LJS_0227.pt")
        else:
            error("no matched checkpoint")
        if os.path.exists("univnet/config/config.json"):
            hp = OmegaConf.load("univnet/config/config.json")
        else:
            hp = OmegaConf.create(checkpoint['hp_str'])

        vocoder = univnet.Generator(hp)
        saved_state_dict = checkpoint['model_g']
        new_state_dict = {}

        for k, v in saved_state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict['module.' + k]
            except:
                new_state_dict[k] = v
        vocoder.load_state_dict(new_state_dict)
        vocoder.eval(inference=True)
        vocoder.to(device)

    return vocoder

def vocoder_infer(mels,
                  vocoder,
                  name,
                  max_wav_value=32768.0,
                  lengths=None):
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
            print("wavs:",wavs)
        elif name == "waveglow":
            #mels = torch.load("/media/dell/disk/LJSpeech/batch_mel/train/1_mel.pt")[0]
            print(mels.size())
            print(type(mels))
            mels = torch.autograd.Variable(mels.cuda())
            #mels = torch.unsqueeze(mels, 0)
            mels = mels.half() if False else mels
            print(mels.size())
            print(type(mels))
            print(mels)
            wavs = vocoder.infer(mels, sigma=0.6)
            '''
            for audio in wavs:
                if denoiser_strength > 0:
                    audio = denoiser(audio, denoiser_strength)
            '''
        elif name == "univnet":
            print(mels.shape)
            wavs = vocoder.inference_batch(mels)
            print(wavs.shape)
            print(wavs)

    if name == "univnet":
        wavs = wavs.cpu().numpy() / max_wav_value
    else:
        wavs = wavs.cpu().numpy()
    wavs = [wav for wav in wavs]

    return wavs


def plot_spectrogram(spectrogram, savepath):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()

def read_wav_np(path):
    sr, wav = read(path)
    print(wav)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav


model_name = "univnet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
stft = TacotronSTFT(filter_length=1024,
                            hop_length=256,
                            win_length=1024,
                            n_mel_channels=80,
                            sampling_rate=22050,
                            mel_fmin=0.0,
                            mel_fmax=8000.0,
                            center=False,
                            device=device)
id = "LJ029-0021"
mels = torch.load("./data/LJSpeech-1.1/wavs/{}.mel".format(id)).unsqueeze(0)
sr, audio = read_wav_np("./data/LJSpeech-1.1/wavs/{}.wav".format(id))
vocoder = get_vocoder(model_name, "LJSpeech", device)
wavs = np.array(vocoder_infer(mels, vocoder, model_name))
print(audio)
print(wavs)
spec_real = stft.linear_spectrogram(torch.Tensor(audio).to(device).unsqueeze(0)).cpu()
spec_fake = stft.linear_spectrogram(torch.Tensor(wavs).to(device)).cpu()
spec_fake = librosa.amplitude_to_db(spec_fake, ref=np.max,top_db=80.)
spec_real = librosa.amplitude_to_db(spec_real, ref=np.max,top_db=80.)
plot_spectrogram(spec_fake.squeeze(), "./output/vocoder/{}_{}_pre.png".format(id, model_name))
plot_spectrogram(spec_real.squeeze(), "./output/vocoder/{}_tgt.png".format(id))