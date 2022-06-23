from logging import error
import os
import json

import torch
import numpy as np

import hifigan
import univnet
import glow
from model import FastSpeech2, ScheduledOptim

from omegaconf import OmegaConf
#from denoiser import Denoiser


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(model, train_config, model_config,
                                         args.restore_step)
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

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
                "/media/dell/disk/LJSpeech/chkpt/LJS/LJS_0277.pt")
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
                  model_config,
                  preprocess_config,
                  lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
            print(wavs.shape)
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
        wavs = wavs.cpu().numpy()
    else:
        wavs = (wavs.cpu().numpy() *
                preprocess_config["preprocessing"]["audio"]["max_wav_value"]
                ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][:lengths[i]]

    return wavs
