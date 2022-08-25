import utils
import argparse
import json
import glob
import os
import numpy as np
import librosa
import torch
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from text import cleaned_text_to_sequence
from pinyin_dict import pinyin_dict

def get_phoneme4pinyin(pinyins):
    result = []
    for pinyin in pinyins:
        if pinyin[:-1] in pinyin_dict:
            tone = pinyin[-1]
            a = pinyin[:-1]
            a1, a2 = pinyin_dict[a]
            result += [a1, a2 + tone, "#0"]
    result.append("sil")
    return result

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

pinyin_parser = Pinyin(MyConverter())

def get_text(phones):
    text_norm = cleaned_text_to_sequence(phones)
    # baker 应该将add_blank设置为false
    # [0, 19, 81, 3, 14, 51, 3, 0, 1]
    # [0, 0, 0, 19, 0, 81, 0, 3, 0, 14, 0, 51, 0, 3, 0, 0, 0, 1, 0]
    # text_norm = commons.intersperse(text_norm, 0)
    # if hps.data.add_blank:
    #     text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/baker_bigvgan_vits.json",
                        help='JSON file for configuration')
    parser.add_argument('-i', '--input_path', type=str, default="./dataset/baker")
    parser.add_argument('-o', '--output_path', type=str, default="./dataset/baker_npz")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = utils.HParams(**config)

    speaker = 'baker'
    os.makedirs(os.path.join(args.output_path, speaker, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, speaker, 'test'), exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(args.input_path, '*.wav')))
    for wav in wavs[:80]:
        if not os.path.exists(wav.replace('.wav', '.txt')):
            continue
        data = preprocess_wav(wav, hparams)
        np.savez(os.path.join(args.output_path, speaker, 'test', os.path.basename(wav).replace('.wav', '.npz')),
                 **data, allow_pickle=False)

    for wav in wavs[80:]:
        if not os.path.exists(wav.replace('.wav', '.txt')):
            continue
        data = preprocess_wav(wav, hparams)
        np.savez(os.path.join(args.output_path, speaker, 'train', os.path.basename(wav).replace('.wav', '.npz')),
                 **data, allow_pickle=False)


def preprocess_wav(wav, hparams):
    audio, _ = librosa.load(wav, sr=16000)
    audio = audio * hparams.data.max_wav_value
    text_file = wav.replace('wav', 'txt')
    with open(text_file, encoding='utf8') as f:
        text = f.readline().rstrip()
    token = cleaned_text_to_sequence(text)

    data = {
        'audio': audio,
        'token': token,
        'text': text
    }
    return data


if __name__ == "__main__":
    main()
