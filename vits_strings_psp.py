import os
import numpy as np
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile
import torch
import commons
import utils
from model_vits_with_bigvgan import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
import prosody_txt
from pinyin_dict import pinyin_dict


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def load_pinyin_dict():
    my_dict = {}
    with open("./misc/pypinyin-local.dict", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            pinyin = cuts[1:]
            tmp = []
            for one in pinyin:
                onelist = [one]
                tmp.append(onelist)
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


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


def chinese_to_phonemes(pinyin_parser, text, single_zw):
    all = 'sil'
    zw_index = 0
    py_list_all = pinyin_parser.pinyin(text, style=Style.TONE3, errors="ignore")
    py_list = [single[0] for single in py_list_all]
    for single in single_zw:
        if single == '#':
            all = all[:-2]
            all += single
        elif single.isdigit():
            all += single
        else:
            pyname = pinyin_dict.get(py_list[zw_index][:-1])
            all += ' ' + pyname[0] + ' ' + pyname[1] + py_list[zw_index][-1] + ' ' + '#0'
            zw_index += 1
    all = all + ' ' + 'sil' + ' ' + 'eos'
    return all


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def get_text(phones, hps):
    text_norm = cleaned_text_to_sequence(phones)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


load_pinyin_dict()
pinyin_parser = Pinyin(MyConverter())

# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/baker_bigvgan_vits.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/baker/G_400000.pth", net_g, None)

# check directory existence
if not os.path.exists("./vits_out"):
    os.makedirs("./vits_out")

if __name__ == "__main__":
    n = 0
    yl_model = prosody_txt.init_model()
    fo = open("vits_strings.txt", "r+")
    while (True):
        try:
            message = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (message == None):
            break
        if (message == ""):
            break
        n = n + 1
        single_zw = ''
        prosody_txt.run_auto_labels(yl_model, message)
        with open('temp.txt', 'r') as r:
            for line in r.readlines():
                line = line.strip()
                single_zw += line + '#3'
        single_zw = single_zw[:-1] + '4'
        print(single_zw)
        phonemes = chinese_to_phonemes(pinyin_parser, message, single_zw)
        input_ids = get_text(phonemes, hps)
        with torch.no_grad():
            x_tst = input_ids.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([input_ids.size(0)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0, noise_scale_w=0, length_scale=1)[0][
                0, 0].data.cpu().float().numpy()
        save_wav(audio, f"./vits_out/{n}_baker_0815.wav", hps.data.sampling_rate)
        print(message)
        print(phonemes)
        print(input_ids)
    fo.close()
