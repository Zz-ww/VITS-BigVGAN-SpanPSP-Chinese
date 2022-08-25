import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from glob import glob
import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
# from text import text_to_sequence, cleaned_text_to_sequence
import torchaudio

class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, is_train=False):

        self.spk_path = glob(os.path.join(audiopaths_and_text,'*'))

        print("Speaker num", len(self.spk_path))
        self.is_train = is_train
        self.npzs, self.spk_label = self.get_npz_path(self.spk_path)

        print("Total data len: ", len(self.npzs))
        # self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 1000)

        c = list(zip(self.npzs, self.spk_label))
        random.seed(1234)
        random.shuffle(c)
        self.npzs, self.spk_label = zip(*c)

        self._filter()
        print("filtered data len: ", len(self.npzs))

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        npz_new = []
        lengths = []
        spk_new = []
        for npz, spk in zip(self.npzs, self.spk_label):
            temp = np.load(npz)
            # if len(temp['audio'])//256 < 400:
            npz_new.append(npz)
            lengths.append(len(temp['audio']) // (self.hop_length))
            spk_new.append(spk)

        self.lengths = lengths
        self.npzs = npz_new
        self.spk_label = spk_new



    def get_npz_path(self, spk_path):
        npz_path = []
        speaker_label = []
        i = 0
        for spk in spk_path:
            if self.is_train:
                temp_path = glob(os.path.join(spk, os.path.join("train", "*.npz")))
                npz_path += temp_path
                speaker_label += [i]*len(temp_path)

            else:
                temp_path = glob(os.path.join(spk, os.path.join("test", "*.npz")))
                npz_path += temp_path
                speaker_label += [i]*len(temp_path)
            i +=1
        return npz_path, speaker_label

    def get_audio_text_pair(self, audiopath_and_text, spk_label):

        files = np.load(audiopath_and_text)
        text = self.add_blank_token(files['token'])
        spec, wav = self.get_audio(files['audio'], audiopath_and_text, language=0)
        spk_id = spk_label

        return (text, spec, wav, spk_id)

    def get_audio(self, audio, filename, language=0):

        audio = torch.FloatTensor(audio.astype(np.float32))
        audio_norm = audio / self.max_wav_value  * 0.95
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".npz", "." + str(language) + "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def add_blank_token(self, text):
        if self.add_blank:
            text = commons.intersperse(text, 0)
        text_norm = torch.LongTensor(text)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.npzs[index], self.spk_label[index])

    def __len__(self):
        return len(self.npzs)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length

        # _, ids_sorted_decreasing = torch.sort(
        #     torch.LongTensor([x[1].size(1) for x in batch]),
        #     dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        sid = torch.LongTensor(len(batch))

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))


        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()


        for i in range(len(batch)):
            row = batch[i]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, 0
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths,  sid

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
