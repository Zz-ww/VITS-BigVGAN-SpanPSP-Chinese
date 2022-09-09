

# VITS+BigVGAN+SpanPSP 中文TTS


本项目实现vits+BigVGAN端到端的中文TTS 模型，推理阶段加入中文韵律预测模型，实现的带韵律预测的中文TTS模型。

## 1.环境准备(Environment)

- Python 3.7 or higher.
- Pytorch 1.9.0, or any compatible version.
- NLTK 3.2, torch-struct 0.4, transformers 4.3.0, or compatible.
- pytokenizations 0.7.2 or compatible.
- 安装espeak库：apt-get install espeak
- 编译monotonic_align:

```
cd monotonic_align
python setup.py build_ext --inplace
```

## 2.数据(Dataset)

使用标贝16k，10000句中文女声数据进行tts训练。

## 3.项目结构(Repository structure)

```
VITS-BigVGAN-Chinese
├──benepar
|   ├──integrations
|   ├── ...
├──bert-base-chinese
|   ├──config.json
|   ├──pytorch_model.bin
|   └──vocab.txt
├──configs
|   ├──baker_bigvgan_vits.json
├──dataset
|   ├──baker
|       ├──000001.txt
|       ├──000001.wav
|       ├── ...
|   ├──baker_npz
|   ├──000001-010000.txt
|   └──baker_train.txt
├──logs
├──misc
├──monotonic_align
├──text
├──vits_out
├──weights
|   ├──pretrained_SpanPSP_Databaker.pt
├──...
├──README.md
```

### bert-base-chinese

> Link: https://huggingface.co/bert-base-chinese

## 4.数据预处理(Data preprocessing)

使用标贝16k女声数据集

```
cd dataset
#解压baker的txt
tar -xvf baker.tar
#拷贝你的baker音频文件到./dataset/baker下
cp /你的标贝数据路径/*.wav  ./baker/
cd ..
python preprocess.py
```

预处理数据格式如下所示：

```
├──dataset
|   ├──baker #预处理文件
|       ├──000001.txt
|       ├──000001.wav
|       ├── ...
|   ├──baker_npz #预处理后文件保存路径
|   ├──000001-010000.txt #baker原始标注文件
|   └──baker_train.txt #baker转音素韵律后标注文件
```

000001.wav中文标注

```
卡尔普#2陪外孙#1玩滑梯#4。
```

转为dataset/baker/000001.txt

```
sil k a2 #0 ^ er2 #0 p u3 #2 p ei2 #0 ^ uai4 #0 s uen1 #1 ^ uan2 #0 h ua2 #0 t i1 #4 sil eos
```

注：如需使用其他数据集，需按标贝格式和韵律标注方式制作数据集。

## 5.模型训练(Training)

```
python train_vits_with_bigvgan.py -c configs/baker_bigvgan_vits.json  -m baker
```

## 6.模型推理(Inference)

模型tts合成vits_strings.txt文本内容

```
python vits_strings_psp.py
```

其中vits_strings.txt文本内容在推理阶段使用SpanPSP的模型进行语录预测：

```
  #输入：
  猴子用尾巴荡秋千
  #韵律预测结果：
  猴子#2用#1尾巴#2荡秋千#4。 
```

## 7.合成效果(Results)

标贝数据集合成效果：

```
#输入：
云雾有时宛如玉带平卧峰峦山涧，有时炊烟袅绕，薄雾轻旋。
 #韵律预测结果：
云雾#2有时#1宛如#1玉带#2平卧#2峰峦#1山涧#3有时#2炊烟#1袅绕#3薄雾#1轻旋#4
 #合成结果：
 ./vits_out/baker_output.wav
```

## 8.预训练模型（Pretrained model）

#### weights/pretrained_SpanPSP_Databaker.pt:

https://pan.baidu.com/s/1Cox0ouFCUKJLemysiLZ4vQ 提取码:7ell

#### bert-base-chinese:

https://pan.baidu.com/s/1twX20z1O_xqMVyq_le4E5g 提取码:p12j

#### ./logs/baker/G_745000.pth:

链接：https://pan.baidu.com/s/1qxR1AdQAFrOR1QItofwLGA  提取码：n8cs



## 扩展(Extension)

#### 最近在试验基于4k视频的人物换装和物品替换，感兴趣的可以去看看视频展示的效果，我会在空闲时逐步整理代码更新项目：

https://github.com/Zz-ww/Virtual-try-on-4kVideo



## 参考(Reference)

- BigVGAN: https://arxiv.org/abs/2206.04658
- BigVGAN:https://github.com/sh-lee-prml/BigVGAN
- VITS: https://github.com/jaywalnut310/vits
- SpanPSP:https://github.com/thuhcsi/SpanPSP
- vits_chinese:https://github.com/lutianxiong/vits_chinese
