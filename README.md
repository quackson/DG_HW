训练数据：

- 存放在data文件夹的LJSpeech1.1
- 将附件中的数据解压到该文件夹中

环境配置

```perl
pip install -r requirements.txt
```

模型训练

- Hifi-Gan训练：

  - ```shell
    cd hifigan
    bash train.sh
    ```

- Univnet训练：

  - ```shell
    cd univnet
    bash LJS_16.sh
    ```

音频生成

- 可以选择四种不同的vocoder

- 在FastSpeech2根目录下运行：替换YOUR_DESIRED_TEXT为需要生成的文本

- ```perl
  #melGan vocoder
  python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/melgan.yaml -t config/LJSpeech/train.yaml
  #hifi-gan vocoder
  python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/hifigan.yaml -t config/LJSpeech/train.yaml
  #waveglow vocoder
  python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/waveglow.yaml -t config/LJSpeech/train.yaml
  #univnet vocoder
  python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/univ.yaml -t config/LJSpeech/train.yaml
  ```

- 根目录下有不同vocoder的sh脚本，可运行修改

输出结果：

- output文件夹中，wav文件名和输入的text一样