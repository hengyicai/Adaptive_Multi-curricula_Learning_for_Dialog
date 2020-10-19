# Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation

This repo contains preliminary code of the AAAI2020 paper named "[Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation](https://arxiv.org/abs/2003.00639)".

This codebase is built upon the [ParlAI](https://parl.ai/) project. 
Check `parlai/agents/adaptive_learning` for experimental models implementation.
RL-based multi-curriculum learning lies in `parlai/tasks/adaptive_learning`.
Running scripts can be found in `projects/adaptive_learning`.

## Framework Overview

<p align="center">
<img src="https://github.com/hengyicai/Adaptive_Multi-curricula_Learning_for_Dialog/blob/master/Fig-ModelArch.png" alt="Framework Overview" width="600"/>
</p>

## Requirements
- Python3
- Pytorch 1.2 or newer

Dependencies of the core modules are listed in requirement.txt.

## Dataset
The datasets used in the paper can be download from [here](https://drive.google.com/file/d/1Lj9R55u-xk1IVJ6uNXOUU0YjKvIURJAY/view?usp=sharing). 
Put it in `data/` and unzip it using `tar -xzvf AdaptiveLearning.tar.gz`

## Installing
```
git clone git@github.com:hengyicai/Adaptive_Multi-curricula_Learning_for_Dialog.git ~/Adaptive_Multi-curricula_Learning_for_Dialog
cd ~/Adaptive_Multi-curricula_Learning_for_Dialog; python setup.py develop
echo "export PARLAI_HOME=~/Adaptive_Multi-curricula_Learning_for_Dialog" >> ~/.bashrc; source ~/.bashrc
```

## Running

```
cd ~/Adaptive_Multi-curricula_Learning_for_Dialog
bash projects/adaptive_learning/shell/run.sh
```

The last line of `projects/adaptive_learning/shell/run.sh` specifies preliminary arguments for the training:
```
# train_model  MODEL_NAME  TASK_NAME  SUB_TASK  T  VALIDATION_EVERY_N_SECS  VALIDATION_EVERY_N_EPOCHS  NUM_EPOCHS
train_model seq2seq personachat_h3 combine 11000 -1 0.2 30
```

This run will apply the multi-curriculum learning framework on `Seq2seq` model using dataset `PersonaChat`. The duration of curriculum learning is `11000` steps. 

Applying the single `specificity` curriculum dialogue learning on model `CVAE` using dataset `DailyDialog`, with curriculum learning duration `8000`:
```
train_model cvae daily_dialog specificity 8000 -1 0.2 30
```

See `projects/adaptive_learning/shell/run.sh` for details.

## Citation
If you find our code/models or ideas useful in your research, please consider citing the paper:
```Tex

@InProceedings{Hengyi_2020_AAAI,
  author={Hengyi Cai and Hongshen Chen and Cheng Zhang and Yonghao Song and Xiaofang Zhao and Yangxi Li and Dongsheng Duan and Dawei Yin},
  title={Learning from Easy to Complex: Adaptive Multi-curricula Learning for Neural Dialogue Generation},
  booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2020}
}

```
