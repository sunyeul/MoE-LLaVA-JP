#!/bin/zsh

cd ~/workspace/MoE-LLaVA-JP
poetry run huggingface-cli download --repo-type dataset liuhaotian/LLaVA-CC3M-Pretrain-595K --local-dir /home/junhyeong.kim/workspace/MoE-LLaVA-JP/dataset/LLaVA-CC3M-Pretrain-595K