#!/bin/zsh

cd ~/workspace/MoE-LLaVA-JP
poetry run huggingface-cli download --repo-type dataset turing-motors/LLaVA-Instruct-150K-JA --local-dir /home/junhyeong.kim/workspace/MoE-LLaVA-JP/dataset/