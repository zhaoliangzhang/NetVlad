#!/bin/bash

if [ ! -d "model/" ]; then
	echo "path doesn't exist"
	mkdir model
else
	echo "path exist"
fi

if [ ! -f "model/dpu_pytorch.elf" ]; then
	wget https://cloud.tsinghua.edu.cn/f/75d445e1657a4804a01a/?dl=1 -O ./model/dpu_pytorch.elf
else
	echo "elf file exist"
fi

if [ ! -f "model/vlad_weight" ]; then
	wget https://cloud.tsinghua.edu.cn/f/4dbaa4de8df747cd8df5/?dl=1 -O ./model/vlad_weight
else
	echo "weight file exist"
fi
