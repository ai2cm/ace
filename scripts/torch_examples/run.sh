#!/usr/bin/env bash

torchrun --nproc-per-node 2 distributed_loading_demo.py
