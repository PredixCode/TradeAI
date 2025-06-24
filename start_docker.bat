@echo off
call docker run --gpus all -v ${PWD}/ai/models:/app/ai/models -v ${PWD}/market/util:/app/market/util trade-ai --mode train-lstm
:: Modes:
::   - 'train-vae', 'train-lstm', 'train-trader', 'full-pipeline', 'eval-vae', 'eval-lstm', 'backtest'
