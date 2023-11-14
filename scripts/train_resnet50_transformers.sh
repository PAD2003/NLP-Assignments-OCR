if [ ! -d "aug_epoch" ]; then
    mkdir "aug_epoch"
fi
export CUDA_VISIBLE_DEVICES=2
python -m  src.train logger=wandb +logger.wandb.name="resnet50_transformers" data.num_workers=32 trainer=gpu trainer.max_epochs=35