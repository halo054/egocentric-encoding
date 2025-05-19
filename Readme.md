Follow ImageBind-Lora https://github.com/fabawi/ImageBind-LoRA for model downloading and environment setup.

4 RTX A5000 or GPUs with larger VRAM is required for training.

"train_ego_naive.py" is the script to do the egocentric encoder training.

"get_embeddings.py" is the script to extract and store embeddings.

"retrieval.py" is the script to test retrieval score.

The dataset we use is Ego-Exo4d, other Dataset may be used with customized torch dataset and dataloader.
