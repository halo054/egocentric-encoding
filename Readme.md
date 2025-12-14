![Modality_Alignment](https://github.com/user-attachments/assets/f831bc08-f087-45e1-a57b-9389f0fa089f)


Final_Report.pdf contains the research report for our ego-exo alignment. For some reasons Github may not show the pdf correctly when viewing it online. A backup for the Research Report available [here](https://drive.google.com/file/d/1CN2i8wW-55-RkZs-F6qSn_9UUa0RPy1r/view?usp=sharing).

Weights can be downloaded [here](https://drive.google.com/file/d/17D-BIo4xzCRvQCfAuCAFB-vlJ54pQ8DG/view?usp=sharing).

Follow [ImageBind-Lora](https://github.com/fabawi/ImageBind-LoRA) for model downloading and environment setup.

4 RTX A5000 or GPUs with larger VRAM is required for training.

- `train_ego_naive.py` - is the script to do the egocentric encoder training.

- `get_embeddings.py` - is the script to extract and store embeddings.

- `retrieval.py` - is the script to test retrieval score.

The dataset we use is Ego-Exo4d, other Dataset may be used with customized torch dataset and dataloader.
