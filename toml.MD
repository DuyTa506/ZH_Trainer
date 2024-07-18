### This is the explained hyperparameters for config.toml file :
- First, i provided the meta config , which in [meta] , the hyperparameters like below :
    - name = "wav2vec2_xlsr_pinyin"  (name of the saved path checkpoint and logs)
    - pretrained_path = "TencentGameMate/chinese-wav2vec2-large" (the pretrained model from huggingface / or any hugging_face liked weights)
    - seed = 42 (random initialize seed for every run)
    - epochs = 10
    - save_dir = "saved/" 
    - gradient_accumulation_steps = 2 (Num step to accumulate the gradient, instead of update any backward pass in training loop)
    - use_amp = false # Whether to use Automatic Mixed Precision for speeding up - https://pytorch.org/docs/stable/amp.html (Please check hardware compatibility)
    - device_ids= “0,1,2,3,4,5,6,7” # set the gpu devices on which you want to train your model
    - sr = 16000 (audio sample rate)
    - max_clip_grad_norm = 5.0 # torch.nn.utils.clip_grad_norm_ (avoid exploding gradients , the final gradien will be normalize by this factor)
    - train_ratio = 0.9 (training size)
- Second is for [special token] : 

    - bos_token = "<bos>"
    - eos_token = "<eos>"
    - unk_token = "<unk>"
    - pad_token = "<pad>"
- Next is for huggingface configurations , please pass this because of private project
    - local_dir = "huggingface-hub_xlsr_pinyn" # where your repo places in local (the huggingface type weights after each training)
- Next is for training and validation dataset and dataloader configuration : 
   - Remember that the path parameter can be multiple folers, with separated by comma . E.g : "path1,path2,path3"
   - volume = 1.0 (Percentage of each datasets you want to train)
   - model_type = "pinyin" (choose between "pinyin" and "hanzi" )
   - batch_size = 2
   - nb_workers = 8 (processed for filterd audio ) differ from num_workers = 2 (processed for dataloader and need to configure -shm in docker containers)
- Next to : [optimizer] :
    - Modify your learning rate in here. Now, this project is using AdamW optimizer, you can change different optim by change the code in train.py
- Next  is [scheduler]
    Learning rate scheduler is a type of schedule learning  rate for efficent learning.Support Onecycle and linear 
- The last is trainer : 
    -    validation_interval = 5000  (After 5000 steps, we will make a validation, for example)











