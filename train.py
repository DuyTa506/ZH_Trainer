import argparse
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import toml
import warnings
import datetime
warnings.filterwarnings('ignore')

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import *
from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from torch.utils.data import random_split
#---------------------Self module---------------
from filter import filter_token

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '55555'
    #os.environ['GLOO_SOCKET_IFNAME']= 'enp49s0f1'   ## please provide the network by check ifconfig in OS
    os.environ['NCCL_SOCKET_IFNAME']= 'enp49s0f1'
    # initialize the process group
    dist.init_process_group("nccl" ,rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600 * 5))
    torch.cuda.set_device(rank)
    dist.barrier()

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, resume, preload):
    os.environ['CUDA_VISIBLE_DEVICES']=config["meta"]["device_ids"]
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    setup(rank, world_size)

    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    use_amp = config["meta"]["use_amp"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    save_dir =  os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/checkpoints')
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/log_dir')
    
    if rank == 0:
        # Creatr dirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/' + config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["meta"]["seed"] += rank
    set_seed(config["meta"]["seed"])
    config['val_dataset']['args']['sr'] = config['meta']['sr']
    config['train_dataset']['args']['sr'] = config['meta']['sr']

    config['train_dataset']['args']['rank'] = rank
    config['val_dataset']['args']['rank'] = rank

    config["train_dataset"]["args"]["dist"] = dist
    config["val_dataset"]["args"]["dist"] = dist

    config["train_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    config["val_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    
    
    #Begin filters
##############################
    train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])

    vocab_dict = train_base_ds.get_vocab_dict()
    with open('vocab.json', 'w+',encoding='utf-8' ) as f:
        json.dump(vocab_dict, f,ensure_ascii=False)
        f.close()
    dist.barrier()
    # Create processor
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
                                    **config["special_tokens"],
                                    word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['meta']['pretrained_path'])
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    default_collate = DefaultCollate(processor, config['meta']['sr'])
    #data_collator = DataCollatorCTCWithPadding(processor=processor,padding=True)
    

    
    # Create train dataloader
    

    train_ds = train_base_ds.get_data()
    
    train_size = int(config["meta"]["train_ratio"] * len(train_ds))
    valid_size = len(train_ds) - train_size

    train_ds, val_ds = random_split(train_ds, [train_size, valid_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        **config["train_dataset"]["sampler"]
    )
    train_dl = DataLoader(
        dataset=train_ds,
        **config["train_dataset"]["dataloader"],
        sampler = train_sampler,
        collate_fn=default_collate
    )

    # Create val dataloader
    #val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    #val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        **config["val_dataset"]["sampler"]
    )
    val_dl = DataLoader(
        dataset=val_ds,
        **config["val_dataset"]["dataloader"],
        sampler = val_sampler,
        collate_fn=default_collate
    )

    print("Done initialize dataset : Train samples: {}, Test samples: {}".format(len(train_ds),len(val_ds)))
    # Load pretrained model
    if "TencentGameMate" in config['meta']['pretrained_path'] :
        print("-----------------Training with TencentGameMate pretrained model------------------")
    #for Tencent Pretrained
        model = Wav2Vec2ForCTC.from_pretrained(
        config['meta']['pretrained_path'], 
        ctc_loss_reduction="mean", 
         pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
       #gradient_checkpointing=False
        )
    #for XLR-S
    else :

    #model = Wav2Vec2ForCTC.from_pretrained(config['meta']['pretrained_path'])
        model = Wav2Vec2ForCTC.from_pretrained(
        config['meta']['pretrained_path'], 
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        )
    model.config.ctc_zero_infinity = True
    #model.gradient_checkpointing_enable()
    # freeze the wav2vec feature encoder, if you have small dataset, this helps a lot
    model.freeze_feature_extractor()
    # DDP for multi-processing
    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # Set up metric, scheduler, optmizer
    compute_metric = Metric(processor)
    #Can we use LionW optimizer ?
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = config["optimizer"]["lr"]
    )
    steps_per_epoch = (len(train_dl)//gradient_accumulation_steps) + (len(train_dl)%gradient_accumulation_steps != 0)
    # can use Linear Scheduler instead
    if config["scheduler"]["type"] == "linear": 
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,end_factor=0.25,total_iters=5)
    elif config["scheduler"]["type"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config["scheduler"]["max_lr"], 
            epochs=epochs, 
            steps_per_epoch = steps_per_epoch)


    if rank == 0:
        print("Number of training utterances: ", len(train_ds))
        print("Number of validation utterances: ", len(val_ds))

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        n_gpus = world_size,
        config = config,
        resume = resume,
        preload = preload,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        model = model,
        compute_metric = compute_metric,
        processor = processor,
        train_dl = train_dl,
        val_dl = val_dl,
        train_sampler = train_sampler,
        val_sampler = val_sampler,
        optimizer = optimizer,
        scheduler = scheduler,
        save_dir = save_dir,
        log_dir = log_dir,
        gradient_accumulation_steps = gradient_accumulation_steps,
        use_amp = use_amp,
        max_clip_grad_norm = max_clip_grad_norm
    )
    trainer.train()


    cleanup()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', default="config.toml", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', action="store_true",
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-p', '--preload', default=None, type=str,
                      help='Path to pretrained Model')            
    
    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config['meta']["device_ids"].split(','))
    
    
    nb_workers = config["create_data"]["nb_workers"]
    data_path = os.path.join(config["create_data"]["init_pq"], "train.parquet")
    token_max = config["create_data"]["token_max"]
    token_min = config["create_data"]["token_min"]
    dump_tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
                                    **config["special_tokens"],
                                    word_delimiter_token="|")
    
    filter_token(data_path, nb_workers,token_max,token_min, dump_tokenizer)
    del dump_tokenizer
    mp.spawn(
        main,
        args = (n_gpus, config, args.resume, args.preload),
        nprocs = n_gpus,
        join = True
    )



