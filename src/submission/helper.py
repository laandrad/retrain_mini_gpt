from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)


def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part c]: Make some model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model


def initialize_synthesizer_model(mconf):
    attention_model = None
    ### TODO
    ### [part g]: Make some other model here

    ### START CODE HERE
    ### END CODE HERE
    return attention_model


def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model):
    ### TODO:
    ### [part c] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu')) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    ### START CODE HERE
    print(reading_params_path)
    tconf = TrainerConfig()
    tconf.batch_size = 256
    tconf.learning_rate = 6e-4
    tconf.lr_decay = True
    tconf.warmup_tokens = 512 * 20
    tconf.final_tokens = 200 * len(pretrain_dataset) * block_size
    tconf.num_workers = 4
    if not reading_params_path:
        tconf.max_epochs = 75
    else:
        tconf.max_epochs = 10
        state_dict = torch.load(reading_params_path)
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
        model.load_state_dict(torch.load(state_dict))

    finetuning_corpus = open(finetune_corpus_path, 'r').read()
    finetuning_data = NameDataset(finetuning_corpus, pretrain_dataset)
    trainer_obj = Trainer(model, finetuning_data, None, tconf)
    ### END CODE HERE
    return tconf, trainer_obj


def pretrain(pretrain_dataset, block_size, model):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=4

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    tconf = TrainerConfig()
    tconf.batch_size = 128
    tconf.learning_rate = 6e-3
    tconf.lr_decay = True
    tconf.warmup_tokens = 512 * 20
    tconf.final_tokens = 200 * len(pretrain_dataset) * block_size
    tconf.num_workers = 4
    tconf.max_epochs = 650

    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)
    ### END CODE HERE
    return tconf, trainer_obj


def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part c]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    trainer_obj.train()
    torch.save(trainer_obj.model.state_dict(), writing_params_path)
    ### END CODE HERE
    return
