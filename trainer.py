import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from omegaconf import DictConfig
import hydra

from utils.importer import import_experiment_from_config, import_dataloader_from_config

@hydra.main(config_path="config", config_name="config.yaml")
def run_experiment(config: DictConfig):
    experiment_config = config.experiment_config
    
    # import data module/loader
    data_module = import_dataloader_from_config(experiment_config)
    data = data_module(experiment_config)

    # import experiment from config
    experiment_class = import_experiment_from_config(experiment_config)

    # load from checkpoint if path is provided in config
    if config.checkpoint is not None:
        experiment = experiment_class.load_from_checkpoint(config.checkpoint, data.x_plot)
    else:
        experiment = experiment_class(experiment_config, data.x_plot)

    # logging and monitors
    log_dir = f'{experiment_config.name}/'
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    checkpoint_callback = ModelCheckpoint(monitor=experiment_config.monitor, 
                                          mode=experiment_config.monitor_mode, 
                                          save_last=True, save_top_k=25, every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # create trainer and fit model
    trainer = pl.Trainer(logger=tb_logger, 
                         gradient_clip_val=experiment_config.gradient_clip_val,
                         callbacks=[checkpoint_callback, lr_monitor], 
                         accelerator='gpu', devices=config.gpus, max_epochs=experiment_config.max_epochs, 
                         check_val_every_n_epoch=1, overfit_batches=0, 
                         num_sanity_val_steps=0)#, precision=16)
    trainer.fit(experiment, data)

if __name__ == '__main__':
    run_experiment()