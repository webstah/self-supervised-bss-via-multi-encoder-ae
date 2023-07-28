def import_experiment_from_config(config):
    experiment_module = __import__(config.experiment[0], 
        fromlist=[config.experiment[0].split('.')[-1]])

    return getattr(experiment_module, config.experiment[1])

def import_model_from_config(config):
    experiment_module = __import__(config.model[0], 
        fromlist=[config.model[0].split('.')[-1]])

    return getattr(experiment_module, config.model[1])

def import_dataloader_from_config(config):
    experiment_module = __import__(config.dataloader[0], 
        fromlist=[config.dataloader[0].split('.')[-1]])

    return getattr(experiment_module, config.dataloader[1])