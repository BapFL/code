from utils.utils import parse_args
from algorithm.FedPer.server import SERVER as FedPer_SERVER
from algorithm.BapFL_FedPer.server import SERVER as BapFL_FedPer_SERVER
from algorithm.BapFL_Plus_FedPer.server import SERVER as BapFL_Plus_SERVER
from algorithm.Gen_BapFL_FedPer.server import SERVER as Gen_BapFL_SERVER
from algorithm.LGFedAvg.server import SERVER as LGFedAvg_SERVER
from algorithm.BapFL_LGFedAvg.server import SERVER as BapFL_LGFedAvg_SERVER
from algorithm.BapFL_Plus_LGFedAvg.server import SERVER as BapFL_Plus_LGFedAvg_SERVER
from algorithm.Gen_BapFL_LGFedAvg.server import SERVER as Gen_BapFL_LGFedAvg_SERVER

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        import wandb
        """
        You Can Backdoor Personalized Federated Learning
        """
        wandb.init(project="CIKM23-BapFL", entity="xxx")
        wandb.watch_called = False
        config = wandb.config
        config.update(args)
    else:
        config = args
    attacker_ids = [int(item.strip()) for item in config.attackers_ids.split(',')]
    config.attackers = attacker_ids
    if config.alpha in [1, 5, 10000]:
        config.alpha = int(config.alpha)
    print("Attacker population: ", config.attackers)
    server = None
    if config.algorithm == 'FedPer':
        server = FedPer_SERVER(config=config)
    elif config.algorithm == 'BapFL_FedPer':
        server = BapFL_FedPer_SERVER(config=config)
    elif config.algorithm == 'BapFL_Plus_FedPer':
        server = BapFL_Plus_SERVER(config=config)
    elif config.algorithm == 'Gen_BapFL_FedPer':
        server = Gen_BapFL_SERVER(config=config)
    elif config.algorithm == 'LGFedAvg':
        server = LGFedAvg_SERVER(config=config)
    elif config.algorithm == 'BapFL_LGFedAvg':
        server = BapFL_LGFedAvg_SERVER(config=config)
    elif config.algorithm == 'BapFL_Plus_LGFedAVg':
        server = BapFL_Plus_LGFedAvg_SERVER(config=config)
    elif config.algorithm == 'Gen_BapFL_LGFedAvg':
        server = Gen_BapFL_LGFedAvg_SERVER(config=config)
    server.federate()
