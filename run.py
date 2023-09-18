from utils.utils import parse_args
from algorithm.fedper.server import SERVER as FedPer_SERVER
from algorithm.bapfl.server import SERVER as BapFL_SERVER

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        import wandb

        run = wandb.init(project="Your Project Name", entity="Your Identity")
        wandb.watch_called = False
        config = wandb.config
        config.update(args)
    else:
        config = args
    server = None
    if config.algorithm == 'blackbox':
        server = FedPer_SERVER(config=config)
    elif config.algorithm == 'bapfl':
        server = BapFL_SERVER(config=config)
    print("Attacker population: ", server.attackers)
    server.federate()
