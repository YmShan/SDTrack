class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/users/shanym/SDTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/users/shanym/SDTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/users/shanym/SDTrack/pretrained_models'
        self.eotb_dir_train = '/data/dataset/FE108_3C/train'
        self.visevent_train = '/data/dataset/VisEvent/train/'
        self.felt_train = '/data/dataset/FELT/train/'
