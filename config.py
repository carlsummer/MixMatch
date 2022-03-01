# semi-supervised learning:
# 1. model structure
# 2. hype setting are important!
class Hyperparameters:
    device = "cuda"
    classes_num = 10
    n_labeled = 250
    seed = 1234
    # model
    T = 0.5
    K = 2
    alpha = 0.75
    lambda_u = 75
    # exp
    batch_size = 64
    init_lr = 0.002
    epochs = 1000
    verbose_step = 300
    save_step = 300


HP = Hyperparameters()
