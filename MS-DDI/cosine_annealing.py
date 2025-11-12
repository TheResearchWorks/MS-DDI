import math

def cosine_annealing_with_warmup(epoch, total_epochs=100, warmup_epochs=5, 
                              initial_lr=0.0001, min_lr=1e-6):

    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * initial_lr
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine_decay
