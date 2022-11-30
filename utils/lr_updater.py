
class LrUpdater:
    def __init__(self, mode, optimizer, power=0.9, lr_decay=0.5, step_interval=None, step_every=None, min_lr=0):
        self.mode = mode
        self.base_lr = []
        # poly-related
        self.power = power
        # step-related
        self.lr_decay = lr_decay
        self.step_interval = step_interval
        self.step_every= step_every
    
        self.min_lr = min_lr
        
        for group in optimizer.param_groups: 
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]

    def adjust_lr_poly(self, optimizer, cur_it, its):
        for i, g in  enumerate(optimizer.param_groups):
            coeff = (1 - cur_it / its)**self.power
            _lr = (self.base_lr[i] - self.min_lr) * coeff + self.min_lr
            g['lr'] = _lr

    def adjust_lr_step(self, optimizer, cur_it, its):
        for i, g in enumerate(optimizer.param_groups):
            if self.step_every is not None:
                power = cur_it//self.step_every
            if self.step_interval is not None and cur_it in self.step_interval:
                power = self.step_interval.index(cur_it) + 1

            _lr = self.base_lr[i] * (self.lr_decay)**(power)

            g['lr'] = _lr

    def adjust_lr_fixed(self, optimizer, cur_it, its):
    
        for g in optimizer.param_groups:
            g['lr'] = self.base_lr