class Styler:
    def __init__(self):
        self.colors = {'loss': 'C0', 'accuracy': 'C1', 'default': 'C7'}
        self.linestyles = {'train': '-', 'test': '--'}
    
    def color(self, key):
        if 'loss' in key:
            return self.colors['loss']
        else:
            return self.colors['accuracy']
    
    def linestyle(self, key):
        if 'test' in key or 'val' in key:
            return self.linestyles['test']
        else:
            return self.linestyles['train']
