class Styler:
    def __init__(self):
        self.colors = {'primary': 'C0', 'secondary': 'C1', 'legend': 'C7'}
        self.linestyles = {'train': '-', 'test': '--'}
    
    def color(self, key):
        if 'loss' in key:
            return self.colors['primary']
        else:
            return self.colors['secondary']
    
    def linestyle(self, key):
        if 'test' in key or 'val' in key:
            return self.linestyles['test']
        else:
            return self.linestyles['train']
