from autograd import elementwise_grad as egrad, numpy as anp
from .tools import plot_history, visualize_network

class ANN:
    def __init__(self, nodes, activations, loss):
        self.layers = []
        self._layers = []
        self._info = [nodes, activations]
        self._activations = {'linear': self._linear, 'relu': self._relu, 'sigmoid': self._sigmoid}
        self._loss = {'mse': self._mse, 'bce': self._bce}[loss]
        self._optimizers = {'sgd': self._sgd, 'momentum': self._momentum, 'adadelta': self._adadelta, 'adam': self._adam, 'radam': self._radam}
        self._colors = {'loss': 'C0', 'accuracy': 'C1'}
        self._linestyles = {'train': '-', 'test': '--'}
        for i, a in enumerate(activations):
            self.layers.append({'w': anp.random.uniform(-1, 1, [nodes[i], nodes[i+1]]), 'b': anp.zeros(nodes[i+1]), 'a': a})
            self._layers.append(0)
    
    def _linear(self, z):
        return z
    
    def _relu(self, z):
        return anp.maximum(0, z)
    
    def _sigmoid(self, z):
        return 1 / (1 + anp.exp(-z))
    
    def _mse(self, p, y):
        return (y - p) ** 2
    
    def _bce(self, p, y):
        return -(y * anp.log(p + self._params[-1]['eps']) + (1 - y) * anp.log(1 - p + self._params[-1]['eps']))
    
    def _accuracy(self, p, y):
        return anp.mean(y == p)
    
    def _forward(self, x):
        for i, layer in enumerate(self.layers):
            z = anp.dot(x, layer['w']) + layer['b']
            y = self._activations[layer['a']](z)
            x = y
            self._layers[i] = {'y': y, 'z': z}
    
    def _backward(self, x, y):
        p = self.predict(x)
        dy = egrad(self._loss)(p, y) / x.shape[0]
        for i in range(len(self.layers))[::-1]:
            if i == 0:
                y_prev = x
            else:
                y_prev = self._layers[i-1]['y']
            dz = anp.multiply(dy, egrad(self._activations[self.layers[i]['a']])(self._layers[i]['z']))
            dw = anp.dot(y_prev.T, dz)
            db = anp.sum(dz, axis=0)
            dy = anp.dot(dz, self.layers[i]['w'].T)
            self._optimizer(i, dw, db)
    
    def _sgd(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self.layers[i][c] -= self._params[-1]['alpha'] * d
    
    def _momentum(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self._params[i][f'momentum_v{c}'] = self._params[-1]['beta'] * self._params[i][f'momentum_v{c}'] + self._params[-1]['alpha'] * d
            self.layers[i][c] -= self._params[i][f'momentum_v{c}']
    
    def _adadelta(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self._params[i][f'adadelta_E{c}'] = self._params[-1]['gamma'] * self._params[i][f'adadelta_E{c}'] + (1 - self._params[-1]['gamma']) * d ** 2
            self._params[i][f'adadelta_a{c}'] = self._params[-1]['gamma'] * self._params[i][f'adadelta_a{c}'] + (1 - self._params[-1]['gamma']) * self._params[i][f'adadelta_d{c}'] ** 2
            self._params[i][f'adadelta_d{c}'] = anp.sqrt(self._params[i][f'adadelta_a{c}'] + self._params[-1]['eps']) / anp.sqrt(self._params[i][f'adadelta_E{c}'] + self._params[-1]['eps']) * d
            self.layers[i][c] -= self._params[i][f'adadelta_d{c}']
    
    def _adam(self, i, dw, db):
        t = i + 1
        for c, d in zip(['w', 'b'], [dw, db]):
            self._params[i][f'adam_m{c}'] = self._params[-1]['beta'] * self._params[i][f'adam_m{c}'] + (1 - self._params[-1]['beta']) * d
            self._params[i][f'adam_v{c}'] = self._params[-1]['beta2'] * self._params[i][f'adam_v{c}'] + (1 - self._params[-1]['beta2']) * d ** 2
            m_hat = self._params[i][f'adam_m{c}'] / (1 - anp.power(self._params[-1]['beta'], t))
            v_hat = anp.sqrt(self._params[i][f'adam_v{c}'] / (1 - anp.power(self._params[-1]['beta2'], t))) + self._params[-1]['eps']
            self.layers[i][c] -= self._params[-1]['alpha'] * m_hat / v_hat
    
    def _radam(self, i, dw, db):
        t = i + 1
        for c, d in zip(['w', 'b'], [dw, db]):
            self._params[i][f'radam_m{c}'] = self._params[-1]['beta'] * self._params[i][f'radam_m{c}'] + (1 - self._params[-1]['beta']) * d
            self._params[i][f'radam_v{c}'] = self._params[-1]['beta2'] * self._params[i][f'radam_v{c}'] + (1 - self._params[-1]['beta2']) * d ** 2
            m_hat = self._params[i][f'radam_m{c}'] / (1 - anp.power(self._params[-1]['beta'], t))
            p = self._params[-1]['p_inf'] - 2 * t * anp.power(self._params[-1]['beta2'], t) / (1 - anp.power(self._params[-1]['beta2'], t))
            if p > 4:
                v_hat = anp.sqrt(self._params[i][f'radam_v{c}'] / (1 - anp.power(self._params[-1]['beta2'], t))) + self._params[-1]['eps']
                r = anp.sqrt((p - 4) * (p - 2) * self._params[-1]['p_inf'] / (self._params[-1]['p_inf'] - 4) * (self._params[-1]['p_inf'] - 2) * p)
                self.layers[i][c] -= self._params[-1]['alpha'] * r * m_hat / v_hat
            else:
                self.layers[i][c] -= self._params[-1]['alpha'] * m_hat
    
    def _save_metrics(self, x_train, y_train, x_test, y_test):
        self.history['train loss'].append(anp.mean(self._loss(self.predict(x_train), y_train)))
        self.history['test loss'].append(anp.mean(self._loss(self.predict(x_test), y_test)))
        self.history['train accuracy'].append(self._accuracy(self.predict(x_train, True), y_train))
        self.history['test accuracy'].append(self._accuracy(self.predict(x_test, True), y_test))
    
    def _display_metrics(self, i, metrics):
        if type(metrics) == str:
            metrics = [metrics]
        string = str(i)
        for metric in metrics:
            for k, v in self.history.items():
                if metric in k:
                    string += ' ' + k + ': ' + str(round(self.history[k][i], 6))
        print(string)
    
    def fit(self, x_train, y_train, x_test, y_test, epochs, optimizer='sgd', alpha=0.1, beta=0.9, beta2=0.999, gamma=0.9, verbose=None):
        self._optimizer = self._optimizers[optimizer]
        self._params = [{'eps': 1e-7, 'alpha': alpha, 'beta': beta, 'beta2': beta2, 'gamma': gamma, 'p_inf': 2 / (1 - beta2) - 1}]
        for i in range(len(self.layers)):
            self._params.insert(0, {'momentum_vw': 0, 'momentum_vb': 0,
                                     'adadelta_Ew': 0, 'adadelta_Eb': 0, 'adadelta_aw': 0, 'adadelta_ab': 0, 'adadelta_dw': 0, 'adadelta_db': 0,
                                     'adam_mw': 0, 'adam_mb': 0, 'adam_vw': 0, 'adam_vb': 0,
                                     'radam_mw': 0, 'radam_mb': 0, 'radam_vw': 0, 'radam_vb': 0})
        self.history = {'train loss': [], 'test loss': [], 'train accuracy': [], 'test accuracy': []}
        for i in range(epochs):
            self._backward(x_train, y_train)
            self._save_metrics(x_train, y_train, x_test, y_test)
            if verbose:
                self._display_metrics(i, verbose)
    
    def predict(self, x, categorical=False):
        self._forward(x)
        p = self._layers[-1]['y']
        if categorical:
            if p.shape[1] == 1:
                return anp.round(p)
            else:
                p_cat = anp.zeros_like(p)
                p_cat[anp.arange(p.shape[0]), anp.argmax(p, axis=1)] = 1
                return p_cat
        else:
            return self._layers[-1]['y']
    
    def plot(self, metric=None, save=False):
        plot_history(self.history, metric, save)
    
    def visualize(self, save=False):
        visualize_network(self._info[0], self._info[1], save)
