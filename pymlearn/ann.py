from autograd import elementwise_grad as egrad, numpy as anp
from .tools import plot_history, visualize_network

class ANN:
    def __init__(self, nodes, activations, loss):
        self.layers = []
        self.__layers = []
        self.__info = [nodes, activations]
        self.__activations = {'linear': self.__linear, 'relu': self.__relu, 'sigmoid': self.__sigmoid}
        self.__loss = {'mse': self.__mse, 'bce': self.__bce}[loss]
        self.__optimizers = {'sgd': self.__sgd, 'momentum': self.__momentum, 'adadelta': self.__adadelta, 'adam': self.__adam, 'radam': self.__radam}
        self.__colors = {'loss': 'C0', 'accuracy': 'C1'}
        self.__linestyles = {'train': '-', 'test': '--'}
        for i, a in enumerate(activations):
            self.layers.append({'w': anp.random.uniform(-1, 1, [nodes[i], nodes[i+1]]), 'b': anp.zeros(nodes[i+1]), 'a': a})
            self.__layers.append(0)
    
    def __linear(self, z):
        return z
    
    def __relu(self, z):
        return anp.maximum(0, z)
    
    def __sigmoid(self, z):
        return 1 / (1 + anp.exp(-z))
    
    def __mse(self, p, y):
        return (y - p) ** 2
    
    def __bce(self, p, y):
        return -(y * anp.log(p + self.__params[-1]['eps']) + (1 - y) * anp.log(1 - p + self.__params[-1]['eps']))
    
    def __accuracy(self, p, y):
        return anp.mean(y == p)
    
    def __forward(self, x):
        for i, layer in enumerate(self.layers):
            z = anp.dot(x, layer['w']) + layer['b']
            y = self.__activations[layer['a']](z)
            x = y
            self.__layers[i] = {'y': y, 'z': z}
    
    def __backward(self, x, y):
        p = self.predict(x)
        dy = egrad(self.__loss)(p, y) / x.shape[0]
        for i in range(len(self.layers))[::-1]:
            if i == 0:
                y_prev = x
            else:
                y_prev = self.__layers[i-1]['y']
            dz = anp.multiply(dy, egrad(self.__activations[self.layers[i]['a']])(self.__layers[i]['z']))
            dw = anp.dot(y_prev.T, dz)
            db = anp.sum(dz, axis=0)
            dy = anp.dot(dz, self.layers[i]['w'].T)
            self.__optimizer(i, dw, db)
    
    def __sgd(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self.layers[i][c] -= self.__params[-1]['alpha'] * d
    
    def __momentum(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self.__params[i][f'momentum_v{c}'] = self.__params[-1]['beta'] * self.__params[i][f'momentum_v{c}'] + self.__params[-1]['alpha'] * d
            self.layers[i][c] -= self.__params[i][f'momentum_v{c}']
    
    def __adadelta(self, i, dw, db):
        for c, d in zip(['w', 'b'], [dw, db]):
            self.__params[i][f'adadelta_E{c}'] = self.__params[-1]['gamma'] * self.__params[i][f'adadelta_E{c}'] + (1 - self.__params[-1]['gamma']) * d ** 2
            self.__params[i][f'adadelta_a{c}'] = self.__params[-1]['gamma'] * self.__params[i][f'adadelta_a{c}'] + (1 - self.__params[-1]['gamma']) * self.__params[i][f'adadelta_d{c}'] ** 2
            self.__params[i][f'adadelta_d{c}'] = anp.sqrt(self.__params[i][f'adadelta_a{c}'] + self.__params[-1]['eps']) / anp.sqrt(self.__params[i][f'adadelta_E{c}'] + self.__params[-1]['eps']) * d
            self.layers[i][c] -= self.__params[i][f'adadelta_d{c}']
    
    def __adam(self, i, dw, db):
        t = i + 1
        for c, d in zip(['w', 'b'], [dw, db]):
            self.__params[i][f'adam_m{c}'] = self.__params[-1]['beta'] * self.__params[i][f'adam_m{c}'] + (1 - self.__params[-1]['beta']) * d
            self.__params[i][f'adam_v{c}'] = self.__params[-1]['beta2'] * self.__params[i][f'adam_v{c}'] + (1 - self.__params[-1]['beta2']) * d ** 2
            m_hat = self.__params[i][f'adam_m{c}'] / (1 - anp.power(self.__params[-1]['beta'], t))
            v_hat = anp.sqrt(self.__params[i][f'adam_v{c}'] / (1 - anp.power(self.__params[-1]['beta2'], t))) + self.__params[-1]['eps']
            self.layers[i][c] -= self.__params[-1]['alpha'] * m_hat / v_hat
    
    def __radam(self, i, dw, db):
        t = i + 1
        for c, d in zip(['w', 'b'], [dw, db]):
            self.__params[i][f'radam_m{c}'] = self.__params[-1]['beta'] * self.__params[i][f'radam_m{c}'] + (1 - self.__params[-1]['beta']) * d
            self.__params[i][f'radam_v{c}'] = self.__params[-1]['beta2'] * self.__params[i][f'radam_v{c}'] + (1 - self.__params[-1]['beta2']) * d ** 2
            m_hat = self.__params[i][f'radam_m{c}'] / (1 - anp.power(self.__params[-1]['beta'], t))
            p = self.__params[-1]['p_inf'] - 2 * t * anp.power(self.__params[-1]['beta2'], t) / (1 - anp.power(self.__params[-1]['beta2'], t))
            if p > 4:
                v_hat = anp.sqrt(self.__params[i][f'radam_v{c}'] / (1 - anp.power(self.__params[-1]['beta2'], t))) + self.__params[-1]['eps']
                r = anp.sqrt((p - 4) * (p - 2) * self.__params[-1]['p_inf'] / (self.__params[-1]['p_inf'] - 4) * (self.__params[-1]['p_inf'] - 2) * p)
                self.layers[i][c] -= self.__params[-1]['alpha'] * r * m_hat / v_hat
            else:
                self.layers[i][c] -= self.__params[-1]['alpha'] * m_hat
    
    def __save_metrics(self, x_train, y_train, x_test, y_test):
        self.history['train loss'].append(anp.mean(self.__loss(self.predict(x_train), y_train)))
        self.history['test loss'].append(anp.mean(self.__loss(self.predict(x_test), y_test)))
        self.history['train accuracy'].append(self.__accuracy(self.predict(x_train, True), y_train))
        self.history['test accuracy'].append(self.__accuracy(self.predict(x_test, True), y_test))
    
    def __display_metrics(self, i, metrics):
        if type(metrics) == str:
            metrics = [metrics]
        string = str(i)
        for metric in metrics:
            for k, v in self.history.items():
                if metric in k:
                    string += ' ' + k + ': ' + str(round(self.history[k][i], 6))
        print(string)
    
    def fit(self, x_train, y_train, x_test, y_test, epochs, optimizer='sgd', alpha=0.1, beta=0.9, beta2=0.999, gamma=0.9, verbose=None):
        self.__optimizer = self.__optimizers[optimizer]
        self.__params = [{'eps': 1e-7, 'alpha': alpha, 'beta': beta, 'beta2': beta2, 'gamma': gamma, 'p_inf': 2 / (1 - beta2) - 1}]
        for i in range(len(self.layers)):
            self.__params.insert(0, {'momentum_vw': 0, 'momentum_vb': 0,
                                     'adadelta_Ew': 0, 'adadelta_Eb': 0, 'adadelta_aw': 0, 'adadelta_ab': 0, 'adadelta_dw': 0, 'adadelta_db': 0,
                                     'adam_mw': 0, 'adam_mb': 0, 'adam_vw': 0, 'adam_vb': 0,
                                     'radam_mw': 0, 'radam_mb': 0, 'radam_vw': 0, 'radam_vb': 0})
        self.history = {'train loss': [], 'test loss': [], 'train accuracy': [], 'test accuracy': []}
        for i in range(epochs):
            self.__backward(x_train, y_train)
            self.__save_metrics(x_train, y_train, x_test, y_test)
            if verbose:
                self.__display_metrics(i, verbose)
    
    def predict(self, x, categorical=False):
        self.__forward(x)
        p = self.__layers[-1]['y']
        if categorical:
            if p.shape[1] == 1:
                return anp.round(p)
            else:
                p_cat = anp.zeros_like(p)
                p_cat[anp.arange(p.shape[0]), anp.argmax(p, axis=1)] = 1
                return p_cat
        else:
            return self.__layers[-1]['y']
    
    def plot(self, metric=None, save=False):
        plot_history(self.history, metric, save)
    
    def visualize(self, save=False):
        visualize_network(self.__info[0], self.__info[1], save)
