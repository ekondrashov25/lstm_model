import numpy as np

class LSTMModel:
    def __init__(self, input_size, output_size, hidden_size, learning_rate=1e-1, seq_length=25):
        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # forget gate layer
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1)) # maybe filled with np.ones(()) for better perfomance

        # input gate layer
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))

        # also input gate layer (matricies with updated parameters)
        self.W_c = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))

        # output gate layer
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))

        # for output
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))

        # initialize memory for adam optimization
        self.mem = {k: np.zeros_like(v) for k, v in self.get_params().items()}

    def get_params(self):
        return {
            'W_f': self.W_f, 'b_f': self.b_f,
            'W_i': self.W_i, 'b_i': self.b_i,
            'W_c': self.W_c, 'b_c': self.b_c,
            'W_o': self.W_o, 'b_o': self.b_o,
            'W_hy': self.W_hy, 'b_y': self.b_y,
        }

    def forward(self, inputs, targets, h_prev, C_prev):
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        is_, fs, os, gs = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(C_prev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            x_t = np.zeros((self.input_size, 1))
            x_t[inputs[t]] = 1
            xs[t] = x_t

            concat_vectors = np.vstack((hs[t-1], x_t))

            i = self.sigmoid(self.W_i @ concat_vectors + self.b_i) # input layer
            f = self.sigmoid(self.W_f @ concat_vectors + self.b_f) # forget layer
            o = self.sigmoid(self.W_o @ concat_vectors + self.b_o) # output layer
            C_hat = np.tanh(self.W_c @ concat_vectors + self.b_c)

            is_[t], fs[t], os[t], gs[t] = i, f, o, C_hat

            # # here we add info into cell state
            # cs[t] = f * cs[t-1] + i * C_hat
            # ys[t] = o * np.tanh(cs[t]) # the same thing as h in article

            # ys[t] = self.W_hy.dot(hs[t]) + self.b_y
            # ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            # # loss (negative log-likelihood)
            # loss += -np.log(ps[t][targets[t], 0])

            # here we add info into cell state
            cs[t] = f * cs[t-1] + i * C_hat
            hs[t] = o * np.tanh(cs[t])

            y = self.W_hy.dot(hs[t]) + self.b_y
            ys[t] = y
            ps[t] = np.exp(y) / np.sum(np.exp(y))
            loss += -np.log(ps[t][targets[t], 0])

        cache = (xs, hs, cs, ps, is_, fs, os, gs)
        return loss, cache

    def backward(self, cache, inputs, targets):
        xs, hs, cs, ps, is_, fs, os, gs = cache

        grads = {k: np.zeros_like(v) for k, v in self.get_params().items()}
        dh_next = np.zeros_like(hs[0])
        dc_next = np.zeros_like(cs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            grads["W_hy"] += dy.dot(hs[t].T)
            grads['b_y'] += dy
            dh = self.W_hy.T.dot(dy) + dh_next

            c = cs[t]
            g = gs[t]

            c_prev = cs[t-1]
            h_prev = hs[t-1]

            concat_vectors = np.vstack((h_prev, xs[t]))

            #output gate
            o = os[t]
            do = dh * np.tanh(c)
            do_raw = self.dsigmoid(o) * do
            grads['W_o'] += do_raw.dot(concat_vectors.T)
            grads['b_o'] += do_raw

            # cell state gradients
            dc = dh * o * self.dtanh(np.tanh(c))
            dc += dc_next

            # forget gate gradients
            f = fs[t]
            df = dc * c_prev
            df_raw = self.dsigmoid(f) * df
            grads['W_f'] += df_raw.dot(concat_vectors.T)
            grads['b_f'] += df_raw

            # input gate
            i = is_[t]
            di = dc * g
            di_raw = self.dsigmoid(i) * di
            grads['W_i'] += di_raw.dot(concat_vectors.T)
            grads['b_i'] += di_raw

            # candidate gradients
            dg = dc * i
            dg_raw = self.dtanh(g) * dg
            grads['W_c'] += dg_raw.dot(concat_vectors.T)
            grads['b_c'] += dg_raw

            dconcat = self.W_i.T.dot(di_raw) + self.W_f.T.dot(df_raw) + self.W_o.T.dot(do_raw) + self.W_c.T.dot(dg_raw)

            dh_next = dconcat[:self.hidden_size, :]
            dc_next = f * dc

        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads

    def sample(self, seed_idx, n, h, c, temperature=1.0):
        x = np.zeros((self.input_size, 1))
        x[seed_idx] = 1
        ixes = []

        for _ in range(n):
            concat = np.vstack((h, x))

            i = self.sigmoid(self.W_i @ concat + self.b_i)
            f = self.sigmoid(self.W_f @ concat + self.b_f)
            o = self.sigmoid(self.W_o @ concat + self.b_o)
            g = np.tanh(self.W_c @ concat + self.b_c)

            c = f * c + i * g
            h = o * np.tanh(c)

            y = self.W_hy @ h + self.b_y
            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))

            ix = np.random.choice(range(self.input_size), p=p.ravel())
            x = np.zeros((self.input_size, 1))
            x[ix] = 1
            ixes.append(ix)

        return ixes

    def update_params(self, grads):
        # Adagrad update
        for k, v in self.get_params().items():
            self.mem[k] += grads[k] * grads[k]
            self.get_params()[k] += -self.learning_rate * grads[k] / (np.sqrt(self.mem[k]) + 1e-8)
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(y):
        return y * (1 - y)

    @staticmethod
    def dtanh(y):
        return 1 - y * y