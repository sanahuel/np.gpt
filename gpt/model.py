import numpy as np

def softmax(x, derivative=False):
    exps = np.exp(x - x.max(axis=-1, keepdims=True))
    if derivative: 
        return exps / np.sum(exps, axis=-1, keepdims=True) * (1 - exps / np.sum(exps, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def relu(x, derivative = False):
    if derivative: return np.where(x>0, 1, 0)
    return np.maximum(0, x)

class TokenEmbedding:
    """Converts token IDs to embeddings and adds positional encoding"""
    def __init__(self, vocab_size, d_model, max_seq_len, lr=0.01):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Cache for backpropagation
        self.cache = None

        # Learning rate
        self.lr = lr

        # Token embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding
        position = np.arange(self.max_seq_len)[:, np.newaxis] #(max_seq_length, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)) #(d_model/2)
        encoding = np.zeros((self.max_seq_len, self.d_model))
        encoding[:, 0::2] = np.sin(position * div_term) #Even positions -> sine
        encoding[:, 1::2] = np.cos(position * div_term) #Odd positions -> cosine
        self.position_encoding = encoding #(max_seq_length, d_model)
    
    def forward(self, tokens: np.ndarray) -> np.ndarray:
        # Tokens shape: (batch_size, seq_length)
        # Output shape: (batch_size, seq_length, d_model)
        # Token embedding -> look up operation based on token ID
        # Positional encoding -> add positional information up to seq_lenght
        self.cache = tokens
        return self.embedding[tokens] + self.position_encoding[:tokens.shape[1]]
    
    def backward(self, delta):
        # delta shape: (batch_size, seq_length, d_model)
        # Output shape: (vocab_size, d_model)
        # self.cache contains the token IDs used in the forward pass
        dEmbeddings = np.zeros_like(self.embedding)

        # flatten cache -> from (batch_size, seq_length) to 1D list of IDs
        # reshape delta -> from (batch_size, seq_length, d_model) to 2D matrix, each row corresponding with an ID
        # dEmbedding[token ID] += delta[token ID]
        np.add.at(dEmbeddings, self.cache.flatten(), delta.reshape(-1, self.d_model))

        self.update_weights(dEmbeddings)
    
    def update_weights(self, dEmbeddings):
        self.embedding -= self.lr * dEmbeddings

class MultiHeadAttention:
    """Implements multi-head self-attention"""
    def __init__(self, d_model, num_heads, lr=0.01):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Cache for backpropagation
        self.cache = None
        self.cache_Q = None
        self.cache_K = None
        self.cache_V = None
        self.cache_attention_weights = None
        self.cache_attention_output = None

        # Learning rate
        self.lr = lr

        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_out = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask):
        # x shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)
        self.cache = x
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Spliting Q, K, V into multiple heads -> From (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_head)
        # Transpose -> To perform self-attention across the heads dimension (batch_size, num_heads, seq_len, d_head)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        self.cache_Q = Q
        self.cache_K = K
        self.cache_V = V

        # Scaled dot product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        if mask is not None: scores += mask
        weights = softmax(scores)
        self.cache_attention_weights = weights

        att_output = weights @ V

        # Restore original shape
        att_output = att_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        self.cache_attention_output = att_output

        # Linear projection
        return att_output @ self.W_out
    
    def backward(self, delta):
        # delta shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)
        batch_size, seq_len, _ = self.cache.shape

        # First gradient for W_out
        dW_out = np.einsum("ijk,ijl->kl", self.cache_attention_output, delta)

        # Then backprop through the layer
        delta_out = delta @ self.W_out.T
        delta_out = delta_out.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # Backprop through the attention layer
        d_att_weights = delta_out @ self.cache_V.transpose(0, 1, 3, 2) # dOut -> dWeights
        sum_term = np.sum(self.cache_attention_weights * d_att_weights, axis=-1, keepdims=True)
        d_scores = self.cache_attention_weights * (d_att_weights - sum_term) / np.sqrt(self.d_head)        

        dQ = d_scores @ self.cache_K#.transpose(0, 1, 3, 2)
        dK = d_scores.transpose(0, 1, 3, 2) @ self.cache_Q
        dV = self.cache_attention_weights.transpose(0, 1, 3, 2) @ delta_out 

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        dW_q = np.einsum("ijk,ijl->kl", self.cache, dQ)
        dW_k = np.einsum("ijk,ijl->kl", self.cache, dK)
        dW_v = np.einsum("ijk,ijl->kl", self.cache, dV)

        dx = np.einsum("ijl,lk->ijk", dQ, self.W_q) + np.einsum("ijl,lk->ijk", dK, self.W_k) + np.einsum("ijl,lk->ijk", dV, self.W_v)

        self.update_weights(dW_q, dW_k, dW_v, dW_out)

        return dx
    
    def update_weights(self, dW_q, dW_k, dW_v, dW_out):
        self.W_q -= self.lr * dW_q
        self.W_k -= self.lr * dW_k
        self.W_v -= self.lr * dW_v
        self.W_out -= self.lr * dW_out

class FeedForward:
    """Implements the feed-forward network after attention"""
    def __init__(self, d_model, d_hidden, lr=0.01):
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Cache for backpropagation
        self.cache = None
        self.cache_act = None
        self.cache_hidden = None

        # Learning rate
        self.lr = lr

        self.W_1 = np.random.randn(d_model, d_hidden) * 0.02
        self.b_1 = np.zeros(d_hidden)
        self.W_2 = np.random.randn(d_hidden, d_model) * 0.02
        self.b_2 = np.zeros(d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)
        self.cache = x
        x = x @ self.W_1 + self.b_1
        self.cache_act = x
        x = relu(x)
        self.cache_hidden = x
        x = x @ self.W_2 + self.b_2
        return x

    def backward(self, delta):
        batch_size, seq_len, _ = delta.shape
        
        # Reshape -> Flatten batch size dimension
        delta_reshaped = delta.reshape(-1, self.d_model) # (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        hidden_reshaped = self.cache_hidden.reshape(-1, self.d_hidden) # (batch_size, seq_len, d_hidden) -> (batch_size * seq_len, d_hidden)
        hidden_act_reshaped = self.cache_act.reshape(-1, self.d_hidden) # (batch_size, seq_len, d_hidden) -> (batch_size * seq_len, d_hidden)
        cache_reshaped = self.cache.reshape(-1, self.d_model) # (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        
        # 2nd layer
        dW2 = hidden_reshaped.T @ delta_reshaped
        db2 = np.sum(delta_reshaped, axis=0)
        dx = delta_reshaped @ self.W_2.T
        
        # ReLU
        dx = dx * relu(hidden_act_reshaped, derivative=True)  
        
        # 1st layer
        dW1 = cache_reshaped.T @ dx
        db1 = np.sum(dx, axis=0)
        dx = (dx @ self.W_1.T).reshape(batch_size, seq_len, self.d_model) # Reshape to original shape
                
        self.update_weights(dW1, db1, dW2, db2)
        
        return dx
    
    def update_weights(self, dW1, db1, dW2, db2):
        self.W_1 -= self.lr * dW1
        self.W_2 -= self.lr * dW2
        self.b_1 -= self.lr * db1
        self.b_2 -= self.lr * db2


    
class LayerNorm:
    """Implements Layer Normalization."""
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.cache = None
        self.mean = None
        self.variance = None
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)
        self.cache = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - self.mean) / np.sqrt(self.variance + self.eps)        
        return x_norm
    
    def backward(self, delta):
        # delta shape: (batch_size, seq_length, d_model)
        # no trainable parameters -> just backpropating delta through the normalization operation
        m = delta.shape[-1] # number of features in the embedding dimension d_model
        dx_norm = delta / np.sqrt(self.variance + self.eps)
        dvariance = np.sum(delta * (self.cache - self.mean) * -0.5 * (self.variance + self.eps) ** -1.5, axis=-1, keepdims=True)
        dmean = np.sum(-dx_norm, axis=-1, keepdims=True) + dvariance * np.sum(-2 * (self.cache - self.mean), axis=-1, keepdims=True) / m
        return (dx_norm + (dvariance*2*(self.cache - self.mean)/m) + (dmean/m))
        
class TransformerBlock:
    """Implements a Transformer block"""
    def __init__(self, d_model, num_heads, d_ff, lr=0.01):
        self.mh_att = MultiHeadAttention(d_model, num_heads, lr)
        self.ff = FeedForward(d_model, d_ff, lr)
        self.layer_norm1 = LayerNorm()
        self.layer_norm2 = LayerNorm()

    def forward(self, x, mask):
        # x shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)
        skip = x
        x = self.mh_att.forward(x, mask)
        x = self.layer_norm1.forward(x + skip)
        skip = x
        x = self.ff.forward(x)
        x = self.layer_norm2.forward(x + skip)
        return x
    
    def backward(self, delta):
        # delta shape: (batch_size, seq_length, d_model)
        # output shape: (batch_size, seq_length, d_model)

        # 2nd Layer Norm
        delta_norm2 = self.layer_norm2.backward(delta)
        delta_skip2 = delta_norm2
        
        # Feed Forward
        delta_ff = self.ff.backward(delta_norm2)
        
        # 1sr Layer Norm (+ 2nd skip connection delta)
        delta_norm1 = self.layer_norm1.backward(delta_ff + delta_skip2)
        delta_skip1 = delta_norm1
        
        # Multi Head Attention
        delta_att = self.mh_att.backward(delta_norm1)
        
        # + 1st skip connection delta
        dx = delta_att + delta_skip1
        
        return dx

    
class GPT:
    """Trainable GPT model"""
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads, d_ff, num_blocks, lr=0.01):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_blocks = num_blocks

        # Cache for backpropagation
        self.cache = None

        # Learning rate
        self.lr = lr
        
        # Components
        self.embeddings = TokenEmbedding(vocab_size, d_model, max_seq_len, lr)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, d_ff, lr) for _ in range(num_blocks)]
        self.layer_norm = LayerNorm()
        self.linear_out = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length)
        # output shape: (batch_size, seq_length, vocab_size)
        x = self.embeddings.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x, mask)
        x = self.layer_norm.forward(x)

        self.cache = x

        return softmax(x @ self.linear_out)
    
    def backward(self, probs, y):
        # probs shape: (batch_size, seq_length, vocab_size)
        # y shape: (batch_size, seq_length)

        delta = probs - y

        # First through linear layer
        dLinear = np.einsum("ijk,ijl->kl", self.cache, delta)   # Gradient for linear layer
        delta = delta @ self.linear_out.T                       # Backpropagate through layer
        
        delta = self.layer_norm.backward(delta)
        
        for block in reversed(self.transformer_blocks):
            delta = block.backward(delta)
            
        self.embeddings.backward(delta)
        
        # Update linear layer
        self.linear_out -= self.lr * dLinear

    def train(self, train_data, num_epochs, batch_size, lr=None):
        # train_data -> array of (X,Y) sequences of tokens IDs
        # Arrange batch -> Create mask -> One-hot encode Y -> Forward -> Backward
        print(f"Training for {num_epochs} epochs...")
        if lr: self.lr = lr

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle every epoch
            np.random.shuffle(train_data)            
            # Split training data in batches

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]

                if len(batch) < batch_size: continue # If len(train_data) % batch_size != 0 -> skip last batch
                    
                x_batch = np.array([x for x, _ in batch])
                y_batch = np.array([y for _, y in batch])

                # Mask -> model only considers past tokens
                seq_length = x_batch.shape[1]
                mask = np.tril(np.ones((seq_length, seq_length))) * -1e9

                # Y from Token IDs -> one-hot encoding
                y_onehot = np.zeros((y_batch.shape[0], y_batch.shape[1], self.vocab_size))
                for i in range(y_batch.shape[0]):
                    for j in range(y_batch.shape[1]):
                        y_onehot[i, j, y_batch[i, j]] = 1

                # Forward
                #probs = self.forward(x_batch, mask)
                probs = self.forward(x_batch)

                # Cross-entropy loss
                loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / (y_batch.shape[0] * y_batch.shape[1])

                # Backward pass
                self.backward(probs, y_onehot)

                total_loss += loss
                num_batches += 1
                
                # Print progress
                if num_batches % 500 == 0:
                    print(f"    Epoch [{epoch + 1}], Batch {num_batches}, Loss: {loss:.4f}")
            
            # Print epoch summary
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}] completed. Average loss: {avg_loss:.4f}")
    

    def generate(self, context, max_length, temperature=1.0):
        tokens = context.copy()

        for _ in range(max_length):
            # Create attention mask -> only past tokens are considered
            seq_length = tokens.shape[1]
            mask = np.tril(np.ones((seq_length, seq_length))) * -1e9

            # Get logits without softmax
            x = self.embeddings.forward(tokens)
            for block in self.transformer_blocks:
                x = block.forward(x, mask)
            x = self.layer_norm.forward(x)
            logits = x @ self.linear_out
            
            # Apply temperature and softmax only once
            next_token_logits = logits[:, -1, :] / temperature
            probs = softmax(next_token_logits)
            
            # Sample next token -> Random choice based on probability distribution
            next_token = np.random.choice(self.vocab_size, p=probs[0])

            # Append -> Now this token is part of the context for next iteration
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)

        return tokens
    


######### TESTS ##########

# # Test parameters
# vocab_size = 1000
# d_model = 64
# max_seq_len = 20
# num_heads = 4
# d_ff = 128
# num_blocks = 2
# batch_size = 4
# sequence_length = 10

# # Initialize model
# model = GPT(vocab_size=vocab_size, 
#             d_model=d_model,
#             max_seq_len=max_seq_len,
#             num_heads=num_heads,
#             d_ff=d_ff,
#             num_blocks=num_blocks)

# # Test forward pass
# print("Testing forward pass...")
# x = np.random.randint(0, vocab_size, (batch_size, sequence_length))
# output = model.forward(x)
# assert output.shape == (batch_size, sequence_length, vocab_size), f"Expected shape {(batch_size, sequence_length, vocab_size)}, got {output.shape}"
# print(" Output shape OK")
# assert np.allclose(np.sum(output, axis=-1), 1), "Softmax outputs should sum to 1"
# print(" Softmax OK")

# # Test backward pass
# print("Testing backward pass...")
# y = np.random.randint(0, vocab_size, (batch_size, sequence_length))
# y_onehot = np.zeros((batch_size, sequence_length, vocab_size))
# for i in range(batch_size):
#     for j in range(sequence_length):
#         y_onehot[i, j, y[i, j]] = 1
# model.backward(output, y_onehot)

# # Test train method
# print("Testing train method...")
# train_data = []
# num_sequences = 20
# for _ in range(num_sequences):
#     x = np.random.randint(0, vocab_size, (sequence_length,))
#     y = np.roll(x, -1)  # Next token prediction
#     train_data.append((x, y))  # Directly append x and y

# try:
#     model.train(train_data, num_epochs=10, batch_size=2)
#     print("Training test passed")
# except Exception as e:
#     print(f"Training test failed with error: {str(e)}")

# # Test generate method
# print("Testing generate method...")
# context = np.random.randint(0, vocab_size, (1, 5))  # Start with 5 tokens
# generated = model.generate(context, max_length=10, temperature=0.8)
# print(f"Generated sequence: {generated}")
# assert generated.shape == (1, 15), f"Expected shape (1, 15), got {generated.shape}"
# assert np.array_equal(generated[:, :5], context), "Generated sequence should contain original context"

# print("All tests completed")

# def gradient_check_embedding():
#     # Setup
#     vocab_size, d_model, max_seq_len = 10, 8, 5
#     np.random.seed(42)
#     tokens = np.random.randint(0, vocab_size, (2, 3))
#     deltas = np.random.randn(2, 3, d_model)  # Random delta for backward pass
#     epsilon = 1e-5

#     # Initialize embedding layer
#     emb = TokenEmbedding(vocab_size, d_model, max_seq_len)

#     # Forward pass
#     output = emb.forward(tokens)

#     # Backward pass
#     emb.backward(deltas)
#     analytical_gradient = emb.embedding

#     # Numerical gradient
#     num_grad = np.zeros_like(analytical_gradient)
#     for i in range(vocab_size):
#         for j in range(d_model):
#             # +epsilon perturbation
#             emb.embedding[i, j] += epsilon
#             loss_plus = np.sum((emb.forward(tokens) - deltas)**2)
#             emb.embedding[i, j] -= epsilon

#             # -epsilon perturbation
#             emb.embedding[i, j] -= epsilon
#             loss_minus = np.sum((emb.forward(tokens) - deltas)**2)
#             emb.embedding[i, j] += epsilon

#             # Numerical gradient
#             num_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

#     # Compare gradients
#     relative_error = np.abs(analytical_gradient - num_grad) / (np.abs(analytical_gradient) + np.abs(num_grad) + 1e-12)
#     assert np.allclose(analytical_gradient, num_grad, atol=1e-5), f"Gradient check failed. Max relative error: {np.max(relative_error)}"
#     print("TokenEmbedding gradient check passed.")

# gradient_check_embedding()