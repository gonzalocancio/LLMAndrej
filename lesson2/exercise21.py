import torch 
import torch.nn.functional as F

# Load data

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Training data (trigrams)

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append((ix1, ix2))
        ys.append(ix3)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"Shape of xs: {xs.shape}, Shape of ys: {ys.shape}")

# Initialize parameters for a simple linear model
g = torch.Generator().manual_seed(2147783647)
W = torch.rand((27 * 27, 27), generator=g, requires_grad=True)  # Input size = 27x27

# Gradient descent loop
for k in range(400):
    # Forward pass
    xenc = F.one_hot(xs[:, 0], num_classes=27).float() @ torch.eye(27)  # One-hot encoding for first character
    
    xenc2 = F.one_hot(xs[:, 1], num_classes=27).float() @ torch.eye(27)  # One-hot encoding for second character
    
    xenc = (xenc.unsqueeze(2) @ xenc2.unsqueeze(1)).reshape(-1, 27 * 27)  # Combine the two inputs
    
    logits = xenc @ W  # Logits
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(xs)), ys].log().mean()

    # Print loss
    print(f"Step {k + 1}: Loss = {loss.item()}")

    # Backward pass
    W.grad = None
    loss.backward()

    # Update parameters
    if k < 100:
        W.data -= 50 * W.grad
    elif k < 200:
        W.data -= 30 * W.grad
    elif k < 300:
        W.data -= 15 * W.grad
    else:
        W.data -= 5 * W.grad

def generate_name(W, itos, max_length=20):
    name = []
    context = [0, 0]  # Start with two '.' (index 0) for the trigram model
    
    for _ in range(max_length):
        # One-hot encode the context
        x1 = F.one_hot(torch.tensor([context[-2]]), num_classes=27).float() @ torch.eye(27)
        x2 = F.one_hot(torch.tensor([context[-1]]), num_classes=27).float() @ torch.eye(27)
        xenc = (x1.unsqueeze(2) @ x2.unsqueeze(1)).reshape(-1, 27 * 27)

        # Compute probabilities
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)

        # Sample the next character index
        ix_next = torch.multinomial(probs, num_samples=1).item()
        
        # If the end character is predicted, stop
        if ix_next == 0:
            break
        
        # Append the character to the name
        name.append(itos[ix_next])
        
        # Update the context
        context.append(ix_next)

    return ''.join(name)

for _ in range(10):
    print(generate_name(W, itos, 20))
