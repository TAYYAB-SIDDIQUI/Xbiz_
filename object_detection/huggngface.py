import torch

# Load the full model directly
model = torch.load('best.pt')
model.eval()

# Example usage
sample_input = torch.randn(1, 10)  # Update based on expected input
output = model(sample_input)
print(output)

