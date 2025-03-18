import torch
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

'''
functions used for analysing distributions of the vectors in the embedding matrix.
'''


def get_tokens_from_vectors(embedding_matrix, batch_size, num_batches, do_rms=False, model_norm=None):

    '''
    generates token IDs from randomly sampled vectors based on a given embedding matrix.
    '''

    input_dim = embedding_matrix.shape[1]
    emb_mean = torch.mean(torch.mean(embedding_matrix, dim=0))
    emb_std = torch.mean(torch.std(embedding_matrix, dim=0))
    torch.manual_seed(1234)
    vectors = torch.normal(mean=emb_mean, std=emb_std, size=(num_batches * batch_size, input_dim))

    tokens = []
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i : i + batch_size].cuda()
        if do_rms:
            batch  = model_norm(batch)
        predictions = torch.matmul(batch, embedding_matrix.T)
        token_ids = torch.argmax(predictions, dim=1).tolist()
        tokens.extend(token_ids)
    return tokens

def plot_dist(tokens):
    token_counts = Counter(tokens)
    sorted_tokens = sorted(token_counts.items())
    tokens, counts = zip(*sorted_tokens)

    # plt.figure(figsize=(12, 6))
    plt.bar(tokens, counts, width=1.0, edgecolor="black", color='blue')
    plt.title("Token Distribution")
    plt.xlabel("Token Index")
    plt.ylabel("Frequency")
    plt.show()

    return token_counts

def plot_emb_dist(emb):
    # Convert it into a NumPy array for easier plotting
    numpy_tensor = emb.detach().cpu().numpy()

    # Select a subset of positions to visualize (for simplicity, use every 100th position)
    positions_to_plot = range(0, 4096, 10)  # Change step size depending on how dense you want the plot

    # Prepare the figure
    plt.figure(figsize=(12, 8))

    # Plot the distributions for selected positions
    for pos in positions_to_plot:
        sns.kdeplot(numpy_tensor[:, pos], linewidth=1) # label=f"Position {pos}",

    # Add titles and labels
    plt.title("Distribution of Values at Different Positions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize='small')
    plt.show()