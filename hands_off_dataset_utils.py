
import torch
import legacy
import dnnlib


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_style_gan(network_pkl: str):
    """

    :return:
    """
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    return G

def save_weights(generator,path):
    """

    Args:
        genrator:

    Returns:

    """
    torch.save({
        'g_ema': generator.state_dict(),
    }, path)

if __name__ == '__main__':
    G = load_style_gan("/mnt/d/projects/hands-off-dataset/pretrained_models/style_gan_models/network-snapshot-001000.pkl")
    save_weights(G,"/mnt/d/projects/hands-off-dataset/pretrained_models/style_gan_models/network-snapshot-001000_official_generator.pt")
