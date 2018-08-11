import torch
import torch.nn.modules.loss


def weighted_binary_cross_entropy(output, target, pos_weight):
    # eps = 1e-8  # 1 - eps -> big eats small
    # output = torch.clamp(output, min=eps, max=1-eps)
    if pos_weight is not None:
        loss = (target * torch.sigmoid(output).log()) * pos_weight + \
               ((1 - target) * torch.sigmoid(1 - output).log())
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    loss = torch.neg(torch.mean(loss))
    return loss


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * torch.mean(weighted_binary_cross_entropy(preds, labels, pos_weight))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
