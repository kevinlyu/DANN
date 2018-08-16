import torch
from torch.autograd import Variable
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize(feature_extractor, class_classifier, domain_discriminator, source_dataloader, target_dataloader):

    print("TSNE Processing")

    feature_extractor.eval()
    class_classifier.eval()
    domain_discriminator.eval()

    # random sample source domain
    iterator = iter(source_dataloader)
    source_data, source_label = iterator.next()
    source_data = Variable(source_data.cuda())
    source_tag = Variable(torch.zeros(
        (source_label.size()[0])).type(torch.LongTensor))

    # random sample target domain
    iterator = iter(target_dataloader)
    target_data, target_label = iterator.next()
    target_data = Variable(target_data.cuda())
    target_tag = Variable(torch.ones(
        (target_label.size()[0])).type(torch.LongTensor))

    source_embedding = feature_extractor(source_data)
    target_embedding = feature_extractor(target_data)

    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)

    dann_tsne = tsne.fit_transform(np.concatenate(
        (source_embedding.cpu().detach().numpy(), target_embedding.cpu().detach().numpy())))

    plot_embedding(dann_tsne, np.concatenate((source_label, target_label)), np.concatenate((source_tag, target_tag)), "DANN")

def plot_embedding(X, y, d, title=None):

    # normalize
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X-x_min)/(x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.bwr(d[i]/1.), fontdict={"weight":"bold", "size":9})

        plt.xticks([])
        plt.yticks([])

        if title is not None:
            plt.title(title)
        
        plt.show()
