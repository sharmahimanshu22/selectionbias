from matplotlib import pyplot as plt
import io
import torch
import pdb
from PIL import Image
import numpy as np
from DataGen.plots.CIEllipse import CIEllipse
from DataGen.plots import sortedplot as sp


    
    

def VisualizeEmbeddings(x, y, sample_index, model, gaussianMixLatentSpace):
    num_components_true = np.unique(y).shape[0]
    num_gaussianpoints_per_component = gaussianMixLatentSpace.points_per_component
    X = [x[(y==i).flatten()] for i in range(num_components_true)]
    X = [x[np.random.choice(x.shape[0], num_gaussianpoints_per_component, replace=True)] for x in X]
    with torch.no_grad():
        Z = [model(xx)[0].detach().numpy() for xx in X]
        Gaussianpoints = gaussianMixLatentSpace.transform_gaussians(sample_index)
    plt.figure()
    #colors = plt.cm.get_cmap('tab10', num_components_true)
    for z in Z:
        plt.scatter(z[:, 0], z[:, 1], label='Direct', alpha=0.5)
        mean = np.mean(z, axis=0)
        cov = np.cov(z.T)
        CIEllipse(mean, cov, plt.gca(), n_std=1.0, facecolor='none', edgecolor='black') # true
    for comp_index, gaussian in zip(gaussianMixLatentSpace.Sample2Component[sample_index], Gaussianpoints):
        #plt.scatter(gaussian[:, 0], gaussian[:, 1], label='Indirect', alpha=0.2)
        # mean = gaussianMixLatentSpace.Mu[comp_index].detach().numpy()
        # cov = gaussianMixLatentSpace.Cov[comp_index].detach().numpy()
        mu = gaussianMixLatentSpace.GMM.compDist[comp_index].mu
        cov = gaussianMixLatentSpace.GMM.compDist[comp_index].cov
        CIEllipse(mu, cov, plt.gca(), n_std=1.0, facecolor='none', edgecolor='red')  # estimated
    plt.title('Embeddings')
    plt.legend()
     # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image = np.array(image)

    # Convert the NumPy array to a PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1)  # Change the order of dimensions to [C, H, W]

    return image


def VisualizeMatching(z, gaussianpoints):
    plt.scatter(z[:, 0], z[:, 1], color='r', label='Direct')
    plt.scatter(gaussianpoints[:, 0], gaussianpoints[:, 1], color='b', label='Indirect')
    #[plt.plot([zz[0],gg[0]], [zz[1], gg[1]], color='grey') for zz, gg in zip(z, gaussianpoints)]
    plt.title('Matching')
    plt.legend()
    
        # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

        # Convert the buffer to a PIL Image
    image = Image.open(buf)

        # Convert the PIL Image to a NumPy array
    image = np.array(image)

        # Convert the NumPy array to a PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1)  # Change the order of dimensions to [C, H, W]
    return image


def ResponsibilityErrorPlot(Responsibilities_true, Responsibilities_pred):
    plt.figure()
    for i in range(len(Responsibilities_true)):
        sp.sortedplot(Responsibilities_true[i], Responsibilities_pred[i], label='Comp'+str(i))
    
    sp.title('Responsibility Error')
    sp.xlabel('True')
    sp.ylabel('Predicted')
    sp.xlim([0, 1])
    sp.ylim([0, 1])
    sp.legend()
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image = np.array(image)

    # Convert the NumPy array to a PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1)  # Change the order of dimensions to [C, H, W]

    return image




    
def VisualizeInputData(X, Y):
    for i, (x,y) in enumerate(zip(X, Y)):
        if len(X) == 2:
            plt.subplot(2,1,i)
        if len(X) == 3:
            plt.subplot(2,2,i)
        plt.title('Sample '+str(i))
        for c in np.unique(y):
            xx= x[(y==c).flatten()]
            ix = np.random.choice(xx.shape[0], 100, replace=True)
            plt.scatter(xx[ix, 0], xx[ix, 1], label='C'+str(c), alpha=0.5) 
            cov = np.cov(xx.T)
            mean = np.mean(xx, axis=0)
            CIEllipse(mean, cov, plt.gca(), n_std=1.0, facecolor='none', edgecolor='k')
    plt.legend()
    plt.show()
    
