import os

class Context:
    def __init__(self, n_samples, n_pos_compos, n_neg_comps, dim, sample_to_pos_comp_idces,sample_to_pos_comps_mix_prop,
                 sample_to_neg_comp_idces,sample_to_neg_comps_mix_prop, sample_sizes, frozendata, loaddir, savedir):
        self.n_samples = n_samples
        self.n_pos_comps = n_pos_compos
        self.n_neg_comps = n_neg_comps
        self.input_dim = dim
        self.sample_to_pos_comp_idces = sample_to_pos_comp_idces
        self.sample_to_pos_comps_mix_prop = sample_to_pos_comps_mix_prop
        self.sample_to_neg_comp_idces = sample_to_neg_comp_idces
        self.sample_to_neg_comps_mix_prop = sample_to_neg_comps_mix_prop
        self.sample_sizes = sample_sizes
        self.frozendata = frozendata
        if loaddir is not None:
            self.loaddir = os.path.join(os.getcwd(), loaddir)
        if savedir is not None:
            self.savedir = os.path.join(os.getcwd(), savedir)



def get_input_context_two_sample_two_comps(args = None):
    frozendata = False
    loaddir = None
    savedir = None
    if args is not None:
        frozendata = args.usefrozen
        loaddir = args.datadir
        savedir = args.storedir
    n_samples = 2
    n_pos_comps = 1
    n_neg_comps = 1
    input_dim = 2
    sample_to_pos_comp_idces = [[0], [0]]
    sample_to_pos_comps_mix_prop = [[0.8], [0.2]]
    sample_to_neg_comp_idces = [[0], [0]]
    sample_to_neg_comps_mix_prop = [[0.2], [0.8]]
    sample_sizes = [10000, 10000]
    return Context(n_samples, n_pos_comps,n_neg_comps, input_dim, sample_to_pos_comp_idces, 
                   sample_to_pos_comps_mix_prop, sample_to_neg_comp_idces, sample_to_neg_comps_mix_prop, 
                   sample_sizes, frozendata, loaddir, savedir)


def get_input_context_one_sample_two_comps(args = None):  
    frozendata = False
    loaddir = None
    savedir = None
    if args is not None:
        frozendata = args.usefrozen
        loaddir = args.datadir
        savedir = args.storedir
    n_samples = 1
    n_pos_comps = 1
    n_neg_comps = 1
    input_dim = 2
    sample_to_pos_comp_idces = [[0]]
    sample_to_pos_comps_mix_prop = [[0.8]]
    sample_to_neg_comp_idces = [[0]]
    sample_to_neg_comps_mix_prop = [[0.2]]
    sample_sizes = [20000]
    return Context(n_samples, n_pos_comps,n_neg_comps, input_dim, sample_to_pos_comp_idces, 
                   sample_to_pos_comps_mix_prop, sample_to_neg_comp_idces, sample_to_neg_comps_mix_prop, 
                   sample_sizes, frozendata, loaddir, savedir)



class HyperParameters:
    def __init__(self, sample_sizes, test_size, batch_size, gmls_sigma, encoded_dim_autoencoder, num_layers_autoencoder, width_autoencoder, 
                 learning_rate, n_epochs,warmup_epochs):
        self.sample_sizes_training = [e*(1-test_size) for e in sample_sizes] # a hyper parameter dependent on input sample sizes. isn't that bad ?
        self.batch_size = batch_size
        self.gmls_sigma = gmls_sigma
        self.encoded_dim_autoencoder = encoded_dim_autoencoder
        self.num_layers_autoencoder = num_layers_autoencoder
        self.width_autoencoder = width_autoencoder
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs


def get_hyperparameters(input_context):
    sample_sizes = input_context.sample_sizes
    test_size = 0.2
    batch_size = 200
    gmls_sigma = 0.1
    encoded_dim_autoencoder = 2
    num_layers_autoencoder = 10
    width_autoencoder = 10
    learning_rate = 0.001
    n_epochs = 1000
    warmup_epochs = 20
    return HyperParameters(sample_sizes,test_size, batch_size,  gmls_sigma, encoded_dim_autoencoder, 
                           num_layers_autoencoder, width_autoencoder, learning_rate,
                           n_epochs, warmup_epochs)

