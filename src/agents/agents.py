import os
import sys
import copy
import torch
import dotmap
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from src.datasets.mnist import (
    load_many_rotated_mnist,
    DEFAULT_ROTATIONS,
    UNSEEN_ROTATIONS,
    DEFAULT_ROTATIONS_SPARSE,
    UNSEEN_ROTATIONS_SPARSE,
    DEFAULT_ROTATIONS_DISJOINT,
    UNSEEN_ROTATIONS_DISJOINT,
    ALL_ROTATIONS,
    DEFAULT_ROTATIONS_DICT,
    UNSEEN_ROTATIONS_DICT,
    load_many_scaled_mnist,
    DEFAULT_SCALES,
    UNSEEN_SCALES,
    DEFAULT_SCALES_SPARSE,
    UNSEEN_SCALES_SPARSE,
    DEFAULT_SCALES_DISJOINT,
    UNSEEN_SCALES_DISJOINT,
    ALL_SCALES,
    DEFAULT_SCALES_DICT,
    UNSEEN_SCALES_DICT,
    load_many_sheared_mnist,
    DEFAULT_SHEARS,
    UNSEEN_SHEARS,
    DEFAULT_SHEARS_SPARSE,
    UNSEEN_SHEARS_SPARSE,
    DEFAULT_SHEARS_DISJOINT,
    UNSEEN_SHEARS_DISJOINT,
    ALL_SHEARS,
    DEFAULT_SHEARS_DICT,
    UNSEEN_SHEARS_DICT,
)
from src.datasets.norb import (
    load_norb_numpy,
    ALL_ELEVATION,
    DEFAULT_ELEVATION,
    UNSEEN_ELEVATION,
    DEFAULT_ELEVATION_DISJOINT,
    UNSEEN_ELEVATION_DISJOINT,
    DEFAULT_ELEVATION_SPARSE,
    UNSEEN_ELEVATION_SPARSE,
    DEFAULT_ELEVATION_DICT,
    UNSEEN_ELEVATION_DICT,
    ALL_AZIMUTH,
    DEFAULT_AZIMUTH,
    UNSEEN_AZIMUTH,
    DEFAULT_AZIMUTH_DISJOINT,
    UNSEEN_AZIMUTH_DISJOINT,
    DEFAULT_AZIMUTH_SPARSE,
    UNSEEN_AZIMUTH_SPARSE,
    DEFAULT_AZIMUTH_DICT,
    UNSEEN_AZIMUTH_DICT,
    ALL_LIGHTING,
    DEFAULT_LIGHTING,
    UNSEEN_LIGHTING,
    DEFAULT_LIGHTING_DISJOINT,
    UNSEEN_LIGHTING_DISJOINT,
    DEFAULT_LIGHTING_SPARSE,
    UNSEEN_LIGHTING_SPARSE,
    DEFAULT_LIGHTING_DICT,
    UNSEEN_LIGHTING_DICT,
)
from src.datasets.utils import \
    BagOfDatasets, sample_minibatch, sample_minibatch_from_cache
from src.models.meta import MetaVAE
from src.models.vhe import HomoEncoder
from src.models.ns import Statistician
from src.models.vhe_vamprior import HomoEncoder_VampPrior
from src.models.meta_conv import ConvMetaVAE
from src.models.vhe_conv import ConvHomoEncoder
from src.models.ns_conv import ConvStatistician
from src.models.vhe_vamprior_conv import ConvHomoEncoder_VampPrior
from src.objectives.elbo import (
    bernoulli_elbo_loss,
    gaussian_elbo_loss
)
from src.utils import (
    save_checkpoint as save_snapshot,
    AverageMeter,
)
from src.setup import print_cuda_statistics


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)

        self._choose_device()
        self._create_model()
        self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0
        self.current_loss = 0
        self.current_val_loss = 0
        self.best_val_loss = np.inf

        self.val_losses = []

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size, shuffle=True,
                            num_workers=self.config.data_loader_workers)

        return loader, dataset_size

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.test()
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model testing
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')


class MetaAgent(BaseAgent):
    """
    Primary agent for training multimodal models and baselines using
    a supervised dataset of (modality1, modality2) pairs. Assumes a 
    MLP for all models.

    @param config: DotMap
                   configuration settings
    """
    def _load_datasets(self):
        if self.config.data_params.dataset == 'rotate':
            concept_name = 'mnist'
            rotations = DEFAULT_ROTATIONS_DICT[self.config.data_params.split]
            train_datasets = load_many_rotated_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=True,
                rotations=rotations,
            )
            test_datasets = load_many_rotated_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=False,
                rotations=rotations,
            )
        elif self.config.data_params.dataset == 'scale':
            concept_name = 'mnist'
            scales = DEFAULT_SCALES_DICT[self.config.data_params.split]
            train_datasets = load_many_scaled_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=True,
                scales=scales,
            )
            test_datasets = load_many_scaled_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=False,
                scales=scales,
            )
        elif self.config.data_params.dataset == 'shear':
            concept_name = 'mnist'
            shears = DEFAULT_SHEARS_DICT[self.config.data_params.split]
            train_datasets = load_many_sheared_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=True,
                shears=shears,
            )
            test_datasets = load_many_sheared_mnist(
                self.config.data_params.data_dir,
                self.config.data_params.image_size,
                train=False,
                shears=shears,
            )
        elif self.config.data_params.dataset == 'norb':
            concept_name = self.config.data_params.concept
            split = self.config.data_params.split
            image_size = self.config.data_params.image_size

            if concept_name == 'elevation':
                values = DEFAULT_ELEVATION_DICT[split]
            elif concept_name == 'azimuth':
                values = DEFAULT_AZIMUTH_DICT[split]
            elif concept_name == 'lighting':
                values = DEFAULT_LIGHTING_DICT[split]
            else:
                raise Exception('NORB split {} not recognized.')

            train_datasets = [
                load_norb_numpy(i, concept=concept_name, image_size=image_size, train=True)
                for i in values
            ]
            test_datasets = [
                load_norb_numpy(i, concept=concept_name, image_size=image_size, train=False)
                for i in values
            ]
        else:
            raise Exception('dataset {} not recognized'.format(
                self.config.data_params.dataset))

        # im going to cache each dataset
        train_caches = []
        for i in range(len(train_datasets)):
            cache_file_i = os.path.join(self.config.data_params.data_dir,
                                        self.config.data_params.dataset,
                                        'cache',
                                        'train_cache_{}_{}_{}.npy'.format(
                                            i, self.config.data_params.split,
                                            concept_name))
            if os.path.isfile(cache_file_i):
                self.logger.info("found cached dataset {}/{}. loading...".format(i+1, len(train_datasets)))
                cache_i = np.load(cache_file_i)
            else:
                self.logger.info("cache-ing each train dataset {}/{}".format(i+1, len(train_datasets)))
                loader_i = DataLoader(train_datasets[i], batch_size=100, shuffle=False)
                pbar = tqdm(total=len(loader_i))
                cache_i = []
                for image, label in loader_i:
                    cache_i.append(image.numpy())
                    pbar.update()
                pbar.close()
                cache_i = np.concatenate(cache_i, axis=0)
                np.save(cache_file_i, cache_i)
                self.logger.info("saved cache to filesystem.")
            train_caches.append(cache_i)

        test_caches = []
        for i in range(len(test_datasets)):
            cache_file_i = os.path.join(self.config.data_params.data_dir,
                                        self.config.data_params.dataset,
                                        'cache',
                                        'test_cache_{}_{}_{}.npy'.format(
                                            i, self.config.data_params.split,
                                            concept_name))
            if os.path.isfile(cache_file_i):
                self.logger.info("found cached dataset {}/{}. loading...".format(i+1, len(test_datasets)))
                cache_i = np.load(cache_file_i)
            else:
                self.logger.info("cache-ing each test dataset {}/{}".format(i+1, len(test_datasets)))
                loader_i = DataLoader(test_datasets[i], batch_size=100, shuffle=False)
                pbar = tqdm(total=len(loader_i))
                cache_i = []
                for image, label in loader_i:
                    cache_i.append(image.numpy())
                    pbar.update()
                pbar.close()
                cache_i = np.concatenate(cache_i, axis=0)
                np.save(cache_file_i, cache_i)
                self.logger.info("saved cache to filesystem.")
            test_caches.append(cache_i)

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.train_caches = train_caches
        self.test_caches = test_caches
        self.n_train_datasets = len(self.train_datasets)
        self.n_test_datasets = len(self.test_datasets)
        self.train_dataset = BagOfDatasets(train_datasets)
        self.test_dataset = BagOfDatasets(test_datasets)

    def _create_model(self):
        if self.config.model_params.name == 'meta':
            self.model = MetaVAE(
                self.n_train_datasets,
                self.config.model_params.z_dim,
                hidden_dim=400,
            )
        elif self.config.model_params.name == 'vhe':
            self.model = HomoEncoder(
                300, self.config.model_params.z_dim,
                hidden_dim=400,
            )
        elif self.config.model_params.name == 'ns':
            self.model = Statistician(
                300, self.config.model_params.z_dim,
                hidden_dim_statistic=400,
                hidden_dim=400,
            )
        elif self.config.model_params.name == 'vhe_vamprior':
            self.model = HomoEncoder_VampPrior(
                300, self.config.model_params.z_dim, self.device,
                hidden_dim=400,
            )
        else:
            raise Exception('model name {} not recognized.'.format(
                self.config.model_params.name))
        self.model = self.model.to(self.device)

    def _create_optimizer(self):
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim_params.lr,
        )

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()
        epoch_loss = AverageMeter()

        for batch_i, data_list in enumerate(self.train_loader):
            if self.config.model_params.name == 'ns':
                # forward pass for NS is a little special
                data = torch.stack([x[0] for x in data_list]).float()
                data = data.to(self.device)
                out = self.model(data)
                # if self.config.data_params.dataset == '3dshapes':
                #     loss = self.model.gaussian_elbo_loss_sets(out)
                # else:
                loss = self.model.bernoulli_elbo_loss_sets(out)
            else:
                # forward pass for VHE and Meta are the same
                loss = 0
                for i in range(self.n_train_datasets):
                    data_i, _ = data_list[i]
                    data_i = data_i.float()
                    data_i = data_i.to(self.device)
                    batch_size = len(data_i)
                    sample_list_i = sample_minibatch_from_cache(
                        self.train_caches[i], batch_size,
                        self.config.model_params.n_data_samples)
                    sample_list_i = sample_list_i.float()
                    sample_list_i = sample_list_i.to(self.device)

                    if self.config.model_params.name == 'meta':
                        out_i = self.model(data_i, sample_list_i, i)
                        # if self.config.data_params.dataset == '3dshapes':
                        #     loss_i = gaussian_elbo_loss(*out_i)
                        # else:
                        loss_i = bernoulli_elbo_loss(*out_i)
                    elif self.config.model_params.name == 'vhe':
                        out_i = self.model(data_i, sample_list_i)
                        # if self.config.data_params.dataset == '3dshapes':
                        #     loss_i = self.model.gaussian_elbo(out_i)
                        # else:
                        loss_i = self.model.bernoulli_elbo(out_i)
                    elif self.config.model_params.name == 'vhe_vamprior':
                        out_i = self.model(data_i, sample_list_i)
                        loss_i = self.model.bernoulli_elbo(out_i)
                    else:
                        raise Exception('model name {} not recognized.'.format(
                            self.config.model_params.name))

                    loss += loss_i

                loss /= float(self.n_train_datasets)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss.update(loss.item(), self.config.optim_params.batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})

            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def test(self):
        num_batches = self.test_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        self.model.eval()
        epoch_losses = [
            AverageMeter() for _ in range(self.n_test_datasets)]

        with torch.no_grad():
            for batch_i, data_list in enumerate(self.test_loader):
                if self.config.model_params.name == 'ns':
                    batch_size = data_list[0][0].size(0)
                    data = torch.stack([x[0] for x in data_list])
                    data = data.to(self.device)
                    out = self.model(data)
                    loss = self.model.estimate_marginal(data, n_samples=10)
                    epoch_losses[0].update(loss.item(), batch_size)
                else:
                    for i in range(self.n_test_datasets):
                        data_i, _ = data_list[i]
                        data_i = data_i.float()
                        data_i = data_i.to(self.device)
                        batch_size = len(data_i)
                        sample_list_i = sample_minibatch_from_cache(
                            self.test_caches[i], batch_size,
                            self.config.model_params.n_data_samples)
                        sample_list_i = sample_list_i.float()
                        sample_list_i = sample_list_i.to(self.device)
                        if self.config.model_params.name == 'meta':
                            loss = self.model.estimate_marginal(
                                data_i, sample_list_i, i, n_samples=10)
                        elif self.config.model_params.name == 'vhe':
                            loss = self.model.estimate_marginal(
                                data_i, sample_list_i, n_samples=10)
                        elif self.config.model_params.name == 'vhe_vamprior':
                            loss = self.model.estimate_marginal(
                                data_i, sample_list_i, n_samples=10)

                    epoch_losses[i].update(loss.item(), batch_size)
                tqdm_batch.update()

        self.current_val_iteration += 1
        self.current_val_loss = sum(meter.avg for meter in epoch_losses)
        if self.current_val_loss < self.best_val_loss:
            self.best_val_loss = self.current_val_loss
        self.val_losses.append(self.current_val_loss)
        tqdm_batch.close()

    def load_checkpoint(self, filename):
        filename = os.path.join(self.config.checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.current_val_iteration = checkpoint['val_iteration']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            # we do not load optims for now
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesn't exists: [{}]".format(filename))
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_loss': self.current_val_loss,
            'val_losses': np.array(self.val_losses),
            'config': self.config,
        }
        is_best = ((self.current_val_loss == self.best_val_loss) or
                    not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)


class ConvMetaAgent(MetaAgent):
    """
    Like MetaAgent but uses larger convolutional layers.
    """

    def _create_model(self):
        if self.config.model_params.name == 'meta':
            self.model = ConvMetaVAE(
                self.n_train_datasets,
                self.config.model_params.z_dim,
                n_hidden=3,
                hidden_dim=400,
                n_channels=self.config.data_params.n_channels,
            )
        elif self.config.model_params.name == 'vhe':
            self.model = ConvHomoEncoder(
                300, self.config.model_params.z_dim,
                n_hidden=3,
                hidden_dim=400,
                n_channels=self.config.data_params.n_channels,
            )
        elif self.config.model_params.name == 'ns':
            self.model = ConvStatistician(
                300, self.config.model_params.z_dim,
                hidden_dim_statistic=400,
                n_hidden=3,
                hidden_dim=400,
                n_channels=self.config.data_params.n_channels,
            )
        elif self.config.model_params.name == 'vhe_vamprior':
            self.model = ConvHomoEncoder_VampPrior(
                300, self.config.model_params.z_dim, self.device, 
                n_hidden=3,
                hidden_dim=400,
                n_channels=self.config.data_params.n_channels,
            )
        else:
            raise Exception('model name {} not recognized.'.format(
                self.config.model_params.name))
        self.model = self.model.to(self.device)


class PredictorAgent(BaseAgent):
    r"""Agent class for predicting labels using unsupervised
    embeddings as a downstream task.

    Question: why wouldn't we just learn a supervised model?

    @param config: DotMap
                   configuration settings
    """
    def __init__(self, config):
        self.logger = logging.getLogger("Agent")
        self.meta_agent = self._load_meta_agent(
            config.meta_checkpoint_dir, config.gpu_device)
        self.meta_config = self.meta_agent.config
        self.config = config
        self._load_meta_model()

        self.unseen = True  # change to false for training datasets
        if self.unseen:
            self._load_unseen_datasets()
        else:
            self._load_datasets()
        self._choose_device()

    def _load_meta_agent(self, exp_dir, gpu_device,
                         checkpoint_name='model_best.pth.tar'):
        checkpoint_path = os.path.join(exp_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
        config = checkpoint['config']
        # overwrite GPU since we might want to use a different GPU
        config.gpu_device = gpu_device

        self.logger.info("Loading trained Agent from filesystem")
        AgentClass = globals()[config.agent]
        agent = AgentClass(config)
        agent.load_checkpoint(checkpoint_name)
        agent.model.eval()

        return agent

    def _load_datasets(self):
        self.train_datasets = copy.deepcopy(self.meta_agent.train_datasets)
        self.test_datasets  = copy.deepcopy(self.meta_agent.test_datasets)
        self.train_caches   = copy.deepcopy(self.meta_agent.train_caches)
        self.test_caches    = copy.deepcopy(self.meta_agent.test_caches)
        self.n_train_datasets = len(self.train_datasets)
        self.n_test_datasets = len(self.test_datasets)

    def _load_unseen_datasets(self):
        # load UNSEEN datasets...
        if self.meta_config.data_params.dataset == 'rotate':
            concept_name = 'mnist'
            rotations = UNSEEN_ROTATIONS_DICT[self.meta_config.data_params.split]
            train_datasets = load_many_rotated_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=True,
                rotations=rotations,
            )
            test_datasets = load_many_rotated_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=False,
                rotations=rotations,
            )
        elif self.meta_config.data_params.dataset == 'scale':
            concept_name = 'mnist'
            scales = UNSEEN_SCALES_DICT[self.meta_config.data_params.split]
            train_datasets = load_many_scaled_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=True,
                scales=scales,
            )
            test_datasets = load_many_scaled_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=False,
                scales=scales,
            )
        elif self.meta_config.data_params.dataset == 'shear':
            concept_name = 'mnist'
            shears = UNSEEN_SHEARS_DICT[self.meta_config.data_params.split]
            train_datasets = load_many_sheared_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=True,
                shears=shears,
            )
            test_datasets = load_many_sheared_mnist(
                self.meta_config.data_params.data_dir,
                self.meta_config.data_params.image_size,
                train=False,
                shears=shears,
            )
        elif self.meta_config.data_params.dataset == 'norb':
            concept_name = self.meta_config.data_params.concept
            split = self.meta_config.data_params.split
            image_size = self.meta_config.data_params.image_size

            if concept_name == 'elevation':
                values = UNSEEN_ELEVATION_DICT[split]
            elif concept_name == 'azimuth':
                values = UNSEEN_AZIMUTH_DICT[split]
            elif concept_name == 'lighting':
                values = UNSEEN_LIGHTING_DICT[split]
            else:
                raise Exception('NORB split {} not recognized.')

            train_datasets = [
                load_norb_numpy(i, concept=concept_name, image_size=image_size, train=True)
                for i in values
            ]
            test_datasets = [
                load_norb_numpy(i, concept=concept_name, image_size=image_size, train=False)
                for i in values
            ]
        else:
            raise Exception('dataset {} not recognized'.format(
                self.meta_config.data_params.dataset))

        # im going to cache each dataset
        train_caches = []
        for i in range(len(train_datasets)):
            cache_file_i = os.path.join(self.meta_config.data_params.data_dir,
                                        self.meta_config.data_params.dataset,
                                        'cache',
                                        # don't overwrite other caches...
                                        'unseen_train_cache_{}_{}_{}.npy'.format(
                                            i, self.meta_config.data_params.split,
                                            concept_name))
            if os.path.isfile(cache_file_i):
                self.logger.info("found cached unseen dataset {}/{}. loading...".format(i+1, len(train_datasets)))
                cache_i = np.load(cache_file_i)
            else:
                self.logger.info("cache-ing each unseen train dataset {}/{}".format(i+1, len(train_datasets)))
                loader_i = DataLoader(train_datasets[i], batch_size=100, shuffle=False)
                pbar = tqdm(total=len(loader_i))
                cache_i = []
                for image, label in loader_i:
                    cache_i.append(image.numpy())
                    pbar.update()
                pbar.close()
                cache_i = np.concatenate(cache_i, axis=0)
                np.save(cache_file_i, cache_i)
                self.logger.info("saved cache to filesystem.")
            train_caches.append(cache_i)

        test_caches = []
        for i in range(len(test_datasets)):
            cache_file_i = os.path.join(self.meta_config.data_params.data_dir,
                                        self.meta_config.data_params.dataset,
                                        'cache',
                                        # don't overwrite other caches...
                                        'unseen_test_cache_{}_{}_{}.npy'.format(
                                            i, self.meta_config.data_params.split,
                                            concept_name))
            if os.path.isfile(cache_file_i):
                self.logger.info("found cached unseen dataset {}/{}. loading...".format(i+1, len(test_datasets)))
                cache_i = np.load(cache_file_i)
            else:
                self.logger.info("cache-ing each unseen test dataset {}/{}".format(i+1, len(test_datasets)))
                loader_i = DataLoader(test_datasets[i], batch_size=100, shuffle=False)
                pbar = tqdm(total=len(loader_i))
                cache_i = []
                for image, label in loader_i:
                    cache_i.append(image.numpy())
                    pbar.update()
                pbar.close()
                cache_i = np.concatenate(cache_i, axis=0)
                np.save(cache_file_i, cache_i)
                self.logger.info("saved cache to filesystem.")
            test_caches.append(cache_i)

        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.train_caches = train_caches
        self.test_caches = test_caches
        self.n_train_datasets = len(self.train_datasets)
        self.n_test_datasets = len(self.test_datasets)
        self.train_dataset = BagOfDatasets(train_datasets)
        self.test_dataset = BagOfDatasets(test_datasets)

    def _get_embeddings_and_labels(self, dataset, cache):
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=100, shuffle=False)
            code_dataset, labels = [], []
            pbar = tqdm(total=len(loader))
            for batch_idx, (data, label) in enumerate(loader):
                batch_size = len(data)
                data = data.to(self.device)
                if self.meta_config.model_params.name == 'ns':
                    data = data.unsqueeze(0)
                    codes = self.meta_model.extract_codes(data)
                else:
                    sample_list = sample_minibatch_from_cache(
                        cache, batch_size,
                        self.meta_config.model_params.n_data_samples)
                    sample_list = sample_list.to(self.device)
                    codes = self.meta_model.extract_codes(data, sample_list)
                codes = codes.cpu().numpy()
                code_dataset.append(codes)
                labels.append(label.cpu().numpy())
                pbar.update()
            pbar.close()
        code_dataset = np.concatenate(code_dataset)
        labels = np.concatenate(labels)
        return code_dataset, labels

    def _load_meta_model(self):
        self.meta_model = copy.deepcopy(self.meta_agent.model)
        self.meta_model = self.meta_model.eval()
        for param in self.meta_model.parameters():
            param.requires_grad = False

    def run(self):
        train_accuracies = np.zeros(self.n_train_datasets)
        test_accuracies = np.zeros(self.n_train_datasets)

        for i in range(self.n_train_datasets):
            print('Training LogReg {}/{}'.format(i+1, self.n_train_datasets))
            train_dataset = self.train_datasets[i]
            test_dataset  = self.test_datasets[i]
            train_cache   = self.train_caches[i]
            test_cache    = self.test_caches[i]
            n_train = float(len(train_dataset))
            n_test  = float(len(test_dataset))
            print(' --> building training embeddings')
            train_codes, train_labels = self._get_embeddings_and_labels(train_dataset, train_cache)
            print(' --> building testing embeddings')
            test_codes, test_labels = self._get_embeddings_and_labels(test_dataset, test_cache)

            clf = LogisticRegression().fit(train_codes, train_labels)
            train_preds = clf.predict(train_codes)
            test_preds  = clf.predict(test_codes)
            train_acc = np.sum(train_preds == train_labels) / n_train
            test_acc  = np.sum(test_preds == test_labels) / n_test
            train_accuracies[i] = train_acc
            test_accuracies[i] = test_acc
            print(' --> train acc: {} | test acc: {}'.format(train_acc, test_acc))

        train_name = 'unseen_train_acc.npy' if self.unseen else 'train_acc.npy'
        test_name = 'unseen_test_acc.npy' if self.unseen else 'test_acc.npy'
        np.save(os.path.join(self.config.checkpoint_dir, train_name),
                train_accuracies)
        np.save(os.path.join(self.config.checkpoint_dir, test_name),
                test_accuracies)

    def finalise(self):
        pass

