import os.path

import torch
import numpy as np
from augment_data_gen.tabDiffusion.dataloader import TabularDataset, SubsetWithAttributes
from augment_data_gen.tabDiffusion.modules import MLPDiffusion
from augment_data_gen.tabDiffusion.trainer import DiffusionTrainer
from augment_data_gen.tabDiffusion.tabdiffusion import GaussianMultinomialDiffusion


class tabDiffusionPipeline:
    def __init__(self, X_num, y, X_cat=None, task_type='reg', is_y_cond=False, epochs=20000, num_timesteps=1000,
                 gaussian_loss_type='kl', lr=1e-3, batch_size=4096, hidden_dim_layer=[512, 512], dim_t=128, fold=5,
                 dropout=0.0, eval_freq=10000, print_freq=5000, model_dir=None, data_hash=None, train_hash=None,
                 device=torch.device('cpu')):
        self.model_dir, self.data_hash, self.train_hash = model_dir, data_hash, train_hash
        self.model_path, self.train_skip = os.path.join(self.model_dir, self.data_hash + '.pth'), False
        self.X_num, self.X_cat, self.y = X_num, X_cat, y
        self.eval_freq, self.print_freq = eval_freq, print_freq
        self.n, self.numerical_feature_num = self.X_num.shape
        if X_cat is not None:
            def unique_value_count(arr):
                unique_elements = np.unique(arr, axis=0)
                unique_counts = np.apply_along_axis(len, 0, unique_elements)
                return unique_counts

            self.category_feature_num_list = np.apply_along_axis(unique_value_count, 0, self.X_cat)
        else:
            self.category_feature_num_list = np.array([0])
        self.category_feature_num = np.sum(self.category_feature_num_list)
        self.is_y_cond = is_y_cond
        self.unique_label_num = 0 if task_type == 'reg' else np.unique(self.y).shape[0]

        # dataset parameters
        self.fold = fold
        self.val_dataset, self.train_dataset, self.dataset = None, None, None
        self.train_split()

        # denoise function parameters
        self.task_type = task_type
        self.dim_t = dim_t
        self.num_timesteps = num_timesteps
        self.hidden_dim_layer = hidden_dim_layer
        self.dropout = dropout
        self.init_denoise_fn()

        # diffusion model parameters
        self.gaussian_loss_type = gaussian_loss_type
        self.lr = lr
        self.device = device
        self.init_diffusion_model()

        # trainer parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_trainer()

    def init_denoise_fn(self):
        self.denoise_fn = MLPDiffusion(dim_in=self.numerical_feature_num + self.category_feature_num,
                                       unique_label_num=self.unique_label_num, task_type=self.task_type,
                                       is_y_cond=self.is_y_cond, dim_t=self.dim_t,
                                       dim_hidden_layers=self.hidden_dim_layer, dropout=self.dropout)

    def init_diffusion_model(self):
        self.diffusion_model = GaussianMultinomialDiffusion(category_feature_num_list=self.category_feature_num_list,
                                                            numerical_features_num=self.numerical_feature_num,
                                                            denoise_fn=self.denoise_fn,
                                                            num_timesteps=self.num_timesteps,
                                                            gaussian_loss_type=self.gaussian_loss_type,
                                                            device=self.device)

    def init_trainer(self):
        self.trainer = DiffusionTrainer(self.diffusion_model, self.train_dataset, self.val_dataset, device=self.device)

    def train_split(self):
        self.dataset = TabularDataset(self.X_num, self.X_cat, self.y)
        # 手动分割
        train_size = int((1 - 1 / self.fold) * self.n)
        indices = torch.randperm(self.n).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_dataset = SubsetWithAttributes(torch.utils.data.Subset(self.dataset, train_indices))
        self.val_dataset = SubsetWithAttributes(torch.utils.data.Subset(self.dataset, val_indices))

    def save_model(self):
        torch.save({
            'model_state_dict': self.trainer.model.state_dict(),  # parameters of diffusion model
            'best_valid_loss': self.trainer.best_valid_loss,
            'train_hash': self.train_hash,
            'data_hash': self.data_hash
        }, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            train_hash, data_hash = checkpoint['train_hash'], checkpoint['data_hash']
            if self.train_hash == train_hash and self.data_hash == data_hash:
                self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
                self.best_valid_loss = checkpoint['best_valid_loss']
                self.train_skip = True

    def train(self):
        # 3. 训练
        self.load_model()
        if not self.train_skip:
            self.trainer.train(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, print_freq=self.print_freq,
                               eval_freq=self.eval_freq, model_path=self.model_path)
            self.save_model()

    def sample(self, n_samples):
        y = torch.rand(self.n).float()
        out_dict = {'y': y}
        samples, _ = self.diffusion_model.sample(num_samples=n_samples, y_dist=y)
        # Convert samples to float32 if needed
        if samples.dtype == torch.float64:
            samples = samples.float()
        return samples
