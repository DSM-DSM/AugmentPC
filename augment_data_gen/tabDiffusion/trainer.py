import torch
from augment_data_gen.tabDiffusion.dataloader import FastTensorDataLoader
import logging


class DiffusionTrainer:
    """扩散模型训练器"""

    def __init__(self, model, train_dataset, val_dataset=None, device=torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def prepare_dataloader(self, dataset, batch_size, shuffle):
        """创建优化的数据加载器"""
        X = []
        if dataset.X_num is not None:
            X.append(dataset.X_num)
        if dataset.X_cat is not None:
            X.append(dataset.X_cat)
        X = torch.cat(X, dim=1) if len(X) > 1 else X[0]
        return FastTensorDataLoader(X, dataset.y, batch_size=batch_size, shuffle=shuffle)

    def train(self, epochs, batch_size=128, lr=1e-3, weight_decay=0.0, print_freq=5000, eval_freq=1000,
              model_path='best_model.pth'):
        """训练循环"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = self.prepare_dataloader(self.train_dataset, batch_size, shuffle=True)

        best_valid_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss, gloss, mloss = 0, 0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # 准备输入字典（根据你的模型需求）
                out_dict = {'y': y}

                # 前向传播和损失计算
                optimizer.zero_grad()
                loss_multi, loss_gauss = self.model.mixed_loss(x, out_dict)
                loss = loss_multi + loss_gauss

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                gloss += loss_gauss
                mloss += loss_multi

            avg_train_loss = total_loss / len(train_loader)
            avg_mloss = mloss / len(train_loader)
            avg_gloss = gloss / len(train_loader)
            if (epoch + 1) % print_freq == 0:
                logging.info(
                    f'Epoch {epoch + 1} - Total Train Loss: {avg_train_loss:.4f}, GLoss: {avg_gloss:.4f}, MLoss: {avg_mloss:.4f}')

            # 验证阶段
            if self.val_dataset and (epoch + 1) % eval_freq == 0:
                val_loss = self.validate(batch_size)
                logging.info(f'Epoch {epoch + 1} - Val Loss: {val_loss:.4f}')

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
        self.best_valid_loss = best_valid_loss

    def validate(self, batch_size):
        """验证循环"""
        self.model.eval()
        val_loader = self.prepare_dataloader(self.val_dataset, batch_size, shuffle=False)
        total_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out_dict = {'y': y}

                loss_multi, loss_gauss = self.model.mixed_loss(x, out_dict)
                total_loss += (loss_multi + loss_gauss).item()

        return total_loss / len(val_loader)
