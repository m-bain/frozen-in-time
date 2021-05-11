import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from model.model import sim_matrix
from itertools import cycle


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, init_val=False):
        super().__init__(model, loss, metrics, optimizer, config, writer, init_val=init_val)
        self.init_val = True
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader.batch_size
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            if (batch_idx + 1) * self.batch_size > self.max_samples_per_epoch:
                break
            if isinstance(data['video'], list):
                data['video'] = [x.to(self.device) for x in data['video']]
            else:
                data['video'] = data['video'].to(self.device)
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}

            self.optimizer.zero_grad()
            text_embeds, video_embeds = self.model(data)
            output = sim_matrix(text_embeds, video_embeds)
            loss = self.loss(output)
            loss.backward()
            self.optimizer.step()
            if self.writer is not None:
                self.writer.log_scalar(f'loss_train', loss.detach().item())

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach().item()))

            self.optimizer.zero_grad()

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        # self.valid_data_loader.dataset.__getitem__(0)
        meta_arr = []
        text_embed_arr = []
        vid_embed_arr = []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                meta_arr.append(data['meta'])
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                text_embed, vid_embed = self.model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                sims_batch = sim_matrix(text_embed, vid_embed)
                loss = self.loss(sims_batch)
                total_val_loss += loss.item()

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

        # TODO: this needs a clean
        if self.writer is not None:
            self.writer.log_scalar(f'loss_val', total_val_loss / len(self.valid_data_loader))
        nested_metrics = {}
        for metric in self.metrics:
            metric_name = metric.__name__
            res = metric(sims)
            verbose(epoch=epoch, metrics=res, name=self.valid_data_loader.dataset_name,
                    mode=metric_name)
            nested_metrics[metric_name] = res

            if self.writer is not None:
                to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                            name=self.valid_data_loader.dataset_name)
                for key, val in to_write.items():
                    self.writer.log_scalar(key, val)

            if self.visualizer is not None:
                meta_arr_cat = {key: [] for key in meta_arr[0]}
                for meta in meta_arr:
                    for key, val in meta.items():
                        meta_arr_cat[key] += val
                self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'nested_val_metrics': nested_metrics
        }

        return res_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
