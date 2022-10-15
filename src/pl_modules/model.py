from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from src.common.utils import PROJECT_ROOT
import src.pl_modules.ALSTM as ALSTM
import src.pl_modules.pmetrics as metrics

class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!

        input_dim = self.hparams.input_dim
        hidden_dim = self.hparams.hidden_dim
        num_layers = self.hparams.num_layers
        dropout_prob = self.hparams.dropout_prob
        self.look_back = self.hparams.look_back
        attention_mode = "TMA"  #self.hparams.attention_mode #TODO: Debug

        #self.model = ALSTM.LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=num_layers, output_dim=1, dropout_prob=dropout_prob, device="cuda:0")
        self.model = ALSTM.SALSTM4(input_dim=input_dim, seq_length=self.hparams.look_back, hidden_dim=hidden_dim, layer_dim=num_layers, output_dim=1, dropout_prob=dropout_prob, attention_mode=attention_mode, device="cuda:0")



    def forward(self, x:torch.Tensor, h0=None, c0=None, base_val=None, pprint=False, stochastic=False, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x,h0,c0,base_val,pprint,stochastic)

    def step(self, batch: Any, batch_idx: int):
        Ed = 0
        Esign = 0
        for r1 in range(1, batch.shape[1]-self.look_back-1):

            x_in = batch[:, r1:r1+self.look_back]

            y = batch[:,r1+self.look_back,0]  #0 => close price

            x_inv = x_in-batch[:, r1-1:r1+self.look_back-1]
            y_out = self(x_inv, base_val=x_in[:,-1,0])
    
            Ed += torch.sum(((y.reshape(-1,1)-y_out)**2/batch.shape[0]))
            Esign += torch.mean(0.25*(torch.sign(y.reshape(-1,1)-x_in[:,-1,0].reshape(-1,1))-torch.sign(y_out-x_in[:,-1,0].reshape(-1,1)))**2)
            
        Esign = Esign/batch.shape[1]
        Ed = Ed/batch.shape[1] + 0.4*Esign
        
        strategy = "Long-only"
        budg0 = 1000
        invest_time = 365 #batch.shape[1]
        tax = 0.001
        delay = 0 #365 
        sperf1, _, _, _, aperf1, _ = metrics.profit_meas(batch, self.look_back, None, budg0=budg0, invest_time=invest_time, tax=tax, strategy=strategy, predictor_mode="openaware", predictor=self, do_print=False, delay=delay)
        sperf2, _, _, _, aperf2, _  = metrics.profit_meas(batch, self.look_back, None, budg0=budg0, invest_time=invest_time, tax=tax, strategy=strategy, predictor_mode="taxaware", predictor=self, do_print=False, delay=delay)
        sperf1ns, _, _, _, aperfns1, _  = metrics.profit_meas(batch, self.look_back, None, budg0=budg0, invest_time=invest_time, tax=tax, strategy=strategy, predictor_mode="openaware", predictor=self, do_print=False, delay=delay, st=False)
        sperf2ns, _, _, _, aperfns2, _ = metrics.profit_meas(batch, self.look_back, None, budg0=budg0, invest_time=invest_time, tax=tax, strategy=strategy, predictor_mode="taxaware", predictor=self, do_print=False, delay=delay, st=False)
        return {
            "loss": Ed,
            "dloss": Esign,
            "s-pmetric-openaware": sperf1[-1],
            "s-pmetric-taxaware":sperf2[-1],
            "d-pmetric-openaware": sperf1ns[-1],
            "d-pmetric-taxaware":sperf2ns[-1],
            "APM-s-openaware": aperf1[-1],
            "APM-s-taxaware": aperf2[-1],
            "APM-d-openaware": aperfns1[-1],
            "APM-d-taxaware": aperfns2[-1],
        }


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out_step = self.step(batch, batch_idx)
        loss = out_step["loss"]
        self.log_dict(
            {
                "train_loss": loss,
                "train_dir_loss": out_step["dloss"],
                "train_s-pmetric-openaware": out_step["s-pmetric-openaware"],
                "train_s-pmetric-taxaware": out_step["s-pmetric-taxaware"],
                "train_d-pmetric-openaware": out_step["d-pmetric-openaware"],
                "train_d-pmetric-taxaware": out_step["d-pmetric-taxaware"],
                "train_APM-s-openaware": out_step["APM-s-openaware"],
                "train_APM-s-taxaware": out_step["APM-s-taxaware"],
                "train_APM-d-openaware": out_step["APM-d-openaware"],
                "train_APM-d-taxaware": out_step["APM-d-taxaware"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out_step = self.step(batch, batch_idx)
        loss = out_step["loss"]
        self.log_dict(
            {
                "val_loss": loss,
                "val_dir_loss": out_step["dloss"],
                "val_s-pmetric-openaware": out_step["s-pmetric-openaware"],
                "val_s-pmetric-taxaware": out_step["s-pmetric-taxaware"],
                "val_d-pmetric-openaware": out_step["d-pmetric-openaware"],
                "val_d-pmetric-taxaware": out_step["d-pmetric-taxaware"],
                "val_APM-s-openaware": out_step["APM-s-openaware"],
                "val_APM-s-taxaware": out_step["APM-s-taxaware"],
                "val_APM-d-openaware": out_step["APM-d-openaware"],
                "val_APM-d-taxaware": out_step["APM-d-taxaware"]
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out_step = self.step(batch, batch_idx)
        loss = out_step["loss"]
        self.log_dict(
            {
                "test_loss": loss,
                "test_dir_loss": out_step["dloss"],
                "test_s-pmetric-openaware": out_step["s-pmetric-openaware"],
                "test_s-pmetric-taxaware": out_step["s-pmetric-taxaware"],
                "test_d-pmetric-openaware": out_step["d-pmetric-openaware"],
                "test_d-pmetric-taxaware": out_step["d-pmetric-taxaware"],
                "test_APM-s-openaware": out_step["APM-s-openaware"],
                "test_APM-s-taxaware": out_step["APM-s-taxaware"],
                "test_APM-d-openaware": out_step["APM-d-openaware"],
                "test_APM-d-taxaware": out_step["APM-d-taxaware"]
            },
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
