Stochastic Attention Based LSTM for Stock prices Forecasting.




<p align="center">
  <img src="imgs/SALSTM.drawio.png"  width="600" title="Model Architecture" />
</p>


# Structure

```bash
.
├── .cache              
├── conf                # hydra compositional config 
│   ├── data
│   ├── default.yaml    # current experiment configuration        
│   ├── hydra
│   ├── logging
│   ├── model
│   ├── optim
│   └── train
├── data                # datasets
├── .env                # system-specific env variables, e.g. PROJECT_ROOT
├── requirements.txt    # basic requirements
├── src
│   ├── common          # common modules and utilities
│   ├── pl_data         # PyTorch Lightning datamodules and datasets
│   ├── pl_modules      # PyTorch Lightning modules
│   ├── run.py          # entry point to run current conf
│   └── ui              # interactive streamlit apps
└── wandb               # local experiments (auto-generated)
```
