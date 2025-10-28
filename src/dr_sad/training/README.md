# TrainerSetup

Utility class for creating PyTorch Lightning trainers with configurable early stopping and learning rate scheduling.

## Usage

```python
from dr_sad.training import TrainerSetup
from dr_sad.pyannet.pyannet import PyanNet

# Basic trainer with early stopping
trainer = TrainerSetup.create_trainer(max_epochs=50)
model = PyanNet()
trainer.fit(model, train_loader, val_loader)
```

## With Learning Rate Scheduler

```python
# Create model with scheduler directly
scheduler_config = {
    "patience": 5,
    "factor": 0.1,
    "mode": "min",
    "monitor": "val_loss"
}
model = PyanNet(scheduler_config=scheduler_config, learning_rate=0.001)

# Or use helper method with all arguments
model = TrainerSetup.create_model_with_scheduler(
    PyanNet,
    scheduler_patience=5,
    scheduler_factor=0.1,
    learning_rate=0.001,
)

trainer = TrainerSetup.create_trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
```
## Methods

### `create_trainer(max_epochs=100, early_stopping_patience=10, **trainer_kwargs)`
Creates a trainer with early stopping and LR monitoring.

### `create_model_with_scheduler(model_class, scheduler_patience=5, scheduler_factor=0.1, learning_rate=1e-3, **model_kwargs)`
Helper to create a model instance with scheduler configuration.

## Parameters

### `create_trainer()`
- `max_epochs` (int): Maximum training epochs (default: 100)
- `early_stopping_patience` (int): Early stopping patience (default: 10)
- `**trainer_kwargs`: Additional PyTorch Lightning Trainer arguments

### `create_model_with_scheduler()`
- `model_class` (PyanNet): Model class to instantiate
- `scheduler_patience` (int): Number of epochs with no improvement after which learning rate will be reduced (default: 5)
- `scheduler_factor` (float): Factor by which the learning rate will be reduced (default: 0.1)
- `learning_rate` (float): Initial learning rate (default: 1e-3)
- `**model_kwargs`: Additional keyword arguments to pass to the model
