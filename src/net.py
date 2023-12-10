import os
import torch
from .patterns import PB_DICT
import logging

INPUT_DIM = 2 * (5 * (len(PB_DICT) - 1) + 1) + 2 * (2 * len(PB_DICT)) + 2
HIDDEN_DIM = 100
OUTPUT_DIM = 1

PRE_INPUT_DIM = 128
PRE_HIDDEN_DIM = 32
PRE_OUTPUT_DIM = 1

INPUT_CHANNELS = 4
HIDDEN_CHANNELS = 32
OUTPUT_CHANNELS = 64
    
class Net(torch.nn.Module):
    def __init__(self, 
                 model_path: str = None, 
                 logger: logging.Logger = logging.getLogger(__name__),
                 lr: float = 0.1,
                 **kwargs):

        self.model_path = model_path
        self.logger = logger
        self.lr = lr
        
        super(Net, self).__init__(**kwargs)
    
    @staticmethod
    def device() -> torch.DeviceObjType:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    def compile_model(self, *layers) -> None: 
        self.model = torch.nn.Sequential(*layers)
        self.model = self.model.to(self.device())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        
        try:
            self.load_model()
        except Exception as e:
            self.logger.error(e)
            self.logger.info('Initializing new model')
    
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def load_model(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = self.model_path
        assert filepath is not None, "Filepath is required"
        assert os.path.exists(filepath), f"Filepath does not exist: {filepath}"
        self.model.load_state_dict(torch.load(filepath))
        
    def save_model(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = self.model_path
        assert filepath is not None, "Filepath is required"
        torch.save(self.model.state_dict(), filepath)

class Conv_Net(Net):
    def __init__(self, M: int, N: int, **kwargs):
        self.board_size = (M, N)
        self.input_channels = INPUT_CHANNELS
        self.hidden_channels = HIDDEN_CHANNELS
        self.output_channels = OUTPUT_CHANNELS
        self.output_dim = OUTPUT_DIM
        
        super(Conv_Net, self).__init__(**kwargs)
                
        self.compile_model(
            torch.nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3),
            torch.nn.BatchNorm2d(self.output_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(self.output_channels, self.output_dim),
            torch.nn.Tanh(),
        )
        
class Dense_Net(Net):
    def __init__(self, **kwargs):
        self.input_dim = INPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.output_dim = OUTPUT_DIM
        
        super(Dense_Net, self).__init__(**kwargs)
        
        self.compile_model(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.Tanh(),
        )
        
        
class Pre_Dense_Net(Net):
    def __init__(self, **kwargs):
        self.input_dim = PRE_INPUT_DIM
        self.hidden_dim = PRE_HIDDEN_DIM
        self.output_dim = PRE_OUTPUT_DIM
        
        super(Pre_Dense_Net, self).__init__(**kwargs)
        
        self.compile_model(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.Tanh(),
        )
        
        