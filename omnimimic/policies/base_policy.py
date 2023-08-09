from abc import ABC, abstractmethod

class BaseTorchPolicy(ABC):
    
    @abstractmethod
    def forward_np(self):
        '''
        Takes numpy data from the dataloader and outputs pytorch object
        '''
        pass

    @abstractmethod
    def compute_loss_np(self):
        '''
        Takes numpy data from the dataloader and outputs pytorch loss object
        '''
        pass

    @abstractmethod
    def get_action(self):
        '''
        Takes numpy data from the dataloader and outputs numpy action
        '''
        pass

