
class BaseDataLoader():
    def __init__(self):
        self.opt = None
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None
