from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, user_item_matrix):
        pass

    @abstractmethod
    def predict(self, user_id, top_n):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _save_model(self):
        pass
