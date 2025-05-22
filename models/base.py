from abc import ABC, abstractmethod
import os

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

    def model_exists(self):
        """Проверяет существование сохраненной модели"""
        return os.path.exists(self.model_path)
