from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, user_item_matrix):
        pass

    @abstractmethod
    def recommend(self, user_id, top_n=10):
        pass
