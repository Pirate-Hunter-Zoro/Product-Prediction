import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender

class NovelModel(SequentialRecommender):
    
    def __init__(self, config, dataset):
        """Instantiate the NovelModel class with the recbole config and dataset

        Args:
            config (dict): Recbole config dict injected by run_recbole (hyperparameters and field names)
            dataset (SequentialDataset): Exposes num(item_id_field), token2id calls as needed
        """
        super().__init__(config, dataset)
    
    def forward(self, item_seq: torch.LongTensor, item_seq_len: torch.LongTensor) -> torch.FloatTensor:
        """Calculate and return hidden dimension state of each item

        Args:
            item_seq (torch.LongTensor): Shape (batch, max_seq_len) (padded with 0's)
            item_seq_len (torch.LongTensor): Shape (batch) of true sequence length of each session

        Returns:
            torch.FloatTensor: Session state
        """
        pass
    
    def calculate_loss(self, interaction) -> torch.FloatTensor:
        """Calculate numeric loss associated with interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Associated loss
        """
        return super().calculate_loss(interaction)
    
    def full_sort_predict(self, interaction) -> torch.FloatTensor:
        """Calculate next item prediction scores associated with this interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Shape (batch, num_items) - scores across full vocab
        """
        return super().full_sort_predict(interaction)