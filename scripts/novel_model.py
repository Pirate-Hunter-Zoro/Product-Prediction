import torch
import torch.nn as nn
from pathlib import Path
from recbole.model.abstract_recommender import SequentialRecommender

from attribute_loader import load_text_embedding, load_price_bins

class NovelModel(SequentialRecommender):
    
    def __init__(self, config, dataset):
        """Instantiate the NovelModel class with the recbole config and dataset

        Args:
            config (dict): Recbole config dict injected by run_recbole (hyperparameters and field names)
            dataset (SequentialDataset): Exposes num(item_id_field), token2id calls as needed
        """
        super().__init__(config, dataset)
        self.n_price_bins = config["n_price_bins"]
        self.hidden_size = config["hidden_size"]
        # The following calls need the n_price_bins and hidden_size attributes initialized
        self.register_buffer("title_embeddings", load_text_embedding(config["TITLE_EMBEDDING_PATH"], dataset))
        self.register_buffer("brand_embeddings", load_text_embedding(config["BRAND_EMBEDDING_PATH"], dataset))
        self.register_buffer("color_embeddings", load_text_embedding(config["COLOR_EMBEDDING_PATH"], dataset))
        self.register_buffer("price_bin_idx", load_price_bins(config["PRICE_BINS_PATH"], dataset, n_bins=self.n_price_bins))
        # Note we have an extra price bin because in case some item had no price bin, it will still have a price embedding
        self.price_embedding = nn.Embedding(self.n_price_bins+1, self.hidden_size)
    
        # Obtain attributes necessary for forward method
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        # Row 0 is the PAD token so it is always constant - no computing gradients
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # batch_first=True ensures we have [batch, seq, dim] just like everything else
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
    
        # Create attributes necessary for cross attention of embeddings
        self.num_heads = config["num_heads"]
        self.text_dim = self.title_embeddings.shape[1]
        # Need a projection from text embedding dimension space to hidden dimension space
        self.title_proj = nn.Linear(self.text_dim, self.hidden_size)
        # Similarly for all other projections
        self.brand_proj = nn.Linear(self.text_dim, self.hidden_size)
        self.color_proj = nn.Linear(self.text_dim, self.hidden_size)
        # Cross attention heads - to operate on the text, brand, and color projects once in hidden dimension space
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, dropout=self.dropout_prob, batch_first=True)
    
    def forward(self, item_seq: torch.LongTensor, item_seq_len: torch.LongTensor) -> torch.FloatTensor:
        """Calculate and return hidden dimension state of each item

        Args:
            item_seq (torch.LongTensor): Shape (batch, max_seq_len) (padded with 0's)
            item_seq_len (torch.LongTensor): Shape (batch) of true sequence length of each session

        Returns:
            torch.FloatTensor: Session state
        """
        raise NotImplementedError(0)
    
    def calculate_loss(self, interaction) -> torch.FloatTensor:
        """Calculate numeric loss associated with interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Associated loss
        """
        raise NotImplementedError()
    
    def full_sort_predict(self, interaction) -> torch.FloatTensor:
        """Calculate next item prediction scores associated with this interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Shape (batch, num_items) - scores across full vocab
        """
        raise NotImplementedError()