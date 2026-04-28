import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from attribute_loader import load_text_embedding, load_price_bins

ATTRIBUTES = {"title", "brand", "price", "color"}

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
        self.active_item_attributes = list(config['attribute_slots'])
        if not (set(self.active_item_attributes) <= ATTRIBUTES):
            raise ValueError(f"Expected attributes to be a subset of {str(ATTRIBUTES)} - fix config.yaml")
        if "title" not in self.active_item_attributes:
            raise ValueError(f"'title' must be present in attributes - fix config.yaml")
        
        
        
        # The following calls need the n_price_bins and hidden_size attributes initialized
        self.register_buffer("title_embeddings", load_text_embedding(config["TITLE_EMBEDDING_PATH"], dataset))
        if "brand" in self.active_item_attributes:
            self.register_buffer("brand_embeddings", load_text_embedding(config["BRAND_EMBEDDING_PATH"], dataset))
        if "color" in self.active_item_attributes:
            self.register_buffer("color_embeddings", load_text_embedding(config["COLOR_EMBEDDING_PATH"], dataset))
        if "price" in self.active_item_attributes:
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
        if 'brand' in self.active_item_attributes:
            self.brand_proj = nn.Linear(self.text_dim, self.hidden_size)
        if 'color' in self.active_item_attributes:
            self.color_proj = nn.Linear(self.text_dim, self.hidden_size)
        # Cross attention heads - to operate on the text, brand, and color projects once in hidden dimension space
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, dropout=self.dropout_prob, batch_first=True)
    
    def forward(self, item_seq: torch.LongTensor, item_seq_len: torch.LongTensor) -> torch.FloatTensor:
        """Calculate and return hidden dimension state of each item

        Args:
            item_seq (torch.LongTensor): Shape (batch, max_seq_len) (padded with 0's)
            item_seq_len (torch.LongTensor): Shape (batch) of true sequence length of each session

        Returns:
            torch.FloatTensor: Shape (batch, hidden_size). GRU session state
                fused (additive) with the cross-attention summary over the
                session's per-item attribute bag.
        """
        # Input is (batch, max_seq_len) - embed each item
        item_seq_emb = self.item_embedding(item_seq) # Output is (batch, max_seq_len, hidden_size)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb) # Prevent model from memorizing any single input row
        # Run through GRU
        gru_output, _ = self.gru(item_seq_emb_dropout) # Shape (batch, max_seq_len, hidden_size)
        # Pull out one vector from each session - the hidden state at the last real item (subtract 1 for the actual index)
        session_state = self.gather_indexes(gru_output, item_seq_len - 1) # Shape (batch, hidden_size) - per-session representation
        
        # AT THIS POINT, we've basically replicated GRU4Rec
        
        # Use embedding projections - input shape (N, in_features), output shape (N, out_features)
        K = len(self.active_item_attributes)
        proj_list = [\
                        self.title_proj(self.title_embeddings) if attr == "title" else\
                            (self.brand_proj(self.brand_embeddings) if attr == "brand" else\
                                (self.color_proj(self.color_embeddings) if attr == "color" else\
                                    # Produce per-item price vector - each item falls in a price bin which gets a vector from our price embedding table
                                    # Input shape (num_items,) - each item falls in a price bin 
                                    # Output shape (num_items, hidden_size)
                                    self.price_embedding(self.price_bin_idx)\
                                )\
                            )\
                        for attr in self.active_item_attributes\
                    ]
        
        # Stack all of the projections - shape (num_items, K, hidden_size), where K is the number of active attributes
        attr_bag = torch.stack(proj_list, dim=1)
        
        # Now for cross attention
        # Recall item_seq is of shape (batch, max_seq_len)
        session_attr_bag = attr_bag[item_seq] # shape (batch, max_seq_len, K, hidden_size)
        session_attr_flat = session_attr_bag.flatten(start_dim=1, end_dim=2) # shape (batch, max_seq_len * K, hidden_size)
        # Add target-seq axis so session_state can act as MHA query (batch_first=True)
        unsqueezed_session_state = session_state.unsqueeze(dim=1) # shape (batch, 1, hidden_size)
        # Create a padding mask
        padding_mask = item_seq == 0 # shape (batch, max_seq_len)
        # But recall we have K attributes we embedded
        padding_mask = padding_mask.repeat_interleave(K, dim=1) # shape (batch, K*max_seq_len)
        # Actual call for cross attention - query, key, value - result is shape (batch, 1, hidden_size)
        attended, _ = self.cross_attn(unsqueezed_session_state, session_attr_flat, session_attr_flat, key_padding_mask=padding_mask, need_weights=False)
        attended = attended.squeeze(dim=1) # (batch, hidden_size)
        
        # Return fused session (from GRU) with attended state
        return session_state + attended
    
    def calculate_loss(self, interaction) -> torch.FloatTensor:
        """Calculate numeric loss associated with interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Associated loss
        """
        item_sequence = interaction[self.ITEM_SEQ] # Shape (batch, max_seq_len)
        item_seq_len = interaction[self.ITEM_SEQ_LEN] # Shape (batch,)
        target = interaction[self.POS_ITEM_ID] # Shape (batch,)
        fused_output = self.forward(item_sequence, item_seq_len) # Shape (batch, hidden_size)
        # Recall that item_embedding.weight is Shape (num_items, hidden_size), so we take its transpose
        logits = torch.matmul(fused_output, self.item_embedding.weight.T) # Shape (batch, num_items)
        # Now we have output predicted "probabilities" (if we soft-maxed) over all next possible items
        return F.cross_entropy(logits, target)
    
    def predict(self, interaction) -> torch.FloatTensor:
        """Predict most likely next product according to output mode logits

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Prediction over entire batch of next product
        """
        item_seq = interaction[self.ITEM_SEQ] # Shape (batch, max_seq_len)
        item_seq_lengths = interaction[self.ITEM_SEQ_LEN] # Shape (batch,)
        item_candidates = interaction[self.ITEM_ID] # Shape (batch,)
        sequence_embeddings = self.forward(item_seq, item_seq_lengths) # Shape (batch, hidden_size)
        candidate_embeddings = self.item_embedding(item_candidates) # Shape (batch, hidden_size)
        # Dot product of sequence embedding with candidate embedding over batch
        return torch.sum(sequence_embeddings * candidate_embeddings, dim=1) # Shape (batch,)
    
    def full_sort_predict(self, interaction) -> torch.FloatTensor:
        """Calculate next item prediction scores associated with this interaction

        Args:
            interaction (RecBole Interaction batch object): Holds item_id_list, item_id_list_length, and target item_id

        Returns:
            torch.FloatTensor: Shape (batch, num_items) - scores across full vocab
        """
        item_sequence = interaction[self.ITEM_SEQ] # Shape (batch, max_seq_len)
        item_seq_len = interaction[self.ITEM_SEQ_LEN] # Shape (batch,)
        fused_output = self.forward(item_sequence, item_seq_len) # Shape (batch, hidden_size)
        # Once again, recall that item_embedding.weight is Shape (num_items, hidden_size), so we take its transpose
        return torch.matmul(fused_output, self.item_embedding.weight.T) # Shape (batch, num_items)