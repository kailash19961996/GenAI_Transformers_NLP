from transformer import *

tgt_vocab_size = 5000
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 512
max_seq_length = 100
dropout = 0.1

# Load the trained model
model = Transformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
model.load_state_dict(torch.load("model_2.pth"))
model.eval()

# Example usage
sp = spm.SentencePieceProcessor()
sp.load('sentencepiece_small_m.model')

def word_to_token_id(word, sp_model):
    # Convert word to token ID using SentencePiece
    return sp_model.piece_to_id(word)

def generate_text(starting_word):
    
    max_length = 100
    ending_word = "</sos>"
    # Convert starting and ending words to token IDs
    starting_token_id = word_to_token_id(starting_word, sp)
    if starting_token_id is None:
        raise ValueError(f"Starting word '{starting_word}' not found in vocabulary.")
    
    ending_token_id = word_to_token_id(ending_word, sp)
    if ending_token_id is None:
        raise ValueError(f"Ending word '{ending_word}' not found in vocabulary.")
    
    
    generated_sequence = [starting_token_id]
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([generated_sequence])
            output = model(input_tensor)
            predicted_token = output.argmax(-1)[:,-1].item()
            generated_sequence.append(predicted_token)
            if predicted_token == ending_token_id:
                break
            
    # Convert token IDs to words using SentencePiece
    generated_text = sp.decode_ids(generated_sequence)
    return generated_text
