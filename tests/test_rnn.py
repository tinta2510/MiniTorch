from minitorch import tensor, Module, Parameter, zeros
from minitorch.nn import LSTM, GRU, RNN, Linear, ReLU, Sigmoid, RParam, dropout
import traceback

# Unit tests for RNN, GRU, LSTM
def test_rnn():
    print("Testing RNN...")
    batch_size = 2
    seq_len = 3
    input_size = 4
    hidden_size = 5
    
    # Create input tensor (seq_len, batch, input_size)
    x = zeros((seq_len, batch_size, input_size))
    for t in range(seq_len):
        for b in range(batch_size):
            for i in range(input_size):
                x[t, b, i] = 0.1 * (t + b + i)
    
    # Create RNN
    rnn = RNN(input_size, hidden_size)
    
    # Forward pass
    try:
        output = rnn(x)
        print(f"RNN output shape: {output.shape}")
        assert output.shape == (seq_len, batch_size, hidden_size)
        print("RNN test passed!")
        return True
    except Exception as e:
        print(f"RNN test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_gru():
    print("Testing GRU...")
    batch_size = 2
    seq_len = 3
    input_size = 4
    hidden_size = 5
    
    # Create input tensor (seq_len, batch, input_size)
    x = zeros((seq_len, batch_size, input_size))
    for t in range(seq_len):
        for b in range(batch_size):
            for i in range(input_size):
                x[t, b, i] = 0.1 * (t + b + i)
    
    # Create GRU
    gru = GRU(input_size, hidden_size)
    
    # Forward pass
    try:
        output = gru(x)
        print(f"GRU output shape: {output.shape}")
        assert output.shape == (seq_len, batch_size, hidden_size)
        print("GRU test passed!")
        return True
    except Exception as e:
        print(f"GRU test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_lstm():
    print("Testing LSTM...")
    batch_size = 2
    seq_len = 3
    input_size = 4
    hidden_size = 5
    
    # Create input tensor (seq_len, batch, input_size)
    x = zeros((seq_len, batch_size, input_size))
    for t in range(seq_len):
        for b in range(batch_size):
            for i in range(input_size):
                x[t, b, i] = 0.1 * (t + b + i)
    
    # Create LSTM
    lstm = LSTM(input_size, hidden_size)
    
    # Forward pass
    try:
        output, (h, c) = lstm(x)
        print(f"LSTM output shape: {output.shape}")
        print(f"LSTM final hidden state shape: {h.shape}")
        print(f"LSTM final cell state shape: {c.shape}")
        assert output.shape == (seq_len, batch_size, hidden_size)
        assert h.shape == (batch_size, hidden_size)
        assert c.shape == (batch_size, hidden_size)
        print("LSTM test passed!")
        return True
    except Exception as e:
        print(f"LSTM test failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running unit tests for RNN, GRU, and LSTM...")
    rnn_passed = test_rnn()
    gru_passed = test_gru()
    lstm_passed = test_lstm()