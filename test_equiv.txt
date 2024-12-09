import torch
import tensorflow as tf
import numpy as np

# Assuming the model files are in the same directory or properly imported
from pytorch_model import OmniGen as PyTorchOmniGen
from tensorflow_model import OmniGen as TensorFlowOmniGen
from transformer import Phi3Config

def create_test_configuration():
    """Create a consistent configuration for both models."""
    config = Phi3Config(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072
    )
    return config

def convert_pytorch_weights_to_tensorflow(pytorch_model, tensorflow_model):
    """
    Convert PyTorch model weights to TensorFlow model weights.
    Note: This is a basic implementation and might need refinement.
    """
    # Get PyTorch state dict
    pytorch_state_dict = pytorch_model.state_dict()
    
    # Prepare TensorFlow weights
    tf_weights = []
    
    # Conversion logic (simplified)
    for name, tensor in pytorch_state_dict.items():
        # Convert to numpy and adjust dimensions if necessary
        converted_tensor = tensor.detach().numpy()
        
        # Special handling for certain layers
        if len(converted_tensor.shape) == 2:  # Linear layer weights
            converted_tensor = converted_tensor.T
        
        tf_weights.append(converted_tensor)
    
    # Set TensorFlow model weights
    tensorflow_model.set_weights(tf_weights)
    
    return tensorflow_model

def prepare_test_inputs(batch_size=2, height=64, width=64, channels=4):
    """
    Prepare consistent input tensors for testing.
    
    Args:
        batch_size (int): Number of images in batch
        height (int): Image height
        width (int): Image width
        channels (int): Number of input channels
    
    Returns:
        Tuple of test inputs for both PyTorch and TensorFlow
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    tf.random.set_seed(42)
    
    # Create random input tensors
    pytorch_x = torch.randn(batch_size, channels, height, width)
    tensorflow_x = tf.convert_to_tensor(pytorch_x.numpy().transpose(0, 2, 3, 1))
    
    # Create timestep
    pytorch_timestep = torch.randint(0, 1000, (batch_size,))
    tensorflow_timestep = pytorch_timestep.numpy()
    
    # Optional: Create additional inputs like input_ids, attention masks
    pytorch_input_ids = torch.randint(0, 1000, (batch_size, 10))
    tensorflow_input_ids = pytorch_input_ids.numpy()
    
    return {
        'pytorch': {
            'x': pytorch_x,
            'timestep': pytorch_timestep,
            'input_ids': pytorch_input_ids,
        },
        'tensorflow': {
            'x': tensorflow_x,
            't': tensorflow_timestep,
            'c': tensorflow_input_ids,
        }
    }

def run_equivalence_test():
    """
    Run comprehensive model equivalence test.
    """
    # Create configuration
    config = create_test_configuration()
    
    # Instantiate models
    pytorch_model = PyTorchOmniGen(config)
    tensorflow_model = TensorFlowOmniGen(config)
    
    # Convert weights from PyTorch to TensorFlow
    tensorflow_model = convert_pytorch_weights_to_tensorflow(pytorch_model, tensorflow_model)
    
    # Prepare test inputs
    inputs = prepare_test_inputs()
    
    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model.forward(
            x=inputs['pytorch']['x'], 
            timestep=inputs['pytorch']['timestep'],
            input_ids=inputs['pytorch']['input_ids'],
            input_img_latents=None,
            input_image_sizes=None,
            attention_mask=None,
            position_ids=None
        )
    
    # Run TensorFlow inference
    tensorflow_output = tensorflow_model(
        x=inputs['tensorflow']['x'], 
        t=inputs['tensorflow']['t'],
        c=inputs['tensorflow']['c']
    )
    
    # Convert outputs to numpy for comparison
    pytorch_np = pytorch_output[0].numpy() if isinstance(pytorch_output, tuple) else pytorch_output.numpy()
    tensorflow_np = tensorflow_output.numpy()
    
    # Compare outputs
    print("Output Shape (PyTorch):", pytorch_np.shape)
    print("Output Shape (TensorFlow):", tensorflow_np.shape)
    
    # Compute comparison metrics
    abs_diff = np.abs(pytorch_np - tensorflow_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Maximum Absolute Difference: {max_diff}")
    print(f"Mean Absolute Difference: {mean_diff}")
    
    # Set tolerance for numerical differences
    tolerance = 1e-5
    
    assert max_diff < tolerance, f"Maximum difference {max_diff} exceeds tolerance {tolerance}"
    print("Model equivalence test passed successfully!")

if __name__ == "__main__":
    run_equivalence_test()