# export/export_onnx.py
import torch
import logging

logger = logging.getLogger("Q-GUARD")


def export_to_onnx(
    model: torch.nn.Module,
    input_dim: int,
    path: str = "qguard.onnx",
    opset_version: int = 14
) -> bool:
    """
    Export a PyTorch model to ONNX format with enterprise-grade safety.

    - Dynamically detects model device (CUDA/CPU)
    - Creates dummy input on the same device
    - Preserves and restores model training state
    - Supports dynamic batch sizes for TensorRT/FastAPI
    - Isolated exception handling prevents pipeline crashes

    Args:
        model: The PyTorch model to export.
        input_dim: Number of input features.
        path: Output ONNX file path.
        opset_version: ONNX opset version (14+ recommended for modern ops).

    Returns:
        bool: True if export succeeded, False otherwise.
    """

    # 1. Detect model device dynamically from actual parameter storage
    try:
        device = next(model.parameters()).device
    except StopIteration:
        logger.warning("Model has no parameters. Defaulting to CPU.")
        device = torch.device("cpu")

    logger.info(f"Detected model device: {device}")

    # 2. Preserve original training state
    was_training = model.training

    try:
        # 3. Switch to eval mode for deterministic tracing
        model.eval()

        # 4. Create dummy input on the SAME device as the model
        dummy_input = torch.randn(1, input_dim, dtype=torch.float32).to(device)

        logger.info(
            f"Exporting ONNX model -> {path} "
            f"| Input dim: {input_dim} "
            f"| Device: {device} "
            f"| Opset: {opset_version}"
        )

        # 5. Export with dynamic batch axes for TensorRT/FastAPI compatibility
        torch.onnx.export(
            model,
            dummy_input,
            path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits", "probs", "uncertainty", "attention_weights"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "probs": {0: "batch_size"},
                "uncertainty": {0: "batch_size"},
                "attention_weights": {0: "batch_size"}
            }
        )

        logger.info(f"ONNX export succeeded -> {path}")
        return True

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False

    finally:
        # 6. Restore original training state regardless of success/failure
        if was_training:
            model.train()
        logger.info(f"Model training state restored: {'train' if was_training else 'eval'}")