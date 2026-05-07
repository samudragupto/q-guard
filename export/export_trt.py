# export/export_trt.py
import logging

logger = logging.getLogger("Q-GUARD")


def export_to_tensorrt(
    onnx_path: str = "qguard.onnx",
    trt_path: str = "qguard_trt.engine",
    fp16: bool = True,
    max_workspace_gb: int = 2
) -> bool:
    """
    Convert ONNX model to TensorRT engine with GPU optimization.

    Args:
        onnx_path: Path to the ONNX model.
        trt_path: Output TensorRT engine path.
        fp16: Enable FP16 precision (2x throughput on modern GPUs).
        max_workspace_gb: Maximum GPU workspace in GB.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    try:
        import tensorrt as trt

        logger.info("TensorRT detected. Converting ONNX -> TensorRT engine...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"TRT Parse Error: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << (30 * max_workspace_gb)
        )

        # FP16 Precision for maximum GPU throughput
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled for TensorRT.")

        # Build and serialize engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("TensorRT engine build failed.")
            return False

        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved -> {trt_path}")
        return True

    except ImportError:
        logger.warning(
            "TensorRT not installed. Skipping TRT conversion. "
            "Use ONNX runtime for inference instead."
        )
        return False

    except Exception as e:
        logger.error(f"TensorRT conversion failed: {e}")
        return False