# export/export_trt.py

import logging

logger = logging.getLogger("Q-GUARD")


def export_to_tensorrt(
    onnx_path: str = "qguard.onnx",
    trt_path: str = "qguard_trt.engine",
    input_dim: int = 8,
    fp16: bool = True,
    max_workspace_gb: int = 2,
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 64
) -> bool:
    """
    Convert ONNX model to TensorRT engine with dynamic batching support.

    Args:
        onnx_path: Path to exported ONNX model
        trt_path: Output TensorRT engine path
        input_dim: Number of input features
        fp16: Enable FP16 acceleration
        max_workspace_gb: GPU workspace size
        min_batch: Minimum batch size
        opt_batch: Optimal batch size
        max_batch: Maximum batch size

    Returns:
        bool: Success status
    """

    try:
        import tensorrt as trt

        logger.info(
            "TensorRT detected. Converting ONNX -> TensorRT engine..."
        )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Create builder
        builder = trt.Builder(TRT_LOGGER)

        # Explicit batch flag required for dynamic batching
        network_flags = (
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Load ONNX model
        with open(onnx_path, "rb") as model_file:

            parsed = parser.parse(model_file.read())

            if not parsed:

                logger.error("Failed to parse ONNX model.")

                for i in range(parser.num_errors):
                    logger.error(
                        f"TensorRT Parse Error [{i}]: "
                        f"{parser.get_error(i)}"
                    )

                return False

        logger.info("ONNX model parsed successfully.")

        # Builder config
        config = builder.create_builder_config()

        # Workspace memory
        workspace_size = max_workspace_gb * (1 << 30)

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size
        )

        logger.info(
            f"TensorRT workspace size: "
            f"{max_workspace_gb} GB"
        )

        # Enable FP16 if GPU supports it
        if fp16 and builder.platform_has_fast_fp16:

            config.set_flag(trt.BuilderFlag.FP16)

            logger.info(
                "FP16 precision enabled for TensorRT."
            )

        # ==========================================
        # DYNAMIC BATCH OPTIMIZATION PROFILE
        # ==========================================

        input_tensor = network.get_input(0)

        input_name = input_tensor.name

        logger.info(
            f"TensorRT input tensor: {input_name}"
        )

        profile = builder.create_optimization_profile()

        profile.set_shape(
            input_name,

            # Minimum shape
            (min_batch, input_dim),

            # Optimal shape
            (opt_batch, input_dim),

            # Maximum shape
            (max_batch, input_dim)
        )

        config.add_optimization_profile(profile)

        logger.info(
            "Optimization profile added: "
            f"min=({min_batch},{input_dim}) | "
            f"opt=({opt_batch},{input_dim}) | "
            f"max=({max_batch},{input_dim})"
        )

        # ==========================================
        # BUILD ENGINE
        # ==========================================

        logger.info(
            "Building TensorRT engine... "
            "(this may take some time)"
        )

        serialized_engine = builder.build_serialized_network(
            network,
            config
        )

        if serialized_engine is None:

            logger.error(
                "TensorRT engine build failed."
            )

            return False

        # Save engine
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(
            f"TensorRT engine saved -> {trt_path}"
        )

        logger.info(
            "TensorRT conversion completed successfully."
        )

        return True

    except ImportError:

        logger.warning(
            "TensorRT not installed. "
            "Skipping TensorRT conversion."
        )

        return False

    except Exception as e:

        logger.error(
            f"TensorRT conversion failed: {str(e)}"
        )

        return False