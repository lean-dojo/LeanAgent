import os
from leanagent import *
from retrieval.fisher_computation_module import FisherComputationModule

new_data_path = os.environ.get("NEW_DATA_PATH", "datasets/new_data")

def main():
    """
    The main function that drives LeanAgent.
    This function performs the following steps:
    1. Logs the start of Fisher Information Matrix calculation for EWC.
    2. Checks if CUDA is available and sets the device accordingly.
    3. Defines the configuration for the model.
    4. Attempts to find and load the latest model checkpoint; if not found, uses the current model state.
    5. Creates a Fisher computation module.
    6. Sets environment variables for NCCL timeout and error handling.
    7. Configures the DDP strategy and sets up the trainer for Fisher computation.
    8. Prepares the data module for retrieval tasks.
    9. Runs the Fisher computation using the trainer and data module.
    10. Saves the Fisher Information Matrix if the process is successful and the current process is the global zero.
    Exceptions:
    - Logs and handles various exceptions that may occur during the process.
    """
    try:
        logger.info("Calculating Fisher Information Matrix for EWC")
        ### FISHER INFORMATION MATRIX FOR NEXT EWC
        
        if not torch.cuda.is_available():
            logger.warning("Indexing the corpus using CPU can be very slow.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
            "lr": 1e-3,
            "warmup_steps": 1000,
            "max_seq_len": 512,
            "num_retrieved": 100,
        }

        try:
            best_model_path = find_latest_checkpoint()
            logger.info(f"Found latest checkpoint: {best_model_path}")
            best_model = PremiseRetriever.load(best_model_path, device, freeze=False, config=config)
        except FileNotFoundError as e:
            logger.error(f"No checkpoint found: {str(e)}")
            logger.warning("Using the current model state.")
            best_model = model

        # Create Fisher computation module
        fisher_module = FisherComputationModule(best_model)

        VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['NCCL_TIMEOUT'] = str(VERY_LONG_TIMEOUT * 1000)

        ddp_strategy = DDPStrategy(timeout=timedelta(seconds=VERY_LONG_TIMEOUT))
        # Setup trainer for Fisher computation
        # Use 4 GPUs for Fisher computation
        fisher_trainer = pl.Trainer(
            accelerator="gpu",
            precision="bf16-mixed",
            strategy=ddp_strategy,
            devices=4,
            max_epochs=1,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
        )

        corpus_path = new_data_path + "/corpus.jsonl"
        data_path = new_data_path + "/random"
        logger.info(f"Data path: {data_path}")
        data_module = RetrievalDataModule(
            data_path=data_path,
            corpus_path=corpus_path,
            num_negatives=3,
            num_in_file_negatives=1,
            model_name="google/byt5-small",
            batch_size=BATCH_SIZE,
            eval_batch_size=64,
            max_seq_len=1024,
            num_workers=4
        )
        data_module.setup(stage='fit')

        try:
            logger.info("right before barrier fisher")
            fisher_trainer.strategy.barrier()
            fisher_trainer.fit(fisher_module, datamodule=data_module)
            logger.info("right after barrier fisher")
            fisher_trainer.strategy.barrier()

            # Save the FIM if needed
            if fisher_trainer.is_global_zero:
                fisher_file_path = os.path.join(RAID_DIR, FISHER_DIR, f"fisher_info_{new_data_path.split('/')[-1]}_distributed.pkl")
                fisher_module.save_fisher_info(fisher_file_path)
                logger.info(f"Fisher Information Matrix saved at {fisher_file_path}")
        except Exception as e:
            print(f"An error occurred during fisher: {str(e)}")
            print(traceback.format_exc())

    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
