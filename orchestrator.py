import os
import torch
from torch.utils.data import DataLoader

from config import get_config
from src.utils.seed_utils import set_seed
from src.utils.device_utils import get_device
from src.utils.logging_utils import setup_logging, logger
from src.utils.plot_utils import plot_losses_curves, plot_single_model_report

from src.data_preprocessing.loader import load_data
from src.data_preprocessing.splitter import split_data
from src.data_preprocessing.preprocess_categorical import preprocess_categorical_data
from src.data_preprocessing.preprocess_target import preprocess_target

from src.preprocessors.word2vec_utils import (
    train_word2vec_model,
    build_vocab_from_w2v,
    create_embedding_matrix_w2v,
)
from src.preprocessors.vocabulary import text_to_sequence

from src.datasets.tokens_dataset import TokensDataset

from src.models.unfrozen_models import UnfrozenCNNWithResiduals

from src.training.early_stopping import EarlyStopping
from src.training.train_model import train_model
from src.training.save_model import save_model

from src.evaluation.evaluate_model import evaluate_model


def main():
    # Load configuration
    cfg = get_config()

    # Setup
    set_seed(cfg.general.random_seed)
    device, _ = get_device()

    g = torch.Generator()
    g.manual_seed(cfg.general.random_seed)

    setup_logging()

    # Load data
    df = load_data(file_path=cfg.paths.data_filepath, logger=logger)

    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df=df,
        test_size=cfg.data.split.test_size,
        valid_size=cfg.data.split.valid_size,
        random_state=cfg.general.random_seed,
        log=False,
        logger=logger,
    )

    # Combine text columns
    title_col = cfg.data.columns.title
    desc_col = cfg.data.columns.description

    X_train_combined = (X_train[title_col] + " " + X_train[desc_col]).tolist()
    X_valid_combined = (X_valid[title_col] + " " + X_valid[desc_col]).tolist()
    X_test_combined = (X_test[title_col] + " " + X_test[desc_col]).tolist()

    # Preprocess categorical data
    X_train_cat, X_valid_cat, X_test_cat, _ = preprocess_categorical_data(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        categorical_columns=cfg.data.columns.categorical,
        high_cardinality_columns=cfg.data.columns.high_cardinality,
        categorical_na_strategy="constant",
        categorical_fill_value="unknown",
        high_card_na_strategy="constant",
        high_card_fill_value="unknown",
        logger=logger,
    )

    # Preprocess target
    y_train_processed, y_valid_processed, y_test_processed, target_scaler = (
        preprocess_target(
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            log=False,
            save_artifacts=True,
            artifact_prefix="",
            logger=logger,
        )
    )

    # Train Word2Vec model
    w2v_model = train_word2vec_model(
        texts=X_train_combined,
        vector_size=cfg.model.embeddings.w2v_embedding_dim,
        window=5,
        min_count=20,
        epochs=10,
        logger=logger,
    )
    vocab_w2v = build_vocab_from_w2v(w2v_model=w2v_model, logger=logger)
    embedding_matrix = create_embedding_matrix_w2v(
        vocab=vocab_w2v, w2v_model=w2v_model, logger=logger
    )

    # Convert text to sequences
    max_seq_len = cfg.model.embeddings.max_seq_length
    X_train_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=max_seq_len,
            clean_tokens_use=False,
        )
        for text in X_train_combined
    ]
    X_valid_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=max_seq_len,
            clean_tokens_use=False,
        )
        for text in X_valid_combined
    ]
    X_test_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=max_seq_len,
            clean_tokens_use=False,
        )
        for text in X_test_combined
    ]

    # Create datasets
    train_dataset = TokensDataset(
        text_indices=X_train_seq, categorical=X_train_cat, targets=y_train_processed
    )
    valid_dataset = TokensDataset(
        text_indices=X_valid_seq, categorical=X_valid_cat, targets=y_valid_processed
    )
    test_dataset = TokensDataset(
        text_indices=X_test_seq, categorical=X_test_cat, targets=y_test_processed
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        generator=g,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        generator=g,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        generator=g,
    )

    # Initialize model
    model = UnfrozenCNNWithResiduals(
        embedding_matrix=embedding_matrix,
        categorical_dim=X_train_cat.shape[1],
        cat_hidden_size=cfg.model.architecture.cat_hidden_size,
        emb_hidden_size=cfg.model.architecture.emb_hidden_size,
        num_filters=cfg.model.architecture.num_filters,
        dropout=cfg.model.architecture.dropout_rate,
    ).to(device)

    # Train model
    model, history, elapsed_time = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        target_scaler=target_scaler,
        n_epochs=cfg.training.epochs,
        lr=cfg.training.learning_rate,
        optimizer_fn=cfg.training.optimizer,
        loss_fn=cfg.training.loss_function,
        device=device,
        early_stopping=EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            verbose=True,
            logger=logger,
        ),
        delta=cfg.model.loss.delta,
        use_lr_scheduler=True,
        scheduler_patience=cfg.training.scheduler.patience,
        scheduler_factor=cfg.training.scheduler.factor,
        logger=logger,
        multi_input=True,
        seed=cfg.general.random_seed,
        log=False,
    )

    # Save and evaluate
    model_name = "final_model"

    plot_losses_curves(
        history_dict=history,
        loss_fn=cfg.training.loss_function,
        save_path=os.path.join(
            cfg.paths.plots_dir, f"{model_name}_training_curves.png"
        ),
        logger=logger,
    )

    save_model(
        model=model,
        model_name=model_name,
        model_dir=cfg.paths.models_dir,
        logger=logger,
    )

    test_evaluation_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        model_name=model_name,
        training_time=elapsed_time,
        target_scaler=target_scaler,
        loss_fn=cfg.training.loss_function,
        device=device,
        logger=logger,
        multi_input=True,
        log=False,
        results_dict=None,
    )

    plot_single_model_report(
        model=model,
        model_name=model_name,
        history=history,
        elapsed_time=elapsed_time,
        test_metrics=test_evaluation_metrics,
        save_dir=cfg.paths.plots_dir,
        loss_fn=cfg.training.loss_function,
        logger=logger,
    )


if __name__ == "__main__":
    main()