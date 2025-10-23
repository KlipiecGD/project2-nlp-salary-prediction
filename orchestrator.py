import os
import logging
import torch
from torch.utils.data import DataLoader


from config.config import (
    FILEPATH,
    MODELS_DIR,
    PLOTS_DIR,
    TEST_SIZE,
    VALID_SIZE,
    CATEGORICAL_COLUMNS,
    HIGH_CARDINALITY_COLUMNS,
    RANDOM_SEED,
    MAX_SEQ_LENGTH,
    NUM_WORKERS,
    BATCH_SIZE,
    EPOCHS,
    CAT_HIDDEN_SIZE,
    EMB_HIDDEN_SIZE,
    W2V_EMBEDDING_DIM,
    NUM_FILTERS,
    DROPOUT_RATE,
    LOSS_FUNCTION,
    OPTIMIZER,
    LEARNING_RATE,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE,
    DELTA,
)
from src.utils.seed_utils import set_seed, seed_worker
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
    set_seed(RANDOM_SEED)
    device, _ = get_device()

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    setup_logging()

    df = load_data(file_path=FILEPATH, logger=logger)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df=df,
        test_size=TEST_SIZE,
        valid_size=VALID_SIZE,
        random_state=RANDOM_SEED,
        log=False,
        logger=logger,
    )

    X_train_combined = (X_train["Title"] + " " + X_train["FullDescription"]).tolist()
    X_valid_combined = (X_valid["Title"] + " " + X_valid["FullDescription"]).tolist()
    X_test_combined = (X_test["Title"] + " " + X_test["FullDescription"]).tolist()

    X_train_cat, X_valid_cat, X_test_cat, _ = preprocess_categorical_data(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        categorical_columns=CATEGORICAL_COLUMNS,
        high_cardinality_columns=HIGH_CARDINALITY_COLUMNS,
        categorical_na_strategy="constant",
        categorical_fill_value="unknown",
        high_card_na_strategy="constant",
        high_card_fill_value="unknown",
        logger=logger,
    )

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

    w2v_model = train_word2vec_model(
        texts=X_train_combined,
        vector_size=W2V_EMBEDDING_DIM,
        window=5,
        min_count=20,
        epochs=10,
        logger=logger,
    )
    vocab_w2v = build_vocab_from_w2v(w2v_model=w2v_model, logger=logger)
    embedding_matrix = create_embedding_matrix_w2v(
        vocab=vocab_w2v, w2v_model=w2v_model, logger=logger
    )

    X_train_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=MAX_SEQ_LENGTH,
            clean_tokens_use=False,
        )
        for text in X_train_combined
    ]
    X_valid_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=MAX_SEQ_LENGTH,
            clean_tokens_use=False,
        )
        for text in X_valid_combined
    ]
    X_test_seq = [
        text_to_sequence(
            text=text,
            vocab=vocab_w2v,
            max_length=MAX_SEQ_LENGTH,
            clean_tokens_use=False,
        )
        for text in X_test_combined
    ]

    train_dataset = TokensDataset(
        text_indices=X_train_seq, categorical=X_train_cat, targets=y_train_processed
    )
    valid_dataset = TokensDataset(
        text_indices=X_valid_seq, categorical=X_valid_cat, targets=y_valid_processed
    )
    test_dataset = TokensDataset(
        text_indices=X_test_seq, categorical=X_test_cat, targets=y_test_processed
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=g,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=g,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    model = UnfrozenCNNWithResiduals(
        embedding_matrix=embedding_matrix,
        categorical_dim=X_train_cat.shape[1],
        cat_hidden_size=CAT_HIDDEN_SIZE,
        emb_hidden_size=EMB_HIDDEN_SIZE,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    model, history, elapsed_time = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        target_scaler=target_scaler,
        n_epochs=EPOCHS,
        lr=LEARNING_RATE,
        optimizer_fn=OPTIMIZER,
        loss_fn=LOSS_FUNCTION,
        device=device,
        early_stopping=EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=logger
        ),
        delta=DELTA,
        use_lr_scheduler=True,
        scheduler_patience=SCHEDULER_PATIENCE,
        scheduler_factor=SCHEDULER_FACTOR,
        logger=logger,
        multi_input=True,
        seed=RANDOM_SEED,
        log=False,
    )

    model_name = "final_model"

    plot_losses_curves(
        history=history,
        loss_fn=LOSS_FUNCTION,
        plot_dir=os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png"),
        logger=logger,
    )

    save_model(
        model=model,
        model_name=model_name,
        model_dir=MODELS_DIR,
        logger=logger,
    )
    test_evaluation_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        model_name=model_name,
        training_time=elapsed_time,
        target_scaler=target_scaler,
        loss_fn=LOSS_FUNCTION,
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
        save_dir=PLOTS_DIR,
        loss_fn=LOSS_FUNCTION,
        logger=logger,
    )


if __name__ == "__main__":
    main()
