"""
Complete BERT-based Model Training for Phishing Detection
Supports both URLs and emails with state-of-the-art accuracy
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Configuration dictionary (adjust parameters as needed)
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 3,
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': 'models/bert_phishing_model',
    'save_strategy': 'epoch',      # keep as is (valid)
    'eval_strategy': 'epoch',      # renamed from 'evaluation_strategy'
    'logging_steps': 100,
    'save_total_limit': 2,
    'load_best_model_at_end': True
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDataset(torch.utils.data.Dataset):
    """Custom Dataset for BERT tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Return dict of tensors (flattened batches)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # flatten batch dim
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_phishing_data():
    """
    Load phishing detection dataset.
    Expected format: CSV file(s) with 'text' and 'label' columns.
    label: 0 = legitimate, 1 = phishing
    """
    try:
        data_sources = [
            'data/phishing_data.csv',
            'data/phishing_urls.csv',
            'data/phishing_emails.csv'
        ]

        all_data = []

        for source in data_sources:
            if os.path.exists(source):
                df = pd.read_csv(source)
                logger.info(f"Loaded {len(df)} samples from {source}")
                all_data.append(df)

        if not all_data:
            logger.warning("No data files found. Creating sample dataset...")
            sample_data = create_sample_data()
            return sample_data['text'].tolist(), sample_data['label'].tolist()

        # Combine datasets
        combined_df = pd.concat(all_data, ignore_index=True)

        # Clean data
        combined_df = combined_df.dropna()
        combined_df = combined_df.drop_duplicates()

        # Ensure labels are integer
        combined_df['label'] = combined_df['label'].astype(int)

        logger.info(f"Total samples: {len(combined_df)}")
        logger.info(f"Legitimate: {sum(combined_df['label'] == 0)}")
        logger.info(f"Phishing: {sum(combined_df['label'] == 1)}")

        return combined_df['text'].tolist(), combined_df['label'].tolist()

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sample_data = create_sample_data()
        return sample_data['text'].tolist(), sample_data['label'].tolist()

def create_sample_data():
    """Create sample phishing and legitimate dataset for initial training/demo."""
    sample_texts = [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.github.com",
        "https://www.wikipedia.org",
        "https://www.stackoverflow.com",
        "https://www.linkedin.com",
        # Legitimate emails
        "Thank you for your recent purchase. Your order will be shipped soon.",
        "Your meeting is scheduled for tomorrow at 2 PM.",
        "Welcome to our newsletter. Here are this week's updates.",
        # Phishing URLs
        "http://paypal-security-update.tk/verify-account-now",
        "https://amazon-update.tk/confirm-payment-details",
        "http://google-security-alert.ml/verify-suspicious-login",
        "https://microsoft-security.ga/urgent-account-verification",
        "http://bank-security-notice.cf/immediate-action-required",
        # Phishing emails
        "URGENT: Your account will be suspended unless you verify immediately!",
        "Click here to claim your $1000 prize now! Limited time offer!",
        "Your PayPal account has been limited. Verify now to restore access.",
        "SECURITY ALERT: Suspicious activity detected. Click to secure account.",
        "Congratulations! You've won the lottery. Send personal details to claim."
    ]

    sample_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Expand sample data to make enough samples for training/demo
    expanded_texts = sample_texts * 100
    expanded_labels = sample_labels * 100

    return pd.DataFrame({'text': expanded_texts, 'label': expanded_labels})

def compute_metrics(eval_pred):
    """Compute evaluation metrics for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    logger.info("Starting BERT phishing detection training...")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Load dataset
    texts, labels = load_phishing_data()

    # Train / validation split (stratified)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=labels
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Load tokenizer and model pretrained on bert-base-uncased
    logger.info("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2
    )

    # Create PyTorch datasets for Trainer
    train_dataset = PhishingDataset(train_texts, train_labels, tokenizer, max_length=CONFIG['max_length'])
    val_dataset = PhishingDataset(val_texts, val_labels, tokenizer, max_length=CONFIG['max_length'])

    # Setup TrainingArguments with new arg name: eval_strategy (not evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(CONFIG['output_dir'], 'logs'),
        logging_steps=CONFIG['logging_steps'],
        eval_strategy=CONFIG['eval_strategy'],     # <-- Changed here
        save_strategy=CONFIG['save_strategy'],
        save_total_limit=CONFIG['save_total_limit'],
        load_best_model_at_end=CONFIG['load_best_model_at_end'],
        metric_for_best_model='f1',
        greater_is_better=True,
        seed=CONFIG['random_state']
    )

    # Initialize Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("Evaluating final model...")
    eval_results = trainer.evaluate()
    logger.info("Final Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"{key}: {value:.4f}")

    # Save final model and tokenizer
    logger.info(f"Saving model and tokenizer to {CONFIG['output_dir']}")
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])

    # Generate detailed classification report on validation set
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    report = classification_report(
        val_labels,
        y_pred,
        target_names=['Legitimate', 'Phishing'],
        output_dict=False
    )
    logger.info("Detailed Classification Report:")
    logger.info(f"\n{report}")

    # Save metrics to JSON file
    metrics = {
        'accuracy': accuracy_score(val_labels, y_pred),
        'precision': precision_recall_fscore_support(val_labels, y_pred, average='binary')[0],
        'recall': precision_recall_fscore_support(val_labels, y_pred, average='binary')[1],
        'f1': precision_recall_fscore_support(val_labels, y_pred, average='binary')[2],
    }

    metrics_file = os.path.join(CONFIG['output_dir'], 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training completed successfully!")
    return model, tokenizer, metrics

if __name__ == "__main__":
    main()
