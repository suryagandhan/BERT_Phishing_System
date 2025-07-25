"""
BERT-based Phishing Detection Prediction Module
Real-time prediction for URLs and emails with confidence scoring
"""

import os
import torch
import json
import time
import logging
from typing import List, Dict, Union
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertPhishingPredictor:
    """
    BERT-based phishing detection predictor for URLs and emails
    Provides real-time predictions with confidence scores
    """
    
    def __init__(self, model_path: str = "models/bert_phishing_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model()
        
        # Load metrics if available
        self.metrics = self._load_metrics()
        
        logger.info(f"Predictor loaded on device: {self.device}")
    
    def _load_model(self):
        """Load the trained BERT model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Attempting to load base BERT model for demonstration...")
            
            # Fallback to base model
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.warning("Using base BERT model - predictions may be less accurate")
    
    def _load_metrics(self) -> Dict:
        """Load training metrics if available"""
        metrics_path = os.path.join(self.model_path, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    def predict_single(self, text: str) -> Dict[str, Union[int, float, str]]:
        """
        Predict phishing probability for a single text input
        
        Args:
            text: URL or email text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                phishing_prob = probabilities[0, 1].item()
                legitimate_prob = probabilities[0, 0].item()
                
                prediction = 1 if phishing_prob > 0.5 else 0
                confidence = max(phishing_prob, legitimate_prob)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'text': text[:100] + '...' if len(text) > 100 else text
            }
        
        processing_time = time.time() - start_time
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': prediction,
            'label': 'phishing' if prediction == 1 else 'legitimate',
            'phishing_probability': round(phishing_prob, 4),
            'legitimate_probability': round(legitimate_prob, 4),
            'confidence': round(confidence, 4),
            'risk_level': self._get_risk_level(phishing_prob),
            'processing_time': round(processing_time, 3)
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict phishing probability for multiple texts
        
        Args:
            texts: List of URLs or email texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Processing batch of {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            result = self.predict_single(text)
            result['batch_index'] = i
            results.append(result)
        
        return results
    
    def _get_risk_level(self, phishing_prob: float) -> str:
        """Determine risk level based on phishing probability"""
        if phishing_prob < 0.3:
            return 'LOW'
        elif phishing_prob < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'max_length': self.max_length,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'metrics': self.metrics
        }
    
    def analyze_url(self, url: str) -> Dict:
        """
        Analyze a URL for phishing indicators
        
        Args:
            url: URL to analyze
            
        Returns:
            Detailed analysis results
        """
        result = self.predict_single(url)
        
        # Add URL-specific analysis
        url_features = self._extract_url_features(url)
        result.update({
            'url_features': url_features,
            'recommendation': self._get_recommendation(result['prediction'], result['confidence'])
        })
        
        return result
    
    def analyze_email(self, email_text: str, subject: str = None) -> Dict:
        """
        Analyze an email for phishing indicators
        
        Args:
            email_text: Email body text
            subject: Email subject (optional)
            
        Returns:
            Detailed analysis results
        """
        # Combine subject and body if both provided
        if subject:
            full_text = f"Subject: {subject}\n\n{email_text}"
        else:
            full_text = email_text
        
        result = self.predict_single(full_text)
        
        # Add email-specific analysis
        email_features = self._extract_email_features(email_text, subject)
        result.update({
            'email_features': email_features,
            'recommendation': self._get_recommendation(result['prediction'], result['confidence'])
        })
        
        return result
    
    def _extract_url_features(self, url: str) -> Dict:
        """Extract basic URL features for phishing analysis, handling edge cases safely."""
        parts = url.split('/')
        
        # Check if URL has a host part to examine
        if len(parts) > 2:
            host_part = parts[2]
            has_ip = any(char.isdigit() for char in host_part)
            # Alternatively, you can add a regex check for IP address format here if needed
        else:
            has_ip = False

        # Count dots minus 1 assuming domain structure, with a minimum of 0 for safety
        subdomain_count = max(url.count('.') - 1, 0)

        # Count suspicious keywords in URL (case insensitive)
        suspicious_keywords = sum(
            1 for word in ['verify', 'urgent', 'click', 'secure', 'update'] 
            if word in url.lower()
        )

        return {
            'length': len(url),
            'has_https': url.lower().startswith('https://'),
            'has_suspicious_tld': any(tld in url.lower() for tld in ['.tk', '.ml', '.ga', '.cf']),
            'has_ip': has_ip,
            'subdomain_count': subdomain_count,
            'suspicious_keywords': suspicious_keywords
        }
        
    def _extract_email_features(self, email_text: str, subject: str = None) -> Dict:
        """Extract basic email features for analysis"""
        text = email_text.lower()
        
        return {
            'length': len(email_text),
            'has_subject': subject is not None,
            'urgency_words': sum(1 for word in ['urgent', 'immediate', 'asap', 'quickly'] 
                               if word in text),
            'financial_words': sum(1 for word in ['bank', 'credit', 'payment', 'account'] 
                                 if word in text),
            'suspicious_phrases': sum(1 for phrase in ['click here', 'verify now', 'act now'] 
                                    if phrase in text),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in email_text if c.isupper()) / len(email_text) if email_text else 0
        }
    
    def _get_recommendation(self, prediction: int, confidence: float) -> str:
        """Get recommendation based on prediction and confidence"""
        if prediction == 1:  # Phishing
            if confidence > 0.9:
                return "BLOCK - High confidence phishing attempt detected"
            elif confidence > 0.7:
                return "WARN - Likely phishing attempt, exercise caution"
            else:
                return "REVIEW - Potential phishing indicators found"
        else:  # Legitimate
            if confidence > 0.9:
                return "ALLOW - High confidence legitimate content"
            elif confidence > 0.7:
                return "ALLOW - Likely legitimate, low risk"
            else:
                return "MONITOR - Uncertain classification, monitor closely"

def main():
    """Test the predictor with sample data"""
    # Initialize predictor
    predictor = BertPhishingPredictor()
    
    # Display model info
    model_info = predictor.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://www.linkedin.com",
        "http://paypal-security-update.tk/verify-account-now",
        "https://amazon-confirm-payment.ga/urgent-verification",
        "http://microsoft-security-alert.ml/suspicious-login-verify"
    ]
    
    print("URL Analysis Results:")
    print("-" * 80)
    
    for url in test_urls:
        result = predictor.analyze_url(url)
        print(f"URL: {result['text']}")
        print(f"Prediction: {result['label'].upper()} ({result['confidence']:.3f} confidence)")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Processing Time: {result['processing_time']}s")
        print("-" * 80)
    
    # Test emails
    test_emails = [
        {
            'subject': 'Meeting Reminder',
            'body': 'Your meeting is scheduled for tomorrow at 2 PM. Please confirm your attendance.'
        },
        {
            'subject': 'URGENT: Account Verification Required',
            'body': 'Your account will be suspended unless you verify immediately! Click here to verify now and avoid account closure.'
        },
        {
            'subject': 'Prize Notification',
            'body': 'Congratulations! You have won $10,000 in our lottery. Send your personal details to claim your prize now!'
        }
    ]
    
    print("\nEmail Analysis Results:")
    print("-" * 80)
    
    for email in test_emails:
        result = predictor.analyze_email(email['body'], email['subject'])
        print(f"Subject: {email['subject']}")
        print(f"Prediction: {result['label'].upper()} ({result['confidence']:.3f} confidence)")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Processing Time: {result['processing_time']}s")
        print("-" * 80)
    
    # Batch prediction test
    all_texts = test_urls + [email['body'] for email in test_emails]
    batch_results = predictor.predict_batch(all_texts)
    
    print(f"\nBatch Prediction Summary:")
    phishing_count = sum(1 for r in batch_results if r['prediction'] == 1)
    print(f"Total texts analyzed: {len(batch_results)}")
    print(f"Phishing detected: {phishing_count}")
    print(f"Legitimate: {len(batch_results) - phishing_count}")
    print(f"Average confidence: {np.mean([r['confidence'] for r in batch_results]):.3f}")

if __name__ == "__main__":
    main()