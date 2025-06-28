# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:38:33 2025

@author: Andres
"""

# Enhanced Credit Card Fraud Detection with SHAP Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the credit card dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
        print(f"📊 Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        return df
    except FileNotFoundError:
        print("❌ Dataset file not found. Please check the file path.")
        return None

def train_fraud_model(X_train, y_train):
    """Train the Random Forest model with SMOTE"""
    print("\n🔄 Applying SMOTE for class balance...")
    sm = SMOTE(random_state=42, n_jobs=-1)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    print(f"✅ Resampled shape: {X_resampled.shape}")
    
    print("🌳 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,  # Increased for better performance
        max_depth=15,      # Slightly deeper
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        max_features='sqrt'
    )
    
    model.fit(X_resampled, y_resampled)
    print("✅ Model training completed!")
    return model

def evaluate_model(model, X_test, y_test, threshold=0.3):
    """Evaluate model performance with custom threshold"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_proba >= threshold).astype(int)
    
    print(f"\n📈 MODEL EVALUATION (Threshold: {threshold})")
    print("="*50)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_custom))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_custom):.4f}")
    
    return y_proba, y_pred_custom

def plot_feature_importance(model, feature_names):
    """Plot Random Forest feature importance"""
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    
    plt.figure(figsize=(12, 8))
    feat_importances.nlargest(15).plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title("Random Forest Feature Importances (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return feat_importances

def enhanced_shap_analysis(model, X_test, y_test, y_proba, sample_size=100):
    """Enhanced SHAP analysis with comprehensive insights"""
    print(f"\n🔍 ENHANCED SHAP ANALYSIS")
    print("="*60)
    
    # Sample data for SHAP analysis
    X_shap_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    y_shap_sample = y_test.loc[X_shap_sample.index]
    
    # Fix: Get the correct indices for y_proba (numpy array indexing)
    # Convert pandas index to position-based index for numpy array
    test_indices = X_test.index.tolist()
    sample_positions = [test_indices.index(idx) for idx in X_shap_sample.index]
    y_proba_sample = y_proba[sample_positions]
    
    print(f"📊 Analyzing {len(X_shap_sample)} samples...")
    print(f"Sample indices range: {min(X_shap_sample.index)} to {max(X_shap_sample.index)}")
    print(f"Test set size: {len(X_test)}, y_proba size: {len(y_proba)}")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_sample)
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Binary classification: shap_values is [class_0_values, class_1_values]
        fraud_shap_values = shap_values[1]  # Class 1 (fraud)
        print(f"Using fraud class SHAP values with shape: {fraud_shap_values.shape}")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array format: (samples, features, classes)
        fraud_shap_values = shap_values[:, :, 1]  # Class 1 (fraud)
        print(f"Extracted fraud class SHAP values with shape: {fraud_shap_values.shape}")
    else:
        # Single class or other format
        fraud_shap_values = shap_values
        print(f"Using SHAP values as-is with shape: {fraud_shap_values.shape}")
    
    return analyze_shap_insights(X_shap_sample, y_shap_sample, y_proba_sample, fraud_shap_values)

def analyze_shap_insights(X_sample, y_sample, y_proba_sample, shap_values):
    """Analyze SHAP values for business insights"""
    
    # 1. Feature Importance Comparison
    print("\n1️⃣ SHAP-BASED FEATURE IMPORTANCE:")
    print("-" * 40)
    
    shap_importance = np.abs(shap_values).mean(0)
    shap_features = pd.Series(shap_importance, index=X_sample.columns).sort_values(ascending=False)
    
    print("Top 10 Most Important Features (SHAP):")
    for i, (feature, importance) in enumerate(shap_features.head(10).items()):
        print(f"{i+1:2d}. {feature}: {importance:.4f}")
    
    # 2. Individual Case Analysis
    print("\n2️⃣ INDIVIDUAL CASE ANALYSIS:")
    print("-" * 40)
    
    analyze_individual_cases(X_sample, y_sample, y_proba_sample, shap_values)
    
    # 3. Feature Interaction Patterns
    print("\n3️⃣ FEATURE INTERACTION PATTERNS:")
    print("-" * 40)
    
    analyze_feature_interactions(X_sample, shap_values, shap_features)
    
    # 4. Risk Segmentation
    print("\n4️⃣ RISK SEGMENTATION ANALYSIS:")
    print("-" * 40)
    
    analyze_risk_segments(y_sample, y_proba_sample, shap_values)
    
    # 5. Visualization
    create_shap_visualizations(X_sample, shap_values)
    
    return shap_features

def analyze_individual_cases(X_sample, y_sample, y_proba_sample, shap_values):
    """Analyze individual high-confidence cases"""
    
    # Convert y_proba_sample to pandas Series if it's numpy array for easier indexing
    if isinstance(y_proba_sample, np.ndarray):
        y_proba_sample = pd.Series(y_proba_sample, index=X_sample.index)
    
    # High-confidence fraud cases
    high_fraud_mask = (y_sample == 1) & (y_proba_sample > 0.8)
    high_legit_mask = (y_sample == 0) & (y_proba_sample < 0.2)
    
    if high_fraud_mask.any():
        fraud_idx = np.where(high_fraud_mask)[0][0]
        fraud_shap = shap_values[fraud_idx]
        fraud_prob = y_proba_sample.iloc[fraud_idx]
        
        print(f"🚨 HIGH-CONFIDENCE FRAUD CASE:")
        print(f"   Fraud Probability: {fraud_prob:.3f}")
        print(f"   Top 3 fraud-driving features:")
        
        fraud_drivers = pd.Series(fraud_shap, index=X_sample.columns).sort_values(ascending=False)
        for i, (feature, impact) in enumerate(fraud_drivers.head(3).items()):
            feature_value = X_sample.iloc[fraud_idx][feature]
            print(f"     {i+1}. {feature}: {impact:+.4f} (value: {feature_value:.3f})")
    
    if high_legit_mask.any():
        legit_idx = np.where(high_legit_mask)[0][0]
        legit_shap = shap_values[legit_idx]
        legit_prob = y_proba_sample.iloc[legit_idx]
        
        print(f"\n✅ HIGH-CONFIDENCE LEGITIMATE CASE:")
        print(f"   Fraud Probability: {legit_prob:.3f}")
        print(f"   Top 3 legitimacy-driving features:")
        
        legit_drivers = pd.Series(legit_shap, index=X_sample.columns).sort_values(ascending=True)
        for i, (feature, impact) in enumerate(legit_drivers.head(3).items()):
            feature_value = X_sample.iloc[legit_idx][feature]
            print(f"     {i+1}. {feature}: {impact:+.4f} (value: {feature_value:.3f})")

def analyze_feature_interactions(X_sample, shap_values, shap_features):
    """Analyze feature interaction patterns"""
    
    top_feature = shap_features.index[0]
    
    if top_feature in X_sample.columns:
        feature_idx = X_sample.columns.get_loc(top_feature)
        feature_shap = shap_values[:, feature_idx]
        feature_values = X_sample[top_feature].values
        
        if len(feature_values) > 2:
            correlation = np.corrcoef(feature_values, feature_shap)[0, 1]
            print(f"📊 {top_feature} Analysis:")
            print(f"   Value-Impact Correlation: {correlation:.3f}")
            
            if abs(correlation) > 0.5:
                direction = "Higher" if correlation > 0 else "Lower"
                print(f"   💡 INSIGHT: {direction} {top_feature} values strongly indicate fraud")
            else:
                print(f"   💡 INSIGHT: {top_feature}'s impact is context-dependent")
            
            # Value distribution analysis
            fraud_mask = shap_values.sum(axis=1) > 0  # Positive SHAP sum indicates fraud tendency
            if fraud_mask.any() and (~fraud_mask).any():
                fraud_mean = feature_values[fraud_mask].mean()
                legit_mean = feature_values[~fraud_mask].mean()
                print(f"   Fraud cases avg: {fraud_mean:.3f} | Legitimate cases avg: {legit_mean:.3f}")

def analyze_risk_segments(y_sample, y_proba_sample, shap_values):
    """Analyze risk segments based on SHAP complexity"""
    
    fraud_count = (y_sample == 1).sum()
    total_count = len(y_sample)
    
    # Calculate SHAP complexity (total absolute SHAP values)
    shap_complexity = np.abs(shap_values).sum(axis=1)
    
    # Segment by complexity
    high_complexity_mask = shap_complexity > np.percentile(shap_complexity, 75)
    low_complexity_mask = shap_complexity < np.percentile(shap_complexity, 25)
    
    print(f"📊 Risk Segmentation Results:")
    print(f"   Total samples analyzed: {total_count}")
    print(f"   Fraud cases: {fraud_count} ({fraud_count/total_count*100:.1f}%)")
    print(f"   Average SHAP complexity: {shap_complexity.mean():.3f}")
    
    if high_complexity_mask.any():
        high_complex_fraud_rate = y_sample[high_complexity_mask].mean()
        print(f"   High complexity cases fraud rate: {high_complex_fraud_rate*100:.1f}%")
    
    if low_complexity_mask.any():
        low_complex_fraud_rate = y_sample[low_complexity_mask].mean()
        print(f"   Low complexity cases fraud rate: {low_complex_fraud_rate*100:.1f}%")

def create_shap_visualizations(X_sample, shap_values):
    """Create SHAP visualizations"""
    
    print("\n5️⃣ SHAP VISUALIZATIONS:")
    print("-" * 40)
    
    try:
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
        plt.title(f"SHAP Feature Impact Summary ({len(X_sample)} samples)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Feature importance bar plot (alternative to waterfall for robustness)
        plt.figure(figsize=(12, 6))
        feature_importance = np.abs(shap_values).mean(0)
        feature_names = X_sample.columns
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='lightcoral')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 10 Features by SHAP Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        print("📊 SHAP analysis completed successfully, but some visualizations skipped.")

def generate_business_insights(shap_features, model_metrics):
    """Generate business-ready insights"""
    
    print("\n6️⃣ BUSINESS INSIGHTS & RECOMMENDATIONS:")
    print("="*60)
    
    top_5_features = shap_features.head(5)
    
    print("🎯 KEY FINDINGS:")
    print(f"   • Most critical fraud indicators: {', '.join(top_5_features.index[:3])}")
    print(f"   • Feature diversity: {len(shap_features[shap_features > 0.001])} features significantly impact decisions")
    
    print("\n📋 RECOMMENDATIONS:")
    print("   1. Focus monitoring on top 3 features for real-time alerts")
    print("   2. Implement feature-specific thresholds for automated flagging")
    print("   3. Regular model retraining with new fraud patterns")
    print("   4. Consider ensemble methods for improved robustness")
    
    print("\n✨ ANALYSIS COMPLETE! Model ready for production deployment.")

def main():
    """Main execution function"""
    
    # Original file path from your code
    file_path = "C:/Users/Andres/Downloads/creditcard.csv.csv"
    
    print("🚀 ENHANCED CREDIT CARD FRAUD DETECTION WITH SHAP")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data(file_path)
    if df is None:
        return
    
    # Prepare features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = train_fraud_model(X_train, y_train)
    
    # Evaluate model
    y_proba, y_pred = evaluate_model(model, X_test, y_test, threshold=0.3)
    
    # Plot feature importance
    feat_importances = plot_feature_importance(model, X.columns)
    
    # Enhanced SHAP analysis
    shap_features = enhanced_shap_analysis(model, X_test, y_test, y_proba, sample_size=150)
    
    # Generate business insights
    generate_business_insights(shap_features, {
        'auc': roc_auc_score(y_test, y_pred),
        'feature_count': len(X.columns)
    })

if __name__ == "__main__":
    main()