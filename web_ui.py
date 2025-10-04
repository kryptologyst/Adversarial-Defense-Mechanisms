"""
Streamlit web UI for adversarial defense experiments.
"""
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import io
import json
from pathlib import Path
import time

from src.models import create_model
from src.attacks import create_attack
from src.defenses import create_defense
from src.database import ExperimentDatabase
from src.visualization import AdversarialVisualizer


# Page configuration
st.set_page_config(
    page_title="Adversarial Defense Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .attack-result {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
    }
    .defense-result {
        background-color: #d1ecf1;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_type: str):
    """Load and cache model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type)
    return model, device


@st.cache_data
def load_mnist_sample():
    """Load a sample of MNIST data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    
    images, labels = next(iter(loader))
    return images[:20], labels[:20]  # Return first 20 samples


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Adversarial Defense Lab</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["modern_cnn", "resnet"],
        help="Choose the neural network architecture"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model(model_type)
    
    st.sidebar.success(f"✅ {model_type.upper()} model loaded on {device}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Interactive Testing", 
        "📊 Experiment Results", 
        "🔬 Attack Comparison",
        "🛡️ Defense Analysis",
        "📈 Database Explorer"
    ])
    
    with tab1:
        interactive_testing(model, device)
    
    with tab2:
        experiment_results()
    
    with tab3:
        attack_comparison(model, device)
    
    with tab4:
        defense_analysis(model, device)
    
    with tab5:
        database_explorer()


def interactive_testing(model, device):
    """Interactive testing tab."""
    st.header("🎯 Interactive Adversarial Testing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Attack Configuration")
        
        attack_type = st.selectbox(
            "Attack Type",
            ["fgsm", "pgd", "bim", "cw"],
            help="Choose the adversarial attack method"
        )
        
        epsilon = st.slider(
            "Epsilon (Attack Strength)",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.01,
            help="Higher values create stronger attacks"
        )
        
        if attack_type == "pgd":
            steps = st.slider("PGD Steps", 1, 50, 10)
            alpha = st.slider("PGD Alpha", 0.001, 0.1, 0.01, step=0.001)
        else:
            steps = None
            alpha = None
    
    with col2:
        st.subheader("Defense Configuration")
        
        use_defense = st.checkbox("Enable Defense", value=False)
        
        if use_defense:
            defense_type = st.selectbox(
                "Defense Type",
                ["jpeg", "gaussian_noise", "gaussian_blur", "median_filter", "bit_depth_reduction"],
                help="Choose the defense mechanism"
            )
            
            if defense_type == "jpeg":
                quality = st.slider("JPEG Quality", 10, 100, 75)
            elif defense_type == "gaussian_noise":
                noise_std = st.slider("Noise Std", 0.01, 0.3, 0.1, step=0.01)
            elif defense_type == "gaussian_blur":
                blur_sigma = st.slider("Blur Sigma", 0.1, 3.0, 1.0, step=0.1)
            else:
                quality = noise_std = blur_sigma = None
        else:
            defense_type = None
            quality = noise_std = blur_sigma = None
    
    # Load sample data
    with st.spinner("Loading MNIST samples..."):
        images, labels = load_mnist_sample()
    
    # Test button
    if st.button("🚀 Run Attack Test", type="primary"):
        with st.spinner("Generating adversarial examples..."):
            # Create attack
            attack_kwargs = {"epsilon": epsilon}
            if attack_type == "pgd":
                attack_kwargs.update({"steps": steps, "alpha": alpha})
            
            attack = create_attack(attack_type, model, device, **attack_kwargs)
            
            # Test on first 5 samples
            test_images = images[:5].to(device)
            test_labels = labels[:5].to(device)
            
            # Generate adversarial examples
            adv_images = attack.attack(test_images, test_labels)
            
            # Apply defense if enabled
            if use_defense:
                defense_kwargs = {}
                if defense_type == "jpeg":
                    defense_kwargs["quality"] = quality
                elif defense_type == "gaussian_noise":
                    defense_kwargs["std"] = noise_std
                elif defense_type == "gaussian_blur":
                    defense_kwargs["sigma"] = blur_sigma
                
                defense = create_defense(defense_type, device, **defense_kwargs)
                defended_images = defense.defend(adv_images)
            else:
                defended_images = adv_images
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                clean_preds = model(test_images).argmax(dim=1)
                adv_preds = model(adv_images).argmax(dim=1)
                defended_preds = model(defended_images).argmax(dim=1)
            
            # Calculate accuracies
            clean_acc = (clean_preds == test_labels).float().mean().item()
            adv_acc = (adv_preds == test_labels).float().mean().item()
            defended_acc = (defended_preds == test_labels).float().mean().item()
            
            # Display results
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Clean Accuracy", f"{clean_acc:.3f}")
            
            with col2:
                st.metric("Adversarial Accuracy", f"{adv_acc:.3f}", 
                        delta=f"{adv_acc - clean_acc:.3f}")
            
            with col3:
                defense_name = defense_type.replace('_', ' ').title() if defense_type else "No Defense"
                st.metric(f"{defense_name} Accuracy", f"{defended_acc:.3f}", 
                        delta=f"{defended_acc - adv_acc:.3f}")
            
            # Visualize examples
            st.subheader("🖼️ Visual Examples")
            
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            
            for i in range(5):
                # Original
                axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title(f"Clean\nPred: {clean_preds[i].item()}")
                axes[0, i].axis('off')
                
                # Adversarial
                axes[1, i].imshow(adv_images[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title(f"Adversarial\nPred: {adv_preds[i].item()}")
                axes[1, i].axis('off')
                
                # Defended
                axes[2, i].imshow(defended_images[i].cpu().squeeze(), cmap='gray')
                axes[2, i].set_title(f"Defended\nPred: {defended_preds[i].item()}")
                axes[2, i].axis('off')
            
            axes[0, 0].set_ylabel("Original", fontsize=12)
            axes[1, 0].set_ylabel("Adversarial", fontsize=12)
            axes[2, 0].set_ylabel("Defended", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)


def experiment_results():
    """Experiment results tab."""
    st.header("📊 Experiment Results")
    
    # Initialize database
    db = ExperimentDatabase()
    
    # Get all experiments
    experiments = db.get_all_experiments()
    
    if not experiments:
        st.warning("No experiments found in database. Run some experiments first!")
        return
    
    # Experiment selector
    exp_names = [f"{exp['id']}: {exp['name']}" for exp in experiments]
    selected_exp = st.selectbox("Select Experiment", exp_names)
    exp_id = int(selected_exp.split(':')[0])
    
    # Get experiment details
    experiment = db.get_experiment(exp_id)
    results = db.get_experiment_results(exp_id)
    performance = db.get_model_performance(exp_id)
    
    # Display experiment info
    st.subheader("🔍 Experiment Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", experiment['model_type'].upper())
        st.metric("Attack Types", len(experiment['attack_types']))
    
    with col2:
        st.metric("Defense Types", len(experiment['defense_types']))
        st.metric("Epsilon Values", len(experiment['epsilon_values']))
    
    with col3:
        st.metric("Total Results", len(results))
        st.metric("Training Epochs", len(performance))
    
    # Training curves
    if performance:
        st.subheader("📈 Training Progress")
        
        epochs = [p['epoch'] for p in performance]
        train_loss = [p['train_loss'] for p in performance if p['train_loss']]
        train_acc = [p['train_accuracy'] for p in performance if p['train_accuracy']]
        val_loss = [p['val_loss'] for p in performance if p['val_loss']]
        val_acc = [p['val_accuracy'] for p in performance if p['val_accuracy']]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
        
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='blue')), row=1, col=1)
        if val_loss:
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='red')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Acc', line=dict(color='blue')), row=1, col=2)
        if val_acc:
            fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Acc', line=dict(color='red')), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    if results:
        st.subheader("📋 Detailed Results")
        
        # Create DataFrame
        df_data = []
        for result in results:
            df_data.append({
                'Attack': result['attack_type'].upper(),
                'Epsilon': result['epsilon'],
                'Defense': result['defense_type'] or 'No Defense',
                'Accuracy': result['accuracy'],
                'Loss': result['loss'] or 0,
                'Samples': result['sample_count'] or 0
            })
        
        df = pd.DataFrame(df_data)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        summary_stats = df.groupby(['Attack', 'Defense'])['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        st.dataframe(summary_stats, use_container_width=True)


def attack_comparison(model, device):
    """Attack comparison tab."""
    st.header("🔬 Attack Comparison")
    
    # Load sample data
    with st.spinner("Loading MNIST samples..."):
        images, labels = load_mnist_sample()
    
    # Configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        attack_types = st.multiselect(
            "Select Attacks to Compare",
            ["fgsm", "pgd", "bim", "cw"],
            default=["fgsm", "pgd"]
        )
    
    with col2:
        epsilon_values = st.multiselect(
            "Select Epsilon Values",
            [0.1, 0.2, 0.3, 0.4, 0.5],
            default=[0.1, 0.2, 0.3]
        )
    
    if st.button("🚀 Compare Attacks", type="primary"):
        if not attack_types or not epsilon_values:
            st.error("Please select at least one attack type and epsilon value.")
            return
        
        with st.spinner("Running attack comparison..."):
            results = []
            
            for attack_type in attack_types:
                for epsilon in epsilon_values:
                    # Create attack
                    attack = create_attack(attack_type, model, device, epsilon=epsilon)
                    
                    # Test on sample
                    test_images = images[:10].to(device)
                    test_labels = labels[:10].to(device)
                    
                    # Generate adversarial examples
                    adv_images = attack.attack(test_images, test_labels)
                    
                    # Get predictions
                    model.eval()
                    with torch.no_grad():
                        clean_preds = model(test_images).argmax(dim=1)
                        adv_preds = model(adv_images).argmax(dim=1)
                    
                    # Calculate accuracy
                    clean_acc = (clean_preds == test_labels).float().mean().item()
                    adv_acc = (adv_preds == test_labels).float().mean().item()
                    
                    results.append({
                        'Attack': attack_type.upper(),
                        'Epsilon': epsilon,
                        'Clean Accuracy': clean_acc,
                        'Adversarial Accuracy': adv_acc,
                        'Drop': clean_acc - adv_acc
                    })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Display results
            st.subheader("📊 Attack Comparison Results")
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                df, 
                x='Attack', 
                y='Adversarial Accuracy', 
                color='Epsilon',
                title='Adversarial Accuracy by Attack Type and Epsilon',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy drop visualization
            fig2 = px.bar(
                df,
                x='Attack',
                y='Drop',
                color='Epsilon',
                title='Accuracy Drop by Attack Type and Epsilon',
                barmode='group'
            )
            st.plotly_chart(fig2, use_container_width=True)


def defense_analysis(model, device):
    """Defense analysis tab."""
    st.header("🛡️ Defense Analysis")
    
    # Load sample data
    with st.spinner("Loading MNIST samples..."):
        images, labels = load_mnist_sample()
    
    # Configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        attack_type = st.selectbox("Attack Type", ["fgsm", "pgd", "bim"])
        epsilon = st.slider("Epsilon", 0.1, 0.5, 0.2, step=0.1)
    
    with col2:
        defense_types = st.multiselect(
            "Select Defenses to Compare",
            ["jpeg", "gaussian_noise", "gaussian_blur", "median_filter", "bit_depth_reduction"],
            default=["jpeg", "gaussian_noise", "gaussian_blur"]
        )
    
    if st.button("🚀 Analyze Defenses", type="primary"):
        if not defense_types:
            st.error("Please select at least one defense type.")
            return
        
        with st.spinner("Analyzing defenses..."):
            # Create attack
            attack = create_attack(attack_type, model, device, epsilon=epsilon)
            
            # Test on sample
            test_images = images[:10].to(device)
            test_labels = labels[:10].to(device)
            
            # Generate adversarial examples
            adv_images = attack.attack(test_images, test_labels)
            
            results = []
            
            # Test without defense
            model.eval()
            with torch.no_grad():
                clean_preds = model(test_images).argmax(dim=1)
                adv_preds = model(adv_images).argmax(dim=1)
            
            clean_acc = (clean_preds == test_labels).float().mean().item()
            adv_acc = (adv_preds == test_labels).float().mean().item()
            
            results.append({
                'Defense': 'No Defense',
                'Accuracy': adv_acc,
                'Improvement': 0.0
            })
            
            # Test with each defense
            for defense_type in defense_types:
                defense = create_defense(defense_type, device)
                defended_images = defense.defend(adv_images)
                
                with torch.no_grad():
                    defended_preds = model(defended_images).argmax(dim=1)
                
                defended_acc = (defended_preds == test_labels).float().mean().item()
                improvement = defended_acc - adv_acc
                
                results.append({
                    'Defense': defense_type.replace('_', ' ').title(),
                    'Accuracy': defended_acc,
                    'Improvement': improvement
                })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Display results
            st.subheader("📊 Defense Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df, use_container_width=True)
            
            with col2:
                # Visualization
                fig = px.bar(
                    df,
                    x='Defense',
                    y='Accuracy',
                    title=f'Defense Effectiveness Against {attack_type.upper()} Attack',
                    color='Accuracy',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Improvement visualization
            fig2 = px.bar(
                df,
                x='Defense',
                y='Improvement',
                title='Accuracy Improvement Over No Defense',
                color='Improvement',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig2, use_container_width=True)


def database_explorer():
    """Database explorer tab."""
    st.header("📈 Database Explorer")
    
    # Initialize database
    db = ExperimentDatabase()
    
    # Get statistics
    stats = db.get_statistics()
    
    st.subheader("📊 Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", stats['experiment_count'])
    
    with col2:
        st.metric("Total Results", stats['result_count'])
    
    with col3:
        st.metric("Attack Types", stats['attack_types_count'])
    
    with col4:
        st.metric("Defense Types", stats['defense_types_count'])
    
    # Attack statistics
    if stats['attack_statistics']:
        st.subheader("🔬 Attack Statistics")
        
        attack_df = pd.DataFrame(stats['attack_statistics'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(attack_df, use_container_width=True)
        
        with col2:
            fig = px.bar(
                attack_df,
                x='attack_type',
                y='avg_accuracy',
                title='Average Accuracy by Attack Type',
                color='avg_accuracy',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Data")
    
    experiments = db.get_all_experiments()
    if experiments:
        exp_names = [f"{exp['id']}: {exp['name']}" for exp in experiments]
        selected_exp = st.selectbox("Select Experiment to Export", exp_names)
        exp_id = int(selected_exp.split(':')[0])
        
        if st.button("📥 Export Experiment Data"):
            export_path = f"experiment_{exp_id}_export.json"
            db.export_experiment_data(exp_id, export_path)
            st.success(f"Data exported to {export_path}")
            
            # Provide download link
            with open(export_path, 'r') as f:
                data = f.read()
            
            st.download_button(
                label="⬇️ Download Export File",
                data=data,
                file_name=export_path,
                mime="application/json"
            )


if __name__ == "__main__":
    main()
