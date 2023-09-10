import streamlit as st
import pandas as pd
import numpy as np
import umap 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import webbrowser


# Set width mode to wide to display plots better
st.set_page_config(layout="wide")
# Streamlit Configuration
st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar
st.sidebar.header("Schizophrenia Data Analysis")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Sliders for UMAP and KMeans parameters
st.sidebar.subheader("UMAP Parameters")
n_neighbors = st.sidebar.slider("Number of Neighbors", 2, 50, 5)
min_dist = st.sidebar.slider("Minimum Distance", 0.0, 1.0, 0.3, 0.1)

st.sidebar.subheader("Clustering Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (KMeans)", 2, 20, 5)
n_dendro_clusters = st.sidebar.slider("Number of Clusters (Dendrogram)", 2, 20, 5)

# Add option to choose linkage method for dendrogram
linkage_methods = ["ward", "single", "complete", "average"]
selected_linkage_method = st.sidebar.selectbox("Linkage Method for Dendrogram", linkage_methods, 0)

# Checkbox to toggle PCA and UMAP visualization
show_pca = st.sidebar.checkbox("Show PCA Visualization", False)
show_umap = st.sidebar.checkbox("Show UMAP Visualization", False)

# Load the data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Function to perform UMAP embedding and K-means clustering
def umap_and_kmeans(band_data, n_neighbors=n_neighbors, min_dist=min_dist, n_clusters=n_clusters):
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(band_data)
    kmeans_labels = KMeans(n_init=4, n_clusters=n_clusters, random_state=42).fit(embedding).labels_
    return embedding, kmeans_labels

# Function to plot UMAP embedding results
def plot_umap_embedding(embedding, kmeans_labels, ax, title):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_labels, cmap='rainbow', s=20)
    # add a text with umap parameters and kmeans cluster number 
    ax.text(0.99, 0.01, f"n_neighbors={n_neighbors}, min_dist={min_dist}, n_clusters={n_clusters}", 
            transform=ax.transAxes, ha='right', va='bottom', size=10)
    ax.set_title(title)

def plot_dendrogram_colored_ticks(band_data, ax, title, method='ward'):
    """
    Plot the dendrogram with correctly colored tick numbers for the "All Subjects" group.
    """
    # Hierarchical clustering
    Z = linkage(band_data, method=method)
    
    # Plot the dendrogram
    ddata = dendrogram(Z, ax=ax, leaf_rotation=90)
    ax.set_title(title + " Dendrogram (" + method + " linkage)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    
    # Color the tick numbers based on control and schizophrenia subjects
    control_indices = data_control.index.to_list()
    schizophrenia_indices = data_schizophrenia.index.to_list()
    
    # Get the x-tick labels (leaf labels) from the dendrogram
    leaf_labels = ddata['leaves']
    
    # Iterate through x-ticks and color them based on the group
    for idx, label in enumerate(ax.get_xticklabels()):
        label_idx = leaf_labels[idx]
        if label_idx in control_indices:
            label.set_color('black')
        elif label_idx in schizophrenia_indices:
            label.set_color('red')

def plot_dendrogram_and_pca_with_correct_colored_ticks(band_data, ax_dendro, title, color_ticks=False, method='ward'):
    """
    Plot the dendrogram with optionally colored tick numbers and PCA visualization on the given axes.
    """
    # Hierarchical clustering
    Z = linkage(band_data, method=method)
    
    # Plot the dendrogram
    ddata = dendrogram(Z, ax=ax_dendro, leaf_rotation=90)
    ax.set_title(str(title) + " Dendrogram (" + str(method) + " linkage)")
    ax_dendro.set_xlabel("Sample Index")
    ax_dendro.set_ylabel("Distance")
    

    if color_ticks:
        # Color the tick numbers based on control and schizophrenia subjects
        control_indices = data_control.index.to_list()
        schizophrenia_indices = data_schizophrenia.index.to_list()

        # Get the x-tick labels (leaf labels) from the dendrogram
        leaf_labels = ddata['leaves']

        # Iterate through x-ticks and color them based on the group
        for idx, label in enumerate(ax_dendro.get_xticklabels()):
            label_idx = leaf_labels[idx]
            if label_idx in control_indices:
                label.set_color('black')
            elif label_idx in schizophrenia_indices:
                label.set_color('red')
    return Z

def plot_band_pca(band_data, Z, ax_pca, title):
    
    # Cut the dendrogram to obtain 3 clusters
    labels = fcluster(Z, t=n_dendro_clusters, criterion='maxclust')
    band_data['Cluster'] = labels
    
    # Use PCA to reduce the data to 2D
    pca = PCA(n_components=2)
    band_pca = pca.fit_transform(band_data.drop('Cluster', axis=1))
    # return band_pca

    # Create a scatter plot for PCA reduced data
    ax_pca.scatter(band_pca[:, 0], band_pca[:, 1], c=band_data['Cluster'], cmap='rainbow')
    ax_pca.set_title(title + " 2D PCA")
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")

# If a CSV file is uploaded
if uploaded_file:
    st.write("Dataset loaded successfully!")
    
    # Load the data
    data = load_data(uploaded_file)

    # Split data into control and schizophrenia groups
    data_control = data[data['Group'] == 0]
    data_schizophrenia = data[data['Group'] == 1]
    data_full = data

    # Combined dendrogram for "All Subjects"
    all_bands_data = pd.concat([
        data.loc[:, data.columns.str.startswith('avpp_delta')],
        data.loc[:, data.columns.str.startswith('avpp_theta')],
        data.loc[:, data.columns.str.startswith('avpp_alpha')],
        data.loc[:, data.columns.str.startswith('avpp_beta')],
        data.loc[:, data.columns.str.startswith('avpp_gamma')]
    ], axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_dendrogram_colored_ticks(all_bands_data, ax, "All Bands Combined", method=selected_linkage_method)
    plt.tight_layout()

    # Save the dendrogram plot to a PNG file
    dendrogram_filename = "Combined_Dendrogram_plot.png"
    fig.savefig(dendrogram_filename, dpi=300)

    # Provide a download button for the dendrogram PNG file
    with open(dendrogram_filename, "rb") as f:
        btn = st.download_button(
            label="Download Combined Dendrogram Plot",
            data=f,
            file_name=dendrogram_filename,
            mime="image/png"
        )

    st.pyplot(fig)
    st.write("EDA - Exploratory Data Analysis")
    # Detect available bands from column names
    bands_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    available_bands = [band for band in bands_list if any(data.columns.str.startswith(f'avpp_{band}'))]

    # Note: Replace all `plt.show()` with `st.pyplot()`
    # Create the plots with dendrogram, PCA, and UMAP visualizations
    nrows = 3 if show_pca and show_umap else 2 if show_pca or show_umap else 1 # Number of rows in the plot
    hight = 15 if show_pca and show_umap else 10 if show_pca or show_umap else 5 # Height of the plot
    for data_group, title in zip([data_schizophrenia, data_control, data_full], ["Schizophrenia", "Control", "All Subjects"]):
        fig, axes = plt.subplots(nrows=nrows, ncols=len(available_bands), figsize=(36, hight))
        fig.suptitle(title, fontsize=25)
        
        # Ensure axes is 2D
        if nrows == 1:
            axes = axes.reshape(1, -1)

        # Create band data based on detected bands for the current data group
        bands = [(band.capitalize(), data_group.loc[:, data_group.columns.str.startswith(f'avpp_{band}')]) for band in available_bands]

        # Configure the axes based on the selected visualizations
        axes_mapping = [0]  # dendrogram axes index is always 0
        if show_pca:
            axes_mapping.append(len(axes_mapping))
        if show_umap:
            axes_mapping.append(len(axes_mapping))

        # Plot dendrogram, PCA, and UMAP visualizations for each band
        for col, (band_name, band_data) in enumerate(bands):
            ax_dendro = axes[axes_mapping[0]][col]
            ax_dendro.set_title(band_name)
            color_ticks = True if title == "All Subjects" else False
            # Dendrogram plots using previous functions
            Z = plot_dendrogram_and_pca_with_correct_colored_ticks(band_data.copy(), ax_dendro, band_name, color_ticks, method=selected_linkage_method)
            
            if show_pca:
                ax_pca = axes[axes_mapping[1]][col]
                plot_band_pca(band_data.copy(), Z, ax_pca, title)
                
            if show_umap:
                ax_umap = axes[axes_mapping[-1]][col]
                embedding, kmeans_labels = umap_and_kmeans(band_data)
                plot_umap_embedding(embedding, kmeans_labels, ax_umap, band_name + " 2D UMAP")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # Save the plot to a PNG file
        plot_filename = f"{title.replace(' ', '_')}_plot.png"
        fig.savefig(plot_filename, dpi=600)
        # plt.show()
        # st.pyplot()
        # st.image(plot_filename, use_column_width=True, clamp=True)
        st.pyplot(fig)
        plt.close(fig)
        
        # Provide a download button for the PNG file
        with open(plot_filename, "rb") as f:
            btn = st.download_button(
                label=f"Download {title} Plot",
                data=f,
                file_name=plot_filename,
                mime="image/png"
            )

