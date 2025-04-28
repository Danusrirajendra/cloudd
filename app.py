import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

# Suppress specific warnings if needed (optional)
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

st.set_page_config(page_title="Customer Clustering App", layout="wide")

# --- Session State Initialization ---
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0 # Start at page 0 (Introduction)

# --- Data Loading and Preprocessing (Run Once) ---
@st.cache_data
def load_data(file_path="Credit Card Customer Data.csv"):
    """Loads the dataset."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Make sure the file is in the correct directory or repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_resource # Use cache_resource for objects like fitted scalers/models
def get_scaler_and_data(data):
    """Preprocesses data and returns scaler and scaled data."""
    try:
        if data.shape[1] < 3:
             st.error("Error: Dataset requires at least 3 columns (identifiers + features).")
             st.stop()
        # Select features starting from the 3rd column (index 2)
        features_to_scale = data.columns[2:]
        x = data[features_to_scale].select_dtypes(include=np.number)

        if x.empty:
            st.error("Error: No numeric features found starting from the third column for scaling.")
            st.stop()
        if x.isnull().sum().any():
             st.warning("Data contains missing values. Filling with median for scaling/clustering.")
             # Impute directly on the slice using .loc to avoid SettingWithCopyWarning
             cols_with_na = x.columns[x.isnull().any()]
             for col in cols_with_na:
                 median_val = x[col].median()
                 x.loc[:, col] = x[col].fillna(median_val)


        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(x)
        data_scaled_df = pd.DataFrame(data_scaled, columns=x.columns)
        return scaler, data_scaled_df, x.columns

    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        st.stop()


@st.cache_resource(show_spinner="Fitting KMeans model...")
def fit_kmeans(data_scaled_df, n_clusters=3):
    """Fits KMeans model."""
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(data_scaled_df)
        return kmeans, y_kmeans
    except Exception as e:
        st.error(f"Error fitting KMeans: {e}")
        st.stop()


@st.cache_resource(show_spinner="Fitting Agglomerative Clustering model...")
def fit_agg_clustering(data_scaled_df, n_clusters=3):
    """Fits Agglomerative Clustering model."""
    try:
        agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
        y_agg = agg_cluster.fit_predict(data_scaled_df)
        return agg_cluster, y_agg
    except Exception as e:
        st.error(f"Error fitting Agglomerative Clustering: {e}")
        st.stop()

# --- Load and Process Data ---
# Wrap data loading and processing in try-except at the top level
try:
    data = load_data()
    scaler, data_scaled_df, feature_columns = get_scaler_and_data(data)
    kmeans, y_kmeans = fit_kmeans(data_scaled_df, n_clusters=3)
    agg_cluster, y_agg = fit_agg_clustering(data_scaled_df, n_clusters=3)
    app_ready = True
except Exception as main_error:
    st.error(f"A critical error occurred during initialization: {main_error}")
    st.stop() # Stop if essential data/models can't be loaded/trained
    app_ready = False


# --- App Title ---
st.title("💳 Credit Card Customer Clustering")
st.markdown("---")

# --- Page Definitions ---
TOTAL_PAGES = 5 # 0: Intro, 1: Overview, 2: KMeans, 3: Hierarchical, 4: Predict

# Check if app initialized correctly before proceeding
if app_ready:

    # PAGE 0: Introduction
    if st.session_state.page_number == 0:
        st.header("Welcome to the Customer Clustering App!")
        st.markdown("""
        This application demonstrates how to segment credit card customers based on their usage behavior using machine learning clustering techniques.

        **What does this app do?**

        1.  **Data Overview:** Displays the raw dataset and a correlation heatmap to understand relationships between features.
        2.  **KMeans Clustering:**
            *   Uses the Elbow Method to suggest an optimal number of clusters.
            *   Applies the KMeans algorithm to group customers into distinct segments (using k=3).
            *   Visualizes the resulting clusters and calculates the Silhouette Score to evaluate cluster quality.
        3.  **Hierarchical Clustering:**
            *   Applies Agglomerative Hierarchical Clustering.
            *   Visualizes the relationships using a Dendrogram.
            *   Shows the resulting clusters (using 3 clusters for comparison) and calculates the Silhouette Score.
        4.  **Predict Cluster for User:** Allows you to input hypothetical customer data and predicts which cluster (based on the KMeans model) this new user would belong to.

        **Goal:**
        The primary goal is to identify distinct groups of customers. Understanding these segments can help businesses tailor marketing strategies, product offerings, and customer service approaches more effectively.

        **Data:**
        The application uses a "Credit Card Customer Data" dataset containing information like average credit limit, total credit cards held, visits to the bank, online visits, and calls made. *Features used for clustering start from the third column.*

        **Navigation:**
        Use the **"Next ➡️"** and **"⬅️ Back"** buttons at the bottom to navigate through the different sections of the analysis.
        """)
        st.info("Click 'Next' to begin exploring the data.")


    # PAGE 1: Data Overview
    elif st.session_state.page_number == 1:
        st.header("Page 2: 📊 Data Overview")
        st.subheader("Raw Dataset Sample")
        st.dataframe(data.head())

        st.subheader("Dataset Description")
        # Ensure description is only on numeric columns to avoid errors
        st.dataframe(data.select_dtypes(include=np.number).describe())

        st.subheader("Correlation Heatmap (Numeric Features Used for Clustering)")
        numeric_features_for_clustering = data[feature_columns].select_dtypes(include=np.number)
        if numeric_features_for_clustering.empty:
            st.warning("No numeric data found in the selected feature columns to generate a correlation heatmap.")
        else:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_features_for_clustering.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
            ax_corr.set_title("Correlation Matrix of Clustering Features")
            st.pyplot(fig_corr)
            plt.close(fig_corr)

    # PAGE 2: KMeans Clustering
    elif st.session_state.page_number == 2:
        st.header("Page 3: 📈 KMeans Clustering")
        st.subheader("Elbow Method to Determine Optimal Clusters")

        @st.cache_data(show_spinner="Calculating WCSS for Elbow Method...")
        def calculate_wcss(scaled_data, k_range=range(1, 11)):
            wcss = []
            try:
                for i in k_range:
                    km = KMeans(n_clusters=i, random_state=42, n_init=10)
                    km.fit(scaled_data)
                    wcss.append(km.inertia_)
                return list(k_range), wcss
            except Exception as e:
                st.error(f"Error calculating WCSS: {e}")
                return [], []

        k_values, wcss_values = calculate_wcss(data_scaled_df)

        if k_values and wcss_values:
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(k_values, wcss_values, marker='o', linestyle='--')
            ax_elbow.set_xlabel("Number of Clusters (k)")
            ax_elbow.set_ylabel("WCSS (Inertia)")
            ax_elbow.set_title("Elbow Method for Optimal k")
            ax_elbow.grid(True)
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)
            st.markdown("Look for the 'elbow' point where the rate of decrease sharply changes.")
        else:
             st.warning("Could not generate Elbow Method plot.")

        st.subheader("KMeans Clustering Result (k=3)")
        if data_scaled_df.shape[1] >= 2:
            try:
                feature1_idx = 0
                feature2_idx = 1
                feature1_name = data_scaled_df.columns[feature1_idx]
                feature2_name = data_scaled_df.columns[feature2_idx]

                fig_kmeans, ax_kmeans = plt.subplots(figsize=(8, 6))

                for i in range(kmeans.n_clusters):
                    ax_kmeans.scatter(data_scaled_df.iloc[y_kmeans == i, feature1_idx],
                                      data_scaled_df.iloc[y_kmeans == i, feature2_idx],
                                      label=f"Cluster {i+1}", alpha=0.7)
                ax_kmeans.scatter(kmeans.cluster_centers_[:, feature1_idx], kmeans.cluster_centers_[:, feature2_idx],
                                  s=250, c='red', marker='X', label='Centroids', edgecolors='k')

                ax_kmeans.set_xlabel(f"{feature1_name} (Scaled)")
                ax_kmeans.set_ylabel(f"{feature2_name} (Scaled)")
                ax_kmeans.set_title(f"KMeans Clusters (k=3) on {feature1_name} vs {feature2_name}")
                ax_kmeans.legend()
                ax_kmeans.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_kmeans)
                plt.close(fig_kmeans)

                try:
                    score = silhouette_score(data_scaled_df, y_kmeans)
                    st.success(f"KMeans Silhouette Score (k=3): {score:.3f}")
                    st.caption("Silhouette Score range is [-1, 1]. Higher values indicate better-defined clusters.")
                except Exception as e:
                    st.warning(f"Could not calculate Silhouette Score: {e}")

            except Exception as e:
                 st.error(f"Error plotting KMeans results: {e}")
        else:
            st.warning("Need at least two features to create a 2D scatter plot for clusters.")


    # PAGE 3: Hierarchical Clustering
    elif st.session_state.page_number == 3:
        st.header("Page 4: 🌲 Hierarchical Clustering")
        st.subheader("Dendrogram (Ward Linkage)")

        @st.cache_data(show_spinner="Generating Dendrogram...")
        def create_dendrogram(scaled_data):
            try:
                if scaled_data.shape[0] > 1:
                    Z = linkage(scaled_data, method='ward')
                    return Z
                else:
                    return None
            except Exception as e:
                st.error(f"Could not generate linkage matrix: {e}")
                return None

        linkage_matrix = create_dendrogram(data_scaled_df)

        if linkage_matrix is not None:
            fig_dendro, ax_dendro = plt.subplots(figsize=(12, 7))
            num_samples = data_scaled_df.shape[0]
            p_truncate = min(30, max(10, num_samples // 20))
            dendrogram(linkage_matrix, truncate_mode='lastp', p=p_truncate, show_leaf_counts=True, ax=ax_dendro, leaf_rotation=90.)
            ax_dendro.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, Truncated)")
            ax_dendro.set_xlabel(f"Cluster Size or Sample Index")
            ax_dendro.set_ylabel("Distance (Ward)")
            ax_dendro.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_dendro)
            plt.close(fig_dendro)
            st.markdown("The dendrogram visualizes how clusters are merged.")
        else:
            st.warning("Cannot generate dendrogram.")


        st.subheader("Agglomerative Clustering Result (k=3)")
        if data_scaled_df.shape[1] >= 2:
            try:
                feature1_idx = 0
                feature2_idx = 1
                feature1_name = data_scaled_df.columns[feature1_idx]
                feature2_name = data_scaled_df.columns[feature2_idx]

                fig_agg, ax_agg = plt.subplots(figsize=(8, 6))

                for i in range(agg_cluster.n_clusters):
                     ax_agg.scatter(data_scaled_df.iloc[y_agg == i, feature1_idx],
                                    data_scaled_df.iloc[y_agg == i, feature2_idx],
                                    label=f"Cluster {i+1}", alpha=0.7)

                ax_agg.set_xlabel(f"{feature1_name} (Scaled)")
                ax_agg.set_ylabel(f"{feature2_name} (Scaled)")
                ax_agg.set_title(f"Agglomerative Clusters (k=3) on {feature1_name} vs {feature2_name}")
                ax_agg.legend()
                ax_agg.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_agg)
                plt.close(fig_agg)

                try:
                    score_agg = silhouette_score(data_scaled_df, y_agg)
                    st.success(f"Agglomerative Silhouette Score (k=3): {score_agg:.3f}")
                    st.caption("Compare this score with the KMeans score.")
                except Exception as e:
                     st.warning(f"Could not calculate Agglomerative Silhouette Score: {e}")

            except Exception as e:
                st.error(f"Error plotting Agglomerative Clustering results: {e}")
        else:
             st.warning("Need at least two features to create a 2D scatter plot for clusters.")


    # PAGE 4: Predict Cluster for User
    elif st.session_state.page_number == 4:
        st.header("Page 5: 🧍 Predict Cluster for New User")
        st.subheader("Enter User Details (Based on Features Used for Clustering):")

        input_values = {}
        if feature_columns is None or feature_columns.empty:
            st.error("Feature columns not loaded. Cannot create input fields.")
            # Disable prediction button if feature columns are missing
            st.button("✨ Predict Cluster", disabled=True)
        else:
            widget_cols = st.columns(2)
            half_point = (len(feature_columns) + 1) // 2
            has_input_error = False # Flag errors during input creation

            for i, col_name in enumerate(feature_columns):
                target_col = widget_cols[0] if i < half_point else widget_cols[1]

                with target_col:
                    try:
                        # 1. Determine default value (heuristic)
                        default_val_num = data[col_name].median() # Start with median
                        # Apply specific overrides
                        if 'limit' in col_name.lower(): default_val_num = 50000.0
                        elif 'card' in col_name.lower(): default_val_num = 3.0
                        elif 'bank' in col_name.lower(): default_val_num = 2.0
                        elif 'online' in col_name.lower(): default_val_num = 3.0
                        elif 'call' in col_name.lower(): default_val_num = 4.0
                        else: default_val_num = float(default_val_num) # Ensure float if no specific override

                        # --- SIMPLIFIED & ROBUST FIX: Use FLOATS consistently ---
                        current_value_fl = float(default_val_num)
                        min_value_fl = 0.0
                        # Determine step: 1.0 if original data looks like integers, else 0.01
                        try:
                            # Check if all non-NA values in the original column are whole numbers
                            is_whole_number_col = pd.api.types.is_numeric_dtype(data[col_name]) and (data[col_name].dropna() % 1 == 0).all()
                        except TypeError:
                             # Handle potential TypeError if column cannot be modulo checked (e.g., non-numeric mixed in)
                            is_whole_number_col = False

                        step_val_fl = 1.0 if is_whole_number_col else 0.01

                        widget_key = f"input_{col_name}"

                        # Create widget - Pass ONLY FLOATS, remove format
                        input_values[col_name] = st.number_input(
                            label=f"{col_name.replace('_', ' ').title()}",
                            value=current_value_fl,
                            min_value=min_value_fl,
                            step=step_val_fl,
                            key=widget_key
                            # No format argument
                        )
                    except Exception as e:
                        st.error(f"Error creating input for '{col_name}': {e}.")
                        input_values[col_name] = None
                        has_input_error = True

            # --- Prediction Logic ---
            # Display message if inputs failed
            if has_input_error:
                st.error("One or more input fields failed to generate. Cannot predict.")

            if st.button("✨ Predict Cluster", disabled=has_input_error):
                if any(v is None for v in input_values.values()):
                     st.error("Cannot predict because one or more input fields had errors.")
                else:
                    try:
                        # Create DataFrame ensuring float type for consistency
                        user_input_dict = {col: float(val) for col, val in input_values.items()}
                        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)

                        # Scale user input
                        user_scaled = scaler.transform(user_df)

                        # Predict
                        predicted_cluster = kmeans.predict(user_scaled)[0]

                        st.success(f"🎉 The new user belongs to **Cluster {predicted_cluster + 1}** (using KMeans)")
                        # Add descriptive text about the clusters here

                        # Plotting
                        st.subheader("Visualizing New User with Existing Clusters")
                        if data_scaled_df.shape[1] >= 2:
                            plot_feature1_idx = 0
                            plot_feature2_idx = 1
                            plot_feature1_name = data_scaled_df.columns[plot_feature1_idx]
                            plot_feature2_name = data_scaled_df.columns[plot_feature2_idx]

                            fig_pred, ax_pred = plt.subplots(figsize=(8, 6))

                            for i in range(kmeans.n_clusters):
                                ax_pred.scatter(data_scaled_df.iloc[y_kmeans == i, plot_feature1_idx],
                                                data_scaled_df.iloc[y_kmeans == i, plot_feature2_idx],
                                                label=f"Cluster {i+1}", alpha=0.5)
                            ax_pred.scatter(kmeans.cluster_centers_[:, plot_feature1_idx], kmeans.cluster_centers_[:, plot_feature2_idx],
                                            s=250, c='black', marker='X', label='Centroids', edgecolors='w')
                            ax_pred.scatter(user_scaled[0, plot_feature1_idx], user_scaled[0, plot_feature2_idx],
                                            c='red', s=300, marker='*', label=f'New User (Cluster {predicted_cluster + 1})', edgecolors='k')

                            ax_pred.set_xlabel(f"{plot_feature1_name} (Scaled)")
                            ax_pred.set_ylabel(f"{plot_feature2_name} (Scaled)")
                            ax_pred.set_title(f"New User Prediction (on {plot_feature1_name} vs {plot_feature2_name})")
                            ax_pred.legend()
                            ax_pred.grid(True, linestyle='--', alpha=0.5)
                            st.pyplot(fig_pred)
                            plt.close(fig_pred)
                        else:
                            st.warning("Need at least two features to create a 2D scatter plot.")

                    except ValueError as ve:
                         if "feature names mismatch" in str(ve).lower():
                             expected_features_str = ", ".join(map(str, scaler.feature_names_in_)) # Ensure strings
                             received_features_str = ", ".join(map(str, user_df.columns.tolist())) # Ensure strings
                             st.error(f"Prediction Error: Feature names mismatch.\nExpected: {expected_features_str}\nReceived: {received_features_str}\nDetails: {ve}")
                         elif "numpy boolean subtract" in str(ve).lower() or "could not convert" in str(ve).lower():
                             st.error(f"Type Error during Scaling/Prediction. Check input data types. Input: {user_input_dict}. Details: {ve}")
                         else:
                             st.error(f"An error occurred during prediction: {ve}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during prediction or plotting: {e}")


    # --- Page Separator and Navigation Buttons ---
    st.markdown("---")
    col_nav1, col_nav2 = st.columns([1, 1])

    with col_nav1:
        if st.session_state.page_number > 0:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.page_number -= 1
                st.rerun()

    with col_nav2:
        if st.session_state.page_number < TOTAL_PAGES - 1:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.page_number += 1
                st.rerun()

# --- Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("App built with [Streamlit](https://streamlit.io)")
# Optional: Display current page number
# st.sidebar.write(f"Page: {st.session_state.page_number + 1} / {TOTAL_PAGES}")
