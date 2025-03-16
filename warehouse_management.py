import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import calendar
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Define the function to run warehouse management
def run_warehouse_management():
    # Step 1: Upload the dataset
    st.header("Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload your transactions CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess the data
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Preprocess the data
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y, %I:%M %p', errors='coerce')  # Ensures all dates are in a consistent format
        df['Month'] = df['Date'].dt.to_period('M')  # Extracts the month from the Date column
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')  # Ensures the Quantity column is numeric
        df = df.dropna(subset=['Quantity'])
        
        # Add day of week and hour for time-based analysis
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Hour'] = df['Date'].dt.hour
        
        # Display Data Distribution Information
        st.header("Data Distribution")
        
        # Time series analysis of transactions
        st.subheader("Transaction Volume Over Time")
        monthly_counts = df.groupby(df['Date'].dt.to_period('M')).size()
        monthly_counts_df = pd.DataFrame({
            'Month': monthly_counts.index.astype(str),
            'Transactions': monthly_counts.values
        })
        
        fig_time_series = px.line(monthly_counts_df, x='Month', y='Transactions', 
                                title='Monthly Transaction Volume',
                                markers=True)
        st.plotly_chart(fig_time_series)
        
        # Add day of week and hourly transaction patterns
        st.subheader("Transaction Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week pattern
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['DayOfWeek'].value_counts().reindex(day_order)
            
            fig_day = px.bar(
                x=day_counts.index, 
                y=day_counts.values,
                labels={'x': 'Day of Week', 'y': 'Transaction Count'},
                title='Transactions by Day of Week',
                color=day_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_day)
        
        with col2:
            # Hour pattern
            hour_counts = df['Hour'].value_counts().sort_index()
            
            fig_hour = px.bar(
                x=hour_counts.index, 
                y=hour_counts.values,
                labels={'x': 'Hour of Day', 'y': 'Transaction Count'},
                title='Transactions by Hour of Day',
                color=hour_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_hour)

        # Feature Engineering
        item_monthly_stats = df.groupby(['Month', 'Item']).agg(
            Total_Quantity=('Quantity', 'sum'),
            Avg_Quantity=('Quantity', 'mean'),
            Movement_Frequency=('Quantity', 'count')
        ).reset_index()

        # Show Top Items by Quantity
        st.subheader("Top Items by Total Quantity")
        top_items = df.groupby('Item')['Quantity'].sum().sort_values(ascending=False).head(10)
        
        fig_top_items = px.bar(
            x=top_items.index,
            y=top_items.values,
            labels={'x': 'Item', 'y': 'Total Quantity'},
            title='Top 10 Items by Total Quantity',
            color=top_items.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_top_items)

        # Scaling the data
        X = item_monthly_stats[['Total_Quantity', 'Avg_Quantity', 'Movement_Frequency']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow and Silhouette Analysis
        st.write("### Elbow and Silhouette Analysis")
        inertia = []
        silhouette_scores = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(range(2, 10), inertia, marker='o')
        ax[0].set_title('Elbow Method')
        ax[0].set_xlabel('Number of clusters')
        ax[0].set_ylabel('Inertia')

        ax[1].plot(range(2, 10), silhouette_scores, marker='o')
        ax[1].set_title('Silhouette Analysis')
        ax[1].set_xlabel('Number of clusters')
        ax[1].set_ylabel('Silhouette Score')

        st.pyplot(fig)

        # KMeans Clustering
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=9, value=3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        item_monthly_stats['Cluster'] = kmeans.fit_predict(X_scaled)

        # Cluster Summary
        cluster_summary = item_monthly_stats.groupby('Cluster').agg(
            Avg_Total_Quantity=('Total_Quantity', 'mean'),
            Avg_Movement_Frequency=('Movement_Frequency', 'mean'),
            Items_Per_Cluster=('Item', 'count')
        ).reset_index()

        st.write("### Cluster Summary")
        st.dataframe(cluster_summary)
        
        # Visualize clusters with PCA
        st.subheader("Cluster Visualization")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = item_monthly_stats['Cluster']
        pca_df['Item'] = item_monthly_stats['Item']
        
        fig_clusters = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            color='Cluster',
            hover_data=['Item'],
            title='PCA Visualization of Item Clusters',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_clusters)
        
        # 3D Visualization of original features
        st.subheader("3D Visualization of Clusters")
        fig_3d = px.scatter_3d(
            item_monthly_stats,
            x='Total_Quantity',
            y='Avg_Quantity',
            z='Movement_Frequency',
            color='Cluster',
            hover_data=['Item'],
            title='3D Visualization of Item Features by Cluster',
            labels={
                'Total_Quantity': 'Total Quantity',
                'Avg_Quantity': 'Average Quantity',
                'Movement_Frequency': 'Movement Frequency'
            },
            opacity=0.7
        )
        st.plotly_chart(fig_3d)

        # Define cluster names based on characteristics
        if num_clusters == 3:
            cluster_descriptions = {
                0: "High-demand and high-frequency items that need frequent restocking.",
                1: "Moderate-demand items with occasional movement.",
                2: "Low-demand items with infrequent movement, often stored longer."
            }
        else:
            # Generate generic descriptions based on characteristics
            cluster_descriptions = {}
            for cluster in range(num_clusters):
                cluster_data = cluster_summary[cluster_summary['Cluster'] == cluster]
                avg_qty = cluster_data['Avg_Total_Quantity'].values[0]
                avg_freq = cluster_data['Avg_Movement_Frequency'].values[0]
                
                # Determine demand level
                if avg_qty > cluster_summary['Avg_Total_Quantity'].median():
                    demand = "High-demand"
                else:
                    demand = "Low-demand"
                    
                # Determine frequency level
                if avg_freq > cluster_summary['Avg_Movement_Frequency'].median():
                    frequency = "high-frequency"
                else:
                    frequency = "low-frequency"
                    
                cluster_descriptions[cluster] = f"{demand} and {frequency} items."

        # Display detailed insights for each cluster
        st.write("### Cluster Insights")
        for cluster in cluster_summary['Cluster']:
            cluster_data = item_monthly_stats[item_monthly_stats['Cluster'] == cluster]
            top_items = cluster_data.sort_values(by='Total_Quantity', ascending=False).head(5)['Item'].tolist()
            representative_items = ', '.join(top_items)

            st.write(f"#### Cluster {cluster} ({cluster_descriptions[cluster]}):")
            st.write(f"- **Total Items**: {len(cluster_data)}")
            st.write(f"- **Average Total Quantity**: {cluster_summary.loc[cluster_summary['Cluster'] == cluster, 'Avg_Total_Quantity'].values[0]:.2f}")
            st.write(f"- **Average Movement Frequency**: {cluster_summary.loc[cluster_summary['Cluster'] == cluster, 'Avg_Movement_Frequency'].values[0]:.2f}")
            st.write(f"- **Representative Items**: {representative_items} ...")
            
            # Add a radar chart for cluster characteristics visualization
            if st.checkbox(f"Show Cluster {cluster} Profile", value=False):
                # Normalize values for radar chart
                radar_data = cluster_summary.loc[cluster_summary['Cluster'] == cluster].iloc[0]
                max_values = cluster_summary[['Avg_Total_Quantity', 'Avg_Movement_Frequency', 'Items_Per_Cluster']].max()
                
                normalized_values = [
                    radar_data['Avg_Total_Quantity'] / max_values['Avg_Total_Quantity'],
                    radar_data['Avg_Movement_Frequency'] / max_values['Avg_Movement_Frequency'],
                    radar_data['Items_Per_Cluster'] / max_values['Items_Per_Cluster']
                ]
                
                categories = ['Total Quantity', 'Movement Frequency', 'Item Count']
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values + [normalized_values[0]],  # Close the loop
                    theta=categories + [categories[0]],  # Close the loop
                    fill='toself',
                    name=f'Cluster {cluster}'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title=f"Characteristics Profile of Cluster {cluster}"
                )
                st.plotly_chart(fig_radar)

        # Step 2: FP-Growth Analysis
        st.header("Step 2: FP-Growth Analysis")
        cluster_options = item_monthly_stats['Cluster'].unique()
        selected_cluster = st.selectbox("Select a Cluster for FP-Growth Analysis", cluster_options)

        # Filter dataset for the selected cluster
        filtered_df = df.merge(item_monthly_stats[['Item', 'Cluster']], on='Item', how='left')
        filtered_df = filtered_df[filtered_df['Cluster'] == selected_cluster]

        if not filtered_df.empty:
            # Group transactions by 'Date' and 'Owner'
            transactions = filtered_df.groupby(['Date', 'Owner'])['Item'].apply(list).reset_index()

            # Convert transactions to one-hot encoding format
            te = TransactionEncoder()
            te_ary = te.fit(transactions['Item']).transform(transactions['Item'])
            fpgrowth_df = pd.DataFrame(te_ary, columns=te.columns_)

            # Add Transaction ID
            fpgrowth_df.insert(0, 'Transaction ID', range(1, len(fpgrowth_df) + 1))

            st.write("### Prepared Dataset for FP-Growth")
            st.dataframe(fpgrowth_df.head())
            
            # Show item co-occurrence matrix
            st.subheader("Item Co-occurrence Heatmap")
            
            # Calculate co-occurrence matrix (limit to top items for clarity)
            top_item_cols = fpgrowth_df.iloc[:, 1:].sum().sort_values(ascending=False).head(15).index
            top_items_df = fpgrowth_df[['Transaction ID'] + list(top_item_cols)]
            
            co_occurrence = np.dot(top_items_df.iloc[:, 1:].T, top_items_df.iloc[:, 1:])
            np.fill_diagonal(co_occurrence, 0)  # Remove self-pairs
            
            fig_heatmap = px.imshow(
                co_occurrence,
                x=top_item_cols,
                y=top_item_cols,
                color_continuous_scale='Viridis',
                title='Item Co-occurrence Matrix',
                labels=dict(x="Item", y="Item", color="Co-occurrence")
            )
            st.plotly_chart(fig_heatmap)

            # Apply FP-Growth algorithm
            min_support = st.slider("Select minimum support threshold", 0.01, 1.0, 0.05)
            frequent_itemsets = fpgrowth(fpgrowth_df.drop(columns=['Transaction ID']), min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                # Convert frozenset to string list before displaying
                display_itemsets = frequent_itemsets.copy()
                display_itemsets['itemsets'] = display_itemsets['itemsets'].apply(lambda x: list(x))
                
                st.write("### Frequent Itemsets")
                st.dataframe(display_itemsets)
                
                # Rest of the visualization code remains the same
                # Convert frozensets to strings for better visualization
                frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                
                # Sort and get top 20 for visualization
                top_frequent = frequent_itemsets.sort_values('support', ascending=False).head(20)
                
                fig_frequent = px.bar(
                    top_frequent,
                    x='support', 
                    y='itemsets_str',
                    orientation='h',
                    title='Top 20 Frequent Itemsets by Support',
                    labels={'support': 'Support', 'itemsets_str': 'Itemsets'},
                    color='support',
                    color_continuous_scale='Viridis'
                )
                fig_frequent.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_frequent)

                # Generate Association Rules
                min_lift = st.slider("Select minimum lift threshold", 0.1, 5.0, 1.0)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

                if not rules.empty:
                    st.write("### Association Rules")
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
                    
                    # Create readable strings for visualization
                    rules['rule'] = rules.apply(lambda row: f"{', '.join(row['antecedents'])} → {', '.join(row['consequents'])}", axis=1)
                    
                    st.dataframe(rules)
                    
                    # Visualize top rules by lift
                    st.subheader("Top Association Rules")
                    
                    top_rules = rules.sort_values('lift', ascending=False).head(15)
                    
                    fig_rules = px.bar(
                        top_rules,
                        x='lift',
                        y='rule',
                        orientation='h',
                        title='Top 15 Association Rules by Lift',
                        labels={'lift': 'Lift', 'rule': 'Rule'},
                        color='confidence',
                        color_continuous_scale='Viridis',
                        hover_data=['support', 'confidence']
                    )
                    fig_rules.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_rules)
                    
                    # Bubble chart for rule metrics
                    st.subheader("Association Rules Metrics")
                    
                    fig_bubble = px.scatter(
                        rules,
                        x='support',
                        y='confidence',
                        size='lift',
                        color='lift',
                        hover_name='rule',
                        title='Association Rules: Support vs Confidence vs Lift',
                        labels={
                            'support': 'Support',
                            'confidence': 'Confidence',
                            'lift': 'Lift'
                        },
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_bubble)

                    # Visualize Association Rules as Network
                    st.header("Visualizing Association Rules")
                    
                    # Create a more stable network layout
                    G = nx.DiGraph()
                    for _, row in rules.iterrows():
                        for ant in row['antecedents']:
                            for cons in row['consequents']:
                                G.add_edge(ant, cons, weight=row['lift'])

                    if len(G.nodes()) > 0:  # Check if the graph has any nodes
                        # Create the figure and axis for the network graph
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)  # Adjusted layout parameters
                        
                        # Use edge weights for edge thickness and color
                        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                        
                        if edge_weights:  # Check if there are any edges
                            # Create colormap for edges
                            cmap = plt.cm.viridis
                            
                            # Calculate node sizes based on degree
                            node_size = [3000 * (1 + G.degree(node) / 10) for node in G.nodes()]
                            
                            try:
                                # Draw the network with error handling
                                edges = nx.draw_networkx_edges(
                                    G, 
                                    pos, 
                                    width=[1 + (w / max(edge_weights) * 5) for w in edge_weights],
                                    edge_color=edge_weights,
                                    edge_cmap=cmap,
                                    ax=ax
                                )
                                
                                nodes = nx.draw_networkx_nodes(
                                    G,
                                    pos,
                                    node_size=node_size,
                                    node_color="lightblue",
                                    ax=ax
                                )
                                
                                labels = nx.draw_networkx_labels(
                                    G,
                                    pos,
                                    font_size=10,
                                    ax=ax
                                )
                                
                                # Add colorbar
                                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
                                sm.set_array([])
                                plt.colorbar(sm, ax=ax, label='Lift')
                                
                                plt.title("Association Rules Network Graph")
                                plt.axis('off')
                                
                                st.pyplot(fig)
                                plt.close()  # Close the figure to free memory
                                
                            except Exception as e:
                                st.error(f"Error in network visualization: {str(e)}")
                        else:
                            st.warning("No edges found in the network graph.")
                    else:
                        st.warning("No nodes found in the network graph.")
                    
                   