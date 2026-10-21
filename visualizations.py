import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

def plot_target_distribution(df, target_col):
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    counts['Label'] = counts[target_col].map({0: 'Good Credit', 1: 'Bad Credit / Default'})
    fig = px.bar(counts, x='Label', y='Count', color='Label',
                 title="Credit Risk Outcome Distribution",
                 color_discrete_map={'Good Credit': '#008080', 'Bad Credit / Default': '#1E293B'})
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_confusion_matrix(cm):
    z = cm
    x = ['Predicted Good', 'Predicted Bad']
    y = ['Actual Good', 'Actual Bad']
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Teal', annotation_text=np.array(z).astype(str))
    fig.update_layout(title_text="Confusion Matrix", plot_bgcolor='white', paper_bgcolor='white')
    return fig
