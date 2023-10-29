import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import multivariate_normal
from plotly.subplots import make_subplots


# Streamlit documentation for UI : https://docs.streamlit.io/
# class structure adapted from https://alpopkes.com/posts/machine_learning/bayesian_linear_regression/
class PosteriorDistribution:
    def __init__(self, prior_mean, prior_cov, noise_var):
        # initializing instance variables required to update posterior distribution

        # adapted from https://alpopkes.com/posts/machine_learning/bayesian_linear_regression/
        self.prior_mean = prior_mean.reshape(2, 1)
        self.prior_cov = prior_cov
        self.prior = multivariate_normal(prior_mean, prior_cov)
        self.noise_var = noise_var
        self.noise_precision = 1 / noise_var
        self.param_posterior = self.prior
        self.posterior_mean = self.prior_mean
        self.posterior_cov = self.prior_cov

    def update_posterior(self, input_features, output_features):
        outputs = output_features[:, np.newaxis]

        # appending 1s to the input_data vector to create the design matrix
        input_matrix = np.stack((np.ones(len(input_features)), input_features), axis=1)

        # computing product of design matrix and its transpose, to be used in computing posterior covariance
        input_matrix_product = input_matrix.T.dot(input_matrix)

        # computing inverse of the prior covariance matrix, used in computing posterior mean and posterior covariance
        inverse_prior_cov = np.linalg.inv(self.prior_cov)

        # final posterior covariance
        self.posterior_cov = np.linalg.inv(
            inverse_prior_cov + self.noise_precision * input_matrix_product
        )

        # final posterior mean
        self.posterior_mean = self.posterior_cov.dot(
            inverse_prior_cov.dot(self.prior_mean)
            + self.noise_precision * input_matrix.T.dot(outputs)
        )
        # final posterior distribution
        self.param_posterior = multivariate_normal(
            self.posterior_mean.flatten(), self.posterior_cov
        )


def plot_graph(model, inputs, outputs, data_point_index, w0, w1):
    # initializing plot layout
    fig = make_subplots(rows=1, cols=2)

    fig.update_layout(
        height=800,
        width=800,
        title_text="line fitting vs posterior distribution after "
                   + str(data_point_index)
                   + " data points",
    )

    # Initializing grid layout for surface and contour plot
    x_coordinate = np.arange(-1, 1, 0.01)
    y_coordinate = np.arange(-1, 1, 0.01)
    [X_coordinate, Y_coordinate] = np.meshgrid(x_coordinate, y_coordinate)
    pos = np.dstack((X_coordinate, Y_coordinate))

    # Plotting 3D Surface plot
    surface = go.Figure(data=[go.Surface(z=model.param_posterior.pdf(pos), x=x_coordinate, y=y_coordinate)],
                        layout=go.Layout(
                            title=go.layout.Title(
                                text="3D surface plot after " + str(data_point_index) + " data points")
                        ))
    surface.update_layout(width=10, height=10)
    st.write(surface)

    # Contour plot
    contour_plot = go.Contour(
        x=x_coordinate,
        y=y_coordinate,
        z=model.param_posterior.pdf(pos),
        colorscale="jet",
        colorbar=dict(title="z value", titleside="right"),
    )

    fig.add_trace(contour_plot, row=1, col=2)

    # Marking true weights used to generate data on the scatter plot
    true_weight_point = go.Scatter(
        x=[w0],
        y=[w1],
        mode="markers",
        marker=dict(
            color="white",
            size=10,
        ),
        name="Weights",
    )

    fig.add_trace(
        true_weight_point,
        row=1,
        col=2,
    )
    # appending input data and target to the dictionary
    data_for_graph = {"features": inputs_set.tolist(), "targets": outputs_set.tolist()}

    # Sampling weights from the posterior distribution to fit multiple lines on true data
    g_w0, g_w1 = model.param_posterior.rvs()

    r_w0, r_w1 = model.param_posterior.rvs()

    # Appending predicted values to the data frame
    g_predicted_value = (g_w0 + g_w1 * np.array(data_for_graph["features"]))
    r_predicted_value = (r_w0 + r_w1 * np.array(data_for_graph["features"]))

    df_for_graph = pd.DataFrame(data_for_graph)
    df_for_graph["green"] = g_predicted_value
    df_for_graph["red"] = r_predicted_value

    # plotting true data scatter plot
    fig2 = px.scatter(df_for_graph, x="features", y="targets")

    # plotting line fitting using predicted values
    fig3 = px.line(df_for_graph, x="features", y="green")
    fig3.update_traces(line_color="#5CFF5C")

    fig4 = px.line(df_for_graph, x="features", y="red")
    fig4.update_traces(line_color="#FF2E2E")

    fig.add_trace(fig2.data[0], row=1, col=1)
    fig.add_trace(fig3.data[0], row=1, col=1)
    fig.add_trace(fig4.data[0], row=1, col=1)

    st.write(fig)


# Start of code for UI dashboard

# start of code for Step 1 : Set parameters to generate true data

# documentation referred for UI component : https://docs.streamlit.io/library/api-reference/widgets/st.select_slider
st.sidebar.title('Interactive Bayesian Linear Regression')
st.sidebar.header('Step 1')
st.sidebar.subheader('Set parameters required to generate data')

weight0 = st.sidebar.slider(
    'Select true weight 0',
    -0.8, 0.8, 0.3, step=0.1)

weight1 = st.sidebar.slider(
    'Select true weight 1',
    -0.8, 0.8, 0.3, step=0.1)

no_of_data_points = st.sidebar.slider(
    'Select no of data points to simulate',
    20, 70, 50, step=5)

sd_data_noise = st.sidebar.slider(
    'Set standard deviation of Gaussian noise',
    0.0, 0.1, 0.03, step=0.01)

# End of code for Step 1


# Start of code for Step 2 : Set parameters for initial prior/posterior distribution
st.sidebar.header('Step 2')
st.sidebar.subheader('Set parameters required posterior update')

mean_11 = st.sidebar.slider(
    'Select value mean for weight 0',
    -0.9, 0.9, 0.2, step=0.1)

mean_12 = st.sidebar.slider(
    'Select mean for weight 1',
    -0.9, 0.9, 0.2, step=0.1)

cov_diag = st.sidebar.slider(
    'Select values for covariance',
    0.1, 0.9, 0.5, step=0.1)

# End of code for Step 2


# Start of code to view the parameters set from Step 1 & 2

column1, column2 = st.columns(2)
with column1:
    st.write('Value of weight 0', weight0)
    st.write('Value of weight 1:', weight1)
    st.write('Total number of data points', no_of_data_points)
    st.write('Standard Deviation of Gaussian noise', sd_data_noise)

with column2:
    st.write('Prior mean vector')
    # https: // docs.streamlit.io / library / api - reference / text / st.latex
    st.latex(r''' 
   
	\begin{bmatrix} ''' + rf''' {mean_11} & {mean_12} \\ ''' + r'''\end{bmatrix}
	
	''')

    st.write('Prior covariance matrix')

    st.latex(r''' 

    	\begin{bmatrix} ''' + rf''' {cov_diag} & 0 \\ 0 & {cov_diag} ''' + r'''\end{bmatrix}

    	''')

# End of code to view the parameters set from Step 1 & 2


# Initializing input and output data
input_data = np.random.uniform(-1, 1, no_of_data_points)
output = weight0 + weight1 * input_data + np.random.normal(0, sd_data_noise, len(input_data))

# On click event for button Generate Data
if st.button('Generate Data'):
    fig1 = px.scatter(x=input_data, y=output)
    st.write(fig1)


# function to update posterior and plot graphs
def start_training(model, input_seq, output_seq, w0, w1, data_point):
    model.update_posterior(inputs_set, outputs_set)
    # calling plot_graph function
    plot_graph(
        model,
        inputs=input_seq,
        outputs=output_seq,
        data_point_index=data_point,
        w0=w0,
        w1=w1,
    )


# On click event for button Start Training
if st.button('Start Training'):

    fig1 = px.scatter(x=input_data, y=output)
    st.write(fig1)

    # initializing prior distribution parameters
    prior_mean = np.array([mean_11, mean_12])
    prior_cov = np.array([[cov_diag, 0.0], [0.0, cov_diag]])

    # initializing posterior distribution
    posterior_distribution = PosteriorDistribution(prior_mean, prior_cov, 0.2)

    # list to iterate over the input data points incrementally
    data_point_intervals = range(1, no_of_data_points, 5)

    # iteratively updating the posterior distribution
    for point in data_point_intervals:
        diff = no_of_data_points - point

        inputs_set = input_data[0:point]
        outputs_set = output[0:point]
        start_training(posterior_distribution, inputs_set, outputs_set, weight0, weight1, point)
        if diff < 5:
            start_training(posterior_distribution, inputs_set, outputs_set, weight0, weight1, no_of_data_points)
