

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st

st.title("LR with Gradient Descent Visulizer")

m=st.slider("No. of Data entires     (more entries makes easier to converge)",10,100,30)
noise=st.slider("Noise factor (high noise means less linear trend)",0,20,5)
learning_rate=st.slider("Learning rate (high learning rate may overshoot solution)",0.01,1.0,0.1)


#Generate data
np.random.seed(42)
X=np.random.rand(m,1)
Y=6*X+9+np.random.randn(m,1)*noise

XM=np.c_[np.ones((m,1)),X]


def initilize():
    st.session_state.theta=np.random.rand(2,1)
    st.session_state.step=0
    st.session_state.thetas=[st.session_state.theta.copy()]

if "step" not in st.session_state:
    initilize()



col1,col2=st.columns(2)

with col1:
    if st.button("Next Iteration"):
        gradient=(2/m)*XM.T.dot(XM.dot(st.session_state.theta)-Y)
        st.session_state.theta-=learning_rate*gradient
        st.session_state.thetas.append(st.session_state.theta.copy())
        st.session_state.step+=1


with col2:
    if st.button("Reset"):
        initilize()



fig,ax=plt.subplots(figsize=(8,6))
ax.plot(X,Y,'o')
# for i in range(0,st.session_state.step):
#     color="red" if i==st.session_state.step else "grey"
#     alpha=1.0 if i==st.session_state.step else 0.2
#     lw=2 if i==st.session_state.step else 1
#     ax.plot(X,XM.dot(st.session_state.thetas[i]),color=color,alpha=alpha,linewidth=lw)

if st.session_state.step > 0:
    for i in range(st.session_state.step):
        color = "red" if i == st.session_state.step - 1 else "gray"
        alpha = 1.0 if i == st.session_state.step - 1 else 0.2
        lw = 2 if i == st.session_state.step - 1 else 1
        ax.plot(X, XM.dot(st.session_state.thetas[i]), color=color, alpha=alpha, linewidth=lw)

st.pyplot(fig)
theta=st.session_state.theta


st.markdown(f"### Current Iteration: ")
st.latex(f"Y={theta[1][0]:.2f}X+{theta[0][0]:.2f}")
st.write(f"Iteration number = {st.session_state.step}")


import streamlit as st

# Display the cost function using LaTeX

st.latex(r'Minimising :   J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2')
st.latex(r'Using  \theta := \theta - \frac{\alpha}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x^{(i)}')





    
